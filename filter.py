"""
Real-time plot + CSV logging for Shimmer EXG (ADS1292R) via pyshimmer,
with EMG envelope, cycle detection, and normalized intensity X.

Requirements:
  pip install pyserial pyshimmer matplotlib pandas

Usage:
  python shimmer_live_plot.py --port COM26 --out shimmer_exg_log.csv --window 10
"""

# ===================== Imports =====================
import argparse
import time
import signal
import sys
from datetime import datetime
from pathlib import Path
from collections import deque
import numpy as np
import queue

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from serial import Serial
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
# ===================================================


# --------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM26", help="Serial port (e.g., COM26, /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE, help="Baud rate")
    p.add_argument("--out", default="", help="CSV output path")
    p.add_argument("--window", type=float, default=10.0, help="Plot window length (seconds)")
    p.add_argument("--fps", type=float, default=30.0, help="Plot refresh rate (frames/sec)")
    p.add_argument("--env_lp_hz", type=float, default=4.0, help="Envelope LPF cutoff (Hz)")
    p.add_argument("--hpf_hz", type=float, default=20.0, help="High-pass cutoff (Hz)")
    p.add_argument("--lpf_hz", type=float, default=450.0, help="Low-pass cutoff (Hz)")
    p.add_argument("--min_period", type=float, default=0.30, help="Min cycle period (s)")
    p.add_argument("--max_period", type=float, default=1.50, help="Max cycle period (s)")
    p.add_argument("--x_clip", type=float, default=1.5, help="Clip X to this upper bound")
    p.add_argument("--rate", type=float, default=500.0, help="Estimated sample rate for buffer sizing (Hz)")
    return p.parse_args()


# --------------------- Simple 1st-order IIR helpers ---------------------
def lp_alpha(fc, dt):
    # Low-pass filter coefficient: y += a*(x - y)
    if fc <= 0:
        return 1.0
    rc = 1.0 / (2.0 * np.pi * fc)
    return dt / (rc + dt)

def hp_alpha(fc, dt):
    # High-pass filter coefficient: y[n] = a*(y[n-1] + x[n] - x[n-1])
    if fc <= 0:
        return 0.0
    rc = 1.0 / (2.0 * np.pi * fc)
    return rc / (rc + dt)


# --------------------- Real-time EMG Processor ---------------------
class RealtimeEMGProcessor:
    """
    Minimal real-time EMG processing chain:
      raw -> HPF(~20Hz) -> LPF(~450Hz) ~ band-pass
          -> full-wave rectification -> envelope LPF (~4Hz)

    Cycle detection on envelope:
      - dynamic threshold = 0.2 * moving P95
      - local maxima above threshold with refractory time (min_period)
    Generalized intensity X:
      - mean envelope per cycle / moving P95  (clipped to [0, x_clip])
    """
    def __init__(self, env_lp_hz=4.0, hpf_hz=20.0, lpf_hz=450.0,
                 min_period=0.30, max_period=1.50, x_clip=1.5, win_sec=10.0):
        self.env_lp_hz = env_lp_hz
        self.hpf_hz = hpf_hz
        self.lpf_hz = lpf_hz
        self.min_period = min_period
        self.max_period = max_period
        self.x_clip = x_clip
        self.win_sec = win_sec

        # internal states
        self.prev_t = None
        self.hp_y = 0.0
        self.hp_xprev = 0.0
        self.lp_y = 0.0
        self.env_y = 0.0

        self.env_history = deque()
        self.env_times = deque()

        self.last_peak_t = None
        self.last_val = None
        self.rising = False

        self.cycle_id = -1
        self.in_cycle = False
        self.cycle_env_sum = 0.0
        self.cycle_nsamp = 0

        self.X_series = deque()   # (t, X)
        self.peaks = deque()      # (t_peak, env_peak)

    def _update_dt(self, t):
        if self.prev_t is None:
            self.prev_t = t
            return None
        dt = max(1e-4, min(0.1, t - self.prev_t))
        self.prev_t = t
        return dt

    def _hp(self, x, dt):
        a = hp_alpha(self.hpf_hz, dt)
        y = a * (self.hp_y + x - self.hp_xprev)
        self.hp_y = y
        self.hp_xprev = x
        return y

    def _lp(self, x, dt):
        a = lp_alpha(self.lpf_hz, dt)
        self.lp_y += a * (x - self.lp_y)
        return self.lp_y

    def _env_lp(self, x, dt):
        a = lp_alpha(self.env_lp_hz, dt)
        self.env_y += a * (x - self.env_y)
        return self.env_y

    def _prune_history(self, tnow):
        while self.env_times and (tnow - self.env_times[0] > self.win_sec):
            self.env_times.popleft()
            self.env_history.popleft()

    def _moving_p95(self):
        if not self.env_history:
            return None
        vals = np.fromiter((v for v in self.env_history), dtype=float)
        return float(np.percentile(vals, 95))

    def process(self, t, raw):
        """
        Returns dict with:
          filtered, envelope, is_peak(bool), cycle_id, X
        """
        dt = self._update_dt(t)
        if dt is None:
            return dict(filtered=raw, envelope=0.0, is_peak=False, cycle_id=-1, X=None)

        # band-pass filtering
        x_hp = self._hp(raw, dt)
        x_bp = self._lp(x_hp, dt)

        # rectification and envelope
        rect = abs(x_bp)
        env = self._env_lp(rect, dt)

        # keep history
        self.env_times.append(t)
        self.env_history.append(env)
        self._prune_history(t)

        # dynamic threshold
        p95 = self._moving_p95()
        thr = 0.2 * p95 if (p95 is not None and p95 > 0) else 0.0

        is_peak = False
        X_out = None

        # peak detection
        if self.last_val is None:
            self.last_val = env

        if env > self.last_val:
            self.rising = True
        elif env < self.last_val:
            if self.rising and self.last_val > thr:
                t_candidate = t
                ok_period = True
                if self.last_peak_t is not None:
                    period = t_candidate - self.last_peak_t
                    ok_period = (self.min_period <= period <= self.max_period)
                if ok_period:
                    is_peak = True
                    self.last_peak_t = t_candidate
                    self.peaks.append((t_candidate, self.last_val))

                    # finalize last cycle
                    if self.in_cycle and self.cycle_nsamp > 0 and p95 and p95 > 0:
                        cycle_mean = self.cycle_env_sum / self.cycle_nsamp
                        X = min(self.x_clip, max(0.0, cycle_mean / p95))
                        X_out = X
                        self.X_series.append((t_candidate, X))

                    # start new cycle
                    self.cycle_id += 1
                    self.in_cycle = True
                    self.cycle_env_sum = 0.0
                    self.cycle_nsamp = 0
            self.rising = False

        if self.in_cycle:
            self.cycle_env_sum += env
            self.cycle_nsamp += 1

        self.last_val = env

        return dict(filtered=x_bp, envelope=env, is_peak=is_peak, cycle_id=self.cycle_id, X=X_out)


# --------------------- Data buffer ---------------------
class RingBuffer:
    """Keeps last W seconds of data for plotting, full log for CSV."""
    def __init__(self, window_sec=10.0, est_rate_hz=500.0):
        # keep at least 2x window to avoid 'left gap' when timing fluctuates
        maxlen = max(200, int(window_sec * est_rate_hz * 2.0))
        self.t_rel = deque(maxlen=maxlen)
        self.ch1   = deque(maxlen=maxlen)
        self.ch2   = deque(maxlen=maxlen)
        self.filt1 = deque(maxlen=maxlen)
        self.env1  = deque(maxlen=maxlen)
        self.x_t   = deque(maxlen=maxlen)
        self.x_val = deque(maxlen=maxlen)
        self.peak_ts = deque(maxlen=2000)
        self.csv_rows = []
        self.t0 = None

    def add(self, t_abs, status, ch1, ch2, ts_counter, filt=None, env=None, X=None, is_peak=False):
        if self.t0 is None:
            self.t0 = t_abs
        t_rel = t_abs - self.t0

        self.t_rel.append(t_rel)
        self.ch1.append(ch1)
        self.ch2.append(ch2)
        self.filt1.append(filt if filt is not None else np.nan)
        self.env1.append(env if env is not None else np.nan)

        if is_peak:
            self.peak_ts.append(t_rel)
        if X is not None:
            self.x_t.append(t_rel)
            self.x_val.append(X)

        self.csv_rows.append(
            {
                "t_abs_unix": t_abs,
                "t_rel_s": t_rel,
                "exg_status": status,
                "exg_ch1_24bit": ch1,
                "exg_ch2_24bit": ch2,
                "device_timestamp": ts_counter,
                "emg_ch1_filtered": filt if filt is not None else "",
                "emg_ch1_envelope": env if env is not None else "",
                "emg_cycle_peak": 1 if is_peak else 0,
                "emg_X": X if X is not None else "",
            }
        )


# --------------------- Shimmer handler ---------------------
q = queue.Queue(maxsize=50000)

def make_handler(q_out: queue.Queue):
    def handler(pkt: DataPacket) -> None:
        vals = pkt._values
        item = {
            "t_abs": time.time(),
            "ts_counter": vals.get(EChannelType.TIMESTAMP, None),
            "status": vals.get(EChannelType.EXG_ADS1292R_1_STATUS, None),
            "ch1": vals.get(EChannelType.EXG_ADS1292R_1_CH1_24BIT, None),
            "ch2": vals.get(EChannelType.EXG_ADS1292R_1_CH2_24BIT, None),
        }
        try:
            q_out.put_nowait(item)
        except queue.Full:
            try:
                q_out.get_nowait()
            except queue.Empty:
                pass
            try:
                q_out.put_nowait(item)
            except queue.Full:
                pass
    return handler


# --------------------- Main ---------------------
def main():
    args = parse_args()

    # Output path
    if args.out:
        out_csv = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = Path(f"shimmer_exg_{stamp}.csv")

    q_local = queue.Queue(maxsize=10000)
    rb = RingBuffer(window_sec=args.window, est_rate_hz=args.rate)
    proc = RealtimeEMGProcessor(
        env_lp_hz=args.env_lp_hz,
        hpf_hz=args.hpf_hz,
        lpf_hz=args.lpf_hz,
        min_period=args.min_period,
        max_period=args.max_period,
        x_clip=args.x_clip,
        win_sec=args.window
    )

    running = {"ok": True}
    def stop(*_): running["ok"] = False
    signal.signal(signal.SIGINT, stop)

    ser = None
    dev = None

    try:
        print(f"Opening {args.port} @ {args.baud} ...")
        ser = Serial(args.port, args.baud, timeout=1)
        dev = ShimmerBluetooth(ser)
        dev.initialize()
        name = dev.get_device_name()
        print(f"Connected: {name}")

        dev.add_stream_callback(make_handler(q_local))
        dev.start_streaming()
        print("Streaming started. Close the plot window or press Ctrl+C to stop.")

        # --------------------- Matplotlib live plot ---------------------
        plt.style.use("default")
        fig, (ax0, ax1, ax2) = plt.subplots(
            3, 1, figsize=(9.5, 8.2), sharex=True,
            gridspec_kw={"height_ratios": [1.2, 1.2, 0.8], "hspace": 0.12}
        )

        # (1) Raw & Filtered
        raw_line, = ax0.plot([], [], label="CH1 Raw (24-bit)")
        filt_line, = ax0.plot([], [], label="CH1 Filtered (BP)")
        ax0.set_ylabel("ADC units")
        ax0.set_title("Raw vs Filtered (CH1)")
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="upper right")

        # (2) Envelope & Peaks
        env_line, = ax1.plot([], [], label="Envelope (LPF)")
        ax1.set_ylabel("Envelope")
        ax1.set_title("Envelope & Activation Cycles (CH1)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper right")
        x_text = ax1.text(0.99, 0.95, "X: --", transform=ax1.transAxes,
                          ha="right", va="top", fontsize=10,
                          bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        # (3) X values
        X_scatter = ax2.plot([], [], "o-", markersize=3, label="X (per cycle)")[0]
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("X")
        ax2.set_ylim(0.0, max(1.2, args.x_clip))
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")

        vlines = []

        def clear_vlines():
            nonlocal vlines
            for ln in vlines:
                ln.remove()
            vlines = []

        def on_close(_event): stop()
        fig.canvas.mpl_connect('close_event', on_close)

        # --------------------- Update function (with window slicing) ---------------------
        def update(_frame):
            drained = 0
            while True:
                try:
                    item = q_local.get_nowait()
                except queue.Empty:
                    break

                t_abs = item["t_abs"]
                ch1 = item["ch1"]
                ch2 = item["ch2"]
                if ch1 is None:
                    continue

                res = proc.process(t_abs, float(ch1))
                rb.add(
                    t_abs=t_abs,
                    status=item["status"],
                    ch1=ch1, ch2=(ch2 if ch2 is not None else 0),
                    ts_counter=item["ts_counter"],
                    filt=res["filtered"],
                    env=res["envelope"],
                    X=res["X"],
                    is_peak=res["is_peak"]
                )
                drained += 1

            if drained > 0 and len(rb.t_rel) > 1:
                # desired window: last args.window seconds
                tmax = rb.t_rel[-1]
                t_desired_left = max(0.0, tmax - args.window)

                # if buffer doesn't contain data that far back, clamp to earliest available
                tmin_available = rb.t_rel[0]
                t_left = max(t_desired_left, tmin_available)
                t_right = tmax  # keep right edge at the newest sample

                # Slice once with a single index range to keep all series aligned
                t_all = np.asarray(rb.t_rel)
                i0 = int(np.searchsorted(t_all, t_left, side="left"))
                t_win = t_all[i0:]

                y_raw = np.asarray(rb.ch1)[i0:]
                y_flt = np.asarray(rb.filt1)[i0:]
                y_env = np.asarray(rb.env1)[i0:]

                # (1) Raw & Filtered
                raw_line.set_data(t_win, y_raw)
                filt_line.set_data(t_win, y_flt)
                if y_raw.size:
                    y_all = np.concatenate([y_raw, y_flt[np.isfinite(y_flt)]])
                    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
                    margin = max(10.0, 0.1 * (y_max - y_min + 1.0))
                    ax0.set_ylim(y_min - margin, y_max + margin)

                # (2) Envelope
                env_line.set_data(t_win, y_env)
                if np.isfinite(y_env).any():
                    ev = y_env[np.isfinite(y_env)]
                    ev_min, ev_max = float(np.min(ev)), float(np.max(ev))
                    emargin = 0.05 * (ev_max - ev_min + 1e-6)
                    ax1.set_ylim(max(0.0, ev_min - emargin), ev_max + emargin)

                # Peaks (only those inside the visible window)
                for ln in list(vlines): ln.remove()
                vlines.clear()
                for tp in rb.peak_ts:
                    if t_left <= tp <= t_right:
                        vlines.append(ax1.axvline(tp, color="k", alpha=0.25,
                                                  linestyle="--", linewidth=1.0))

                # (3) X points (only inside the visible window)
                x_t = np.asarray(rb.x_t)
                x_v = np.asarray(rb.x_val)
                if x_t.size:
                    j0 = int(np.searchsorted(x_t, t_left, side="left"))
                    X_scatter.set_data(x_t[j0:], x_v[j0:])
                    if x_v[j0:].size:
                        ax2.set_ylim(0.0, max(args.x_clip, 1.1 * float(np.max(x_v[j0:]))))
                        x_text.set_text(f"X: {float(x_v[j0:][-1]):.2f}")
                    else:
                        x_text.set_text("X: --")
                else:
                    X_scatter.set_data([], [])
                    x_text.set_text("X: --")

                # Apply shared x-limits so the visible window is always 'full' and scrolls
                ax0.set_xlim(t_left, max(t_left + 1e-3, t_right))

            return raw_line, filt_line, env_line, X_scatter

        interval_ms = int(1000.0 / max(1.0, args.fps))
        ani = FuncAnimation(fig, update, interval=interval_ms, blit=False)

        while running["ok"]:
            plt.pause(0.01)

    except Exception as e:
        sys.stderr.write(f"\n[Error] {e}\n")

    finally:
        print("\nStopping...")
        try:
            if dev is not None:
                dev.stop_streaming()
                dev.shutdown()
        except Exception as e:
            sys.stderr.write(f"[warn] shutdown: {e}\n")
        try:
            if ser is not None and ser.is_open:
                ser.close()
        except Exception as e:
            sys.stderr.write(f"[warn] serial close: {e}\n")

        try:
            if len(rb.csv_rows) > 0:
                df = pd.DataFrame(rb.csv_rows)
                df.to_csv(out_csv, index=False)
                print(f"Saved CSV: {out_csv.resolve()}")
            else:
                print("No samples captured; CSV not written.")
        except Exception as e:
            sys.stderr.write(f"[Error] writing CSV: {e}\n")

        print("Done.")


if __name__ == "__main__":
    main()
