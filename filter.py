"""
Real-time plot (+ optional CSV logging) for Shimmer EXG (ADS1292R) via pyshimmer,
with EMG envelope, cycle detection, normalized intensity X,
and activation frequency/peaks visualization.

Requirements:
  pip install pyserial pyshimmer matplotlib pandas

Usage:
  # CSVなし（デフォルト）
  python shimmer_live_plot.py --port COM26 --window 10

  # CSVあり
  python shimmer_live_plot.py --port COM26 --csv --out shimmer_exg_log.csv --window 10
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

    # CSV出力のON/OFF（デフォルトOFF）
    p.add_argument("--csv", action="store_true", help="Enable CSV logging (default: off)")
    p.add_argument("--out", default="", help="CSV output path (used only when --csv is set)")

    p.add_argument("--window", type=float, default=10.0, help="Plot window length (seconds)")
    p.add_argument("--fps", type=float, default=30.0, help="Plot refresh rate (frames/sec)")
    p.add_argument("--env_lp_hz", type=float, default=4.0, help="Envelope LPF cutoff (Hz)")
    p.add_argument("--hpf_hz", type=float, default=20.0, help="High-pass cutoff (Hz)")
    p.add_argument("--lpf_hz", type=float, default=450.0, help="Low-pass cutoff (Hz)")
    p.add_argument("--min_period", type=float, default=0.30, help="Min cycle period (s)")
    p.add_argument("--x_clip", type=float, default=1.5, help="Clip X to this upper bound")
    p.add_argument("--rate", type=float, default=500.0, help="Estimated sample rate for buffer sizing (Hz)")
    p.add_argument("--stats_win", type=float, default=3.0, help="Short stats window for adaptive thresholds (s)")
    p.add_argument("--peak_ema_tau", type=float, default=2.0, help="Time constant (s) for EMA of accepted peak amplitudes")
    p.add_argument("--thr_on_k", type=float, default=0.35, help="ON threshold scale vs ref (dimensionless)")
    p.add_argument("--thr_off_k", type=float, default=0.25, help="OFF threshold scale vs ref (dimensionless)")
    p.add_argument("--min_prom_k", type=float, default=0.02, help="Min peak prominence as fraction of ref")
    p.add_argument("--min_width_s", type=float, default=0.10, help="Minimum burst width (s)")
    p.add_argument("--use_iqr_thr", action="store_true", help="Use median+IQR instead of P95 for ref")
    p.add_argument("--freq_k_peaks", type=int, default=5, help="Number of recent intervals to median for cadence")
    return p.parse_args()


# --------------------- Simple 1st-order IIR helpers ---------------------
def lp_alpha(fc, dt):
    if fc <= 0:
        return 1.0
    rc = 1.0 / (2.0 * np.pi * fc)
    return dt / (rc + dt)

def hp_alpha(fc, dt):
    if fc <= 0:
        return 0.0
    rc = 1.0 / (2.0 * np.pi * fc)
    return rc / (rc + dt)


# --------------------- Real-time EMG Processor ---------------------
class RealtimeEMGProcessor:
    """
    Real-time EMG chain with adaptive thresholds:
      raw -> HPF(~20Hz) -> LPF(~450Hz) -> rect -> envelope LPF(~4Hz)
    Hysteretic burst detector with:
      - short-window stats (P95 or median+IQR) updated every sample
      - EMA of accepted peak amplitudes (fast adaptation)
      - dual thresholds (ON/OFF), min width, prominence, min-period constraint
    X = mean envelope per accepted burst / (normalization ref), clipped.

    Cadence (activation frequency) is computed as the inverse of the
    median of the last K accepted inter-peak intervals (K = freq_k_peaks).
    """

    def __init__(self, env_lp_hz=4.0, hpf_hz=20.0, lpf_hz=450.0,
                 min_period=0.30, x_clip=1.5, win_sec=10.0,
                 stats_win=3.0, peak_ema_tau=2.0,
                 thr_on_k=0.35, thr_off_k=0.25, min_prom_k=0.10,
                 min_width_s=0.10, use_iqr_thr=False, freq_k_peaks=5):
        # filter params
        self.env_lp_hz = env_lp_hz
        self.hpf_hz = hpf_hz
        self.lpf_hz = lpf_hz

        # thresholds / constraints
        self.min_period = float(min_period)
        self.x_clip = x_clip
        self.win_sec = win_sec
        self.stats_win = stats_win
        self.use_iqr_thr = use_iqr_thr
        self.thr_on_k = thr_on_k
        self.thr_off_k = thr_off_k
        self.min_prom_k = min_prom_k
        self.min_width_s = min_width_s
        self.peak_ema_tau = peak_ema_tau
        self.freq_k_peaks = max(2, int(freq_k_peaks))

        # filter states
        self.prev_t = None
        self.hp_y = 0.0
        self.hp_xprev = 0.0
        self.lp_y = 0.0
        self.env_y = 0.0

        # histories
        self.env_times = deque()
        self.env_history = deque()
        self.env_times_short = deque()
        self.env_history_short = deque()

        # detection states
        self.active = False
        self.burst_start_t = None
        self.peak_val = -np.inf
        self.peak_t = None
        self.last_peak_t = None
        self.last_val = None

        # per-cycle accumulation
        self.cycle_id = -1
        self.in_cycle = False
        self.cycle_env_sum = 0.0
        self.cycle_nsamp = 0

        # outputs
        self.X_series = deque()
        self.peaks = deque()          # (t_peak, peak_val)
        self.peak_times = deque()     # t_peak only (for cadence)
        self.last_activation_hz = None
        self.peak_amp_ema = None

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

    def _prune_histories(self, tnow):
        while self.env_times and (tnow - self.env_times[0] > self.win_sec):
            self.env_times.popleft(); self.env_history.popleft()
        while self.env_times_short and (tnow - self.env_times_short[0] > self.stats_win):
            self.env_times_short.popleft(); self.env_history_short.popleft()

    def _short_stats(self):
        if not self.env_history_short:
            return None, None
        vals = np.fromiter((v for v in self.env_history_short), dtype=float)
        med = float(np.median(vals))
        if self.use_iqr_thr:
            q25, q75 = np.percentile(vals, [25, 75])
            ref = float(max(1e-9, q75 - q25))
        else:
            ref = float(np.percentile(vals, 95))
        return ref, med

    @staticmethod
    def _ema_update(ema, value, dt, tau):
        if tau <= 1e-6:
            return float(value)
        alpha = 1.0 - np.exp(-dt / tau)
        return (1.0 - alpha) * (ema if ema is not None else value) + alpha * value

    def _update_cadence(self):
        """Update last_activation_hz from median of recent inter-peak intervals."""
        if len(self.peak_times) < 2:
            self.last_activation_hz = None
            return
        times = np.array(self.peak_times, dtype=float)
        if times.size < 2:
            self.last_activation_hz = None
            return
        # 直近のK-1間隔を使用し、min_period以上のものだけで中央値
        diffs = np.diff(times)[-max(1, self.freq_k_peaks - 1):]
        diffs = diffs[np.isfinite(diffs)]
        diffs = diffs[diffs >= self.min_period]
        if diffs.size == 0:
            self.last_activation_hz = None
            return
        med_T = float(np.median(diffs))
        if med_T > 1e-6:
            self.last_activation_hz = 1.0 / med_T
        else:
            self.last_activation_hz = None

    def process(self, t, raw):
        dt = self._update_dt(t)
        if dt is None:
            return dict(filtered=float(raw), envelope=0.0,
                        is_peak=False, cycle_id=self.cycle_id,
                        X=None, activation_hz=None, peak_val=None)

        # filtering
        x_hp = self._hp(raw, dt)
        x_bp = self._lp(x_hp, dt)
        rect = abs(x_bp)
        env = self._env_lp(rect, dt)

        # histories
        self.env_times.append(t); self.env_history.append(env)
        self.env_times_short.append(t); self.env_history_short.append(env)
        self._prune_histories(t)

        ref_s, base = self._short_stats()
        if ref_s is None or ref_s <= 0.0:
            self.last_val = env
            return dict(filtered=x_bp, envelope=env,
                        is_peak=False, cycle_id=self.cycle_id,
                        X=None, activation_hz=self.last_activation_hz, peak_val=None)

        ref = max(1e-9, 0.5 * ref_s + 0.5 * (self.peak_amp_ema or ref_s))
        thr_on = base + self.thr_on_k * ref
        thr_off = base + self.thr_off_k * ref

        is_peak = False; X_out = None

        if not self.active:
            if env >= thr_on:
                self.active = True
                self.burst_start_t = t
                self.peak_val = env; self.peak_t = t
                self.cycle_id += 1
                self.in_cycle = True
                self.cycle_env_sum = env; self.cycle_nsamp = 1
        else:
            if env > self.peak_val:
                self.peak_val = env; self.peak_t = t
            self.cycle_env_sum += env; self.cycle_nsamp += 1

            if env <= thr_off:
                # 判定：幅とプロミネンスに加えて「min_period以上のデッドタイム」
                width_ok = (t - self.burst_start_t) >= self.min_width_s
                prom_ok = (self.peak_val - base) >= (self.min_prom_k * ref)
                period_ok = True
                if self.last_peak_t is not None:
                    period_ok = (self.peak_t - self.last_peak_t) >= self.min_period

                if width_ok and prom_ok and period_ok:
                    is_peak = True
                    self.peaks.append((self.peak_t, self.peak_val))
                    self.peak_times.append(self.peak_t)
                    amp_above = max(1e-9, self.peak_val - base)
                    self.peak_amp_ema = self._ema_update(self.peak_amp_ema, amp_above, dt, self.peak_ema_tau)

                    if self.in_cycle and self.cycle_nsamp > 0 and ref > 0:
                        cycle_mean = self.cycle_env_sum / self.cycle_nsamp
                        X = min(self.x_clip, max(0.0, cycle_mean / ref))
                        X_out = X; self.X_series.append((self.peak_t, X))

                    # 更新：直近のピーク列から中央値周期でcadence算出
                    self._update_cadence()
                    self.last_peak_t = self.peak_t

                self.active = False; self.in_cycle = False
                self.cycle_env_sum = 0.0; self.cycle_nsamp = 0

        self.last_val = env
        return dict(filtered=x_bp, envelope=env,
                    is_peak=is_peak, cycle_id=self.cycle_id,
                    X=X_out, activation_hz=self.last_activation_hz,
                    peak_val=(self.peak_val if is_peak else None))


# --------------------- Data buffer ---------------------
class RingBuffer:
    def __init__(self, window_sec=10.0, est_rate_hz=500.0, csv_enabled=False):
        maxlen = max(200, int(window_sec * est_rate_hz * 2.0))
        self.t_rel = deque(maxlen=maxlen)
        self.ch1 = deque(maxlen=maxlen)
        self.ch2 = deque(maxlen=maxlen)
        self.filt1 = deque(maxlen=maxlen)
        self.env1 = deque(maxlen=maxlen)
        self.x_t = deque(maxlen=maxlen)
        self.x_val = deque(maxlen=maxlen)
        self.peak_ts = deque(maxlen=2000)
        self.peak_xy = deque(maxlen=2000)
        self.csv_enabled = csv_enabled
        self.csv_rows = [] if csv_enabled else None
        self.t0 = None

    def add(self, t_abs, status, ch1, ch2, ts_counter,
            filt=None, env=None, X=None, is_peak=False,
            activation_hz=None, peak_val=None):
        if self.t0 is None: self.t0 = t_abs
        t_rel = t_abs - self.t0
        self.t_rel.append(t_rel); self.ch1.append(ch1); self.ch2.append(ch2)
        self.filt1.append(filt if filt is not None else np.nan)
        self.env1.append(env if env is not None else np.nan)
        if is_peak:
            self.peak_ts.append(t_rel)
            if peak_val is not None: self.peak_xy.append((t_rel, float(peak_val)))
        if X is not None:
            self.x_t.append(t_rel); self.x_val.append(float(X))

        if self.csv_enabled:
            self.csv_rows.append({
                "t_abs_unix": t_abs, "t_rel_s": t_rel, "exg_status": status,
                "exg_ch1_24bit": ch1, "exg_ch2_24bit": ch2, "device_timestamp": ts_counter,
                "emg_ch1_filtered": filt if filt is not None else "",
                "emg_ch1_envelope": env if env is not None else "",
                "emg_cycle_peak": 1 if is_peak else 0,
                "emg_X": X if X is not None else "",
                "emg_activation_hz": activation_hz if activation_hz is not None else "",
                "emg_peak_val": peak_val if peak_val is not None else ""
            })


# --------------------- Shimmer handler ---------------------
def make_handler(q_out: queue.Queue):
    def handler(pkt: DataPacket) -> None:
        vals = pkt._values
        item = {"t_abs": time.time(),
                "ts_counter": vals.get(EChannelType.TIMESTAMP, None),
                "status": vals.get(EChannelType.EXG_ADS1292R_1_STATUS, None),
                "ch1": vals.get(EChannelType.EXG_ADS1292R_1_CH1_24BIT, None),
                "ch2": vals.get(EChannelType.EXG_ADS1292R_1_CH2_24BIT, None)}
        try:
            q_out.put_nowait(item)
        except queue.Full:
            # drop oldest, keep newest
            try: q_out.get_nowait()
            except queue.Empty: pass
            try: q_out.put_nowait(item)
            except queue.Full: pass
    return handler


# --------------------- Main ---------------------
def main():
    args = parse_args()

    # CSVファイル名（--csv のときだけ意味あり）
    out_csv = None
    if args.csv:
        out_csv = Path(args.out) if args.out else Path(f"shimmer_exg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    q_local = queue.Queue(maxsize=10000)
    rb = RingBuffer(window_sec=args.window, est_rate_hz=args.rate, csv_enabled=args.csv)
    proc = RealtimeEMGProcessor(env_lp_hz=args.env_lp_hz, hpf_hz=args.hpf_hz,
                                lpf_hz=args.lpf_hz, min_period=args.min_period,
                                x_clip=args.x_clip, win_sec=args.window,
                                stats_win=args.stats_win, peak_ema_tau=args.peak_ema_tau,
                                thr_on_k=args.thr_on_k, thr_off_k=args.thr_off_k,
                                min_prom_k=args.min_prom_k, min_width_s=args.min_width_s,
                                use_iqr_thr=args.use_iqr_thr, freq_k_peaks=args.freq_k_peaks)

    running = {"ok": True}
    def stop(*_): running["ok"] = False
    signal.signal(signal.SIGINT, stop)

    ser = None; dev = None
    try:
        print(f"Opening {args.port} @ {args.baud} ...")
        ser = Serial(args.port, args.baud, timeout=1)
        dev = ShimmerBluetooth(ser); dev.initialize()
        print(f"Connected: {dev.get_device_name()}")
        dev.add_stream_callback(make_handler(q_local)); dev.start_streaming()
        print("Streaming started. Close the plot window or press Ctrl+C to stop.")
        if args.csv:
            print(f"[CSV] Logging enabled -> {out_csv}")

        # ----- plot -----
        plt.style.use("default")
        fig, (ax0, ax1, ax2) = plt.subplots(
            3, 1, figsize=(9.5, 8.2), sharex=True,
            gridspec_kw={"height_ratios": [1.2, 1.2, 0.8], "hspace": 0.12}
        )

        raw_line, = ax0.plot([], [], label="CH1 Raw")
        filt_line, = ax0.plot([], [], label="CH1 Filtered")
        ax0.set_ylabel("ADC units"); ax0.set_title("Raw vs Filtered (CH1)")
        ax0.grid(True); ax0.legend()

        env_line, = ax1.plot([], [], label="Envelope")
        # ピークは点のみ（シンプル）
        peak_markers = ax1.plot([], [], "o", ms=5, label="Activation peaks")[0]
        ax1.set_ylabel("Envelope"); ax1.set_title("Envelope & Peaks")
        ax1.grid(True); ax1.legend()

        x_text = ax1.text(0.99, 0.95, "X: --", transform=ax1.transAxes, ha="right", va="top")
        f_text = ax1.text(0.99, 0.85, "f_act: -- Hz", transform=ax1.transAxes, ha="right", va="top")

        X_scatter = ax2.plot([], [], "o-", ms=3, label="X")[0]
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("X")
        ax2.grid(True); ax2.legend()

        # Envelope ylim smoothing state
        ylim_state = {"ev_min": None, "ev_max": None}

        def on_close(_): stop()
        fig.canvas.mpl_connect('close_event', on_close)

        def update(_frame):
            drained = 0
            max_drain = 4000  # safety cap for per-frame work
            while drained < max_drain:
                try:
                    item = q_local.get_nowait()
                except queue.Empty:
                    break
                t_abs = item["t_abs"]; ch1 = item["ch1"]; ch2 = item["ch2"]
                if ch1 is None:
                    continue
                res = proc.process(t_abs, float(ch1))
                rb.add(t_abs, item["status"], ch1, ch2 or 0, item["ts_counter"],
                       filt=res["filtered"], env=res["envelope"], X=res["X"],
                       is_peak=res["is_peak"], activation_hz=res["activation_hz"],
                       peak_val=res.get("peak_val"))
                drained += 1

            if drained > 0 and len(rb.t_rel) > 1:
                tmax = rb.t_rel[-1]
                t_left = max(0.0, tmax - args.window)
                t_right = tmax

                t_all = np.asarray(rb.t_rel)
                i0 = int(np.searchsorted(t_all, t_left))
                t_win = t_all[i0:]
                y_raw = np.asarray(rb.ch1)[i0:]
                y_flt = np.asarray(rb.filt1)[i0:]
                y_env = np.asarray(rb.env1)[i0:]

                # raw / filtered
                raw_line.set_data(t_win, y_raw); filt_line.set_data(t_win, y_flt)
                if y_raw.size:
                    y_all = np.concatenate([y_raw, y_flt[np.isfinite(y_flt)]])
                    if y_all.size:
                        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
                        pad = 0.1 * (abs(y_max - y_min) + 1.0)
                        ax0.set_ylim(y_min - pad, y_max + pad)

                # envelope w/ smoothed ylim
                env_line.set_data(t_win, y_env)
                if np.isfinite(y_env).any():
                    ev = y_env[np.isfinite(y_env)]
                    ev_min_raw, ev_max_raw = float(np.min(ev)), float(np.max(ev))
                    pad = 0.05 * (abs(ev_max_raw - ev_min_raw) + 1e-6)
                    target_min = max(0.0, ev_min_raw - pad)
                    target_max = ev_max_raw + pad
                    if ylim_state["ev_min"] is None:
                        ylim_state["ev_min"], ylim_state["ev_max"] = target_min, target_max
                    else:
                        alpha = 0.2
                        ylim_state["ev_min"] = (1 - alpha) * ylim_state["ev_min"] + alpha * target_min
                        ylim_state["ev_max"] = (1 - alpha) * ylim_state["ev_max"] + alpha * target_max
                    ax1.set_ylim(ylim_state["ev_min"], ylim_state["ev_max"])

                # peak markers (points only)
                px, py = [], []
                for tt, vv in rb.peak_xy:
                    if t_left <= tt <= t_right:
                        px.append(tt); py.append(vv)
                peak_markers.set_data(px, py)

                # X scatter
                x_t = np.asarray(rb.x_t); x_v = np.asarray(rb.x_val)
                if x_t.size:
                    j0 = int(np.searchsorted(x_t, t_left))
                    X_scatter.set_data(x_t[j0:], x_v[j0:])
                    if x_v[j0:].size:
                        ax2.set_ylim(0.0, max(args.x_clip, 1.1 * float(np.max(x_v[j0:]))))
                        x_text.set_text(f"X: {x_v[j0:][-1]:.2f}")
                # cadence text（逐次更新）
                if proc.last_activation_hz is not None:
                    f_text.set_text(f"f_act: {proc.last_activation_hz:.2f} Hz")
                else:
                    f_text.set_text("f_act: -- Hz")

                # 明示的に3軸のxlimを同期
                for ax in (ax0, ax1, ax2):
                    ax.set_xlim(t_left, max(t_left + 1e-3, t_right))

                return raw_line, filt_line, env_line, X_scatter, peak_markers

        ani = FuncAnimation(fig, update, interval=int(1000 / max(1, args.fps)),
                            blit=False, cache_frame_data=False)
        while running["ok"]:
            plt.pause(0.01)

    finally:
        print("\nStopping...")
        try:
            if dev is not None:
                dev.stop_streaming(); dev.shutdown()
        except Exception as e:
            print(f"Device stop error: {e}")
        try:
            if ser is not None:
                ser.close()
        except Exception as e:
            print(f"Serial close error: {e}")

        if args.csv and rb.csv_rows and len(rb.csv_rows) > 0:
            try:
                pd.DataFrame(rb.csv_rows).to_csv(out_csv, index=False)
                print(f"Saved CSV: {out_csv.resolve()}")
            except Exception as e:
                print(f"CSV save error: {e}")
        else:
            if args.csv:
                print("CSV enabled but no samples captured.")
            else:
                print("CSV logging disabled (use --csv to enable).")
        print("Done.")


if __name__ == "__main__":
    main()
