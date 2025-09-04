"""
Real-time plot + CSV logging for Shimmer EXG (ADS1292R) via pyshimmer.

Requirements:
  pip install pyserial pyshimmer matplotlib pandas

Usage:
  python shimmer_live_plot.py --port COM26 --out shimmer_exg_log.csv --window 10
    --port    : serial port (default COM26)
    --out     : output CSV path (default: auto timestamped)
    --window  : plot window in seconds (default 10)
"""


# ===================== Imports (place these at the very top) =====================
# Standard library
import argparse
import time
import signal
import sys
from datetime import datetime
from pathlib import Path
from collections import deque
import queue  # <- queue must be imported here, at top-level (not inside a function)

# Third-party
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from serial import Serial
from pyshimmer import ShimmerBluetooth, DEFAULT_BAUDRATE, DataPacket, EChannelType
# ================================================================================


# --------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="COM26", help="Serial port (e.g., COM26, /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUDRATE, help="Baud rate")
    p.add_argument("--out", default="", help="CSV output path")
    p.add_argument("--window", type=float, default=10.0, help="Plot window length (seconds)")
    p.add_argument("--fps", type=float, default=30.0, help="Plot refresh rate (frames/sec)")
    return p.parse_args()

# --------------------- Data structures ---------------------
class RingBuffer:
    """Keeps last W seconds of data for plotting, unbounded log for CSV."""
    def __init__(self, window_sec=10.0, est_rate_hz=500.0):
        maxlen = max(100, int(window_sec * est_rate_hz))
        self.t_rel = deque(maxlen=maxlen)   # relative time (s) for plotting
        self.ch1   = deque(maxlen=maxlen)
        self.ch2   = deque(maxlen=maxlen)

        # For CSV, keep full log:
        self.csv_rows = []

        self.t0 = None

    def add(self, t_abs, status, ch1, ch2, ts_counter):
        if self.t0 is None:
            self.t0 = t_abs
        t_rel = t_abs - self.t0

        self.t_rel.append(t_rel)
        self.ch1.append(ch1)
        self.ch2.append(ch2)

        self.csv_rows.append(
            {
                "t_abs_unix": t_abs,
                "t_rel_s": t_rel,
                "exg_status": status,
                "exg_ch1_24bit": ch1,
                "exg_ch2_24bit": ch2,
                "device_timestamp": ts_counter,
            }
        )

# --------------------- Shimmer handler ---------------------
# Add import for queue
import queue

# Increase queue capacity to avoid blocking
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
            # Non-blocking insert into queue
            q_out.put_nowait(item)
        except queue.Full:
            # If full, drop the oldest item and push the new one
            try:
                q_out.get_nowait()
            except queue.Empty:
                pass
            try:
                q_out.put_nowait(item)
            except queue.Full:
                # If still full, discard the item (avoid blocking handler thread)
                pass
    return handler

# (Optional) Increase update frequency for smoother plotting
# Example: run with --fps 45 or set default to 45


# --------------------- Main ---------------------
def main():
    args = parse_args()

    # Output path
    if args.out:
        out_csv = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = Path(f"shimmer_exg_{stamp}.csv")

    # Thread-safe queue from handler -> plot loop
    q = queue.Queue(maxsize=10000)
    rb = RingBuffer(window_sec=args.window, est_rate_hz=500.0)

    # Graceful shutdown flag
    running = {"ok": True}

    def stop(*_):
        running["ok"] = False

    # Register Ctrl+C
    signal.signal(signal.SIGINT, stop)

    # Create Shimmer device
    ser = None
    dev = None

    try:
        print(f"Opening {args.port} @ {args.baud} ...")
        ser = Serial(args.port, args.baud, timeout=1)
        dev = ShimmerBluetooth(ser)
        dev.initialize()

        name = dev.get_device_name()
        print(f"Connected: {name}")

        dev.add_stream_callback(make_handler(q))
        dev.start_streaming()
        print("Streaming started. Close the plot window or press Ctrl+C to stop.")

        # --------------------- Matplotlib live plot ---------------------
        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(9, 4.8))
        line1, = ax.plot([], [], label="EXG CH1 (24-bit raw)")
        line2, = ax.plot([], [], label="EXG CH2 (24-bit raw)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Raw value")
        ax.set_title("Shimmer EXG live")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        # Close event to stop streaming
        def on_close(_event):
            stop()

        fig.canvas.mpl_connect('close_event', on_close)

        # Animation update
        def update(_frame):
            # Drain queue quickly to keep up
            drained = 0
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                rb.add(
                    t_abs=item["t_abs"],
                    status=item["status"],
                    ch1=item["ch1"],
                    ch2=item["ch2"],
                    ts_counter=item["ts_counter"],
                )
                drained += 1

            if drained > 0 and len(rb.t_rel) > 2:
                line1.set_data(rb.t_rel, rb.ch1)
                line2.set_data(rb.t_rel, rb.ch2)
                # autoscale view to visible window
                ax.set_xlim(max(0, rb.t_rel[-1] - args.window), rb.t_rel[-1] + 0.001)
                # y autoscale based on current window data
                y_all = list(rb.ch1) + list(rb.ch2)
                y_min, y_max = min(y_all), max(y_all)
                margin = max(10, 0.05 * (y_max - y_min + 1))
                ax.set_ylim(y_min - margin, y_max + margin)
            return line1, line2

        interval_ms = int(1000.0 / max(1.0, args.fps))
        ani = FuncAnimation(fig, update, interval=interval_ms, blit=False)

        # Event loop
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

        # Write CSV
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
