#!/usr/bin/env python3
"""Muse S Neural Dashboard — entry point.

Usage:
    python main.py                  # connect to first Muse S found
    python main.py --name Muse-XXXX # connect by device name
    python main.py --simulate       # run with synthetic data (no headband)
"""

import argparse
import signal
import sys
import time
import threading
import numpy as np

from data_store import MuseDataStore
from config import (
    EEG_SAMPLE_RATE, PPG_SAMPLE_RATE, IMU_SAMPLE_RATE,
    DASH_HOST, DASH_PORT, DASH_DEBUG,
)


def simulate_data(store: MuseDataStore, stop_event: threading.Event):
    """Generate synthetic EEG/PPG/IMU data for dashboard development."""
    t = 0.0
    dt_eeg = 1.0 / EEG_SAMPLE_RATE
    dt_ppg = 1.0 / PPG_SAMPLE_RATE
    dt_imu = 1.0 / IMU_SAMPLE_RATE

    store.connected = True
    store.device_name = "SIMULATED"

    while not stop_event.is_set():
        # EEG: alpha oscillation + noise
        n_eeg = int(EEG_SAMPLE_RATE * 0.05)
        ts_eeg = np.arange(n_eeg) * dt_eeg + t
        eeg = np.zeros((4, n_eeg))
        for ch in range(4):
            alpha = 15 * np.sin(2 * np.pi * 10 * ts_eeg + ch * 0.5)
            beta = 5 * np.sin(2 * np.pi * 22 * ts_eeg + ch * 0.3)
            theta = 8 * np.sin(2 * np.pi * 6 * ts_eeg + ch * 0.7)
            noise = np.random.randn(n_eeg) * 3
            eeg[ch] = alpha + beta + theta + noise
        store.eeg.append(eeg, ts_eeg)

        # PPG: heartbeat-like signal at ~72 bpm
        n_ppg = int(PPG_SAMPLE_RATE * 0.05)
        ts_ppg = np.arange(n_ppg) * dt_ppg + t
        hr_freq = 1.2  # 72 bpm
        ppg = np.zeros((3, n_ppg))
        ppg[0] = 1000 + np.random.randn(n_ppg) * 10
        ppg[1] = 5000 + 500 * np.sin(2 * np.pi * hr_freq * ts_ppg) + np.random.randn(n_ppg) * 20
        ppg[2] = 3000 + 300 * np.sin(2 * np.pi * hr_freq * ts_ppg + 0.3) + np.random.randn(n_ppg) * 15
        store.ppg.append(ppg, ts_ppg)

        # IMU
        n_imu = int(IMU_SAMPLE_RATE * 0.05)
        ts_imu = np.arange(n_imu) * dt_imu + t
        acc = np.zeros((3, n_imu))
        acc[0] = np.random.randn(n_imu) * 0.02
        acc[1] = np.random.randn(n_imu) * 0.02
        acc[2] = 1.0 + np.random.randn(n_imu) * 0.02  # gravity on Z
        store.acc.append(acc, ts_imu)

        gyro = np.random.randn(3, n_imu) * 2.0
        store.gyro.append(gyro, ts_imu)

        t += 0.05
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description="Muse S Neural Dashboard")
    parser.add_argument("--name", type=str, default="", help="Muse device name (e.g. Muse-XXXX)")
    parser.add_argument("--simulate", action="store_true", help="Use synthetic data (no headband)")
    parser.add_argument("--host", type=str, default=DASH_HOST)
    parser.add_argument("--port", type=int, default=DASH_PORT)
    args = parser.parse_args()

    store = MuseDataStore()
    stop_event = threading.Event()
    conn = None

    def shutdown(sig=None, frame=None):
        print("\nShutting down...")
        stop_event.set()
        if conn:
            conn.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.simulate:
        print("Starting in SIMULATION mode (no Muse headband needed)")
        sim_thread = threading.Thread(target=simulate_data, args=(store, stop_event), daemon=True)
        sim_thread.start()
    else:
        from connection import MuseConnection
        conn = MuseConnection(store, serial_number=args.name)
        print("Connecting to Muse S...")
        try:
            info = conn.connect()
            print(info)
            conn.start_streaming()
            print("Streaming started. Launching dashboard...")
        except Exception as e:
            print(f"Connection failed: {e}")
            print("Tip: make sure your Muse S is on and in range.")
            print("Run with --simulate to test the dashboard without a headband.")
            sys.exit(1)

    from dashboard import create_app
    app = create_app(store)
    print(f"\nDashboard live at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=DASH_DEBUG)


if __name__ == "__main__":
    main()
