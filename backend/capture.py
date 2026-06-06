#!/usr/bin/env python3
"""Capture real Muse S data to .npz files for offline simulation/testing."""

import time
import sys
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

BOARD_ID = BoardIds.MUSE_S_BOARD.value
EEG_SR = 256


def capture(duration_s: int = 30, output: str = "data/session.npz"):
    params = BrainFlowInputParams()
    params.timeout = 5
    BoardShim.enable_board_logger()
    board = BoardShim(BOARD_ID, params)

    print("Connecting...", flush=True)
    board.prepare_session()
    print("Connected! Starting stream.", flush=True)
    board.start_stream(num_samples=450000)

    eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
    ts_channel = BoardShim.get_timestamp_channel(BOARD_ID)

    all_eeg = []
    all_ts = []

    print(f"Recording {duration_s}s of EEG data...", flush=True)
    start = time.time()
    while time.time() - start < duration_s:
        data = board.get_board_data(num_samples=256)
        if data.shape[1] > 0:
            eeg = data[eeg_channels[:4], :]
            ts = data[ts_channel, :]
            all_eeg.append(eeg)
            all_ts.append(ts)
            elapsed = time.time() - start
            total = sum(e.shape[1] for e in all_eeg)
            print(f"  {elapsed:.0f}s / {duration_s}s — {total} samples", end="\r", flush=True)
        time.sleep(0.1)

    board.stop_stream()
    board.release_session()

    eeg_arr = np.concatenate(all_eeg, axis=1)
    ts_arr = np.concatenate(all_ts)

    import os
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez(output, eeg=eeg_arr, timestamps=ts_arr, sample_rate=EEG_SR,
             channels=["TP9", "AF7", "AF8", "TP10"])
    print(f"\nSaved {eeg_arr.shape[1]} samples ({eeg_arr.shape[1]/EEG_SR:.1f}s) to {output}")


if __name__ == "__main__":
    dur = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    out = sys.argv[2] if len(sys.argv) > 2 else "data/session.npz"
    capture(dur, out)
