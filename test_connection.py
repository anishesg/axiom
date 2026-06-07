#!/usr/bin/env python3
"""Minimal Muse S test — no config, just connect and read."""

import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

BOARD_ID = BoardIds.MUSE_S_BOARD.value

def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    # no serial_number — auto-discover
    params.timeout = 15

    board = BoardShim(BOARD_ID, params)

    print("Scanning for Muse S...")
    board.prepare_session()
    print("Connected!")

    print("Starting stream (default preset)...")
    board.start_stream()

    # Poll every second to see when data arrives
    for sec in range(10):
        time.sleep(1)
        data = board.get_current_board_data(256)
        print(f"  t={sec+1}s: {data.shape[1]} samples in buffer")
        if data.shape[1] > 0 and sec >= 3:
            eeg_ch = BoardShim.get_eeg_channels(BOARD_ID)
            print(f"\n=== GOT DATA ===")
            print(f"Total channels: {data.shape[0]}")
            print(f"EEG channel indices: {eeg_ch}")
            for i, idx in enumerate(eeg_ch[:4]):
                d = data[idx]
                nonzero = np.count_nonzero(d)
                print(f"  ch[{idx}] {['TP9','AF7','AF8','TP10'][i]}: {nonzero} nonzero, mean={np.mean(d):.1f}, std={np.std(d):.1f}")
            # Print ALL rows to see what's populated
            print(f"\nAll rows summary:")
            for r in range(data.shape[0]):
                d = data[r]
                nz = np.count_nonzero(d)
                if nz > 0:
                    print(f"  row[{r}]: {nz} nonzero, mean={np.mean(d):.2f}, std={np.std(d):.2f}")
            break

    if data.shape[1] == 0:
        print("\nNo data received after 10s. Trying get_board_data()...")
        final = board.get_board_data()
        print(f"  Final shape: {final.shape}")

    board.stop_stream()
    board.release_session()
    print("Done.")

if __name__ == "__main__":
    main()
