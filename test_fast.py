#!/usr/bin/env python3
"""Fast connection test — short timeout, no frills."""

import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

BOARD_ID = BoardIds.MUSE_S_BOARD.value

params = BrainFlowInputParams()
params.timeout = 5

BoardShim.enable_board_logger()
board = BoardShim(BOARD_ID, params)

t0 = time.time()
print("Connecting...")
board.prepare_session()
print(f"Connected in {time.time()-t0:.1f}s")

board.start_stream()
time.sleep(2)

data = board.get_board_data()
eeg_ch = BoardShim.get_eeg_channels(BOARD_ID)
print(f"Samples: {data.shape[1]} | Channels: {data.shape[0]}")
for i, idx in enumerate(eeg_ch[:4]):
    d = data[idx]
    print(f"  {['TP9','AF7','AF8','TP10'][i]}: std={np.std(d):.1f}")

board.stop_stream()
board.release_session()
print("Done")
