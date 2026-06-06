"""BrainFlow connection manager for Muse S."""

import time
import threading
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets

from config import BOARD_ID
from data_store import MuseDataStore


class MuseConnection:

    def __init__(self, store: MuseDataStore, serial_number: str = ""):
        self._store = store
        self._serial_number = serial_number
        self._board: BoardShim | None = None
        self._pump_thread: threading.Thread | None = None
        self._running = False

    def connect(self) -> str:
        params = BrainFlowInputParams()
        if self._serial_number:
            params.serial_number = self._serial_number
        params.timeout = 5

        BoardShim.enable_board_logger()
        self._board = BoardShim(BOARD_ID, params)

        t0 = time.time()
        self._board.prepare_session()
        elapsed = time.time() - t0

        self._store.connected = True
        self._store.device_name = self._serial_number or "MuseS"
        return f"Connected in {elapsed:.1f}s"

    def start_streaming(self):
        if not self._board:
            raise RuntimeError("Call connect() first")

        self._board.start_stream(num_samples=450000)
        self._running = True
        self._pump_thread = threading.Thread(target=self._pump_loop, daemon=True)
        self._pump_thread.start()

    def stop(self):
        self._running = False
        if self._pump_thread:
            self._pump_thread.join(timeout=2.0)
        if self._board:
            try:
                self._board.stop_stream()
            except Exception:
                pass
            try:
                self._board.release_session()
            except Exception:
                pass
        self._store.connected = False

    def _pump_loop(self):
        eeg_channels = BoardShim.get_eeg_channels(BOARD_ID)
        ts_channel = BoardShim.get_timestamp_channel(BOARD_ID)

        try:
            acc_channels = BoardShim.get_accel_channels(BOARD_ID, BrainFlowPresets.AUXILIARY_PRESET)
            gyro_channels = BoardShim.get_gyro_channels(BOARD_ID, BrainFlowPresets.AUXILIARY_PRESET)
            imu_ts = BoardShim.get_timestamp_channel(BOARD_ID, BrainFlowPresets.AUXILIARY_PRESET)
            has_imu = True
        except Exception:
            has_imu = False

        while self._running:
            try:
                eeg_data = self._board.get_board_data(num_samples=512)
                if eeg_data.shape[1] > 0:
                    self._store.eeg.append(
                        eeg_data[eeg_channels[:4], :],
                        eeg_data[ts_channel, :],
                    )
            except Exception:
                pass

            if has_imu:
                try:
                    imu_data = self._board.get_board_data(
                        num_samples=128, preset=BrainFlowPresets.AUXILIARY_PRESET
                    )
                    if imu_data.shape[1] > 0:
                        self._store.acc.append(imu_data[acc_channels[:3], :], imu_data[imu_ts, :])
                        self._store.gyro.append(imu_data[gyro_channels[:3], :], imu_data[imu_ts, :])
                except Exception:
                    pass

            time.sleep(0.04)
