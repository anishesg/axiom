"""
EEG data acquisition from Muse S headband using BrainFlow.

This module handles:
- Bluetooth connection to Muse S
- Real-time data streaming
- Buffer management for downstream processing
"""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable
from threading import Thread, Event

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter

logger = logging.getLogger(__name__)


# Muse S channel information
MUSE_S_BOARD_ID = BoardIds.MUSE_S_BOARD
MUSE_S_SAMPLE_RATE = 256  # Hz
MUSE_S_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]  # Standard 10-20 positions


@dataclass
class MuseConfig:
    """Configuration for Muse S connection."""

    board_id: int = MUSE_S_BOARD_ID
    serial_port: str = ""  # Usually empty for Bluetooth
    mac_address: str = ""  # Optional: specific device MAC
    timeout: int = 30  # Connection timeout in seconds (increased for reliable BLE discovery)
    buffer_size: int = 450000  # ~30 minutes of data at 256 Hz * 4 channels


@dataclass
class EEGSample:
    """A single EEG data sample with metadata."""

    timestamp: float  # Unix timestamp
    channels: np.ndarray  # Shape: (4,) for TP9, AF7, AF8, TP10
    sample_index: int  # BrainFlow sample counter


class MuseAcquisition:
    """
    Handles real-time EEG streaming from Muse S headband.

    Usage:
        muse = MuseAcquisition()
        muse.connect()
        muse.start_stream()

        while running:
            data = muse.get_data(n_samples=256)  # Get 1 second of data
            process(data)

        muse.stop()
    """

    def __init__(self, config: Optional[MuseConfig] = None):
        self.config = config or MuseConfig()
        self.board: Optional[BoardShim] = None
        self.is_connected = False
        self.is_streaming = False

        # Channel indices for Muse S (from BrainFlow)
        self._eeg_channels: list[int] = []
        self._timestamp_channel: int = 0
        self._sample_index_channel: int = 0

        # Callback for real-time processing
        self._data_callback: Optional[Callable[[np.ndarray], None]] = None
        self._callback_thread: Optional[Thread] = None
        self._stop_event = Event()

        # Enable BrainFlow logging for debugging
        BoardShim.enable_dev_board_logger()

    def connect(self) -> bool:
        """
        Establish Bluetooth connection to Muse S.

        Returns:
            True if connection successful, False otherwise.
        """
        if self.is_connected:
            logger.warning("Already connected to Muse S")
            return True

        try:
            # Set up connection parameters
            params = BrainFlowInputParams()
            params.serial_port = self.config.serial_port
            params.mac_address = self.config.mac_address
            params.timeout = self.config.timeout

            # Create board instance
            self.board = BoardShim(self.config.board_id, params)

            # Get channel configuration
            self._eeg_channels = BoardShim.get_eeg_channels(self.config.board_id)
            self._timestamp_channel = BoardShim.get_timestamp_channel(self.config.board_id)

            # Try to get sample index channel (may not exist on all boards)
            try:
                self._sample_index_channel = BoardShim.get_package_num_channel(self.config.board_id)
            except Exception:
                self._sample_index_channel = -1

            logger.info(f"EEG channels: {self._eeg_channels}")
            logger.info(f"Timestamp channel: {self._timestamp_channel}")

            # Prepare session (establishes Bluetooth connection)
            logger.info("Connecting to Muse S... (ensure headband is on and nearby)")
            self.board.prepare_session()

            self.is_connected = True
            logger.info("Successfully connected to Muse S")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Muse S: {e}")
            self.board = None
            return False

    def start_stream(self, callback: Optional[Callable[[np.ndarray], None]] = None) -> bool:
        """
        Start EEG data streaming.

        Args:
            callback: Optional function called with each data chunk.
                     Receives numpy array of shape (channels, samples).

        Returns:
            True if streaming started successfully.
        """
        if not self.is_connected:
            logger.error("Not connected to Muse S. Call connect() first.")
            return False

        if self.is_streaming:
            logger.warning("Already streaming")
            return True

        try:
            self.board.start_stream(self.config.buffer_size)
            self.is_streaming = True
            logger.info("Started EEG stream")

            # Set up callback thread if provided
            if callback:
                self._data_callback = callback
                self._stop_event.clear()
                self._callback_thread = Thread(target=self._callback_loop, daemon=True)
                self._callback_thread.start()

            return True

        except Exception as e:
            logger.error(f"Failed to start stream: {e}")
            return False

    def _callback_loop(self):
        """Background thread for real-time callback processing."""
        while not self._stop_event.is_set():
            try:
                # Get latest data (non-blocking, returns whatever is available)
                data = self.get_data(n_samples=64)  # ~250ms chunks

                if data is not None and data.shape[1] > 0:
                    self._data_callback(data)
                else:
                    # No data yet, wait briefly
                    time.sleep(0.05)

            except Exception as e:
                logger.error(f"Error in callback loop: {e}")
                time.sleep(0.1)

    def get_data(self, n_samples: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get EEG data from the buffer.

        Args:
            n_samples: Number of samples to retrieve. If None, gets all available.

        Returns:
            Numpy array of shape (5, n_samples) where rows are:
            [TP9, AF7, AF8, TP10, timestamp]
            Returns None if not streaming.
        """
        if not self.is_streaming or self.board is None:
            return None

        try:
            if n_samples is None:
                # Get all available data
                raw_data = self.board.get_board_data()
            else:
                # Get specific number of samples
                raw_data = self.board.get_current_board_data(n_samples)

            if raw_data.shape[1] == 0:
                return None

            # Extract EEG channels and timestamp
            eeg_data = raw_data[self._eeg_channels, :]
            timestamps = raw_data[self._timestamp_channel, :]

            # Stack into output array: [TP9, AF7, AF8, TP10, timestamp]
            output = np.vstack([eeg_data, timestamps])

            return output

        except Exception as e:
            logger.error(f"Error getting data: {e}")
            return None

    def get_sample_rate(self) -> int:
        """Get the sampling rate of the connected device."""
        if self.board is None:
            return MUSE_S_SAMPLE_RATE
        return BoardShim.get_sampling_rate(self.config.board_id)

    def get_channel_names(self) -> list[str]:
        """Get the names of EEG channels."""
        return MUSE_S_CHANNELS.copy()

    def stop_stream(self):
        """Stop EEG data streaming."""
        if self._callback_thread is not None:
            self._stop_event.set()
            self._callback_thread.join(timeout=2.0)
            self._callback_thread = None

        if self.is_streaming and self.board is not None:
            try:
                self.board.stop_stream()
                logger.info("Stopped EEG stream")
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
            self.is_streaming = False

    def disconnect(self):
        """Disconnect from Muse S and release resources."""
        self.stop_stream()

        # Always try to release the board if it exists, regardless of is_connected flag.
        # This handles cases where timeout interrupted prepare_session() before
        # is_connected was set to True, leaving BrainFlow resources in a bad state.
        if self.board is not None:
            try:
                self.board.release_session()
                logger.info("Disconnected from Muse S")
            except Exception as e:
                logger.warning(f"Error releasing board session: {e}")
            finally:
                self.board = None

        self.is_connected = False

    @classmethod
    def force_cleanup_all(cls):
        """Force cleanup any stale BrainFlow sessions.

        Call this before creating a new MuseAcquisition if previous sessions
        may have been interrupted (e.g., timeout during BLE discovery).
        """
        try:
            BoardShim.release_all_sessions()
            logger.info("Force released all BrainFlow sessions")
        except Exception as e:
            logger.warning(f"Error during force cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


class SimulatedMuse:
    """
    Simulated Muse S for development without hardware.

    Generates synthetic EEG-like data including:
    - Background alpha oscillations (8-12 Hz)
    - Eye blink artifacts
    - Noise
    """

    def __init__(self, sample_rate: int = 256):
        self.sample_rate = sample_rate
        self.is_connected = True
        self.is_streaming = False
        self._time_offset = 0.0
        self._callback: Optional[Callable] = None
        self._thread: Optional[Thread] = None
        self._stop_event = Event()

    def connect(self) -> bool:
        logger.info("Using simulated Muse S (no hardware)")
        return True

    def start_stream(self, callback: Optional[Callable] = None) -> bool:
        self.is_streaming = True
        self._time_offset = time.time()

        if callback:
            self._callback = callback
            self._stop_event.clear()
            self._thread = Thread(target=self._generate_loop, daemon=True)
            self._thread.start()

        return True

    def _generate_loop(self):
        """Generate synthetic EEG data."""
        while not self._stop_event.is_set():
            data = self._generate_chunk(64)
            if self._callback:
                self._callback(data)
            time.sleep(64 / self.sample_rate)  # Real-time pacing

    def _generate_chunk(self, n_samples: int) -> np.ndarray:
        """Generate synthetic EEG chunk."""
        t = np.linspace(
            self._time_offset,
            self._time_offset + n_samples / self.sample_rate,
            n_samples,
        )
        self._time_offset += n_samples / self.sample_rate

        # Alpha oscillation (8-12 Hz) - stronger on temporal channels
        alpha_freq = 10.0
        alpha_tp = 15 * np.sin(2 * np.pi * alpha_freq * t)  # TP9, TP10
        alpha_af = 5 * np.sin(2 * np.pi * alpha_freq * t)   # AF7, AF8 (weaker)

        # Background noise
        noise = np.random.randn(4, n_samples) * 3

        # Combine into channels [TP9, AF7, AF8, TP10]
        data = np.array([
            alpha_tp + noise[0],   # TP9
            alpha_af + noise[1],   # AF7
            alpha_af + noise[2],   # AF8
            alpha_tp + noise[3],   # TP10
        ])

        # Randomly inject blink artifact (5% chance per chunk)
        if np.random.random() < 0.05:
            blink_pos = n_samples // 2
            blink_width = int(0.2 * self.sample_rate)  # 200ms blink
            blink_start = max(0, blink_pos - blink_width // 2)
            blink_end = min(n_samples, blink_pos + blink_width // 2)

            # Blinks are large deflections on AF7/AF8
            blink_shape = 100 * np.exp(-0.5 * ((np.arange(blink_end - blink_start) - blink_width // 2) / (blink_width / 4)) ** 2)
            data[1, blink_start:blink_end] += blink_shape  # AF7
            data[2, blink_start:blink_end] += blink_shape  # AF8

        # Add timestamps
        timestamps = t
        output = np.vstack([data, timestamps])

        return output

    def get_data(self, n_samples: Optional[int] = None) -> Optional[np.ndarray]:
        n = n_samples or 256
        return self._generate_chunk(n)

    def get_sample_rate(self) -> int:
        return self.sample_rate

    def get_channel_names(self) -> list[str]:
        return MUSE_S_CHANNELS.copy()

    def stop_stream(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.is_streaming = False

    def disconnect(self):
        self.stop_stream()
        self.is_connected = False


def create_acquisition(use_simulation: bool = False) -> MuseAcquisition | SimulatedMuse:
    """
    Factory function to create appropriate acquisition instance.

    Args:
        use_simulation: If True, return simulated Muse for development.

    Returns:
        MuseAcquisition or SimulatedMuse instance.
    """
    if use_simulation:
        return SimulatedMuse()
    return MuseAcquisition()


# CLI for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Muse S acquisition")
    parser.add_argument("--simulate", action="store_true", help="Use simulated data")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    def print_data(data: np.ndarray):
        """Callback to print data statistics."""
        eeg = data[:4, :]  # First 4 rows are EEG channels
        print(f"Samples: {data.shape[1]}, "
              f"Mean: [{eeg[0].mean():.1f}, {eeg[1].mean():.1f}, {eeg[2].mean():.1f}, {eeg[3].mean():.1f}], "
              f"Std: [{eeg[0].std():.1f}, {eeg[1].std():.1f}, {eeg[2].std():.1f}, {eeg[3].std():.1f}]")

    muse = create_acquisition(use_simulation=args.simulate)

    if muse.connect():
        muse.start_stream(callback=print_data)
        print(f"Streaming for {args.duration} seconds... (Ctrl+C to stop)")

        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            print("\nStopping...")

        muse.disconnect()
    else:
        print("Failed to connect to Muse S")
