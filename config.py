"""Central configuration for Muse S dashboard."""

# -- BrainFlow --
BOARD_ID = 39  # BoardIds.MUSE_S_BOARD (native BLE, no dongle)
MUSE_CONFIG = "p51"  # 4 EEG + IMU + PPG

# -- EEG --
EEG_SAMPLE_RATE = 256
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
EEG_BUFFER_SECONDS = 10
EEG_DISPLAY_SECONDS = 5

# -- PPG --
PPG_SAMPLE_RATE = 64
PPG_CHANNELS = ["PPG_AMBIENT", "PPG_IR", "PPG_RED"]
PPG_BUFFER_SECONDS = 30

# -- IMU --
IMU_SAMPLE_RATE = 52
ACC_CHANNELS = ["ACC_X", "ACC_Y", "ACC_Z"]
GYRO_CHANNELS = ["GYRO_X", "GYRO_Y", "GYRO_Z"]
IMU_BUFFER_SECONDS = 10

# -- Frequency Bands (Hz) --
FREQ_BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 50.0),
}

# -- Signal Processing --
NOTCH_FREQ = 60.0  # US power line frequency
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 50.0
FILTER_ORDER = 4

# -- Dashboard --
DASH_UPDATE_INTERVAL_MS = 100  # 10 Hz refresh
DASH_HOST = "127.0.0.1"
DASH_PORT = 8050
DASH_DEBUG = False

# -- Electrode positions (10-20 system, normalized to unit circle for topo map) --
ELECTRODE_POSITIONS = {
    "TP9":  (-0.81, -0.31),
    "AF7":  (-0.41,  0.71),
    "AF8":  ( 0.41,  0.71),
    "TP10": ( 0.81, -0.31),
}
