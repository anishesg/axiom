"""
User profile management for personalized EEG pattern learning.

Stores:
- Baseline EEG statistics from resting state
- Personalized RVQ codebooks
- Intent mappings
- Adaptive detection thresholds
"""

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Optional


@dataclass
class BaselineStats:
    """Statistics from resting EEG for normalization."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    collection_time: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0
    n_windows: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "collection_time": self.collection_time,
            "duration_seconds": self.duration_seconds,
            "n_windows": self.n_windows,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BaselineStats":
        """Create from dict."""
        return cls(
            feature_mean=np.array(data["feature_mean"]),
            feature_std=np.array(data["feature_std"]),
            collection_time=data.get("collection_time", ""),
            duration_seconds=data.get("duration_seconds", 0.0),
            n_windows=data.get("n_windows", 0),
        )


@dataclass
class DetectionThresholds:
    """Personalized thresholds for artifact detection."""

    blink_amplitude: float = 50.0        # μV threshold for blink detection
    blink_duration_min: float = 0.05     # Minimum blink duration (seconds)
    blink_duration_max: float = 0.4      # Maximum blink duration (seconds)
    clench_amplitude: float = 30.0       # μV threshold for jaw clench
    clench_duration_min: float = 0.15    # Minimum clench duration
    long_clench_duration: float = 0.8    # Duration for "long" clench
    double_blink_window: float = 0.8     # Window for double-blink detection
    triple_blink_window: float = 1.2     # Window for triple-blink detection

    def to_dict(self) -> dict:
        return {
            "blink_amplitude": self.blink_amplitude,
            "blink_duration_min": self.blink_duration_min,
            "blink_duration_max": self.blink_duration_max,
            "clench_amplitude": self.clench_amplitude,
            "clench_duration_min": self.clench_duration_min,
            "long_clench_duration": self.long_clench_duration,
            "double_blink_window": self.double_blink_window,
            "triple_blink_window": self.triple_blink_window,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DetectionThresholds":
        return cls(**data)


@dataclass
class CalibrationProgress:
    """Tracks calibration session progress."""

    baseline_collected: bool = False
    states_calibrated: list = field(default_factory=list)
    actions_calibrated: list = field(default_factory=list)
    rvq_trained: bool = False
    intent_mapper_trained: bool = False
    last_calibration: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "baseline_collected": self.baseline_collected,
            "states_calibrated": self.states_calibrated,
            "actions_calibrated": self.actions_calibrated,
            "rvq_trained": self.rvq_trained,
            "intent_mapper_trained": self.intent_mapper_trained,
            "last_calibration": self.last_calibration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationProgress":
        return cls(
            baseline_collected=data.get("baseline_collected", False),
            states_calibrated=data.get("states_calibrated", []),
            actions_calibrated=data.get("actions_calibrated", []),
            rvq_trained=data.get("rvq_trained", False),
            intent_mapper_trained=data.get("intent_mapper_trained", False),
            last_calibration=data.get("last_calibration"),
        )

    def is_complete(self) -> bool:
        """Check if all calibration steps are complete."""
        return (
            self.baseline_collected and
            self.rvq_trained and
            self.intent_mapper_trained
        )


@dataclass
class UserProfile:
    """
    Complete user profile for personalized EEG communication.

    Contains all calibration data needed for accurate intent detection.
    """

    user_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Baseline normalization
    baseline: Optional[BaselineStats] = None

    # Detection thresholds
    thresholds: DetectionThresholds = field(default_factory=DetectionThresholds)

    # Calibration progress
    progress: CalibrationProgress = field(default_factory=CalibrationProgress)

    # RVQ configuration
    use_rvq: bool = True
    rvq_n_levels: int = 3
    rvq_codebook_sizes: tuple = (32, 64, 128)

    # Intent mappings (stored separately as they can be large)
    intent_names: list = field(default_factory=list)

    # Research-style baseline values (from calibration)
    baseline_alpha: Optional[float] = None  # Alpha power during relaxation
    baseline_beta: Optional[float] = None   # Beta power during focus

    # Session statistics
    total_sessions: int = 0
    total_messages_sent: int = 0
    avg_accuracy: float = 0.0

    def save(self, directory: Path):
        """
        Save user profile to directory.

        Creates:
        - profile.json: Main profile data
        - baseline.npy: Baseline statistics (if collected)
        - rvq/: RVQ codebooks (saved by EEGTokenizer)
        - intents.json: Intent mappings (saved by EEGTokenizer)
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save main profile
        profile_data = {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat(),
            "thresholds": self.thresholds.to_dict(),
            "progress": self.progress.to_dict(),
            "use_rvq": self.use_rvq,
            "rvq_n_levels": self.rvq_n_levels,
            "rvq_codebook_sizes": list(self.rvq_codebook_sizes),
            "intent_names": self.intent_names,
            "baseline_alpha": self.baseline_alpha,
            "baseline_beta": self.baseline_beta,
            "total_sessions": self.total_sessions,
            "total_messages_sent": self.total_messages_sent,
            "avg_accuracy": self.avg_accuracy,
        }

        with open(directory / "profile.json", "w") as f:
            json.dump(profile_data, f, indent=2)

        # Save baseline if available
        if self.baseline is not None:
            baseline_data = self.baseline.to_dict()
            with open(directory / "baseline.json", "w") as f:
                json.dump(baseline_data, f, indent=2)
            np.save(directory / "baseline_mean.npy", self.baseline.feature_mean)
            np.save(directory / "baseline_std.npy", self.baseline.feature_std)

    @classmethod
    def load(cls, directory: Path) -> "UserProfile":
        """Load user profile from directory."""
        directory = Path(directory)

        # Load main profile
        with open(directory / "profile.json") as f:
            data = json.load(f)

        profile = cls(
            user_id=data["user_id"],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            thresholds=DetectionThresholds.from_dict(data.get("thresholds", {})),
            progress=CalibrationProgress.from_dict(data.get("progress", {})),
            use_rvq=data.get("use_rvq", True),
            rvq_n_levels=data.get("rvq_n_levels", 3),
            rvq_codebook_sizes=tuple(data.get("rvq_codebook_sizes", [32, 64, 128])),
            intent_names=data.get("intent_names", []),
            baseline_alpha=data.get("baseline_alpha"),
            baseline_beta=data.get("baseline_beta"),
            total_sessions=data.get("total_sessions", 0),
            total_messages_sent=data.get("total_messages_sent", 0),
            avg_accuracy=data.get("avg_accuracy", 0.0),
        )

        # Load baseline if available
        baseline_path = directory / "baseline.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline_data = json.load(f)

            # Load numpy arrays
            mean_path = directory / "baseline_mean.npy"
            std_path = directory / "baseline_std.npy"
            if mean_path.exists() and std_path.exists():
                baseline_data["feature_mean"] = np.load(mean_path).tolist()
                baseline_data["feature_std"] = np.load(std_path).tolist()

            profile.baseline = BaselineStats.from_dict(baseline_data)

        return profile

    def update_baseline(self, windows: list[np.ndarray], duration: float):
        """
        Update baseline statistics from resting EEG windows.

        Args:
            windows: List of EEG windows from resting state
            duration: Total duration in seconds
        """
        from eleven.features import MultiScaleFeatureExtractor

        extractor = MultiScaleFeatureExtractor()
        all_features = []

        for window in windows:
            features = extractor.extract(window, normalize=False)
            all_features.append(features)

        all_features = np.array(all_features)

        self.baseline = BaselineStats(
            feature_mean=np.mean(all_features, axis=0),
            feature_std=np.std(all_features, axis=0),
            duration_seconds=duration,
            n_windows=len(windows),
        )

        self.progress.baseline_collected = True
        self.updated_at = datetime.now().isoformat()

    def update_thresholds(
        self,
        natural_blinks: list[float],
        deliberate_blinks: list[float],
        jaw_clenches: list[float],
    ):
        """
        Update detection thresholds from calibration data.

        Args:
            natural_blinks: Amplitudes from natural blinks
            deliberate_blinks: Amplitudes from deliberate blinks
            jaw_clenches: Amplitudes from jaw clenches
        """
        # Set blink threshold between natural and deliberate
        if natural_blinks and deliberate_blinks:
            natural_max = np.percentile(natural_blinks, 90)
            deliberate_min = np.percentile(deliberate_blinks, 10)
            # Threshold should be above natural but below deliberate
            self.thresholds.blink_amplitude = (natural_max + deliberate_min) / 2

        # Set clench threshold
        if jaw_clenches:
            self.thresholds.clench_amplitude = np.percentile(jaw_clenches, 10) * 0.8

        self.updated_at = datetime.now().isoformat()

    def mark_calibration_complete(self):
        """Mark calibration as complete."""
        self.progress.rvq_trained = True
        self.progress.intent_mapper_trained = True
        self.progress.last_calibration = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

    def record_session(self, messages_sent: int, accuracy: float):
        """Record session statistics."""
        self.total_sessions += 1
        self.total_messages_sent += messages_sent

        # Update running average accuracy
        if self.total_sessions == 1:
            self.avg_accuracy = accuracy
        else:
            self.avg_accuracy = (
                self.avg_accuracy * (self.total_sessions - 1) + accuracy
            ) / self.total_sessions

        self.updated_at = datetime.now().isoformat()


class ProfileManager:
    """
    Manages multiple user profiles.

    Handles profile creation, loading, and persistence.
    """

    def __init__(self, profiles_dir: Path):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._current_profile: Optional[UserProfile] = None

    def list_profiles(self) -> list[str]:
        """List all available user profile IDs."""
        profiles = []
        for path in self.profiles_dir.iterdir():
            if path.is_dir() and (path / "profile.json").exists():
                profiles.append(path.name)
        return profiles

    def create_profile(self, user_id: str) -> UserProfile:
        """Create a new user profile."""
        profile_dir = self.profiles_dir / user_id
        if profile_dir.exists():
            raise ValueError(f"Profile {user_id} already exists")

        profile = UserProfile(user_id=user_id)
        profile.save(profile_dir)
        return profile

    def load_profile(self, user_id: str) -> UserProfile:
        """Load an existing user profile."""
        profile_dir = self.profiles_dir / user_id
        if not profile_dir.exists():
            raise ValueError(f"Profile {user_id} not found")

        return UserProfile.load(profile_dir)

    def save_profile(self, profile: UserProfile):
        """Save a user profile."""
        profile_dir = self.profiles_dir / profile.user_id
        profile.save(profile_dir)

    def delete_profile(self, user_id: str):
        """Delete a user profile."""
        import shutil
        profile_dir = self.profiles_dir / user_id
        if profile_dir.exists():
            shutil.rmtree(profile_dir)

    def get_or_create(self, user_id: str) -> UserProfile:
        """Get existing profile or create new one."""
        try:
            return self.load_profile(user_id)
        except ValueError:
            return self.create_profile(user_id)

    @property
    def current(self) -> Optional[UserProfile]:
        """Get currently loaded profile."""
        return self._current_profile

    def set_current(self, user_id: str):
        """Set the current active profile."""
        self._current_profile = self.load_profile(user_id)

    def clear_current(self):
        """Clear the current profile."""
        if self._current_profile is not None:
            self.save_profile(self._current_profile)
        self._current_profile = None


# CLI for testing
if __name__ == "__main__":
    import tempfile

    print("Testing UserProfile...")

    # Create temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ProfileManager(Path(tmpdir))

        # Create profile
        profile = manager.create_profile("test_user")
        print(f"Created profile: {profile.user_id}")

        # Update baseline (with fake data)
        fake_features = np.random.randn(100, 150)  # 100 windows, 150 features
        profile.baseline = BaselineStats(
            feature_mean=np.mean(fake_features, axis=0),
            feature_std=np.std(fake_features, axis=0),
            duration_seconds=30.0,
            n_windows=100,
        )
        profile.progress.baseline_collected = True
        print(f"Baseline collected: {profile.baseline.n_windows} windows")

        # Update thresholds
        profile.update_thresholds(
            natural_blinks=[30, 35, 40, 45],
            deliberate_blinks=[80, 90, 100, 110],
            jaw_clenches=[40, 50, 60],
        )
        print(f"Blink threshold: {profile.thresholds.blink_amplitude:.1f} μV")

        # Save and reload
        manager.save_profile(profile)
        loaded = manager.load_profile("test_user")
        print(f"Loaded profile: {loaded.user_id}")
        print(f"Baseline loaded: {loaded.baseline is not None}")
        print(f"Blink threshold: {loaded.thresholds.blink_amplitude:.1f} μV")

        # List profiles
        profiles = manager.list_profiles()
        print(f"Available profiles: {profiles}")

        print("\nAll tests passed!")
