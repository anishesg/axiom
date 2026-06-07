"""
Vector Quantization encoder for EEG signals.

Converts continuous EEG feature vectors into discrete tokens
from a learned codebook. This enables:
- Noise reduction (VQ acts as sparsity filter)
- Personalization (user-specific codebooks)
- Discrete state machine logic for communication
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json


@dataclass
class VQConfig:
    """Configuration for Vector Quantizer."""

    codebook_size: int = 64     # Number of discrete tokens
    feature_dim: int = 91       # Dimension of feature vectors (from FeatureExtractor)
    commitment_cost: float = 0.25  # Weight for commitment loss (if training)


@dataclass
class RVQConfig:
    """Configuration for Residual Vector Quantizer (NeuroRVQ-inspired)."""

    n_levels: int = 3                          # Number of quantization levels
    codebook_sizes: tuple = (32, 64, 128)      # Codebook size per level
    feature_dim: int = 150                     # Feature dimension (multi-scale)
    commitment_cost: float = 0.25              # VQ commitment cost
    ema_decay: float = 0.99                    # EMA decay for online adaptation
    use_ema: bool = True                       # Enable online codebook adaptation


class VectorQuantizer:
    """
    Vector Quantizer using K-means style codebook.

    Maps continuous feature vectors to discrete tokens by finding
    the nearest codebook entry.
    """

    def __init__(self, config: Optional[VQConfig] = None):
        self.config = config or VQConfig()

        # Initialize codebook randomly (will be trained/calibrated)
        self.codebook = np.random.randn(
            self.config.codebook_size,
            self.config.feature_dim
        ) * 0.1

        # Track codebook usage for analysis
        self.usage_counts = np.zeros(self.config.codebook_size)

        # Optional: learned token meanings
        self.token_labels: dict[int, str] = {}

    def encode(self, features: np.ndarray) -> int:
        """
        Encode feature vector to nearest codebook token.

        Args:
            features: Shape (feature_dim,) feature vector

        Returns:
            Token index (0 to codebook_size-1)
        """
        # Compute distances to all codebook entries
        distances = np.linalg.norm(self.codebook - features, axis=1)

        # Find nearest
        token = int(np.argmin(distances))

        # Track usage
        self.usage_counts[token] += 1

        return token

    def encode_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Encode batch of feature vectors.

        Args:
            features: Shape (batch_size, feature_dim)

        Returns:
            Token indices of shape (batch_size,)
        """
        # Compute all pairwise distances
        # features: (B, D), codebook: (K, D)
        # result: (B, K)
        diff = features[:, np.newaxis, :] - self.codebook[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)

        tokens = np.argmin(distances, axis=1)

        for t in tokens:
            self.usage_counts[t] += 1

        return tokens

    def decode(self, token: int) -> np.ndarray:
        """
        Decode token back to codebook vector.

        Args:
            token: Token index

        Returns:
            Codebook vector of shape (feature_dim,)
        """
        return self.codebook[token].copy()

    def fit(self, features: np.ndarray, n_iterations: int = 50):
        """
        Train codebook using K-means algorithm.

        Args:
            features: Shape (n_samples, feature_dim) training data
            n_iterations: Number of K-means iterations
        """
        n_samples = features.shape[0]
        k = self.config.codebook_size

        # Initialize with random samples (K-means++)
        indices = np.random.choice(n_samples, k, replace=False)
        self.codebook = features[indices].copy()

        for _ in range(n_iterations):
            # Assign samples to nearest centroid
            assignments = self.encode_batch(features)

            # Update centroids
            for i in range(k):
                mask = assignments == i
                if np.any(mask):
                    self.codebook[i] = np.mean(features[mask], axis=0)

        # Reset usage counts after training
        self.usage_counts = np.zeros(k)

    def save(self, path: Path):
        """Save codebook to file."""
        path = Path(path)
        np.save(path.with_suffix(".npy"), self.codebook)

        # Save metadata
        metadata = {
            "codebook_size": self.config.codebook_size,
            "feature_dim": self.config.feature_dim,
            "token_labels": self.token_labels,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: Path):
        """Load codebook from file."""
        path = Path(path)
        self.codebook = np.load(path.with_suffix(".npy"))

        # Load metadata
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            self.config.codebook_size = metadata["codebook_size"]
            self.config.feature_dim = metadata["feature_dim"]
            self.token_labels = metadata.get("token_labels", {})

    def get_usage_stats(self) -> dict:
        """Get codebook usage statistics."""
        total = np.sum(self.usage_counts)
        if total == 0:
            return {"total": 0, "active_tokens": 0, "entropy": 0}

        probs = self.usage_counts / total
        nonzero = probs > 0

        entropy = -np.sum(probs[nonzero] * np.log2(probs[nonzero]))
        max_entropy = np.log2(self.config.codebook_size)

        return {
            "total": int(total),
            "active_tokens": int(np.sum(nonzero)),
            "entropy": float(entropy),
            "max_entropy": float(max_entropy),
            "utilization": float(entropy / max_entropy) if max_entropy > 0 else 0,
        }


class ResidualVectorQuantizer:
    """
    Residual Vector Quantizer (RVQ) inspired by NeuroRVQ.

    Uses multiple quantization levels where each level encodes the
    residual error from the previous level. This achieves:
    - Progressive refinement of encoding
    - Better reconstruction accuracy
    - Hierarchical pattern capture (coarse → fine)
    """

    def __init__(self, config: Optional[RVQConfig] = None):
        self.config = config or RVQConfig()

        # Validate config
        assert len(self.config.codebook_sizes) == self.config.n_levels, \
            "codebook_sizes must have n_levels entries"

        # Create VQ level for each quantization stage
        self.levels: list[VectorQuantizer] = []
        for i, size in enumerate(self.config.codebook_sizes):
            level_config = VQConfig(
                codebook_size=size,
                feature_dim=self.config.feature_dim,
                commitment_cost=self.config.commitment_cost,
            )
            self.levels.append(VectorQuantizer(level_config))

        # EMA statistics for online adaptation
        if self.config.use_ema:
            self._ema_cluster_size = [
                np.zeros(size) for size in self.config.codebook_sizes
            ]
            self._ema_dw = [
                np.zeros((size, self.config.feature_dim))
                for size in self.config.codebook_sizes
            ]

        # Track reconstruction errors per level
        self.reconstruction_errors: list[float] = []

    def encode(self, features: np.ndarray) -> list[int]:
        """
        Encode feature vector through all RVQ levels.

        Args:
            features: Shape (feature_dim,) feature vector

        Returns:
            List of token indices, one per level
        """
        tokens = []
        residual = features.copy()

        for level in self.levels:
            token = level.encode(residual)
            tokens.append(token)

            # Compute residual for next level
            quantized = level.decode(token)
            residual = residual - quantized

        return tokens

    def encode_batch(self, features: np.ndarray) -> np.ndarray:
        """
        Encode batch of feature vectors.

        Args:
            features: Shape (batch_size, feature_dim)

        Returns:
            Token indices of shape (batch_size, n_levels)
        """
        batch_size = features.shape[0]
        all_tokens = np.zeros((batch_size, self.config.n_levels), dtype=int)
        residuals = features.copy()

        for level_idx, level in enumerate(self.levels):
            tokens = level.encode_batch(residuals)
            all_tokens[:, level_idx] = tokens

            # Compute residuals for next level
            quantized = np.array([level.decode(t) for t in tokens])
            residuals = residuals - quantized

        return all_tokens

    def decode(self, tokens: list[int]) -> np.ndarray:
        """
        Decode tokens back to feature vector.

        Args:
            tokens: List of token indices, one per level

        Returns:
            Reconstructed feature vector of shape (feature_dim,)
        """
        reconstructed = np.zeros(self.config.feature_dim)

        for level, token in zip(self.levels, tokens):
            reconstructed += level.decode(token)

        return reconstructed

    def fit(self, features: np.ndarray, n_iterations: int = 50):
        """
        Train all RVQ levels sequentially.

        Each level is trained on the residual from the previous level.

        Args:
            features: Shape (n_samples, feature_dim) training data
            n_iterations: Number of K-means iterations per level
        """
        residuals = features.copy()
        self.reconstruction_errors = []

        for level_idx, level in enumerate(self.levels):
            # Train this level on current residuals
            level.fit(residuals, n_iterations)

            # Compute quantized values
            tokens = level.encode_batch(residuals)
            quantized = np.array([level.decode(t) for t in tokens])

            # Compute reconstruction error at this level
            error = np.mean(np.linalg.norm(residuals - quantized, axis=1))
            self.reconstruction_errors.append(error)

            # Compute residuals for next level
            residuals = residuals - quantized

        # Reset usage counts after training
        for level in self.levels:
            level.usage_counts = np.zeros(level.config.codebook_size)

    def update_ema(self, features: np.ndarray, tokens: np.ndarray):
        """
        Update codebooks using exponential moving average.

        Enables online adaptation to user's patterns.

        Args:
            features: Shape (batch_size, feature_dim)
            tokens: Shape (batch_size, n_levels) token indices
        """
        if not self.config.use_ema:
            return

        decay = self.config.ema_decay
        residuals = features.copy()

        for level_idx, level in enumerate(self.levels):
            level_tokens = tokens[:, level_idx]

            # Update EMA statistics
            for i, token in enumerate(level_tokens):
                self._ema_cluster_size[level_idx][token] = (
                    decay * self._ema_cluster_size[level_idx][token] + (1 - decay)
                )
                self._ema_dw[level_idx][token] = (
                    decay * self._ema_dw[level_idx][token] +
                    (1 - decay) * residuals[i]
                )

            # Update codebook centroids
            for k in range(level.config.codebook_size):
                if self._ema_cluster_size[level_idx][k] > 0:
                    level.codebook[k] = (
                        self._ema_dw[level_idx][k] /
                        self._ema_cluster_size[level_idx][k]
                    )

            # Compute residuals for next level
            quantized = np.array([level.decode(t) for t in level_tokens])
            residuals = residuals - quantized

    def get_reconstruction_error(self, features: np.ndarray) -> dict:
        """
        Compute reconstruction error for given features.

        Args:
            features: Shape (n_samples, feature_dim)

        Returns:
            Dict with error metrics per level and total
        """
        tokens = self.encode_batch(features)
        reconstructed = np.array([self.decode(t) for t in tokens])

        total_error = np.mean(np.linalg.norm(features - reconstructed, axis=1))

        # Per-level errors
        level_errors = []
        residuals = features.copy()
        for level_idx, level in enumerate(self.levels):
            level_tokens = tokens[:, level_idx]
            quantized = np.array([level.decode(t) for t in level_tokens])
            level_error = np.mean(np.linalg.norm(residuals - quantized, axis=1))
            level_errors.append(level_error)
            residuals = residuals - quantized

        return {
            "total_error": float(total_error),
            "level_errors": level_errors,
            "error_reduction": [
                level_errors[0] / (e + 1e-10) for e in level_errors
            ],
        }

    def save(self, path: Path):
        """Save all RVQ levels to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each level
        for i, level in enumerate(self.levels):
            level.save(path / f"level_{i}")

        # Save config
        config_dict = {
            "n_levels": self.config.n_levels,
            "codebook_sizes": list(self.config.codebook_sizes),
            "feature_dim": self.config.feature_dim,
            "commitment_cost": self.config.commitment_cost,
            "ema_decay": self.config.ema_decay,
            "use_ema": self.config.use_ema,
            "reconstruction_errors": self.reconstruction_errors,
        }
        with open(path / "rvq_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def load(self, path: Path):
        """Load all RVQ levels from directory."""
        path = Path(path)

        # Load config
        with open(path / "rvq_config.json") as f:
            config_dict = json.load(f)

        self.config = RVQConfig(
            n_levels=config_dict["n_levels"],
            codebook_sizes=tuple(config_dict["codebook_sizes"]),
            feature_dim=config_dict["feature_dim"],
            commitment_cost=config_dict["commitment_cost"],
            ema_decay=config_dict.get("ema_decay", 0.99),
            use_ema=config_dict.get("use_ema", True),
        )
        self.reconstruction_errors = config_dict.get("reconstruction_errors", [])

        # Load each level
        self.levels = []
        for i in range(self.config.n_levels):
            level_config = VQConfig(
                codebook_size=self.config.codebook_sizes[i],
                feature_dim=self.config.feature_dim,
            )
            level = VectorQuantizer(level_config)
            level.load(path / f"level_{i}")
            self.levels.append(level)

    def get_combined_token(self, tokens: list[int]) -> int:
        """
        Combine multi-level tokens into single index for intent mapping.

        Uses weighted combination where coarser levels have more weight.

        Args:
            tokens: List of token indices per level

        Returns:
            Single combined token index
        """
        # Primary token from coarsest level
        # This captures the main pattern, finer levels are for reconstruction
        return tokens[0]

    def get_usage_stats(self) -> dict:
        """Get usage statistics across all levels."""
        stats = {
            "levels": [],
            "total_active_tokens": 0,
            "avg_utilization": 0.0,
        }

        for i, level in enumerate(self.levels):
            level_stats = level.get_usage_stats()
            level_stats["level"] = i
            stats["levels"].append(level_stats)
            stats["total_active_tokens"] += level_stats["active_tokens"]

        if stats["levels"]:
            stats["avg_utilization"] = np.mean([
                s["utilization"] for s in stats["levels"]
            ])

        return stats


class IntentMapper:
    """
    Maps VQ tokens to communication intents.

    Learns which tokens correspond to which user intentions
    (yes, no, select, phrases) during calibration.
    """

    def __init__(self, vq: VectorQuantizer):
        self.vq = vq

        # Token -> intent mapping
        # Each intent has a set of associated tokens with confidence scores
        self.intent_tokens: dict[str, dict[int, float]] = {}

        # Prior probabilities for intents
        self.intent_priors: dict[str, float] = {}

    def register_intent(self, intent: str, tokens: list[int], confidence: float = 1.0):
        """
        Register tokens as associated with an intent.

        Args:
            intent: Intent name (e.g., "yes", "no", "phrase_1")
            tokens: List of token indices
            confidence: Confidence weight for these tokens
        """
        if intent not in self.intent_tokens:
            self.intent_tokens[intent] = {}

        for token in tokens:
            current = self.intent_tokens[intent].get(token, 0)
            self.intent_tokens[intent][token] = current + confidence

    def classify(self, token: int) -> tuple[Optional[str], float]:
        """
        Classify a token to an intent.

        Args:
            token: VQ token index

        Returns:
            Tuple of (intent_name, confidence) or (None, 0) if no match
        """
        best_intent = None
        best_score = 0.0

        for intent, token_scores in self.intent_tokens.items():
            if token in token_scores:
                score = token_scores[token]
                # Normalize by total weight for this intent
                total = sum(token_scores.values())
                score = score / total if total > 0 else 0

                if score > best_score:
                    best_score = score
                    best_intent = intent

        return best_intent, best_score

    def classify_sequence(self, tokens: list[int], window: int = 3) -> tuple[Optional[str], float]:
        """
        Classify a sequence of tokens (more robust than single token).

        Uses voting over recent tokens.

        Args:
            tokens: List of recent token indices
            window: Number of recent tokens to consider

        Returns:
            Tuple of (intent_name, confidence)
        """
        recent = tokens[-window:] if len(tokens) >= window else tokens

        intent_votes: dict[str, float] = {}

        for token in recent:
            intent, conf = self.classify(token)
            if intent is not None:
                intent_votes[intent] = intent_votes.get(intent, 0) + conf

        if not intent_votes:
            return None, 0.0

        best_intent = max(intent_votes, key=intent_votes.get)
        # Normalize confidence by window size
        confidence = intent_votes[best_intent] / len(recent)

        return best_intent, confidence

    def save(self, path: Path):
        """Save intent mappings to file."""
        data = {
            "intent_tokens": {
                k: {str(tk): v for tk, v in tokens.items()}
                for k, tokens in self.intent_tokens.items()
            },
            "intent_priors": self.intent_priors,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path):
        """Load intent mappings from file."""
        with open(path) as f:
            data = json.load(f)

        self.intent_tokens = {
            k: {int(tk): v for tk, v in tokens.items()}
            for k, tokens in data["intent_tokens"].items()
        }
        self.intent_priors = data.get("intent_priors", {})


class EEGTokenizer:
    """
    Complete EEG tokenization pipeline.

    Combines feature extraction, VQ encoding, and intent mapping.
    """

    def __init__(
        self,
        sample_rate: int = 256,
        codebook_size: int = 64,
        use_rvq: bool = False,
        rvq_config: Optional[RVQConfig] = None,
    ):
        from eleven.features import FeatureExtractor

        self.sample_rate = sample_rate
        self.use_rvq = use_rvq
        self.feature_extractor = FeatureExtractor(sample_rate)

        if use_rvq:
            # Use multi-level Residual VQ
            if rvq_config is None:
                rvq_config = RVQConfig(
                    n_levels=3,
                    codebook_sizes=(32, 64, 128),
                    feature_dim=self.feature_extractor.feature_dim,
                )
            self.rvq = ResidualVectorQuantizer(rvq_config)
            # For intent mapping, use the RVQ's combined token
            vq_config = VQConfig(
                codebook_size=rvq_config.codebook_sizes[0],
                feature_dim=rvq_config.feature_dim,
            )
            self.vq = self.rvq.levels[0]  # Use first level for intent mapping
        else:
            # Use single-level K-means VQ
            config = VQConfig(
                codebook_size=codebook_size,
                feature_dim=self.feature_extractor.feature_dim,
            )
            self.vq = VectorQuantizer(config)
            self.rvq = None

        self.intent_mapper = IntentMapper(self.vq)

        # Token history for sequence classification
        self._token_history: list[int] = []
        self._rvq_token_history: list[list[int]] = []  # For RVQ multi-level tokens
        self._max_history = 20

    def tokenize(self, window: np.ndarray) -> int:
        """
        Convert EEG window to discrete token.

        Args:
            window: Shape (4, n_samples) filtered EEG data

        Returns:
            Token index (primary token for intent classification)
        """
        features = self.feature_extractor.extract(window)

        if self.use_rvq and self.rvq is not None:
            # Use RVQ for hierarchical encoding
            rvq_tokens = self.rvq.encode(features)
            token = self.rvq.get_combined_token(rvq_tokens)

            # Store full RVQ tokens for reconstruction
            self._rvq_token_history.append(rvq_tokens)
            if len(self._rvq_token_history) > self._max_history:
                self._rvq_token_history.pop(0)
        else:
            # Single-level VQ
            token = self.vq.encode(features)

        self._token_history.append(token)
        if len(self._token_history) > self._max_history:
            self._token_history.pop(0)

        return token

    def get_rvq_tokens(self) -> Optional[list[int]]:
        """Get the most recent RVQ multi-level tokens."""
        if self._rvq_token_history:
            return self._rvq_token_history[-1]
        return None

    def get_reconstruction_quality(self, window: np.ndarray) -> Optional[dict]:
        """
        Get reconstruction quality metrics for RVQ.

        Args:
            window: Shape (4, n_samples) filtered EEG data

        Returns:
            Dict with reconstruction error metrics, or None if not using RVQ
        """
        if not self.use_rvq or self.rvq is None:
            return None

        features = self.feature_extractor.extract(window)
        return self.rvq.get_reconstruction_error(features.reshape(1, -1))

    def get_intent(self) -> tuple[Optional[str], float]:
        """
        Get current intent based on recent tokens.

        Returns:
            Tuple of (intent_name, confidence)
        """
        return self.intent_mapper.classify_sequence(self._token_history)

    def calibrate(self, training_data: dict[str, list[np.ndarray]]):
        """
        Calibrate tokenizer on labeled EEG windows.

        Args:
            training_data: Dict mapping intent names to lists of EEG windows
        """
        # Collect all features for VQ training
        all_features = []
        intent_features: dict[str, list[np.ndarray]] = {}

        for intent, windows in training_data.items():
            intent_features[intent] = []
            for window in windows:
                features = self.feature_extractor.extract(window)
                all_features.append(features)
                intent_features[intent].append(features)

        all_features = np.array(all_features)

        if self.use_rvq and self.rvq is not None:
            # Train RVQ codebook (all levels)
            self.rvq.fit(all_features)

            # Map tokens to intents using primary (first level) tokens
            for intent, features_list in intent_features.items():
                tokens = []
                for features in features_list:
                    rvq_tokens = self.rvq.encode(features)
                    token = self.rvq.get_combined_token(rvq_tokens)
                    tokens.append(token)

                self.intent_mapper.register_intent(intent, tokens)
        else:
            # Train single-level VQ codebook
            self.vq.fit(all_features)

            # Map tokens to intents
            for intent, features_list in intent_features.items():
                tokens = []
                for features in features_list:
                    token = self.vq.encode(features)
                    tokens.append(token)

                self.intent_mapper.register_intent(intent, tokens)

    def adapt_online(self, window: np.ndarray):
        """
        Adapt codebook online using EMA updates.

        Call this periodically during streaming to adapt to user's patterns.

        Args:
            window: Shape (4, n_samples) filtered EEG data
        """
        if not self.use_rvq or self.rvq is None:
            return

        features = self.feature_extractor.extract(window)
        rvq_tokens = self.rvq.encode(features)

        # Update EMA statistics
        self.rvq.update_ema(
            features.reshape(1, -1),
            np.array([rvq_tokens])
        )

    def save(self, directory: Path):
        """Save tokenizer state to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "use_rvq": self.use_rvq,
            "sample_rate": self.sample_rate,
        }
        with open(directory / "tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2)

        if self.use_rvq and self.rvq is not None:
            self.rvq.save(directory / "rvq")
        else:
            self.vq.save(directory / "codebook")

        self.intent_mapper.save(directory / "intents.json")

    def load(self, directory: Path):
        """Load tokenizer state from directory."""
        directory = Path(directory)

        # Load config
        config_path = directory / "tokenizer_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.use_rvq = config.get("use_rvq", False)
            self.sample_rate = config.get("sample_rate", 256)

        if self.use_rvq:
            rvq_path = directory / "rvq"
            if rvq_path.exists():
                self.rvq = ResidualVectorQuantizer()
                self.rvq.load(rvq_path)
                self.vq = self.rvq.levels[0]
        else:
            self.vq.load(directory / "codebook")

        self.intent_mapper.load(directory / "intents.json")

    def get_stats(self) -> dict:
        """Get comprehensive statistics about the tokenizer."""
        stats = {
            "use_rvq": self.use_rvq,
            "sample_rate": self.sample_rate,
            "feature_dim": self.feature_extractor.feature_dim,
            "token_history_length": len(self._token_history),
        }

        if self.use_rvq and self.rvq is not None:
            stats["rvq_stats"] = self.rvq.get_usage_stats()
            stats["reconstruction_errors"] = self.rvq.reconstruction_errors
        else:
            stats["vq_stats"] = self.vq.get_usage_stats()

        return stats


# CLI for testing
if __name__ == "__main__":
    print("Testing Vector Quantizer...")

    # Create test features
    np.random.seed(42)
    n_samples = 500
    feature_dim = 91

    # Simulate 3 clusters (intents)
    cluster_centers = np.random.randn(3, feature_dim) * 2
    test_data = []
    labels = []

    for i, center in enumerate(cluster_centers):
        samples = center + np.random.randn(n_samples // 3, feature_dim) * 0.5
        test_data.append(samples)
        labels.extend([i] * (n_samples // 3))

    test_data = np.vstack(test_data)
    labels = np.array(labels)

    # Train VQ
    config = VQConfig(codebook_size=16, feature_dim=feature_dim)
    vq = VectorQuantizer(config)
    vq.fit(test_data)

    print(f"Codebook shape: {vq.codebook.shape}")

    # Encode test samples
    tokens = vq.encode_batch(test_data)
    print(f"Tokens shape: {tokens.shape}")
    print(f"Unique tokens used: {len(np.unique(tokens))}")

    # Check cluster separation
    for cluster_id in range(3):
        cluster_tokens = tokens[labels == cluster_id]
        unique, counts = np.unique(cluster_tokens, return_counts=True)
        dominant = unique[np.argmax(counts)]
        purity = np.max(counts) / len(cluster_tokens)
        print(f"Cluster {cluster_id}: dominant token={dominant}, purity={purity:.2%}")

    stats = vq.get_usage_stats()
    print(f"\nUsage stats: {stats}")
