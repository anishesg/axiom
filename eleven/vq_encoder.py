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
    ):
        from eleven.features import FeatureExtractor

        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractor(sample_rate)

        # Initialize VQ with appropriate feature dimension
        config = VQConfig(
            codebook_size=codebook_size,
            feature_dim=self.feature_extractor.feature_dim,
        )
        self.vq = VectorQuantizer(config)
        self.intent_mapper = IntentMapper(self.vq)

        # Token history for sequence classification
        self._token_history: list[int] = []
        self._max_history = 20

    def tokenize(self, window: np.ndarray) -> int:
        """
        Convert EEG window to discrete token.

        Args:
            window: Shape (4, n_samples) filtered EEG data

        Returns:
            Token index
        """
        features = self.feature_extractor.extract(window)
        token = self.vq.encode(features)

        self._token_history.append(token)
        if len(self._token_history) > self._max_history:
            self._token_history.pop(0)

        return token

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

        # Train VQ codebook
        self.vq.fit(all_features)

        # Map tokens to intents
        for intent, features_list in intent_features.items():
            tokens = []
            for features in features_list:
                token = self.vq.encode(features)
                tokens.append(token)

            self.intent_mapper.register_intent(intent, tokens)

    def save(self, directory: Path):
        """Save tokenizer state to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.vq.save(directory / "codebook")
        self.intent_mapper.save(directory / "intents.json")

    def load(self, directory: Path):
        """Load tokenizer state from directory."""
        directory = Path(directory)

        self.vq.load(directory / "codebook")
        self.intent_mapper.load(directory / "intents.json")


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
