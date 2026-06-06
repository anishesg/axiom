"""Gaze Tracker — webcam-based eye tracking using MediaPipe Face Mesh.

Uses 478 face landmarks (including 10 iris landmarks) to estimate
where the user is looking on screen.

Pipeline:
  1. MediaPipe Face Mesh → iris + eye landmarks
  2. Extract gaze features (iris position, head pose)
  3. Map to screen coordinates via calibrated ridge regression

Accuracy: ~50-100px on a 2560x1440 display after 9-point calibration.
Sufficient for quadrant/region detection and UI element targeting.
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from sklearn.linear_model import Ridge
import json
import os

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
FOREHEAD = 10

# 3D model points for head pose estimation (relative proportions)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Left eye corner
    (43.3, 32.7, -26.0),      # Right eye corner
    (-28.9, -28.9, -24.1),    # Left mouth corner
    (28.9, -28.9, -24.1),     # Right mouth corner
], dtype=np.float64)

POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]


@dataclass
class GazeResult:
    screen_x: float = 0.0
    screen_y: float = 0.0
    left_iris_x: float = 0.0
    left_iris_y: float = 0.0
    right_iris_x: float = 0.0
    right_iris_y: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    confidence: float = 0.0
    calibrated: bool = False
    region: str = ""


class GazeTracker:
    def __init__(self, screen_w: int = 2560, screen_h: int = 1440,
                 camera_id: int = 0):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.camera_id = camera_id

        self._mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._cap = None
        self._model_x = None
        self._model_y = None
        self._calibration_data = []
        self._calibrated = False

        self._frame_w = 640
        self._frame_h = 480

    def start(self):
        self._cap = cv2.VideoCapture(self.camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._frame_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def stop(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    def _extract_features(self, landmarks) -> np.ndarray:
        """Extract gaze features from face landmarks."""
        left_iris = np.mean([[landmarks[i].x, landmarks[i].y]
                             for i in LEFT_IRIS], axis=0)
        right_iris = np.mean([[landmarks[i].x, landmarks[i].y]
                              for i in RIGHT_IRIS], axis=0)

        left_corner_inner = np.array([landmarks[133].x, landmarks[133].y])
        left_corner_outer = np.array([landmarks[33].x, landmarks[33].y])
        right_corner_inner = np.array([landmarks[362].x, landmarks[362].y])
        right_corner_outer = np.array([landmarks[263].x, landmarks[263].y])

        left_eye_w = np.linalg.norm(left_corner_outer - left_corner_inner)
        right_eye_w = np.linalg.norm(right_corner_outer - right_corner_inner)

        # Iris position relative to eye corners (0=outer, 1=inner)
        left_rel_x = 0.0 if left_eye_w < 1e-6 else (left_iris[0] - left_corner_outer[0]) / left_eye_w
        left_rel_y = left_iris[1] - (left_corner_outer[1] + left_corner_inner[1]) / 2

        right_rel_x = 0.0 if right_eye_w < 1e-6 else (right_iris[0] - right_corner_inner[0]) / right_eye_w
        right_rel_y = right_iris[1] - (right_corner_inner[1] + right_corner_outer[1]) / 2

        # Head pose
        image_points = np.array([
            [landmarks[i].x * self._frame_w, landmarks[i].y * self._frame_h]
            for i in POSE_LANDMARKS
        ], dtype=np.float64)

        focal_length = self._frame_w
        center = (self._frame_w / 2, self._frame_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        _, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix,
                                   dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rvec)
        angles = cv2.decomposeProjectionMatrix(
            np.hstack((rmat, np.zeros((3, 1))))
        )[6]
        pitch, yaw, roll = angles.flatten()[:3]

        return np.array([
            left_rel_x, left_rel_y,
            right_rel_x, right_rel_y,
            left_iris[0], left_iris[1],
            right_iris[0], right_iris[1],
            yaw / 90.0, pitch / 90.0, roll / 90.0,
        ])

    def process_frame(self) -> GazeResult:
        """Capture a frame and estimate gaze position."""
        if not self._cap or not self._cap.isOpened():
            return GazeResult()

        ret, frame = self._cap.read()
        if not ret:
            return GazeResult()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_face.process(frame_rgb)

        if not results.multi_face_landmarks:
            return GazeResult(confidence=0.0)

        landmarks = results.multi_face_landmarks[0].landmark
        features = self._extract_features(landmarks)

        left_iris = np.mean([[landmarks[i].x, landmarks[i].y]
                             for i in LEFT_IRIS], axis=0)
        right_iris = np.mean([[landmarks[i].x, landmarks[i].y]
                              for i in RIGHT_IRIS], axis=0)

        result = GazeResult(
            left_iris_x=float(left_iris[0]),
            left_iris_y=float(left_iris[1]),
            right_iris_x=float(right_iris[0]),
            right_iris_y=float(right_iris[1]),
            head_yaw=float(features[8] * 90),
            head_pitch=float(features[9] * 90),
            head_roll=float(features[10] * 90),
            confidence=0.5,
        )

        if self._calibrated and self._model_x is not None:
            feat = features.reshape(1, -1)
            result.screen_x = float(np.clip(self._model_x.predict(feat)[0],
                                             0, self.screen_w))
            result.screen_y = float(np.clip(self._model_y.predict(feat)[0],
                                             0, self.screen_h))
            result.calibrated = True
            result.confidence = 0.8
            result.region = self._classify_region(result.screen_x, result.screen_y)

        return result

    def add_calibration_point(self, screen_x: float, screen_y: float) -> bool:
        """Add a calibration point — call while user looks at (screen_x, screen_y).

        Returns True when enough points are collected to calibrate.
        """
        if not self._cap or not self._cap.isOpened():
            return False

        features_list = []
        for _ in range(10):
            ret, frame = self._cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._mp_face.process(frame_rgb)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                features_list.append(self._extract_features(landmarks))

        if len(features_list) < 5:
            return False

        avg_features = np.mean(features_list, axis=0)
        self._calibration_data.append((avg_features, screen_x, screen_y))

        return len(self._calibration_data) >= 5

    def calibrate(self) -> bool:
        """Fit the gaze model from collected calibration points."""
        if len(self._calibration_data) < 5:
            return False

        X = np.array([d[0] for d in self._calibration_data])
        y_x = np.array([d[1] for d in self._calibration_data])
        y_y = np.array([d[2] for d in self._calibration_data])

        self._model_x = Ridge(alpha=1.0)
        self._model_y = Ridge(alpha=1.0)
        self._model_x.fit(X, y_x)
        self._model_y.fit(X, y_y)
        self._calibrated = True
        return True

    def save_calibration(self, path: str = "data/gaze_calibration.json"):
        """Save calibration data for reuse across sessions."""
        if not self._calibrated:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "model_x_coef": self._model_x.coef_.tolist(),
            "model_x_intercept": float(self._model_x.intercept_),
            "model_y_coef": self._model_y.coef_.tolist(),
            "model_y_intercept": float(self._model_y.intercept_),
            "screen_w": self.screen_w,
            "screen_h": self.screen_h,
            "n_points": len(self._calibration_data),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_calibration(self, path: str = "data/gaze_calibration.json") -> bool:
        """Load saved calibration."""
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)

        self._model_x = Ridge(alpha=1.0)
        self._model_y = Ridge(alpha=1.0)
        self._model_x.coef_ = np.array(data["model_x_coef"])
        self._model_x.intercept_ = data["model_x_intercept"]
        self._model_y.coef_ = np.array(data["model_y_coef"])
        self._model_y.intercept_ = data["model_y_intercept"]
        self._calibrated = True
        return True

    def _classify_region(self, x: float, y: float) -> str:
        """Classify screen position into a named region."""
        col = "left" if x < self.screen_w / 3 else ("right" if x > 2 * self.screen_w / 3 else "center")
        row = "top" if y < self.screen_h / 3 else ("bottom" if y > 2 * self.screen_h / 3 else "middle")

        if y > self.screen_h * 0.9:
            return "dock"
        if y < self.screen_h * 0.05:
            return "menu_bar"

        return f"{row}-{col}"

    def get_calibration_targets(self, n: int = 9) -> list[tuple[float, float]]:
        """Generate screen positions for calibration targets."""
        margin = 0.1
        if n == 5:
            return [
                (self.screen_w * 0.5, self.screen_h * 0.5),
                (self.screen_w * margin, self.screen_h * margin),
                (self.screen_w * (1 - margin), self.screen_h * margin),
                (self.screen_w * margin, self.screen_h * (1 - margin)),
                (self.screen_w * (1 - margin), self.screen_h * (1 - margin)),
            ]
        # 9-point grid
        cols = [margin, 0.5, 1 - margin]
        rows = [margin, 0.5, 1 - margin]
        return [(self.screen_w * c, self.screen_h * r)
                for r in rows for c in cols]

    def process_synthetic(self, iris_x: float = 0.5, iris_y: float = 0.5,
                          head_yaw: float = 0.0, head_pitch: float = 0.0) -> GazeResult:
        """For simulation — process synthetic gaze data without a camera."""
        features = np.array([
            iris_x, iris_y,
            iris_x, iris_y,
            iris_x, iris_y,
            iris_x, iris_y,
            head_yaw / 90.0, head_pitch / 90.0, 0.0,
        ])

        result = GazeResult(
            left_iris_x=iris_x,
            left_iris_y=iris_y,
            right_iris_x=iris_x,
            right_iris_y=iris_y,
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            confidence=0.8,
        )

        if self._calibrated and self._model_x is not None:
            feat = features.reshape(1, -1)
            result.screen_x = float(np.clip(self._model_x.predict(feat)[0],
                                             0, self.screen_w))
            result.screen_y = float(np.clip(self._model_y.predict(feat)[0],
                                             0, self.screen_h))
            result.calibrated = True
            result.region = self._classify_region(result.screen_x, result.screen_y)

        return result
