# ear_calculator.py
# Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR) computation
# Adaptive calibration, head-pose estimation, and confidence scoring
# Based on: Soukupová & Čech, 2016 — "Real-Time Eye Blink Detection using Facial Landmarks"

import numpy as np
from collections import deque

# ─── MediaPipe Face Mesh landmark indices ─────────────────────
# Left eye (subject's perspective)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
# Right eye (subject's perspective)
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Mouth landmarks (expanded set for better MAR)
UPPER_LIP_TOP = [13]
LOWER_LIP_BOTTOM = [14]
UPPER_LIP_INNER = [82, 312]     # inner upper lip L/R
LOWER_LIP_INNER = [87, 317]     # inner lower lip L/R
LEFT_MOUTH = [78]
RIGHT_MOUTH = [308]

# Head pose landmarks (nose tip, chin, left/right eye corners, forehead)
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
FOREHEAD = 10

# ─── Default thresholds ──────────────────────────────────────
EAR_DROWSY_THRESHOLD = 0.16   # Lowered: Only detect deep closure
EAR_BORDERLINE_LOW   = 0.14
EAR_BORDERLINE_HIGH  = 0.20
MAR_YAWN_THRESHOLD   = 0.60   # Slightly higher yawn threshold


# ─── Math helpers ────────────────────────────────────────────

def _euclidean(p1, p2):
    """Fast euclidean distance between two 2D points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return (dx * dx + dy * dy) ** 0.5


def compute_ear(eye_landmarks):
    """
    Compute Eye Aspect Ratio from 6 landmark points.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Points layout:
        p2  p3
    p1          p4
        p6  p5

    Args:
        eye_landmarks: list of 6 (x, y) tuples
    Returns:
        float: EAR value (typically 0.0 - 0.4)
    """
    p1, p2, p3, p4, p5, p6 = eye_landmarks

    v1 = _euclidean(p2, p6)
    v2 = _euclidean(p3, p5)
    h = _euclidean(p1, p4)

    if h < 1e-6:
        return 0.0

    return (v1 + v2) / (2.0 * h)


def compute_mar(mouth_pts):
    """
    Compute Mouth Aspect Ratio for yawn detection.
    Uses multiple vertical landmarks for stability.

    Args:
        mouth_pts: dict with keys 'upper', 'lower', 'left', 'right'
                   and optional 'upper_inner_l/r', 'lower_inner_l/r'
    Returns:
        float: MAR value
    """
    vertical = _euclidean(mouth_pts['upper'], mouth_pts['lower'])

    # Use inner lip points if available for more precise measurement
    if 'upper_inner_l' in mouth_pts and 'lower_inner_l' in mouth_pts:
        v2 = _euclidean(mouth_pts['upper_inner_l'], mouth_pts['lower_inner_l'])
        v3 = _euclidean(mouth_pts['upper_inner_r'], mouth_pts['lower_inner_r'])
        vertical = (vertical + v2 + v3) / 3.0

    horizontal = _euclidean(mouth_pts['left'], mouth_pts['right'])

    if horizontal < 1e-6:
        return 0.0

    return vertical / horizontal


# ─── Extraction from MediaPipe ───────────────────────────────

def get_ear_from_face_mesh(face_landmarks, frame_w, frame_h):
    """
    Extract both eyes' EAR from MediaPipe FaceMesh landmarks.

    Returns:
        tuple: (left_ear, right_ear, avg_ear, left_eye_pts, right_eye_pts)
    """
    def get_point(idx):
        lm = face_landmarks.landmark[idx]
        return (lm.x * frame_w, lm.y * frame_h)

    left_eye_pts = [get_point(i) for i in LEFT_EYE_IDX]
    right_eye_pts = [get_point(i) for i in RIGHT_EYE_IDX]

    left_ear = compute_ear(left_eye_pts)
    right_ear = compute_ear(right_eye_pts)
    avg_ear = (left_ear + right_ear) / 2.0

    return left_ear, right_ear, avg_ear, left_eye_pts, right_eye_pts


def get_mar_from_face_mesh(face_landmarks, frame_w, frame_h):
    """
    Extract mouth aspect ratio using expanded landmark set.
    """
    def get_point(idx):
        lm = face_landmarks.landmark[idx]
        return (lm.x * frame_w, lm.y * frame_h)

    mouth = {
        'upper': get_point(UPPER_LIP_TOP[0]),
        'lower': get_point(LOWER_LIP_BOTTOM[0]),
        'left': get_point(LEFT_MOUTH[0]),
        'right': get_point(RIGHT_MOUTH[0]),
        'upper_inner_l': get_point(UPPER_LIP_INNER[0]),
        'upper_inner_r': get_point(UPPER_LIP_INNER[1]),
        'lower_inner_l': get_point(LOWER_LIP_INNER[0]),
        'lower_inner_r': get_point(LOWER_LIP_INNER[1]),
    }

    return compute_mar(mouth)


def get_head_pose(face_landmarks, frame_w, frame_h):
    """
    Estimate head pose (pitch, yaw) from key facial landmarks.
    Uses geometric ratios instead of solvePnP for speed.

    Returns:
        tuple: (pitch, yaw) in approximate degrees
               pitch > 0 = looking down, < 0 = looking up
               yaw > 0 = turned right, < 0 = turned left
    """
    def get_point(idx):
        lm = face_landmarks.landmark[idx]
        return (lm.x * frame_w, lm.y * frame_h)

    nose = get_point(NOSE_TIP)
    chin = get_point(CHIN)
    forehead = get_point(FOREHEAD)
    left_eye = get_point(LEFT_EYE_CORNER)
    right_eye = get_point(RIGHT_EYE_CORNER)
    left_cheek = get_point(LEFT_CHEEK)
    right_cheek = get_point(RIGHT_CHEEK)

    # ── Pitch estimation ──
    # Compare nose-to-chin vs nose-to-forehead ratio
    nose_chin = _euclidean(nose, chin)
    nose_forehead = _euclidean(nose, forehead)

    if nose_forehead < 1e-6:
        pitch = 0.0
    else:
        ratio = nose_chin / nose_forehead
        # Neutral ratio is ~1.0-1.2; higher = looking down, lower = looking up
        pitch = (ratio - 1.1) * 45.0  # rough degrees

    # ── Yaw estimation ──
    # Compare distances from nose to left/right eye corners
    nose_left = _euclidean(nose, left_cheek)
    nose_right = _euclidean(nose, right_cheek)
    total = nose_left + nose_right

    if total < 1e-6:
        yaw = 0.0
    else:
        yaw_ratio = nose_right / total  # 0.5 = centered
        yaw = (yaw_ratio - 0.5) * 90.0  # rough degrees

    return pitch, yaw


# ─── Adaptive EAR Calibrator ─────────────────────────────────

class PersonalEARCalibrator:
    """
    Auto-calibrate EAR threshold per session.
    Collects EAR values during the first N frames to establish a personal baseline.
    """

    def __init__(self, calibration_frames=45, threshold_ratio=0.72):
        """
        Args:
            calibration_frames: number of frames to collect for calibration
            threshold_ratio: ratio of baseline to use as drowsy threshold
        """
        self.calibration_frames = calibration_frames
        self.threshold_ratio = threshold_ratio
        self._samples = []
        self._calibrated = False
        self._baseline = 0.0
        self._threshold = EAR_DROWSY_THRESHOLD  # default until calibrated
        self._borderline_low = EAR_BORDERLINE_LOW
        self._borderline_high = EAR_BORDERLINE_HIGH

    @property
    def is_calibrated(self):
        return self._calibrated

    @property
    def baseline(self):
        return self._baseline

    @property
    def threshold(self):
        return self._threshold

    @property
    def borderline_low(self):
        return self._borderline_low

    @property
    def borderline_high(self):
        return self._borderline_high

    @property
    def progress(self):
        """Calibration progress 0.0 - 1.0"""
        if self._calibrated:
            return 1.0
        return len(self._samples) / self.calibration_frames

    def feed(self, ear_value):
        """
        Feed an EAR value during calibration phase.
        Only accepts values > 0.15 (clearly open eyes).

        Returns:
            bool: True if calibration just completed
        """
        if self._calibrated:
            return False

        # Only accept clearly-open-eye values
        if ear_value > 0.15:
            self._samples.append(ear_value)

        if len(self._samples) >= self.calibration_frames:
            self._finalize()
            return True

        return False

    def _finalize(self):
        """Compute calibrated thresholds from collected samples."""
        if not self._samples:
            self._calibrated = True
            return

        # Use median for robustness against outliers
        self._baseline = float(np.median(self._samples))

        # Set thresholds relative to personal baseline
        self._threshold = max(0.15, self._baseline * self.threshold_ratio)
        self._borderline_low = max(0.12, self._threshold - 0.03)
        self._borderline_high = self._baseline * 0.88

        self._calibrated = True

    def get_status_text(self):
        """Get a human-readable calibration status."""
        if self._calibrated:
            return f"Calibrated (baseline: {self._baseline:.2f}, threshold: {self._threshold:.2f})"
        pct = int(self.progress * 100)
        return f"Calibrating... {pct}%"


# ─── Threshold check functions ────────────────────────────────

def is_drowsy(ear, threshold=EAR_DROWSY_THRESHOLD):
    """Check if EAR indicates drowsiness."""
    return ear < threshold


def is_borderline(ear, low=EAR_BORDERLINE_LOW, high=EAR_BORDERLINE_HIGH):
    """Check if EAR is in the borderline zone (might need CNN confirmation)."""
    return low <= ear <= high


def is_yawning(mar, threshold=MAR_YAWN_THRESHOLD):
    """Check if MAR indicates yawning."""
    return mar > threshold
