# detection_engine.py
# Centralized drowsiness detection state machine with weighted scoring

import time
import threading
import numpy as np
from collections import deque

from ear_calculator import (
    PersonalEARCalibrator,
    get_ear_from_face_mesh,
    get_mar_from_face_mesh,
    get_head_pose,
    is_drowsy,
    is_borderline,
    is_yawning,
)

# Detection States
STATE_AWAKE = 0
STATE_WARNING = 1
STATE_ALERT = 2

# Score Weights
SCORE_EAR_CLOSED       = 0.02   # Glacial build-up (very safe)
SCORE_EAR_BORDERLINE_CNN = 0.01
SCORE_YAWN             = 0.005
SCORE_DECAY            = 0.40   # Immediate reset (very aggressive)

# Blink filter — ignore closures shorter than this (1.5 seconds)
MIN_CLOSED_FRAMES = 45   # Deep sustained closure only

# State Thresholds
THRESHOLD_WARNING = 0.50   # Large safety zone
THRESHOLD_ALERT   = 0.85   # Very high threshold
THRESHOLD_AWAKE   = 0.15

# CNN Config
CNN_CACHE_FRAMES = 4


class DrowsinessDetector:
    """
    Core detection engine using weighted multi-signal scoring.
    Accumulates a drowsiness score (0-1) from EAR, CNN, MAR signals.
    """

    def __init__(self, cnn_model=None, img_size=(64, 64)):
        self.cnn_model = cnn_model
        self.img_size = img_size
        self.calibrator = PersonalEARCalibrator(calibration_frames=45, threshold_ratio=0.65)

        # State
        self.state = STATE_AWAKE
        self.drowsiness_score = 0.0
        self.confidence = 1.0

        # Current frame values
        self.current_ear = 0.0
        self.current_mar = 0.0
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.face_detected = False
        self.left_eye_pts = None
        self.right_eye_pts = None
        self.face_landmarks = None

        # History
        self.ear_history = deque(maxlen=90)
        self.mar_history = deque(maxlen=30)
        self.score_history = deque(maxlen=90)
        self.ear_smooth_buf = deque(maxlen=3)

        # CNN threading
        self._cnn_result = None
        self._cnn_frame_age = 999
        self._cnn_lock = threading.Lock()
        self._cnn_running = False

        # Yawn tracking
        self.yawn_frames = 0
        self.total_yawns = 0
        self._was_yawning = False

        # Blink filter
        self._consec_ear_closed = 0

        # Drowsy event tracking
        self.total_drowsy_events = 0
        self._was_alert = False

        # Timing
        self.session_start = time.time()
        self._no_face_frames = 0

    def process_frame(self, frame, face_mesh_results, frame_w, frame_h):
        """Process one video frame and update detection state. Returns current state."""
        self.face_detected = False
        self.face_landmarks = None
        self.left_eye_pts = None
        self.right_eye_pts = None
        self.current_ear = 0.0
        self.current_mar = 0.0

        if face_mesh_results.multi_face_landmarks:
            face_lm = face_mesh_results.multi_face_landmarks[0]
            self.face_landmarks = face_lm
            self.face_detected = True
            self._no_face_frames = 0

            left_ear, right_ear, avg_ear, l_pts, r_pts = get_ear_from_face_mesh(face_lm, frame_w, frame_h)
            self.left_eye_pts = l_pts
            self.right_eye_pts = r_pts

            mar = get_mar_from_face_mesh(face_lm, frame_w, frame_h)
            pitch, yaw = get_head_pose(face_lm, frame_w, frame_h)
            self.current_mar = mar
            self.current_pitch = pitch
            self.current_yaw = yaw

            # Head pose compensation
            pose_penalty = 0.03 if (abs(pitch) > 15 or abs(yaw) > 20) else 0.0
            adjusted_ear = avg_ear + pose_penalty

            # Smoothing
            self.ear_smooth_buf.append(adjusted_ear)
            smoothed_ear = float(np.mean(self.ear_smooth_buf))
            self.current_ear = smoothed_ear

            # Calibration
            self.calibrator.feed(smoothed_ear)
            ear_threshold = self.calibrator.threshold
            bl_low = self.calibrator.borderline_low
            bl_high = self.calibrator.borderline_high

            self.ear_history.append(smoothed_ear)
            self.mar_history.append(mar)
            self._update_confidence()

            # CNN for borderline
            self._cnn_frame_age += 1
            if (self.cnn_model is not None
                    and is_borderline(smoothed_ear, bl_low, bl_high)
                    and self._cnn_frame_age >= CNN_CACHE_FRAMES
                    and not self._cnn_running):
                self._run_cnn_async(frame, face_lm, frame_w, frame_h)

            # Update drowsiness score
            self._update_score(smoothed_ear, mar, ear_threshold, bl_low, bl_high)

            # Yawn tracking
            yawning_now = is_yawning(mar)
            if yawning_now:
                self.yawn_frames += 1
            else:
                if self._was_yawning and self.yawn_frames > 10:
                    self.total_yawns += 1
                self.yawn_frames = 0
            self._was_yawning = yawning_now
        else:
            self._no_face_frames += 1
            self.ear_history.append(0.0)
            if self._no_face_frames > 15:
                self.drowsiness_score = max(0, self.drowsiness_score - 0.02)

        self._update_state()

        if self.state == STATE_ALERT and not self._was_alert:
            self.total_drowsy_events += 1
        self._was_alert = (self.state == STATE_ALERT)

        self.score_history.append(self.drowsiness_score)
        return self.state

    def get_session_duration(self):
        return time.time() - self.session_start

    def get_avg_ear(self):
        vals = [v for v in self.ear_history if v > 0]
        return float(np.mean(vals)) if vals else 0.0

    def _update_score(self, ear, mar, threshold, bl_low, bl_high):
        delta = 0.0
        
        if ear < threshold:
            self._consec_ear_closed += 1
            # Only start penalizing after eyes have been closed for a significant sustained period
            if self._consec_ear_closed >= MIN_CLOSED_FRAMES:
                delta += SCORE_EAR_CLOSED
        elif is_borderline(ear, bl_low, bl_high):
            # No penalty for borderline eyes, just slowly reset the consecutive counter
            self._consec_ear_closed = max(0, self._consec_ear_closed - 1)
            delta -= (SCORE_DECAY * 0.2) # Slight decay even in borderline
        else:
            # Eyes are clearly open: INSTANT reset of consecutive counter and fast score decay
            self._consec_ear_closed = 0
            delta -= SCORE_DECAY

        if is_yawning(mar):
            delta += (SCORE_YAWN * 0.2) # Minimal yawn impact

        if delta > 0:
            delta *= self.confidence

        self.drowsiness_score = float(np.clip(self.drowsiness_score + delta, 0.0, 1.0))

    def _update_state(self):
        score = self.drowsiness_score
        if self.state == STATE_AWAKE:
            if score >= THRESHOLD_ALERT:
                self.state = STATE_ALERT
            elif score >= THRESHOLD_WARNING:
                self.state = STATE_WARNING
        elif self.state == STATE_WARNING:
            if score >= THRESHOLD_ALERT:
                self.state = STATE_ALERT
            elif score < THRESHOLD_AWAKE:
                self.state = STATE_AWAKE
        elif self.state == STATE_ALERT:
            if score < THRESHOLD_WARNING * 0.6:
                self.state = STATE_AWAKE

    def _update_confidence(self):
        confidence = 1.0
        if len(self.ear_history) >= 5:
            recent = list(self.ear_history)[-5:]
            variance = float(np.var(recent))
            if variance > 0.005:
                confidence *= 0.7
            elif variance > 0.002:
                confidence *= 0.85
        if abs(self.current_pitch) > 25:
            confidence *= 0.6
        elif abs(self.current_pitch) > 15:
            confidence *= 0.8
        if abs(self.current_yaw) > 30:
            confidence *= 0.6
        elif abs(self.current_yaw) > 20:
            confidence *= 0.8
        self.confidence = confidence

    def _run_cnn_async(self, frame, face_lm, frame_w, frame_h):
        eye_roi = self._extract_eye_roi(frame, face_lm, frame_w, frame_h)
        if eye_roi is None or eye_roi.size == 0:
            return
        self._cnn_running = True

        def _infer():
            try:
                import cv2
                gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, self.img_size)
                arr = resized.astype("float32") / 255.0
                arr = np.expand_dims(arr, axis=-1)
                arr = np.expand_dims(arr, axis=0)
                pred = self.cnn_model(arr, training=False)
                result = float(pred.numpy()[0][0])
                with self._cnn_lock:
                    self._cnn_result = result
                    self._cnn_frame_age = 0
            except Exception:
                pass
            finally:
                self._cnn_running = False

        threading.Thread(target=_infer, daemon=True).start()

    @staticmethod
    def _extract_eye_roi(frame, face_landmarks, frame_w, frame_h):
        eye_indices = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
        xs, ys = [], []
        for idx in eye_indices:
            lm = face_landmarks.landmark[idx]
            xs.append(int(lm.x * frame_w))
            ys.append(int(lm.y * frame_h))
        pad = 10
        x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
        x2, y2 = min(frame_w, max(xs) + pad), min(frame_h, max(ys) + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()
