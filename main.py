# main.py — Driver Drowsiness Detection: Professional Real-Time Monitor
# Threaded camera capture, modular detection engine, premium UI

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os
import sys
import time
import threading
from collections import deque

from detection_engine import DrowsinessDetector, STATE_AWAKE, STATE_WARNING, STATE_ALERT
from alert_manager import AlertManager
from ui_overlay import draw_hud, STATUS_AWAKE, STATUS_WARNING, STATUS_ALERT
from utils import SessionLogger

# ─── CONFIG ───────────────────────────────────────────────────
MODEL_PATH      = "drowsiness_model_final.h5"
ALARM_SOUND     = "alert.mp3"
FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
IMG_SIZE        = (64, 64)
DISPLAY_SIZE    = (960, 540)
PROCESS_SIZE    = (640, 480)


# ─── MediaPipe Compatibility Adapter ─────────────────────────
# mediapipe >= 0.10.14 removed mp.solutions.face_mesh.
# The Tasks API returns landmarks as plain lists of NormalizedLandmark
# (which still have .x/.y/.z). We wrap them so detection_engine.py,
# ear_calculator.py and ui_overlay.py still see:
#   results.multi_face_landmarks[0].landmark[idx].x

class _FakeLandmarkList:
    """Wraps a list[NormalizedLandmark] to expose a .landmark attribute."""
    __slots__ = ("landmark",)
    def __init__(self, lm_list):
        self.landmark = lm_list  # NormalizedLandmark objects have .x .y .z


class _FakeResults:
    """Makes FaceLandmarkerResult look like old mp.solutions.face_mesh result."""
    __slots__ = ("multi_face_landmarks",)
    def __init__(self, face_landmarker_result):
        faces = face_landmarker_result.face_landmarks
        if faces:
            self.multi_face_landmarks = [_FakeLandmarkList(f) for f in faces]
        else:
            self.multi_face_landmarks = []


# ─── Threaded Camera Capture ─────────────────────────────────
class CameraThread:
    """Captures frames in a background thread for decoupled FPS."""

    def __init__(self, source):
        try:
            src = int(source)
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        except ValueError:
            self.cap = cv2.VideoCapture(source)
        self.source = source

        if not self.cap.isOpened():
            print("ERROR: Could not open camera/video!")
            sys.exit(1)

        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                # Loop video files
                try:
                    int(self.source)
                except ValueError:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            with self._lock:
                self._frame = frame

    def read(self):
        """Get the latest frame (non-blocking)."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


# ─── State Mapping ───────────────────────────────────────────
_UI_STATUS = {
    STATE_AWAKE: STATUS_AWAKE,
    STATE_WARNING: STATUS_WARNING,
    STATE_ALERT: STATUS_ALERT,
}


# ─── Main Application ────────────────────────────────────────
def main(video_source):
    # ── Load CNN model ──
    cnn_model = None
    if os.path.exists(MODEL_PATH):
        try:
            from tensorflow.keras.models import load_model
            cnn_model = load_model(MODEL_PATH)
            print(f"[OK] CNN model loaded: {MODEL_PATH}")
        except Exception as e:
            print(f"[WARN] Model load failed: {e}")
    else:
        print(f"[INFO] No CNN model at {MODEL_PATH}, using EAR-only detection.")

    # ── Initialize components ──
    detector = DrowsinessDetector(cnn_model=cnn_model, img_size=IMG_SIZE)
    alert_mgr = AlertManager(alarm_path=ALARM_SOUND)
    logger = SessionLogger()

    # ── Ensure face model is present ──
    if not os.path.exists(FACE_MODEL_PATH):
        import urllib.request
        print(f"[INFO] Downloading face landmarker model...")
        try:
            urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
            print("[OK] Face landmarker model downloaded.")
        except Exception as e:
            print(f"[ERROR] Cannot download face model: {e}")
            sys.exit(1)

    # ── MediaPipe Tasks FaceLandmarker Setup ──
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision

    base_options = _mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    fl_options   = _mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    face_landmarker = _mp_vision.FaceLandmarker.create_from_options(fl_options)

    # ── Camera ──
    camera = CameraThread(video_source)

    # ── FPS tracking ──
    fps = 0.0
    fps_timer = time.time()
    fps_count = 0
    frame_count = 0

    print("\n" + "=" * 52)
    print("  DRIVER SAFETY MONITOR — Active")
    print("  Press Q to quit | S to stop alarm")
    print("=" * 52 + "\n")

    try:
        while True:
            raw_frame = camera.read()
            if raw_frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            fps_count += 1

            # ── Resize for processing ──
            proc_frame = cv2.resize(raw_frame, PROCESS_SIZE, interpolation=cv2.INTER_AREA)
            h, w = proc_frame.shape[:2]

            # ── FPS calculation ──
            elapsed = time.time() - fps_timer
            if elapsed >= 0.5:
                fps = fps_count / elapsed
                fps_count = 0
                fps_timer = time.time()

            # ── Face landmark detection ──
            rgb      = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results  = _FakeResults(face_landmarker.detect(mp_image))

            # ── Detection engine ──
            state = detector.process_frame(proc_frame, results, w, h)

            # ── Alert system ──
            alert_mgr.update(state, detector.drowsiness_score)

            # ── Logging ──
            if state == STATE_ALERT and not logger.was_alert:
                logger.log_event("Drowsiness ALERT triggered")
            elif state == STATE_WARNING and not logger.was_warning:
                logger.log_event("Drowsiness WARNING")
            logger.was_alert = (state == STATE_ALERT)
            logger.was_warning = (state == STATE_WARNING)

            # ── Resize for display ──
            display = cv2.resize(proc_frame, DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)

            # ── Draw HUD ──
            ui_status = _UI_STATUS.get(state, STATUS_AWAKE)
            draw_hud(
                display,
                status=ui_status,
                ear=detector.current_ear,
                mar=detector.current_mar,
                score=detector.drowsiness_score,
                fps=fps,
                detector=detector,
                frame_count=frame_count,
            )

            # ── Show ──
            cv2.imshow("Driver Safety Monitor", display)

            # ── Key handling ──
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            if key in (ord('s'), ord('S')):
                alert_mgr.force_stop()
                detector.drowsiness_score = 0.0
                detector.state = STATE_AWAKE

    except KeyboardInterrupt:
        pass
    finally:
        # ── Cleanup ──
        camera.release()
        cv2.destroyAllWindows()
        face_landmarker.close()
        alert_mgr.cleanup()
        logger.write_summary(detector)
        print("\n[Session ended]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection System")
    parser.add_argument("--video", default="0",
                        help="Webcam index (0,1,...) or video file path")
    args = parser.parse_args()
    main(args.video)
