# ui_overlay.py — Premium glassmorphism HUD for Driver Drowsiness Detection
import cv2
import numpy as np
import time
import math

# ─── Status Constants ─────────────────────────────────────────
STATUS_AWAKE = 0
STATUS_WARNING = 1
STATUS_ALERT = 2

# ─── Color Palette (BGR) ─────────────────────────────────────
C_BG = (18, 18, 22)
C_PANEL = (28, 28, 34)
C_BORDER = (50, 50, 58)
C_GREEN = (100, 220, 80)
C_GREEN_B = (80, 255, 100)
C_AMBER = (50, 200, 240)
C_AMBER_B = (40, 220, 255)
C_RED = (65, 55, 230)
C_RED_B = (50, 50, 255)
C_WHITE = (235, 235, 235)
C_GRAY = (140, 140, 140)
C_GRAY_D = (80, 80, 80)
C_CYAN = (210, 190, 50)
C_CYAN_D = (140, 130, 35)
C_ORANGE = (40, 150, 235)

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX

# ─── Animation state (module-level for persistence) ──────────
_anim = {
    'score_display': 0.0,
    'ear_display': 0.25,
    'vignette_cache': None,
    'vignette_size': None,
}


def _lerp(current, target, speed=0.15):
    return current + (target - current) * speed


def _draw_rect(frame, x, y, w, h, color, alpha=0.6):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _draw_rounded(frame, x, y, w, h, color, r=10, alpha=0.6):
    overlay = frame.copy()
    cv2.ellipse(overlay, (x+r, y+r), (r, r), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x+w-r, y+r), (r, r), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x+w-r, y+h-r), (r, r), 0, 0, 90, color, -1)
    cv2.ellipse(overlay, (x+r, y+h-r), (r, r), 90, 0, 90, color, -1)
    cv2.rectangle(overlay, (x+r, y), (x+w-r, y+h), color, -1)
    cv2.rectangle(overlay, (x, y+r), (x+w, y+h-r), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _status_color(status):
    if status == STATUS_ALERT:
        return C_RED_B
    elif status == STATUS_WARNING:
        return C_AMBER_B
    return C_GREEN_B


def _ear_color(ear, threshold=0.21):
    if ear >= threshold + 0.04:
        return C_GREEN
    elif ear >= threshold:
        return C_AMBER
    return C_RED


# ─── Arc Gauge ────────────────────────────────────────────────
def _draw_arc_gauge(frame, cx, cy, radius, value, max_val=1.0, status=0, fc=0):
    """Draw a circular arc gauge for drowsiness score."""
    ratio = min(1.0, max(0.0, value / max_val))
    start_angle = 135
    sweep = 270

    # Background arc
    cv2.ellipse(frame, (cx, cy), (radius, radius), 0,
                start_angle, start_angle + sweep, C_GRAY_D, 2, cv2.LINE_AA)

    # Filled arc
    if ratio > 0.01:
        fill_angle = int(sweep * ratio)
        color = _status_color(status)
        thickness = 4 if status == STATUS_ALERT else 3
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0,
                    start_angle, start_angle + fill_angle, color, thickness, cv2.LINE_AA)

    # Center percentage text
    pct = int(ratio * 100)
    txt = f"{pct}%"
    sz = cv2.getTextSize(txt, FONT, 0.55, 1)[0]
    cv2.putText(frame, txt, (cx - sz[0]//2, cy + 6),
                FONT, 0.55, _status_color(status), 1, cv2.LINE_AA)

    # Label below
    cv2.putText(frame, "DROWSY", (cx - 22, cy + radius + 14),
                FONT_S, 0.32, C_GRAY, 1, cv2.LINE_AA)


# ─── Sparkline ────────────────────────────────────────────────
def _draw_sparkline(frame, data, x, y, w, h, color, threshold=None):
    """Draw a mini line chart."""
    if len(data) < 2:
        return
    _draw_rect(frame, x, y, w, h, C_BG, alpha=0.4)

    vals = list(data)
    mn, mx = 0.0, max(0.4, max(vals) * 1.2)
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        px = x + int(i * w / max(1, n - 1))
        py = y + h - int((v - mn) / (mx - mn) * h)
        py = max(y, min(y + h, py))
        pts.append((px, py))

    for i in range(len(pts) - 1):
        cv2.line(frame, pts[i], pts[i+1], color, 1, cv2.LINE_AA)

    if threshold is not None:
        ty = y + h - int((threshold - mn) / (mx - mn) * h)
        ty = max(y, min(y + h, ty))
        cv2.line(frame, (x, ty), (x + w, ty), C_RED, 1, cv2.LINE_AA)


# ─── Status Dot ───────────────────────────────────────────────
def _draw_status_dot(frame, x, y, status, fc):
    color = _status_color(status)
    if status == STATUS_ALERT:
        pulse = int(4 * abs(math.sin(fc * 0.3)))
        r = 8 + pulse
    elif status == STATUS_WARNING:
        pulse = int(2 * math.sin(fc * 0.15))
        r = 7 + pulse
    else:
        r = 7
    glow = tuple(max(0, c - 80) for c in color)
    cv2.circle(frame, (x, y), r + 4, glow, -1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), r, color, -1, cv2.LINE_AA)
    hi = tuple(min(255, c + 50) for c in color)
    cv2.circle(frame, (x - 2, y - 2), max(2, r // 3), hi, -1, cv2.LINE_AA)


# ─── EAR Gauge Bar ───────────────────────────────────────────
def _draw_ear_bar(frame, ear, x, y, w=140, h=10, threshold=0.21):
    max_ear = 0.38
    _draw_rect(frame, x, y, w, h, C_GRAY_D, 0.5)
    ratio = min(1.0, max(0.0, ear / max_ear))
    fw = int(w * ratio)
    if fw > 0:
        cv2.rectangle(frame, (x, y), (x + fw, y + h),
                      _ear_color(ear, threshold), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + w, y + h), C_BORDER, 1, cv2.LINE_AA)
    tx = x + int(w * (threshold / max_ear))
    cv2.line(frame, (tx, y - 2), (tx, y + h + 2), C_WHITE, 1, cv2.LINE_AA)


# ─── Left Dashboard Panel ────────────────────────────────────
def _draw_left_panel(frame, status, ear, mar, score, fps, detector, fc):
    """Draw the main left-side glassmorphism dashboard."""
    fh, fw = frame.shape[:2]
    pw = 195
    ph = fh - 60
    px, py = 8, 35

    # Frosted glass panel
    _draw_rounded(frame, px, py, pw, ph, C_PANEL, r=12, alpha=0.70)

    # Panel border glow based on status
    border_color = _status_color(status)
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + pw, py + ph), border_color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # ── Status section ──
    sx = px + 15
    sy = py + 22
    _draw_status_dot(frame, sx, sy, status, fc)

    labels = {STATUS_AWAKE: "MONITORING", STATUS_WARNING: "CAUTION", STATUS_ALERT: "DANGER"}
    cv2.putText(frame, labels.get(status, "---"), (sx + 18, sy + 5),
                FONT, 0.48, _status_color(status), 1, cv2.LINE_AA)

    # ── Arc gauge ──
    _anim['score_display'] = _lerp(_anim['score_display'], score, 0.12)
    _draw_arc_gauge(frame, px + pw // 2, py + 90, 35,
                    _anim['score_display'], 1.0, status, fc)

    # ── EAR section ──
    ey = py + 140
    _anim['ear_display'] = _lerp(_anim['ear_display'], ear, 0.2)
    disp_ear = _anim['ear_display']
    threshold = detector.calibrator.threshold if detector else 0.21

    cv2.putText(frame, "EAR", (sx, ey), FONT_S, 0.35, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{disp_ear:.3f}", (sx + 35, ey),
                FONT, 0.45, _ear_color(disp_ear, threshold), 1, cv2.LINE_AA)
    _draw_ear_bar(frame, disp_ear, sx, ey + 8, w=pw - 32, h=8, threshold=threshold)

    # ── EAR Sparkline ──
    if detector and len(detector.ear_history) > 2:
        _draw_sparkline(frame, detector.ear_history,
                        sx, ey + 22, pw - 32, 30, C_CYAN, threshold)

    # ── MAR / Yawn ──
    my = ey + 60
    mar_color = C_ORANGE if mar > 0.55 else C_GREEN
    cv2.putText(frame, "MAR", (sx, my), FONT_S, 0.35, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{mar:.2f}", (sx + 35, my), FONT, 0.42, mar_color, 1, cv2.LINE_AA)

    if detector:
        yawn_txt = f"Yawns: {detector.total_yawns}"
        cv2.putText(frame, yawn_txt, (sx + 100, my), FONT_S, 0.30, C_GRAY, 1, cv2.LINE_AA)

    # ── Head Pose ──
    hp_y = my + 22
    cv2.putText(frame, "Pose", (sx, hp_y), FONT_S, 0.32, C_GRAY, 1, cv2.LINE_AA)
    if detector:
        p_txt = f"P:{detector.current_pitch:+.0f} Y:{detector.current_yaw:+.0f}"
        pose_ok = abs(detector.current_pitch) < 20 and abs(detector.current_yaw) < 25
        cv2.putText(frame, p_txt, (sx + 38, hp_y),
                    FONT_S, 0.30, C_GREEN if pose_ok else C_AMBER, 1, cv2.LINE_AA)

    # ── Confidence ──
    conf_y = hp_y + 20
    cv2.putText(frame, "Conf", (sx, conf_y), FONT_S, 0.32, C_GRAY, 1, cv2.LINE_AA)
    if detector:
        conf_pct = int(detector.confidence * 100)
        conf_c = C_GREEN if conf_pct > 70 else (C_AMBER if conf_pct > 40 else C_RED)
        cv2.putText(frame, f"{conf_pct}%", (sx + 38, conf_y),
                    FONT_S, 0.35, conf_c, 1, cv2.LINE_AA)

    # ── Calibration ──
    cal_y = conf_y + 22
    if detector and not detector.calibrator.is_calibrated:
        pct = int(detector.calibrator.progress * 100)
        cv2.putText(frame, f"Calibrating {pct}%", (sx, cal_y),
                    FONT_S, 0.32, C_CYAN, 1, cv2.LINE_AA)
        # Mini progress bar
        bar_w = pw - 32
        _draw_rect(frame, sx, cal_y + 5, bar_w, 4, C_GRAY_D, 0.5)
        fill = int(bar_w * detector.calibrator.progress)
        if fill > 0:
            cv2.rectangle(frame, (sx, cal_y + 5), (sx + fill, cal_y + 9),
                          C_CYAN, -1, cv2.LINE_AA)
    elif detector and detector.calibrator.is_calibrated:
        cv2.putText(frame, f"Thr: {detector.calibrator.threshold:.3f}", (sx, cal_y),
                    FONT_S, 0.28, C_GRAY_D, 1, cv2.LINE_AA)

    # ── Session stats ──
    if detector:
        st_y = ph + py - 55
        dur = detector.get_session_duration()
        h_s, m_s, s_s = int(dur // 3600), int((dur % 3600) // 60), int(dur % 60)
        cv2.putText(frame, f"Session {h_s:02d}:{m_s:02d}:{s_s:02d}", (sx, st_y),
                    FONT_S, 0.32, C_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Alerts: {detector.total_drowsy_events}", (sx, st_y + 16),
                    FONT_S, 0.32, C_RED if detector.total_drowsy_events > 0 else C_GRAY,
                    1, cv2.LINE_AA)

    # ── FPS ──
    fps_y = ph + py - 12
    fps_c = C_GREEN if fps >= 18 else (C_AMBER if fps >= 10 else C_RED)
    cv2.putText(frame, f"FPS {fps:.0f}", (sx, fps_y), FONT_S, 0.35, fps_c, 1, cv2.LINE_AA)


# ─── Top Accent Bar ──────────────────────────────────────────
def _draw_top_bar(frame, status, fc):
    fh, fw = frame.shape[:2]
    _draw_rect(frame, 0, 0, fw, 30, C_BG, 0.65)

    # Accent line
    color = _status_color(status)
    thick = 2
    if status == STATUS_ALERT:
        thick = 2 + int(abs(math.sin(fc * 0.2)) * 2)
    cv2.line(frame, (0, 30), (fw, 30), color, thick, cv2.LINE_AA)

    # Title
    cv2.putText(frame, "DRIVER SAFETY MONITOR", (215, 20),
                FONT, 0.50, C_WHITE, 1, cv2.LINE_AA)

    # Timestamp
    ts = time.strftime("%H:%M:%S")
    cv2.putText(frame, ts, (fw - 85, 20), FONT_S, 0.40, C_GRAY, 1, cv2.LINE_AA)


# ─── Bottom Bar ───────────────────────────────────────────────
def _draw_bottom_bar(frame, detector):
    fh, fw = frame.shape[:2]
    by = fh - 26
    _draw_rect(frame, 0, by, fw, 26, C_BG, 0.60)
    cv2.line(frame, (0, by), (fw, by), C_BORDER, 1, cv2.LINE_AA)

    cv2.putText(frame, "Q", (215, fh - 8), FONT, 0.36, C_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, "Quit", (230, fh - 8), FONT_S, 0.32, C_GRAY, 1, cv2.LINE_AA)
    cv2.putText(frame, "S", (275, fh - 8), FONT, 0.36, C_CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, "Stop Alarm", (290, fh - 8), FONT_S, 0.32, C_GRAY, 1, cv2.LINE_AA)

    if detector:
        avg = detector.get_avg_ear()
        cv2.putText(frame, f"Avg EAR: {avg:.3f}", (fw - 120, fh - 8),
                    FONT_S, 0.32, C_GRAY_D, 1, cv2.LINE_AA)


# ─── Minimal Face Overlay ─────────────────────────────────────────
def _draw_eye_overlay(frame, detector, status, fw, fh):
    """Draw clean, minimal eye outlines only — no cluttered mesh."""
    if detector is None or detector.face_landmarks is None:
        return

    L_EYE = [33, 160, 158, 133, 153, 144]
    R_EYE = [362, 385, 387, 263, 373, 380]

    color = _status_color(status)
    # Dim the color so it’s subtle
    subtle = tuple(max(0, c - 100) for c in color)

    def draw_eye_outline(indices):
        pts = []
        for idx in indices:
            lm = detector.face_landmarks.landmark[idx]
            pts.append((int(lm.x * fw), int(lm.y * fh)))
        # Draw outline connecting all collected points
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], subtle, 1, cv2.LINE_AA)
        if len(pts) > 2:
            cv2.line(frame, pts[-1], pts[0], subtle, 1, cv2.LINE_AA)
        # Corner dots only (inner/outer eye corners)
        if len(pts) >= 4:
            cv2.circle(frame, pts[0], 2, color, -1, cv2.LINE_AA)
            cv2.circle(frame, pts[3], 2, color, -1, cv2.LINE_AA)

    draw_eye_outline(L_EYE)
    draw_eye_outline(R_EYE)


# ─── Alert Overlays ──────────────────────────────────────────
def _draw_vignette(frame, color, intensity=0.4):
    """Draw a colored vignette around frame edges."""
    fh, fw = frame.shape[:2]
    key = (fw, fh)
    if _anim.get('vignette_size') != key:
        Y, X = np.ogrid[:fh, :fw]
        cx, cy = fw / 2, fh / 2
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        mask = np.clip(dist / max_dist, 0, 1).astype(np.float32)
        mask = mask ** 1.5
        _anim['vignette_cache'] = mask
        _anim['vignette_size'] = key

    mask = _anim['vignette_cache'] * intensity
    tint = np.full_like(frame, color, dtype=np.uint8)
    mask3 = np.stack([mask] * 3, axis=-1)
    blended = (frame.astype(np.float32) * (1 - mask3) + tint.astype(np.float32) * mask3)
    np.clip(blended, 0, 255, out=blended)
    frame[:] = blended.astype(np.uint8)


def _draw_alert_overlay(frame, fc):
    """Full alert overlay: red vignette + pulsing border + centered text."""
    fh, fw = frame.shape[:2]
    pulse = abs(math.sin(fc * 0.15))

    # Red vignette
    _draw_vignette(frame, (0, 0, 180), intensity=0.25 + 0.15 * pulse)

    # Pulsing border
    thick = int(3 + pulse * 5)
    cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), C_RED_B, thick, cv2.LINE_AA)

    # Center alert box
    text = "DROWSINESS DETECTED"
    sub = "PULL OVER SAFELY"
    fs1, fs2 = 0.75, 0.45
    sz1 = cv2.getTextSize(text, FONT, fs1, 2)[0]
    sz2 = cv2.getTextSize(sub, FONT_S, fs2, 1)[0]
    box_w = max(sz1[0], sz2[0]) + 40
    box_h = sz1[1] + sz2[1] + 50
    bx = (fw - box_w) // 2
    by = (fh - box_h) // 2

    _draw_rounded(frame, bx, by, box_w, box_h, (10, 10, 40), r=10, alpha=0.80)

    bright = int(200 + 55 * pulse)
    tc = (bright // 4, bright // 4, bright)
    cv2.putText(frame, text, ((fw - sz1[0]) // 2, by + 30 + sz1[1]),
                FONT, fs1, tc, 2, cv2.LINE_AA)
    cv2.putText(frame, sub, ((fw - sz2[0]) // 2, by + 55 + sz1[1] + sz2[1]),
                FONT_S, fs2, C_AMBER, 1, cv2.LINE_AA)


def _draw_warning_overlay(frame, fc):
    """Warning overlay: subtle amber vignette."""
    pulse = abs(math.sin(fc * 0.1))
    _draw_vignette(frame, (0, 100, 200), intensity=0.08 + 0.06 * pulse)


def _draw_no_face(frame, fc):
    fh, fw = frame.shape[:2]
    pulse = 0.3 + 0.15 * abs(math.sin(fc * 0.08))
    text = "No face detected"
    sz = cv2.getTextSize(text, FONT, 0.6, 1)[0]
    tx = (fw - sz[0]) // 2
    ty = fh // 2
    _draw_rounded(frame, tx - 15, ty - sz[1] - 12, sz[0] + 30, sz[1] + 24,
                  C_BG, r=8, alpha=pulse + 0.2)
    cv2.putText(frame, text, (tx, ty), FONT, 0.6, C_GRAY, 1, cv2.LINE_AA)


# ─── Main HUD Entry Point ────────────────────────────────────
def draw_hud(frame, status, ear, mar, score, fps, detector=None, frame_count=0):
    """
    Draw the complete premium HUD overlay.
    """
    # 100% Clean Video Feed — No markings on face.
    
    # Warning vignette (before panels)
    if status == STATUS_WARNING:
        _draw_warning_overlay(frame, frame_count)

    # Panels (Side Dashboard)
    _draw_top_bar(frame, status, frame_count)
    _draw_left_panel(frame, status, ear, mar, score, fps, detector, frame_count)
    _draw_bottom_bar(frame, detector)

    # Alert overlay (Full screen red pulse)
    if status == STATUS_ALERT:
        _draw_alert_overlay(frame, frame_count)

    # No face warning
    if detector and not detector.face_detected:
        _draw_no_face(frame, frame_count)
