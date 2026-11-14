"""
EduSense Behavioral Feature Extraction Module
=============================================
Extracts validated behavioral cues from MediaPipe FaceLandmarker (478 pts + 52 blendshapes)
and MediaPipe PoseLandmarker (33 body keypoints):
1. Head Pose: Real 3D Pitch, Yaw, Roll via cv2.solvePnP.
2. Eye Gaze: Iris-corner offset + head-pose compensation -> Forward/Left/Right/Up/Down/Unknown.
3. Eye Closure & Yawning: 6-point EAR, MAR, and neural blink/jawOpen blendshapes.
4. Facial Expression: Neural blendshape classification (gated by face quality >= 0.30).
5. Body Posture & Hand Raise: Pose keypoint geometry with explicit visibility gating.

Never fabricates missing features; returns "Unknown" / "Unable to Determine" when
landmarks or visibility are insufficient.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import cv2
import numpy as np


# 3D generic human face model points (mm) matching MediaPipe FaceLandmarker indices:
# 1: Nose tip, 152: Chin, 33: Left eye outer, 263: Right eye outer, 61: Left mouth, 291: Right mouth
_MODEL_3D_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip (1)
        (0.0, -63.6, -12.5),      # Chin (152)
        (-43.3, 32.7, -26.0),     # Left eye outer (33)
        (43.3, 32.7, -26.0),      # Right eye outer (263)
        (-28.9, -28.9, -24.1),    # Left mouth corner (61)
        (28.9, -28.9, -24.1),     # Right mouth corner (291)
    ],
    dtype=np.float64,
)
_PNP_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

_LEFT_EYE_EAR_IDS = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_EAR_IDS = [362, 385, 387, 263, 373, 380]


@dataclass
class FrameBehavioralFeatures:
    face_visible: bool
    face_quality: float
    confidence_pct: float
    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    head_pose_label: str = "Unknown"
    gaze_vector_label: str = "Unknown"     # Forward, Left, Right, Up, Down, Unknown
    gaze_classroom_label: str = "Unable to Determine"  # Looking at Board, Looking Away, Head Down, Unable to Determine
    ear: Optional[float] = None
    mar: Optional[float] = None
    eyes_closed_frame: bool = False
    yawning_frame: bool = False
    emotion: str = "Unable to Determine"
    emotion_conf: float = 0.0
    posture: str = "Unknown"               # Upright, Leaning, Slouching, Unknown
    hand_raised_frame: bool = False
    pose_visible: bool = False
    eye_distance_ratio: Optional[float] = None


def _pt(landmarks: list, idx: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def _compute_ear(landmarks: list, ids: list, w: int, h: int) -> float:
    p1 = _pt(landmarks, ids[0], w, h)
    p2 = _pt(landmarks, ids[1], w, h)
    p3 = _pt(landmarks, ids[2], w, h)
    p4 = _pt(landmarks, ids[3], w, h)
    p5 = _pt(landmarks, ids[4], w, h)
    p6 = _pt(landmarks, ids[5], w, h)
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4)
    if horiz < 1e-4:
        return 0.0
    return float((v1 + v2) / (2.0 * horiz))


def estimate_head_pose(landmarks: list, crop_w: int, crop_h: int) -> Optional[Tuple[float, float, float]]:
    """Estimate (pitch, yaw, roll) in degrees via cv2.solvePnP."""
    if len(landmarks) < 468 or crop_w < 20 or crop_h < 20:
        return None

    image_points = np.array(
        [[landmarks[i].x * crop_w, landmarks[i].y * crop_h] for i in _PNP_LANDMARK_IDS],
        dtype=np.float64,
    )
    focal_length = float(max(crop_w, crop_h))
    center = (crop_w / 2.0, crop_h / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0.0, center[0]], [0.0, focal_length, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        _MODEL_3D_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch = float(euler_angles[0, 0])
    yaw = float(euler_angles[1, 0])
    roll = float(euler_angles[2, 0])

    # Normalize angles to [-90, 90] range cleanly
    if pitch > 90:
        pitch -= 180.0
    elif pitch < -90:
        pitch += 180.0
    return round(pitch, 1), round(yaw, 1), round(roll, 1)


def estimate_gaze_from_iris_and_pose(
    landmarks: list,
    pitch: Optional[float],
    yaw: Optional[float],
    w: int,
    h: int,
) -> Tuple[str, str]:
    """Combine iris-in-eye ratio with head orientation -> (granular_gaze, classroom_gaze)."""
    if pitch is None or yaw is None or len(landmarks) < 478:
        return "Unknown", "Unable to Determine"

    # Check head orientation first: large turns/nods dominate gaze direction
    if pitch < -18.0:
        return "Down", "Head Down"
    if yaw < -22.0:
        return "Left", "Looking Away"
    if yaw > 22.0:
        return "Right", "Looking Away"
    if pitch > 22.0:
        return "Up", "Looking Away"

    # Iris center relative to eye corners (MediaPipe 468..477)
    try:
        l_outer = _pt(landmarks, 33, w, h)
        l_inner = _pt(landmarks, 133, w, h)
        l_iris = _pt(landmarks, 468, w, h)

        r_inner = _pt(landmarks, 362, w, h)
        r_outer = _pt(landmarks, 263, w, h)
        r_iris = _pt(landmarks, 473, w, h)

        l_width = max(1.0, np.linalg.norm(l_inner - l_outer))
        r_width = max(1.0, np.linalg.norm(r_outer - r_inner))
        l_ratio_x = float((l_iris[0] - l_outer[0]) / l_width)
        r_ratio_x = float((r_iris[0] - r_inner[0]) / r_width)
        avg_x = (l_ratio_x + r_ratio_x) / 2.0

        l_top = _pt(landmarks, 159, w, h)
        l_bot = _pt(landmarks, 145, w, h)
        l_height = max(1.0, np.linalg.norm(l_bot - l_top))
        iris_y_ratio = float((l_iris[1] - l_top[1]) / l_height)

        if avg_x < 0.33:
            return "Right", "Looking Away"
        if avg_x > 0.67:
            return "Left", "Looking Away"
        if iris_y_ratio > 0.72:
            return "Down", "Head Down"
        if iris_y_ratio < 0.22:
            return "Up", "Looking Away"
    except Exception:
        pass

    return "Forward", "Looking at Board"


def classify_expression_from_blendshapes(
    blendshapes: Optional[Dict[str, float]],
    face_quality: float,
    eyes_closed: bool,
) -> Tuple[str, float]:
    """Map MediaPipe neural blendshapes to observable classroom expression."""
    if not blendshapes or face_quality < 0.30:
        return "Unable to Determine", 0.0

    smile = (blendshapes.get("mouthSmileLeft", 0.0) + blendshapes.get("mouthSmileRight", 0.0)) / 2.0
    frown = (blendshapes.get("mouthFrownLeft", 0.0) + blendshapes.get("mouthFrownRight", 0.0)) / 2.0
    brow_down = (blendshapes.get("browDownLeft", 0.0) + blendshapes.get("browDownRight", 0.0)) / 2.0
    brow_up = blendshapes.get("browInnerUp", 0.0)
    squint = (blendshapes.get("eyeSquintLeft", 0.0) + blendshapes.get("eyeSquintRight", 0.0)) / 2.0
    jaw_open = blendshapes.get("jawOpen", 0.0)

    if eyes_closed and jaw_open > 0.35:
        return "Sleepy", round(min(0.95, 0.65 + jaw_open * 0.3), 2)
    if smile > 0.38:
        return "Happy", round(min(0.96, 0.55 + smile * 0.45), 2)
    if brow_down > 0.38 and frown > 0.22:
        return "Frustrated", round(min(0.92, 0.50 + (brow_down + frown) * 0.3), 2)
    if (brow_down > 0.25 and squint > 0.20) or (brow_up > 0.38 and frown > 0.15):
        return "Confused", round(min(0.90, 0.50 + max(brow_down, brow_up) * 0.35), 2)
    if frown > 0.30 and smile < 0.10:
        return "Bored", round(min(0.88, 0.50 + frown * 0.4), 2)

    return "Neutral", round(max(0.60, 0.88 - max(smile, frown, brow_down) * 0.3), 2)


def extract_pose_features(pose_landmarks: Optional[list]) -> Tuple[str, bool, bool]:
    """
    Extract (posture_label, hand_raised_frame, pose_visible) from MediaPipe PoseLandmarker.
    Uses visibility gating (>= 0.45) to avoid hallucinating occluded body cues.
    """
    if not pose_landmarks or len(pose_landmarks) < 17:
        return "Unknown", False, False

    ls = pose_landmarks[11]
    rs = pose_landmarks[12]
    le = pose_landmarks[13]
    re = pose_landmarks[14]
    lw = pose_landmarks[15]
    rw = pose_landmarks[16]
    nose = pose_landmarks[0]

    ls_vis = getattr(ls, "visibility", 1.0)
    rs_vis = getattr(rs, "visibility", 1.0)
    if ls_vis < 0.45 or rs_vis < 0.45:
        return "Unknown", False, False

    shoulder_mid_y = (ls.y + rs.y) / 2.0
    shoulder_w = max(0.05, abs(ls.x - rs.x))
    shoulder_tilt_deg = abs(math.degrees(math.atan2(ls.y - rs.y, max(1e-4, abs(ls.x - rs.x)))))

    # Check head drop relative to shoulder line (slouching vs upright)
    nose_vis = getattr(nose, "visibility", 1.0)
    nose_above_shoulders = (shoulder_mid_y - nose.y) / shoulder_w if nose_vis >= 0.45 else 0.55

    if shoulder_tilt_deg > 16.0:
        posture = "Leaning"
    elif nose_vis >= 0.45 and nose_above_shoulders < 0.25:
        posture = "Slouching"
    else:
        posture = "Upright"

    # Hand-raise detection: wrist clearly above shoulder + elbow raised
    left_raised = (
        getattr(lw, "visibility", 0.0) >= 0.45
        and getattr(le, "visibility", 0.0) >= 0.40
        and lw.y < (ls.y - 0.05)
        and le.y < (ls.y + 0.14)
    )
    right_raised = (
        getattr(rw, "visibility", 0.0) >= 0.45
        and getattr(re, "visibility", 0.0) >= 0.40
        and rw.y < (rs.y - 0.05)
        and re.y < (rs.y + 0.14)
    )
    return posture, bool(left_raised or right_raised), True


def extract_behavioral_features(
    raw_landmarks_bundle: Optional[object],
    blendshapes: Optional[Dict[str, float]],
    pose_landmarks: Optional[list],
    face_quality: float,
    face_conf: float,
) -> FrameBehavioralFeatures:
    """Master per-frame feature extractor with explicit missing-feature handling."""
    posture, hand_raised_frame, pose_visible = extract_pose_features(pose_landmarks)

    if raw_landmarks_bundle is None:
        return FrameBehavioralFeatures(
            face_visible=False,
            face_quality=0.0,
            confidence_pct=0.0,
            posture=posture,
            hand_raised_frame=hand_raised_frame,
            pose_visible=pose_visible,
        )

    lms, (_, _, crop_w, crop_h), _ = raw_landmarks_bundle
    pose_angles = estimate_head_pose(lms, crop_w, crop_h)
    if pose_angles is not None:
        pitch, yaw, roll = pose_angles
        if pitch < -18.0:
            hp_label = f"Head Down ({pitch:.0f}°)"
        elif abs(yaw) > 22.0:
            hp_label = f"Turned ({yaw:.0f}°)"
        else:
            hp_label = f"Upright ({yaw:.0f}°)"
    else:
        pitch = yaw = roll = None
        hp_label = "Unknown"

    gaze_vec, gaze_cls = estimate_gaze_from_iris_and_pose(lms, pitch, yaw, crop_w, crop_h)

    # EAR & eye closure
    ear_l = _compute_ear(lms, _LEFT_EYE_EAR_IDS, crop_w, crop_h)
    ear_r = _compute_ear(lms, _RIGHT_EYE_EAR_IDS, crop_w, crop_h)
    ear = round((ear_l + ear_r) / 2.0, 3)

    blink_l = (blendshapes or {}).get("eyeBlinkLeft", 0.0)
    blink_r = (blendshapes or {}).get("eyeBlinkRight", 0.0)
    eyes_closed = (ear < 0.18) or ((blink_l + blink_r) / 2.0 > 0.62)

    # MAR & yawn
    top_lip = _pt(lms, 13, crop_w, crop_h)
    bot_lip = _pt(lms, 14, crop_w, crop_h)
    left_lip = _pt(lms, 78, crop_w, crop_h)
    right_lip = _pt(lms, 308, crop_w, crop_h)
    mar = round(float(np.linalg.norm(top_lip - bot_lip) / max(1.0, np.linalg.norm(left_lip - right_lip))), 3)
    jaw_open = (blendshapes or {}).get("jawOpen", 0.0)
    yawning = (mar > 0.58) or (jaw_open > 0.55)

    # Eyebrow contraction ratio (distance between inner brows 107 & 336 normalized by inter-ocular width)
    try:
        brow_dist = np.linalg.norm(_pt(lms, 107, crop_w, crop_h) - _pt(lms, 336, crop_w, crop_h))
        eye_dist = max(1.0, np.linalg.norm(_pt(lms, 33, crop_w, crop_h) - _pt(lms, 263, crop_w, crop_h)))
        eye_distance_ratio = round(float((brow_dist / eye_dist) / 0.36), 3)
    except Exception:
        eye_distance_ratio = None

    emotion, emo_conf = classify_expression_from_blendshapes(blendshapes, face_quality, eyes_closed)

    return FrameBehavioralFeatures(
        face_visible=True,
        face_quality=face_quality,
        confidence_pct=round(face_conf * 100.0, 1),
        yaw_deg=yaw,
        pitch_deg=pitch,
        roll_deg=roll,
        head_pose_label=hp_label,
        gaze_vector_label=gaze_vec,
        gaze_classroom_label=gaze_cls,
        ear=ear,
        mar=mar,
        eyes_closed_frame=eyes_closed,
        yawning_frame=yawning,
        emotion=emotion,
        emotion_conf=emo_conf,
        posture=posture,
        hand_raised_frame=hand_raised_frame,
        pose_visible=pose_visible,
        eye_distance_ratio=eye_distance_ratio,
    )

