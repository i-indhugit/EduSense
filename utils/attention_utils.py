"""
EduSense Attention Status Utility
==================================
Computes human-readable attention tier from a numeric score,
with ±2.5 hysteresis deadband to prevent flickering near boundaries.
"""

from typing import Optional, Tuple


def compute_final_attention_state(
    score: float,
    is_drowsy_consistent: bool = False,
    face_detected: bool = True,
    is_valid_evidence: bool = True,
    current_status: Optional[str] = None
) -> Tuple[str, str]:
    """
    Computes ONE final attention state per student from exact required states:
      - HIGHLY ATTENTIVE
      - ATTENTIVE
      - MODERATE
      - LOW ATTENTION
      - DROWSY
      - UNABLE TO DETERMINE
      (or 'No Face Detected' if face_detected is False)

    Returns:
      (final_state, reason_for_state)
    """
    if not face_detected:
        return "No Face Detected", "No face detected in live camera frame"

    if not is_valid_evidence:
        return "UNABLE TO DETERMINE", "Insufficient evidence or face quality below threshold"

    if is_drowsy_consistent:
        return "DROWSY", "Consistent drowsiness evidence detected over temporal window"

    # Determine target score-based tier
    if score >= 85.0:
        target = "HIGHLY ATTENTIVE"
        reason = "High gaze alignment, upright posture, and alert facial indicators"
    elif score >= 70.0:
        target = "ATTENTIVE"
        reason = "Sustained attention and forward-facing gaze"
    elif score >= 50.0:
        target = "MODERATE"
        reason = "Moderate attention with periodic gaze or pose variance"
    else:
        target = "LOW ATTENTION"
        reason = "Low attention score derived from head turn, looking away, or slouching"

    # Hysteresis smoothing if continuing in an active non-drowsy state
    if current_status in ["HIGHLY ATTENTIVE", "ATTENTIVE", "MODERATE", "LOW ATTENTION"]:
        margin = 2.5
        thresholds = {
            "HIGHLY ATTENTIVE": (85.0, None),
            "ATTENTIVE":        (70.0, 85.0),
            "MODERATE":         (50.0, 70.0),
            "LOW ATTENTION":    (None, 50.0),
        }
        lo, hi = thresholds.get(current_status, (None, None))
        if lo is not None and score < lo - margin:
            return target, reason
        if hi is not None and score >= hi + margin:
            return target, reason
        return current_status, f"Sustained state ({current_status}) within hysteresis margin"

    return target, reason


def compute_simplified_attention_score(
    gaze: str,
    head_pose: str,
    face_quality: float,
    posture: str
) -> Tuple[float, str, str]:
    """
    Simplified Clean Attention Score Calculation:
      Gaze = 40%
      Head Pose = 30%
      Face Visibility = 20%
      Posture = 10%

    Categories:
      80 - 100 = HIGHLY ATTENTIVE
      60 - 79  = ATTENTIVE
      40 - 59  = DISTRACTED
      0  - 39  = LOW ATTENTION
    """
    if face_quality < 0.25:
        return 0.0, "UNABLE TO DETERMINE", "Insufficient face visibility or low quality"

    gaze_score = 100.0 if gaze == "Looking at Board" else (50.0 if gaze == "Looking Away" else 20.0)
    pose_score = 100.0 if "Upright" in head_pose else (50.0 if "Turned" in head_pose else 20.0)
    face_score = min(100.0, max(0.0, face_quality * 100.0))
    posture_score = 100.0 if posture == "Upright" else (50.0 if posture == "Leaning" else 20.0)

    total_score = (0.40 * gaze_score) + (0.30 * pose_score) + (0.20 * face_score) + (0.10 * posture_score)
    total_score = round(max(0.0, min(100.0, total_score)), 1)

    if total_score >= 80.0:
        status = "HIGHLY ATTENTIVE"
        reason = "High gaze alignment, upright posture, and clear face visibility"
    elif total_score >= 60.0:
        status = "ATTENTIVE"
        reason = "Forward gaze and upright head pose"
    elif total_score >= 40.0:
        status = "DISTRACTED"
        reason = "Looking away or turned head pose"
    else:
        status = "LOW ATTENTION"
        reason = "Head down or slouching posture"

    return total_score, status, reason


def compute_attention_status(score: float, current_status: Optional[str] = None) -> str:
    """Backward compatibility helper mapping numeric score to canonical state."""
    state, _ = compute_final_attention_state(
        score=score,
        is_drowsy_consistent=False,
        face_detected=True,
        is_valid_evidence=True,
        current_status=current_status
    )
    return state

