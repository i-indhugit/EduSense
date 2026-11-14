"""
EduSense Real-Time Vision & Analytics Pipeline
==============================================
Processes live frames (from local webcam or browser WebRTC/getUserMedia stream)
through a strict hierarchical pipeline:
  Frame -> Person Detection -> Person Validation -> Pose & Face Detection (inside ROI)
  -> Facial Landmark Validation -> ByteTrack Multi-Person Tracking
  -> Behavioral Feature Extraction -> Temporal Hysteresis -> Feature Fusion.

Guarantees:
- Zero fake/placeholder students or synthetic predictions in live mode.
- Zero stale bounding boxes when a face disappears or is occluded.
- Monotonic persistent Student IDs via Kalman + two-stage ByteTrack assignment.
- Mutual exclusion of contradictory states (e.g. DROWSY overrides ATTENTIVE).
- Explicit "UNABLE TO DETERMINE" / "No Face Detected" states for occluded/low-quality faces.
"""

import base64
import logging
import math
import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from fusion_engine.feature_fusion import FeatureFusionEngine
from inference.detectors import StrictPersonFaceDetector
from inference.features import FrameBehavioralFeatures, extract_behavioral_features
from inference.tracker import ByteTrackManager, PersonTrack, compute_iou
from recommendation.engine import TeacherRecommendationEngine
from utils.attention_utils import compute_final_attention_state

logger = logging.getLogger("EduSensePipeline")

LIVE_MODE = True
MIN_CONFIDENCE_THRESHOLD = 50.0
MAX_MISSING_FRAMES = int(os.getenv("EDUSENSE_TRACK_TIMEOUT_FRAMES", "12"))
POSE_INTERVAL = int(os.getenv("EDUSENSE_POSE_INTERVAL", "2"))
FRAME_MAX_W = int(os.getenv("EDUSENSE_MAX_WIDTH", "960"))


class TrackedFaceState:
    """Isolated per-student temporal state with hysteresis & multi-frame validation."""

    def __init__(self, track_id: int, initial_bbox: Tuple[float, float, float, float]):
        self.track_id = track_id
        self.student_id = f"Student {track_id:02d}"

        self.bbox = initial_bbox
        self.missing_frames = 0

        self.att_history = deque(maxlen=8)
        self.disp_att = 0.0

        self.conf_history = deque(maxlen=8)
        self.disp_conf = 0.0

        self.face_quality_history = deque(maxlen=8)
        self.disp_face_quality = 0.0

        self.emotion_history = deque(maxlen=6)
        self.disp_emotion = "Unable to Determine"

        self.gaze_history = deque(maxlen=5)
        self.disp_gaze = "Unable to Determine"

        self.head_pose_history = deque(maxlen=5)
        self.disp_head_pose = "Unknown"

        self.posture_history = deque(maxlen=6)
        self.disp_posture = "Unknown"

        self.drowsy_window = deque(maxlen=12)
        self.disp_is_drowsy_consistent = False

        self.yawn_window = deque(maxlen=8)
        self.disp_is_yawning_consistent = False

        self.hand_raise_window = deque(maxlen=5)
        self.disp_hand_raised = False

        self.disp_final_state = "UNABLE TO DETERMINE"
        self.disp_reason = "Initializing temporal window"
        self.disp_evidence = ""
        self.latest_raw_features: Dict = {}

    def update_bbox(self, raw_bbox: Tuple[float, float, float, float], alpha: float = 0.30, deadband_px: float = 3.0):
        rx1, ry1, rx2, ry2 = raw_bbox
        cx1, cy1, cx2, cy2 = self.bbox
        nx1 = (alpha * rx1 + (1 - alpha) * cx1) if abs(rx1 - cx1) > deadband_px else cx1
        ny1 = (alpha * ry1 + (1 - alpha) * cy1) if abs(ry1 - cy1) > deadband_px else cy1
        nx2 = (alpha * rx2 + (1 - alpha) * cx2) if abs(rx2 - cx2) > deadband_px else cx2
        ny2 = (alpha * ry2 + (1 - alpha) * cy2) if abs(ry2 - cy2) > deadband_px else cy2
        self.bbox = (nx1, ny1, nx2, ny2)

    def update_student_observations(
        self,
        raw_gaze: str,
        raw_head_pose: str,
        raw_posture: str,
        raw_drowsy_evidence: bool,
        ear_val: Optional[float],
        face_quality: float,
        raw_emotion: str,
        raw_conf: float,
        face_detected: bool = True,
        hand_raised_frame: bool = False,
        yawning_frame: bool = False,
        pitch_deg: Optional[float] = None,
        yaw_deg: Optional[float] = None,
    ):
        self.latest_raw_features = {
            "gaze": raw_gaze,
            "head_pose": raw_head_pose,
            "posture": raw_posture,
            "ear": round(ear_val, 3) if ear_val is not None else None,
            "face_quality": round(face_quality, 3),
            "emotion": raw_emotion,
            "drowsy_evidence_frame": raw_drowsy_evidence,
            "hand_raised_frame": hand_raised_frame,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
        }

        self.posture_history.append(raw_posture)
        self.disp_posture = self._most_frequent(self.posture_history, self.disp_posture)

        self.hand_raise_window.append(1 if hand_raised_frame else 0)
        self.disp_hand_raised = (
            len(self.hand_raise_window) >= 3 and sum(self.hand_raise_window) >= 3
        )

        if not face_detected:
            self.disp_conf = 0.0
            self.disp_face_quality = 0.0
            self.disp_gaze = "Unable to Determine"
            self.disp_head_pose = "Unknown"
            self.disp_emotion = "Unable to Determine"
            self.disp_att = 0.0
            self.disp_is_drowsy_consistent = False
            self.disp_is_yawning_consistent = False
            self.disp_final_state = "UNABLE TO DETERMINE"
            self.disp_reason = "Face occluded or not visible in current person region"
            self.disp_evidence = f"Posture: {self.disp_posture}, Face: Unavailable"
            return

        self.drowsy_window.append(1 if raw_drowsy_evidence else 0)
        self.yawn_window.append(1 if yawning_frame else 0)
        self.gaze_history.append(raw_gaze)
        self.head_pose_history.append(raw_head_pose)
        self.emotion_history.append(raw_emotion)
        self.face_quality_history.append(face_quality)
        self.conf_history.append(raw_conf)

        self.disp_gaze = self._most_frequent(self.gaze_history, raw_gaze)
        self.disp_head_pose = self._most_frequent(self.head_pose_history, raw_head_pose)
        self.disp_emotion = self._most_frequent(self.emotion_history, raw_emotion)

        # Require >= 6 frames before declaring sustained drowsiness (never 1-frame drowsy)
        if len(self.drowsy_window) >= 6:
            drowsy_ratio = sum(self.drowsy_window) / float(len(self.drowsy_window))
            self.disp_is_drowsy_consistent = drowsy_ratio >= 0.60
        else:
            self.disp_is_drowsy_consistent = False

        if len(self.yawn_window) >= 4:
            self.disp_is_yawning_consistent = (sum(self.yawn_window) / float(len(self.yawn_window))) >= 0.60
        else:
            self.disp_is_yawning_consistent = False

        self.disp_face_quality = round(sum(self.face_quality_history) / len(self.face_quality_history), 3)
        self.disp_conf = round(sum(self.conf_history) / len(self.conf_history), 1)

        is_valid_evidence = (
            self.disp_face_quality >= 0.25
            and self.disp_conf >= MIN_CONFIDENCE_THRESHOLD
            and self.disp_gaze != "Unable to Determine"
        )

        if not is_valid_evidence:
            self.disp_att = 0.0
            self.disp_final_state = "UNABLE TO DETERMINE"
            self.disp_reason = "Insufficient face quality or detection confidence"
            self.disp_evidence = f"FaceVis: {int(self.disp_face_quality * 100)}%, Conf: {self.disp_conf:.1f}%"
            return

        gaze_score = 100.0 if self.disp_gaze == "Looking at Board" else (50.0 if self.disp_gaze == "Looking Away" else 20.0)
        pose_score = 100.0 if "Upright" in self.disp_head_pose else (50.0 if "Turned" in self.disp_head_pose else 20.0)
        face_score = min(100.0, max(0.0, self.disp_face_quality * 100.0))
        posture_score = (
            100.0 if self.disp_posture == "Upright"
            else (60.0 if self.disp_posture == "Leaning" else (25.0 if self.disp_posture == "Slouching" else 70.0))
        )

        raw_score = (0.40 * gaze_score) + (0.30 * pose_score) + (0.20 * face_score) + (0.10 * posture_score)
        self.att_history.append(raw_score)
        smoothed_score = round(sum(self.att_history) / len(self.att_history), 1)

        # Canonical 6-state taxonomy with strict DROWSY vs ATTENTIVE mutual exclusion
        prev_state = self.disp_final_state if self.disp_final_state in (
            "HIGHLY ATTENTIVE", "ATTENTIVE", "MODERATE", "LOW ATTENTION", "DROWSY"
        ) else None

        final_state, reason = compute_final_attention_state(
            score=smoothed_score,
            is_drowsy_consistent=self.disp_is_drowsy_consistent,
            face_detected=True,
            is_valid_evidence=True,
            current_status=prev_state,
        )

        # Cap numeric attention estimate when consistently drowsy so score & state never contradict
        if final_state == "DROWSY":
            smoothed_score = min(smoothed_score, 35.0)

        self.disp_att = smoothed_score
        self.disp_final_state = final_state
        self.disp_reason = reason
        self.disp_evidence = (
            f"Gaze: {self.disp_gaze}, HeadPose: {self.disp_head_pose}, "
            f"FaceVis: {int(self.disp_face_quality * 100)}%, Posture: {self.disp_posture}"
        )

    def update_predictions(
        self,
        raw_att: float,
        raw_conf: float,
        raw_emotion: str,
        raw_gaze: str,
        raw_head_pose: str,
        raw_drowsy: bool,
        raw_status: str,
    ):
        """Backward compatibility adapter for unit tests."""
        face_quality = round(max(0.15, min(1.0, (raw_conf - 20.0) / 70.0)), 2) if raw_conf > 0 else 0.15
        posture = "Slouching" if "Down" in raw_head_pose else ("Leaning" if "Turned" in raw_head_pose else "Upright")
        ear_val = 0.15 if raw_drowsy else 0.28
        self.update_student_observations(
            raw_gaze=raw_gaze,
            raw_head_pose=raw_head_pose,
            raw_posture=posture,
            raw_drowsy_evidence=raw_drowsy,
            ear_val=ear_val,
            face_quality=face_quality,
            raw_emotion=raw_emotion,
            raw_conf=raw_conf,
            face_detected=(raw_gaze != "Unable to Determine" or raw_conf > 0),
        )

    def get_debug_info(
        self,
        camera_source: str = "WEBCAM",
        frame_id: int = 0,
        frame_timestamp: str = "",
        detected_person_count: int = 0,
        valid_face_count: int = 0,
        active_track_count: int = 0,
        track_ids: Optional[List[str]] = None,
        face_detected: bool = True,
    ) -> dict:
        drowsy_ratio = sum(self.drowsy_window) / float(len(self.drowsy_window)) if self.drowsy_window else 0.0
        bx1, by1, bx2, by2 = self.bbox
        is_valid = face_detected and (self.disp_face_quality >= 0.25) and (self.disp_conf >= MIN_CONFIDENCE_THRESHOLD)

        return {
            "camera_source": camera_source,
            "frame_id": frame_id,
            "frame_timestamp": frame_timestamp or time.strftime("%H:%M:%S"),
            "detected_person_count": detected_person_count,
            "valid_face_count": valid_face_count,
            "active_track_count": active_track_count,
            "track_ids": track_ids or [self.student_id],
            "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
            "detection_confidence": self.disp_conf,
            "face_detected": face_detected,
            "face_quality": self.disp_face_quality,
            "gaze": self.disp_gaze if is_valid else "Unable to Determine",
            "head_pose": self.disp_head_pose if is_valid else "Unknown",
            "posture": self.disp_posture,
            "drowsiness_probability": round(drowsy_ratio, 2),
            "attention_score": self.disp_att if is_valid else 0.0,
            "final_state": self.disp_final_state,
            "student_id": self.student_id,
            "raw_feature_values": self.latest_raw_features,
            "temporal_values": {
                "smoothed_attention_score": self.disp_att,
                "drowsy_window_count": sum(self.drowsy_window),
                "drowsy_window_total": len(self.drowsy_window),
                "drowsy_window_ratio_pct": round(drowsy_ratio * 100.0, 1),
                "is_drowsy_consistent": self.disp_is_drowsy_consistent,
            },
            "reliability_values": {
                "face_quality": self.disp_face_quality,
                "confidence_pct": self.disp_conf,
                "is_valid_evidence": is_valid,
            },
            "final_attention_score": self.disp_att if is_valid else 0.0,
            "reason_for_state": self.disp_reason,
        }

    @staticmethod
    def _most_frequent(history: deque, current_val):
        if not history:
            return current_val
        counts: Dict = {}
        for item in history:
            counts[item] = counts.get(item, 0) + 1
        best_item, max_count = max(counts.items(), key=lambda x: x[1])
        if max_count >= math.ceil(len(history) * 0.55):
            return best_item
        return current_val


class EduSensePipeline:
    """Production real-time classroom vision & behavioral analytics pipeline."""

    def __init__(self, mode: str = "webcam", camera_index: int = 0, open_camera: bool = True):
        self.mode = mode
        self.camera_index = camera_index
        self.fusion_engine = FeatureFusionEngine()
        self.rec_engine = TeacherRecommendationEngine()

        self.detector = StrictPersonFaceDetector()
        self.tracker = ByteTrackManager(
            high_conf_thresh=0.45,
            low_conf_thresh=0.30,
            iou_thresh=0.20,
            max_missing_frames=MAX_MISSING_FRAMES,
            min_hits_to_confirm=1,  # Confirm immediately on high-conf person for responsive live/image eval
        )
        self._student_states: Dict[int, TrackedFaceState] = {}

        self.frame_counter = 0
        self._latency_window: deque = deque(maxlen=30)
        self._fps_window: deque = deque(maxlen=30)
        self._last_process_ts = time.perf_counter()

        self.camera_connected = False
        self.cap: Optional[cv2.VideoCapture] = None
        if open_camera and mode == "webcam":
            self._open_local_camera()

        self.latest_b64: str = ""
        self.latest_analytics: Dict = self.fusion_engine.process_classroom_frame([])

    def _open_local_camera(self) -> bool:
        try:
            cap = cv2.VideoCapture(self.camera_index)
            if cap is not None and cap.isOpened():
                self.cap = cap
                self.camera_connected = True
                return True
            if cap is not None:
                cap.release()
        except Exception as e:
            logger.warning("Local camera open failed: %s", e)
        self.cap = None
        self.camera_connected = False
        return False

    def reset_tracking(self) -> None:
        self.tracker.reset()
        self._student_states.clear()

    def get_performance_metrics(self) -> Dict[str, float]:
        avg_lat = round(float(sum(self._latency_window) / len(self._latency_window)), 2) if self._latency_window else 0.0
        avg_fps = round(float(sum(self._fps_window) / len(self._fps_window)), 1) if self._fps_window else 0.0
        return {"latency_ms": avg_lat, "fps": avg_fps}

    def process_frame_array(
        self,
        frame_bgr: np.ndarray,
        camera_source: str = "WEBCAM",
        encode_jpeg: bool = True,
    ) -> Tuple[str, Dict]:
        """Run complete detection, tracking, feature extraction, and fusion on a BGR numpy frame."""
        t0 = time.perf_counter()
        self.frame_counter += 1

        # Efficient frame resizing keeping aspect ratio
        orig_h, orig_w = frame_bgr.shape[:2]
        if orig_w > FRAME_MAX_W:
            scale = FRAME_MAX_W / float(orig_w)
            frame = cv2.resize(frame_bgr, (FRAME_MAX_W, int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            frame = frame_bgr.copy()

        fh, fw = frame.shape[:2]
        run_pose = (self.frame_counter % max(1, POSE_INTERVAL) == 1) or POSE_INTERVAL <= 1

        candidates, rejected_debug = self.detector.detect_frame(frame, run_pose=run_pose)
        confirmed_tracks: List[PersonTrack] = self.tracker.update(candidates)

        # Clean up student states for purged tracks
        active_ids = set(self.tracker.tracks.keys())
        for stale_id in [tid for tid in self._student_states if tid not in active_ids]:
            del self._student_states[stale_id]

        # Only tracks visible in THIS frame emit live observations & bounding boxes
        visible_tracks = [t for t in confirmed_tracks if t.visible_this_frame and t.latest_detection is not None]
        live_detected_persons = len(visible_tracks)
        valid_face_count = sum(
            1 for t in visible_tracks if t.latest_detection.face_status == "VALID" and t.latest_detection.face_bbox is not None
        )
        active_tracks_count = len(confirmed_tracks)

        features_list: List[Dict] = []
        track_labels = [t.student_id for t in visible_tracks]
        ts_now = time.strftime("%Y-%m-%d %H:%M:%S")

        for trk in visible_tracks:
            det = trk.latest_detection
            if trk.track_id not in self._student_states:
                initial_box = det.face_bbox if det.face_bbox is not None else trk.smoothed_person_bbox
                self._student_states[trk.track_id] = TrackedFaceState(trk.track_id, initial_box)

            state = self._student_states[trk.track_id]
            face_visible = det.face_status == "VALID" and det.face_bbox is not None

            if face_visible:
                state.update_bbox(det.face_bbox)
                feat: FrameBehavioralFeatures = extract_behavioral_features(
                    raw_landmarks_bundle=det.raw_landmarks,
                    blendshapes=det.blendshapes,
                    pose_landmarks=det.pose_landmarks,
                    face_quality=det.face_quality,
                    face_conf=det.face_conf,
                )
                state.update_student_observations(
                    raw_gaze=feat.gaze_classroom_label,
                    raw_head_pose=feat.head_pose_label,
                    raw_posture=feat.posture,
                    raw_drowsy_evidence=feat.eyes_closed_frame,
                    ear_val=feat.ear,
                    face_quality=feat.face_quality,
                    raw_emotion=feat.emotion,
                    raw_conf=feat.confidence_pct,
                    face_detected=True,
                    hand_raised_frame=feat.hand_raised_frame,
                    yawning_frame=feat.yawning_frame,
                    pitch_deg=feat.pitch_deg,
                    yaw_deg=feat.yaw_deg,
                )
                draw_box = state.bbox
            else:
                # Person visible, face occluded/unavailable/rejected -> DO NOT reuse old face box or invent landmarks
                feat = extract_behavioral_features(
                    raw_landmarks_bundle=None,
                    blendshapes=None,
                    pose_landmarks=det.pose_landmarks,
                    face_quality=0.0,
                    face_conf=0.0,
                )
                state.update_student_observations(
                    raw_gaze="Unable to Determine",
                    raw_head_pose="Unknown",
                    raw_posture=feat.posture,
                    raw_drowsy_evidence=False,
                    ear_val=None,
                    face_quality=0.0,
                    raw_emotion="Unable to Determine",
                    raw_conf=0.0,
                    face_detected=False,
                    hand_raised_frame=feat.hand_raised_frame,
                )
                if det.rejection_reason:
                    state.disp_reason = f"Face excluded ({det.rejection_reason})"
                draw_box = trk.smoothed_person_bbox

            # Draw clean overlays on frame (person box subtle, validated face box bright)
            px1, py1, px2, py2 = [int(max(0, min(fw if i % 2 == 0 else fh, v))) for i, v in enumerate(trk.smoothed_person_bbox)]
            cv2.rectangle(frame, (px1, py1), (px2, py2), (80, 140, 200), 1)
            cv2.putText(
                frame,
                f"{trk.student_id} (Person {trk.person_conf*100:.0f}%)",
                (px1, max(16, py1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (120, 200, 255),
                1,
            )

            if face_visible:
                fx1, fy1, fx2, fy2 = [int(max(0, min(fw if i % 2 == 0 else fh, v))) for i, v in enumerate(draw_box)]
                box_col = (0, 230, 140) if state.disp_final_state in ("HIGHLY ATTENTIVE", "ATTENTIVE") else (
                    (0, 165, 255) if state.disp_final_state == "MODERATE" else (60, 80, 240)
                )
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), box_col, 2)
                cv2.putText(
                    frame,
                    f"{state.disp_final_state} ({state.disp_att:.0f}%)",
                    (fx1, max(16, fy1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    box_col,
                    1,
                )
                out_bbox = [fx1, fy1, fx2, fy2]
            else:
                out_bbox = None

            debug_telemetry = state.get_debug_info(
                camera_source=camera_source,
                frame_id=self.frame_counter,
                frame_timestamp=ts_now,
                detected_person_count=live_detected_persons,
                valid_face_count=valid_face_count,
                active_track_count=active_tracks_count,
                track_ids=track_labels,
                face_detected=face_visible,
            )

            is_valid_conf = face_visible and state.disp_face_quality >= 0.25 and state.disp_conf >= MIN_CONFIDENCE_THRESHOLD
            features_list.append(
                {
                    "student_id": trk.student_id,
                    "track_id": trk.track_id,
                    "bbox": out_bbox,
                    "person_bbox": [px1, py1, px2, py2],
                    "face_detected": face_visible,
                    "face_status": det.face_status,
                    "confidence_pct": state.disp_conf if face_visible else round(trk.person_conf * 100.0, 1),
                    "face_quality": state.disp_face_quality,
                    "attention_pct": state.disp_att if is_valid_conf else 0.0,
                    "attention_status": state.disp_final_state,
                    "final_state": state.disp_final_state,
                    "reason_for_state": state.disp_reason,
                    "gaze_direction": state.disp_gaze if is_valid_conf else "Unable to Determine",
                    "gaze_vector": feat.gaze_vector_label,
                    "head_pose": state.disp_head_pose if is_valid_conf else "Unknown",
                    "pitch_deg": feat.pitch_deg,
                    "yaw_deg": feat.yaw_deg,
                    "roll_deg": feat.roll_deg,
                    "emotion": state.disp_emotion if is_valid_conf else "Unable to Determine",
                    "ear_value": feat.ear,
                    "mar_value": feat.mar,
                    "is_drowsy": state.disp_is_drowsy_consistent,
                    "is_yawning": state.disp_is_yawning_consistent,
                    "hand_raised": state.disp_hand_raised,
                    "posture": state.disp_posture,
                    "pose_visible": feat.pose_visible,
                    "eye_distance_ratio": feat.eye_distance_ratio,
                    "debug_info": debug_telemetry,
                    "_meta_live_detected_persons": live_detected_persons,
                    "_meta_valid_face_count": valid_face_count,
                    "_meta_active_tracks": active_tracks_count,
                }
            )

        # Annotate rejected spoof/uncertain candidates in subtle red dashed-like label for transparency
        for rej_reason, rbox, rconf in rejected_debug:
            rx1, ry1, rx2, ry2 = [int(v) for v in rbox]
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (60, 60, 180), 1)
            cv2.putText(
                frame,
                f"EXCLUDED ({rej_reason[:18]})",
                (rx1, max(14, ry1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (90, 90, 220),
                1,
            )

        # Measure real latency and FPS
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        dt = t1 - self._last_process_ts
        self._last_process_ts = t1
        inst_fps = (1.0 / dt) if dt > 0 else 0.0
        self._latency_window.append(latency_ms)
        self._fps_window.append(min(120.0, inst_fps))
        perf = self.get_performance_metrics()

        # HUD Telemetry
        cv2.rectangle(frame, (10, 10), (360, 96), (12, 12, 16), -1)
        cv2.rectangle(frame, (10, 10), (360, 96), (45, 45, 55), 1)
        cv2.putText(frame, f"SOURCE: {camera_source} | FRAME #{self.frame_counter}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 180), 1)
        cv2.putText(frame, f"PERSONS: {live_detected_persons} | VALID FACES: {valid_face_count} | TRACKS: {active_tracks_count}", (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 230), 1)
        cv2.putText(frame, f"LATENCY: {perf['latency_ms']:.1f} ms | FPS: {perf['fps']:.1f}", (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 200, 255), 1)
        cv2.putText(frame, f"TIME: {ts_now}", (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 150, 165), 1)

        if live_detected_persons == 0:
            cv2.putText(frame, "No Person / Face Detected", (max(40, fw // 2 - 150), fh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 195), 2)

        classroom_analytics = self.fusion_engine.process_classroom_frame(features_list)
        classroom_analytics["live_detected_persons"] = live_detected_persons
        classroom_analytics["valid_face_count"] = valid_face_count
        classroom_analytics["active_tracks"] = active_tracks_count
        classroom_analytics["performance"] = perf
        classroom_analytics["camera_connected"] = True
        classroom_analytics["camera_source"] = camera_source
        classroom_analytics["recommendations"] = self.rec_engine.evaluate_classroom_state(classroom_analytics)

        b64 = ""
        if encode_jpeg:
            _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

        self.latest_b64 = b64
        self.latest_analytics = classroom_analytics
        return b64, classroom_analytics

    def process_next_frame(self) -> Tuple[str, Dict]:
        """Capture one frame from local webcam and run pipeline; returns explicit unavailable state if disconnected."""
        if self.cap is None or not self.cap.isOpened():
            self._open_local_camera()

        ret, frame = (False, None)
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()

        if not ret or frame is None:
            self.camera_connected = False
            # Clear stale tracks on camera disconnection
            self.reset_tracking()
            blank = np.zeros((480, 854, 3), dtype=np.uint8)
            blank[:] = (14, 14, 18)
            cv2.putText(blank, "CAMERA UNAVAILABLE", (260, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 230), 2)
            cv2.putText(blank, "Connect a webcam or enable Browser Camera stream.", (175, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 195), 1)
            _, buf = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

            analytics = self.fusion_engine.process_classroom_frame([])
            analytics["camera_connected"] = False
            analytics["camera_source"] = "DISCONNECTED"
            analytics["performance"] = self.get_performance_metrics()
            analytics["recommendations"] = []
            self.latest_b64 = b64
            self.latest_analytics = analytics
            return b64, analytics

        self.camera_connected = True
        return self.process_frame_array(frame, camera_source="LOCAL_WEBCAM", encode_jpeg=True)

    def get_latest(self) -> Tuple[str, Dict]:
        if not self.latest_b64:
            return self.process_next_frame()
        return self.latest_b64, self.latest_analytics

    def release(self) -> None:
        self.camera_connected = False
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = None
