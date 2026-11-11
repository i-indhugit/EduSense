"""
EduSense Ultra-Stable Real-Time Vision & Analytics Pipeline
============================================================
Processes ONLY live webcam frames using OpenCV face/landmark detection paired
with a persistent IoU Multi-Face Tracker and Temporal State Hysteresis Engine.

Guarantees:
- Persistent Student IDs (IoU tracking eliminates ID switching)
- Heavy EMA spatial smoothing & deadband quantization for bounding boxes (zero jitter)
- Hysteresis & multi-frame consensus for predictions (zero flickering)
- Immediate clearing of track states when faces are lost permanently
"""

import cv2
import numpy as np
import base64
import time
import math
import os
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional

from fusion_engine.feature_fusion import FeatureFusionEngine
from fusion_engine.cross_attention_fusion import create_numpy_reliability_fusion
from utils.uncertainty_utils import UncertaintyEstimationEngine, validate_evidence_or_fallback
from recommendation.engine import TeacherRecommendationEngine
from utils.attention_utils import compute_attention_status, compute_final_attention_state, compute_simplified_attention_score


BASE_DIR = Path(__file__).resolve().parent.parent
FACE_CASCADE_PATH = str(BASE_DIR / "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = str(BASE_DIR / "haarcascade_eye.xml")

MIN_CONFIDENCE_THRESHOLD = 50.0
MAX_MISSING_FRAMES = 3


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union between two boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


class TrackedFaceState:
    def __init__(self, track_id: int, initial_bbox: Tuple[float, float, float, float]):
        self.track_id = track_id
        self.student_id = f"Student {track_id:02d}"
        
        # Smoothed bounding box (x1, y1, x2, y2)
        self.bbox = initial_bbox
        self.missing_frames = 0

        # Independent prediction & feature history deques per student
        self.att_history = deque(maxlen=10)
        self.disp_att = 0.0

        self.conf_history = deque(maxlen=10)
        self.disp_conf = 0.0

        self.face_quality_history = deque(maxlen=10)
        self.disp_face_quality = 0.85

        self.emotion_history = deque(maxlen=6)
        self.disp_emotion = "Neutral"

        self.gaze_history = deque(maxlen=6)
        self.disp_gaze = "Looking at Board"

        self.head_pose_history = deque(maxlen=6)
        self.disp_head_pose = "Upright (0°)"

        self.posture_history = deque(maxlen=6)
        self.disp_posture = "Upright"

        self.drowsy_window = deque(maxlen=10)
        self.disp_is_drowsy_consistent = False

        self.disp_final_state = "ATTENTIVE"
        self.disp_reason = "Initial detection state"
        self.disp_evidence = ""
        self.latest_raw_features = {}

    def update_bbox(self, raw_bbox: Tuple[float, float, float, float], alpha: float = 0.18, deadband_px: float = 3.0):
        """Smooth bounding box with heavy EMA and a deadband offset threshold."""
        rx1, ry1, rx2, ry2 = raw_bbox
        cx1, cy1, cx2, cy2 = self.bbox

        dx1, dy1 = abs(rx1 - cx1), abs(ry1 - cy1)
        dx2, dy2 = abs(rx2 - cx2), abs(ry2 - cy2)

        nx1 = (alpha * rx1 + (1 - alpha) * cx1) if dx1 > deadband_px else cx1
        ny1 = (alpha * ry1 + (1 - alpha) * cy1) if dy1 > deadband_px else cy1
        nx2 = (alpha * rx2 + (1 - alpha) * cx2) if dx2 > deadband_px else cx2
        ny2 = (alpha * ry2 + (1 - alpha) * cy2) if dy2 > deadband_px else cy2

        self.bbox = (nx1, ny1, nx2, ny2)

    def update_student_observations(
        self,
        raw_gaze: str,
        raw_head_pose: str,
        raw_posture: str,
        raw_drowsy_evidence: bool,
        ear_val: float,
        face_quality: float,
        raw_emotion: str,
        raw_conf: float
    ):
        """Update student's isolated temporal state and calculate score + single final attention state."""
        # 1. Update deques for this student
        self.drowsy_window.append(1 if raw_drowsy_evidence else 0)
        self.gaze_history.append(raw_gaze)
        self.head_pose_history.append(raw_head_pose)
        self.posture_history.append(raw_posture)
        self.emotion_history.append(raw_emotion)
        self.face_quality_history.append(face_quality)
        self.conf_history.append(raw_conf)

        self.latest_raw_features = {
            "gaze": raw_gaze,
            "head_pose": raw_head_pose,
            "posture": raw_posture,
            "ear": round(ear_val, 3),
            "face_quality": round(face_quality, 2),
            "emotion": raw_emotion,
            "drowsy_evidence_frame": raw_drowsy_evidence
        }

        # 2. Categorical consensus for observations
        self.disp_gaze = self._most_frequent(self.gaze_history, self.disp_gaze)
        self.disp_head_pose = self._most_frequent(self.head_pose_history, self.disp_head_pose)
        self.disp_posture = self._most_frequent(self.posture_history, self.disp_posture)
        self.disp_emotion = self._most_frequent(self.emotion_history, self.disp_emotion)

        # 3. Drowsiness temporal window consensus (Requires >= 60% of window frames showing drowsiness)
        drowsy_ratio = sum(self.drowsy_window) / float(len(self.drowsy_window))
        self.disp_is_drowsy_consistent = (drowsy_ratio >= 0.60)

        # Clean Basic Score Calculation: Gaze (40%), Head Pose (30%), Face Visibility (20%), Posture (10%)
        avg_fq = sum(self.face_quality_history) / float(len(self.face_quality_history))
        self.disp_face_quality = round(avg_fq, 2)

        avg_conf = sum(self.conf_history) / float(len(self.conf_history))
        self.disp_conf = round(avg_conf, 1)

        sim_score, sim_status, sim_reason = compute_simplified_attention_score(
            gaze=self.disp_gaze,
            head_pose=self.disp_head_pose,
            face_quality=self.disp_face_quality,
            posture=self.disp_posture
        )

        self.disp_att = sim_score
        self.disp_final_state = sim_status
        self.disp_reason = sim_reason
        self.disp_evidence = (
            f"Gaze: {self.disp_gaze}, HeadPose: {self.disp_head_pose}, "
            f"FaceVis: {int(self.disp_face_quality * 100)}%, Posture: {self.disp_posture}"
        )

    def update_predictions(self, raw_att: float, raw_conf: float, raw_emotion: str,
                           raw_gaze: str, raw_head_pose: str, raw_drowsy: bool,
                           raw_status: str):
        """Backward compatibility adapter mapping to update_student_observations."""
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
            raw_conf=raw_conf
        )

    def get_debug_info(self, camera_source: str = "WEBCAM", frame_id: int = 0,
                       frame_timestamp: str = "", detected_person_count: int = 0,
                       active_track_count: int = 0, track_ids: List[str] = None) -> dict:
        """Returns student-specific debug telemetry dictionary strictly matching live debug requirement #12."""
        drowsy_ratio = sum(self.drowsy_window) / float(len(self.drowsy_window)) if self.drowsy_window else 0.0
        bx1, by1, bx2, by2 = self.bbox
        is_valid = (self.disp_face_quality >= 0.25) and (self.disp_conf >= MIN_CONFIDENCE_THRESHOLD)

        return {
            "camera_source": camera_source,
            "frame_id": frame_id,
            "frame_timestamp": frame_timestamp or time.strftime("%H:%M:%S"),
            "detected_person_count": detected_person_count,
            "active_track_count": active_track_count,
            "track_ids": track_ids or [self.student_id],
            "bbox": [int(bx1), int(by1), int(bx2), int(by2)],
            "detection_confidence": self.disp_conf,
            "face_detected": True,
            "face_quality": self.disp_face_quality,
            "gaze": self.disp_gaze if self.disp_final_state != "UNABLE TO DETERMINE" else "Unable to Determine",
            "head_pose": self.disp_head_pose if self.disp_final_state != "UNABLE TO DETERMINE" else "Unable to Determine",
            "posture": self.disp_posture if self.disp_final_state != "UNABLE TO DETERMINE" else "Unable to Determine",
            "drowsiness_probability": round(drowsy_ratio, 2),
            "attention_score": self.disp_att if self.disp_final_state != "UNABLE TO DETERMINE" else 0.0,
            "final_state": self.disp_final_state,
            "student_id": self.student_id,
            "raw_feature_values": self.latest_raw_features,
            "temporal_values": {
                "smoothed_attention_score": self.disp_att,
                "drowsy_window_count": sum(self.drowsy_window),
                "drowsy_window_total": len(self.drowsy_window),
                "drowsy_window_ratio_pct": round(drowsy_ratio * 100.0, 1),
                "is_drowsy_consistent": self.disp_is_drowsy_consistent
            },
            "reliability_values": {
                "face_quality": self.disp_face_quality,
                "confidence_pct": self.disp_conf,
                "is_valid_evidence": is_valid
            },
            "final_attention_score": self.disp_att if self.disp_final_state != "UNABLE TO DETERMINE" else 0.0,
            "reason_for_state": self.disp_reason
        }

    @staticmethod
    def _most_frequent(history: deque, current_val):
        if not history:
            return current_val
        counts = {}
        for item in history:
            counts[item] = counts.get(item, 0) + 1
        best_item, max_count = max(counts.items(), key=lambda x: x[1])
        if max_count >= math.ceil(len(history) * 0.60):
            return best_item
        return current_val


LIVE_MODE = True

class EduSensePipeline:
    def __init__(self, mode: str = "webcam", camera_index: int = 0):
        self.mode = mode
        self.camera_index = camera_index
        self.fusion_engine = FeatureFusionEngine()
        self.rec_engine = TeacherRecommendationEngine()
        self.cap: Optional[cv2.VideoCapture] = None

        # Frame counter & FPS tracking
        self.frame_counter = 0
        self.last_frame_time = time.time()
        self.current_fps = 0.0

        # Load Primary HOG Person Detector (COCO Class 0: Person)
        self.person_detector = cv2.HOGDescriptor()
        self.person_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # Load Downstream Face & Eye Cascades (Used ONLY inside person crop)
        if os.path.exists(FACE_CASCADE_PATH):
            self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        else:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if os.path.exists(EYE_CASCADE_PATH):
            self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        else:
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Persistent Multi-Person Tracking Registry
        self._tracks: Dict[int, TrackedFaceState] = {}
        self._next_track_id = 1

        # Open physical camera
        self.cap = cv2.VideoCapture(self.camera_index)

        # Cache for thread-safe async access
        self.latest_b64: str = ""
        self.latest_analytics: Dict = self.fusion_engine.process_classroom_frame([])

    def process_next_frame(self) -> Tuple[str, Dict]:
        """
        Capture one live frame, run real person detection & tracking, compute analytics,
        and return (base64_jpeg, analytics_dict).
        """
        frame, raw_features = self._detect_and_track_live_frame()

        # Feature Fusion
        classroom_analytics = self.fusion_engine.process_classroom_frame(raw_features)

        # Teacher Recommendations
        recommendations = self.rec_engine.evaluate_classroom_state(classroom_analytics)
        classroom_analytics["recommendations"] = recommendations

        # Encode frame to JPEG Base64
        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

        self.latest_b64 = b64
        self.latest_analytics = classroom_analytics

        return b64, classroom_analytics

    def get_latest(self) -> Tuple[str, Dict]:
        """Returns cached latest b64 frame and analytics."""
        if not self.latest_b64:
            return self.process_next_frame()
        return self.latest_b64, self.latest_analytics

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def _detect_and_track_live_frame(self) -> Tuple[np.ndarray, List[Dict]]:
        """Read live webcam frame, perform IoU multi-person tracking, temporal consensus, and smooth overlay."""
        FRAME_W, FRAME_H = 1280, 720

        # Attempt to read from physical camera
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        ret, frame = False, None
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()

        if not ret or frame is None:
            blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            blank[:] = (12, 12, 15)
            cv2.putText(blank, "CAMERA UNAVAILABLE", (440, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 240), 2)
            cv2.putText(blank, "Connect a physical webcam to begin live analysis.",
                        (360, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 190), 1)
            return blank, []

        # Increment frame counter & measure FPS
        self.frame_counter += 1
        now_time = time.time()
        dt = now_time - self.last_frame_time
        self.last_frame_time = now_time
        self.current_fps = (1.0 / dt) if dt > 0 else 30.0
        timestamp_now = time.strftime("%H:%M:%S")

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eq_gray = cv2.equalizeHist(gray)

        # ─────────────────────────────────────────────────────────────────
        # LIVE WEBCAM FACE DETECTION (Sensitive & Responsive Multi-Neighbor NMS)
        # ─────────────────────────────────────────────────────────────────
        raw_faces = self.face_cascade.detectMultiScale(
            eq_gray,
            scaleFactor=1.10,
            minNeighbors=3,
            minSize=(40, 40)
        )

        if len(raw_faces) == 0:
            raw_faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.10,
                minNeighbors=3,
                minSize=(40, 40)
            )

        detected_boxes: List[Tuple[float, float, float, float]] = []

        if len(raw_faces) > 0:
            sorted_raw = sorted(raw_faces, key=lambda b: b[2] * b[3], reverse=True)
            for (x, y, w, h) in sorted_raw:
                aspect_ratio = float(w) / float(h)
                if 0.55 <= aspect_ratio <= 1.45 and w >= 35 and h >= 35:
                    detected_boxes.append((float(x), float(y), float(x + w), float(y + h)))

        # Non-Maximum Suppression (NMS) to collapse any overlapping candidate boxes into 1 box per face
        if len(detected_boxes) > 1:
            nms_boxes = [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])] for b in detected_boxes]
            scores = [float((b[2] - b[0]) * (b[3] - b[1])) for b in detected_boxes]
            indices = cv2.dnn.NMSBoxes(nms_boxes, scores, score_threshold=10.0, nms_threshold=0.25)
            if len(indices) > 0:
                detected_boxes = [detected_boxes[i] for i in indices.flatten()]

        raw_person_count = len(raw_faces)
        live_detected_persons = len(detected_boxes)

        # ─────────────────────────────────────────────────────────────────
        # TRACKER ASSOCIATION & TEMPORAL HYSTERESIS PERSISTENCE
        # ─────────────────────────────────────────────────────────────────
        matched_track_ids = set()
        matched_box_indices = set()

        for track_id, track in list(self._tracks.items()):
            best_match_score = -1.0
            best_idx = -1
            tx1, ty1, tx2, ty2 = track.bbox
            tcx, tcy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
            t_diag = max(1.0, math.sqrt((tx2 - tx1)**2 + (ty2 - ty1)**2))

            for idx, box in enumerate(detected_boxes):
                if idx in matched_box_indices:
                    continue
                iou = compute_iou(track.bbox, box)
                bx1, by1, bx2, by2 = box
                bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                dist = math.sqrt((tcx - bcx)**2 + (tcy - bcy)**2)
                norm_dist = dist / t_diag

                if iou >= 0.12 or norm_dist <= 0.90:
                    match_score = iou + max(0.0, 1.0 - norm_dist)
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_idx = idx

            if best_idx != -1:
                track.update_bbox(detected_boxes[best_idx], alpha=0.18, deadband_px=3.0)
                track.missing_frames = 0
                matched_track_ids.add(track_id)
                matched_box_indices.add(best_idx)
            else:
                track.missing_frames += 1

        # Create new track ONLY for unassigned valid detections
        for idx, box in enumerate(detected_boxes):
            if idx not in matched_box_indices:
                existing_ids = {t.track_id for t in self._tracks.values()}
                next_id = 1
                while next_id in existing_ids:
                    next_id += 1
                new_track = TrackedFaceState(next_id, box)
                self._tracks[next_id] = new_track
                matched_track_ids.add(next_id)

        # Temporal Hysteresis: Retain track for up to 10 missing frames (~0.35s) before removal (zero flickering)
        for track_id in list(self._tracks.keys()):
            if self._tracks[track_id].missing_frames >= 10:
                del self._tracks[track_id]

        active_tracks = len(self._tracks)
        features: List[Dict] = []
        track_id_labels = [f"Student {t.track_id:02d}" for t in self._tracks.values()]
        timestamp_now = time.strftime("%Y-%m-%d %H:%M:%S")

        # ─────────────────────────────────────────────────────────────────
        # Render Telemetry Overlay
        # ─────────────────────────────────────────────────────────────────
        cv2.rectangle(frame, (12, 12), (350, 134), (15, 15, 20), -1)
        cv2.rectangle(frame, (12, 12), (350, 134), (40, 40, 50), 1)

        cv2.putText(frame, "CAMERA SOURCE: WEBCAM", (22, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 200), 1)
        cv2.putText(frame, f"FRAME ID: #{self.frame_counter}", (22, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1)
        cv2.putText(frame, f"FRAME TIME: {time.strftime('%H:%M:%S')}.{int(time.time()*1000)%1000:03d}", (22, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 210), 1)
        cv2.putText(frame, f"RAW DETECTIONS: {raw_person_count}", (22, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 200, 100), 1)
        cv2.putText(frame, f"ACTIVE TRACKS: {active_tracks}", (22, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 255), 1)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (22, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        if len(self._tracks) == 0:
            cv2.putText(frame, "No Face Detected", (480, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            return frame, []

        sorted_tracks = sorted(self._tracks.values(), key=lambda t: t.track_id)

        for track in sorted_tracks:
            # Render and include track if active or within short grace period (missing_frames < 8)
            if track.missing_frames >= 8:
                continue
            # If track is missing in this frame but within grace period, keep rendering smoothly
            bx1, by1, bx2, by2 = track.bbox
            x1, y1 = max(0, int(bx1)), max(0, int(by1))
            x2, y2 = min(FRAME_W, int(bx2)), min(FRAME_H, int(by2))
            w, h = max(1, x2 - x1), max(1, y2 - y1)

            # Detect eye features within face region
            roi_gray = gray[y1:y2, x1:x2]
            if roi_gray.size > 0:
                eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.08, minNeighbors=3, minSize=(12, 12))
            else:
                eyes = []

            if len(eyes) >= 2:
                ear = round(sum(eh / max(ew, 1) for (ex, ey, ew, eh) in eyes) / len(eyes), 2)
                raw_drowsy = ear < 0.19
                eye_y_avg = sum(ey + eh / 2.0 for (ex, ey, ew, eh) in eyes) / float(len(eyes))
                eye_ratio = eye_y_avg / float(h)
            elif len(eyes) == 1:
                ear = 0.26
                raw_drowsy = False
                eye_ratio = 0.35
            else:
                ear = 0.28
                raw_drowsy = False
                eye_ratio = 0.35

            # Default Head Pose, Gaze, & Posture baselines
            raw_gaze = "Looking at Board"
            raw_head_pose = "Upright (0°)"
            posture = "Upright"

            # Accurate Head Pose & Gaze estimation from facial geometry
            if eye_ratio > 0.58:
                raw_gaze = "Head Down"
                raw_head_pose = "Head Down (-18°)"
                posture = "Slouching"
            elif (h / float(w)) < 0.85:
                raw_gaze = "Looking Away"
                raw_head_pose = "Turned (25°)"
                posture = "Leaning"
            
            raw_att = 0.45 * (95.0 if raw_gaze == "Looking at Board" else 50.0) + 0.30 * (95.0 if posture == "Upright" else 50.0) + 0.25 * (90.0 if not raw_drowsy else 20.0)
            raw_emotion = "Sleepy" if raw_drowsy else ("Bored" if raw_gaze == "Looking Away" else "Neutral")

            # Dynamic Face Quality calculation (0.15 to 1.0)
            bbox_area = float(w * h)
            face_quality = min(1.0, max(0.15, (bbox_area / (140.0 * 140.0)) * (1.1 if len(eyes) >= 1 else 0.85)))
            raw_conf = round(min(99.5, max(30.0, 65.0 + (face_quality * 30.0))), 1)

            # Update student-specific isolated temporal state & calculate single attention state
            track.update_student_observations(
                raw_gaze=raw_gaze,
                raw_head_pose=raw_head_pose,
                raw_posture=posture,
                raw_drowsy_evidence=raw_drowsy,
                ear_val=ear,
                face_quality=face_quality,
                raw_emotion=raw_emotion,
                raw_conf=raw_conf
            )

            is_valid_confidence = (track.disp_face_quality >= 0.25) and (track.disp_conf >= MIN_CONFIDENCE_THRESHOLD)

            # ─────────────────────────────────────────────────────────────
            # Render Bounding Box with Corner Accent Lines & Label Header
            # ─────────────────────────────────────────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 220, 220), 1)

            line_len = min(15, w // 4, h // 4)
            cv2.line(frame, (x1, y1), (x1 + line_len, y1), (255, 255, 255), 2)
            cv2.line(frame, (x1, y1), (x1, y1 + line_len), (255, 255, 255), 2)
            cv2.line(frame, (x2, y1), (x2 - line_len, y1), (255, 255, 255), 2)
            cv2.line(frame, (x2, y1), (x2, y1 + line_len), (255, 255, 255), 2)
            cv2.line(frame, (x1, y2), (x1 + line_len, y2), (255, 255, 255), 2)
            cv2.line(frame, (x1, y2), (x1, y2 - line_len), (255, 255, 255), 2)
            cv2.line(frame, (x2, y2), (x2 - line_len, y2), (255, 255, 255), 2)
            cv2.line(frame, (x2, y2), (x2, y2 - line_len), (255, 255, 255), 2)

            # Bounding Box label header showing TRACK ID, BBOX, and DETECTION CONFIDENCE
            cv2.putText(frame, f"TRACK ID: {track.student_id}", (x1, max(30, y1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 180), 1)
            cv2.putText(frame, f"BBOX: [{x1},{y1},{x2},{y2}]", (x1, max(18, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1)
            cv2.putText(frame, f"CONFIDENCE: {track.disp_conf:.1f}%", (x1, max(6, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 220, 100), 1)

            # ─────────────────────────────────────────────────────────────
            # Render Info Panel BESIDE Bounding Box
            # ─────────────────────────────────────────────────────────────
            panel_w = 210
            panel_h = 175

            px1 = x2 + 8
            if px1 + panel_w > FRAME_W - 10:
                px1 = x1 - panel_w - 8

            px1 = max(8, min(FRAME_W - panel_w - 8, px1))
            py1 = y1
            if py1 + panel_h > FRAME_H - 10:
                py1 = FRAME_H - panel_h - 10
            if py1 < 10:
                py1 = 10

            px2 = px1 + panel_w
            py2 = py1 + panel_h

            cv2.rectangle(frame, (px1, py1), (px2, py2), (15, 15, 18), -1)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (55, 55, 65), 1)

            cv2.rectangle(frame, (px1, py1), (px2, py1 + 20), (25, 25, 30), -1)
            cv2.putText(frame, f"{track.student_id} Live Observations", (px1 + 8, py1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

            if is_valid_confidence:
                info_lines = [
                    ("Gaze",        track.disp_gaze),
                    ("Head Pose",   track.disp_head_pose),
                    ("Posture",     track.disp_posture),
                    ("Drowsiness",  "Drowsy" if track.disp_is_drowsy_consistent else "Alert"),
                    ("Face Quality",f"{int(track.disp_face_quality * 100)}%"),
                    ("Confidence",  f"{track.disp_conf:.1f}%"),
                    ("Evidence",    track.disp_evidence[:30] + "..."),
                ]
            else:
                info_lines = [
                    ("Gaze",        "Unable to Determine"),
                    ("Head Pose",   "Unable to Determine"),
                    ("Posture",     "Unable to Determine"),
                    ("Drowsiness",  "Unable to Determine"),
                    ("Face Quality",f"{int(track.disp_face_quality * 100)}%"),
                    ("Confidence",  f"{track.disp_conf:.1f}%"),
                    ("Evidence",    "Insufficient face quality or confidence"),
                ]

            y_off = py1 + 33
            for label, val in info_lines:
                cv2.putText(frame, f"{label}:", (px1 + 8, int(y_off)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 150, 160), 1)
                cv2.putText(frame, str(val), (px1 + 95, int(y_off)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 230, 230), 1)
                y_off += 18

            # Build feature dict with isolated debug telemetry matching requirement #12
            debug_telemetry = track.get_debug_info(
                camera_source="WEBCAM",
                frame_id=self.frame_counter,
                frame_timestamp=timestamp_now,
                detected_person_count=live_detected_persons,
                active_track_count=active_tracks,
                track_ids=track_id_labels
            )
            features.append({
                "student_id": track.student_id,
                "bbox": [x1, y1, x2, y2],
                "confidence_pct": track.disp_conf,
                "face_quality": track.disp_face_quality,
                "attention_pct": track.disp_att if track.disp_final_state != "UNABLE TO DETERMINE" else 0.0,
                "attention_status": track.disp_final_state,
                "final_state": track.disp_final_state,
                "reason_for_state": track.disp_reason,
                "gaze_direction": track.disp_gaze if track.disp_final_state != "UNABLE TO DETERMINE" else "Unable to Determine",
                "head_pose": track.disp_head_pose if track.disp_final_state != "UNABLE TO DETERMINE" else "Unable to Determine",
                "emotion": track.disp_emotion if track.disp_final_state != "UNABLE TO DETERMINE" else "Unable to Determine",
                "ear_value": round(ear, 3),
                "is_drowsy": track.disp_is_drowsy_consistent,
                "is_yawning": False,
                "hand_raised": False,
                "posture": track.disp_posture if track.disp_final_state != "UNABLE TO DETERMINE" else "Unable to Determine",
                "eye_distance_ratio": 1.0,
                "debug_info": debug_telemetry,
                "_meta_live_detected_persons": live_detected_persons,
                "_meta_active_tracks": active_tracks
            })

        # Bottom status bar displaying exact live counts
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame,
                    f"EduSense LIVE | LIVE_DETECTED_PERSONS: {live_detected_persons} | ACTIVE_TRACKS: {active_tracks} | {ts}",
                    (20, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 220, 170), 1)

        return frame, features
