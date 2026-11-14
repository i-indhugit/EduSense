"""
EduSense Strict Person -> Face -> Landmark -> Pose Detector Suite
=================================================================
Uses pretrained CPU-friendly models (MediaPipe Tasks + OpenCV YuNet):
1. Person & Display/Book Detection: EfficientDet-Lite0 (COCO)
2. Person-Constrained Face Detection: OpenCV YuNet (FaceDetectorYN)
3. 3D Face Landmarks & Blendshapes: MediaPipe FaceLandmarker (478 pts + 52 blendshapes)
4. Body Posture & Hand-Raise Keypoints: MediaPipe PoseLandmarker Lite (33 pts)

Enforces strict spatial & anatomical validation:
- Never runs unconstrained full-frame face detection.
- Rejects faces enclosed in detected TV / laptop / phone / book regions.
- Rejects chest/torso-level printed faces (e.g., T-shirt prints) via upper-body
  bounds and Pose shoulder-line anatomical gating.
- Validates facial landmark geometry (eyes above nose above mouth) and sharpness.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from inference.tracker import DetectionCandidate, compute_iou

logger = logging.getLogger("EduSenseDetectors")

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "models" / "assets"

EFFICIENTDET_PATH = ASSETS_DIR / "efficientdet_lite0.tflite"
YUNET_PATH = ASSETS_DIR / "face_detection_yunet_2023mar.onnx"
FACE_LANDMARKER_PATH = ASSETS_DIR / "face_landmarker.task"
POSE_LANDMARKER_PATH = ASSETS_DIR / "pose_landmarker_lite.task"


def _nms_boxes(
    boxes: List[Tuple[float, float, float, float]],
    scores: List[float],
    iou_thresh: float = 0.45,
) -> List[int]:
    """Pure-NumPy Non-Maximum Suppression returning kept indices."""
    if not boxes:
        return []
    arr = np.array(boxes, dtype=np.float32)
    sc = np.array(scores, dtype=np.float32)
    order = sc.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        kept_rest = []
        for j in rest:
            if compute_iou(tuple(arr[i]), tuple(arr[int(j)])) <= iou_thresh:
                kept_rest.append(int(j))
        order = np.array(kept_rest, dtype=np.int64)
    return keep


class StrictPersonFaceDetector:
    """Hierarchical Person -> Pose -> Face -> Landmark detector."""

    def __init__(
        self,
        person_conf_thresh: float = 0.40,
        face_conf_thresh: float = 0.55,
        min_person_w: int = 36,
        min_person_h: int = 50,
        min_face_size: int = 28,
        enable_pose: bool = True,
    ):
        self.person_conf_thresh = person_conf_thresh
        self.face_conf_thresh = face_conf_thresh
        self.min_person_w = min_person_w
        self.min_person_h = min_person_h
        self.min_face_size = min_face_size
        self.enable_pose = enable_pose

        self.models_loaded = False
        self.load_errors: List[str] = []

        self.object_detector = None
        self.face_detector = None
        self.face_landmarker = None
        self.pose_landmarker = None

        self._init_models()

    def _init_models(self) -> None:
        try:
            if not EFFICIENTDET_PATH.exists():
                raise FileNotFoundError(f"Missing model: {EFFICIENTDET_PATH}")
            self.object_detector = mp_vision.ObjectDetector.create_from_options(
                mp_vision.ObjectDetectorOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=str(EFFICIENTDET_PATH)),
                    score_threshold=min(0.30, self.person_conf_thresh),
                    max_results=25,
                    category_allowlist=["person", "tv", "laptop", "cell phone", "book"],
                )
            )
        except Exception as e:
            self.load_errors.append(f"ObjectDetector: {e}")
            logger.error("Failed to load ObjectDetector: %s", e)

        try:
            if not YUNET_PATH.exists():
                raise FileNotFoundError(f"Missing model: {YUNET_PATH}")
            self.face_detector = cv2.FaceDetectorYN.create(
                str(YUNET_PATH),
                "",
                (320, 320),
                score_threshold=self.face_conf_thresh,
                nms_threshold=0.30,
                top_k=20,
            )
        except Exception as e:
            self.load_errors.append(f"FaceDetectorYN: {e}")
            logger.error("Failed to load FaceDetectorYN: %s", e)

        try:
            if not FACE_LANDMARKER_PATH.exists():
                raise FileNotFoundError(f"Missing model: {FACE_LANDMARKER_PATH}")
            self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(
                mp_vision.FaceLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)),
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    num_faces=1,
                    min_face_detection_confidence=0.45,
                    min_face_presence_confidence=0.45,
                )
            )
        except Exception as e:
            self.load_errors.append(f"FaceLandmarker: {e}")
            logger.error("Failed to load FaceLandmarker: %s", e)

        if self.enable_pose:
            try:
                if not POSE_LANDMARKER_PATH.exists():
                    raise FileNotFoundError(f"Missing model: {POSE_LANDMARKER_PATH}")
                self.pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
                    mp_vision.PoseLandmarkerOptions(
                        base_options=mp_tasks.BaseOptions(model_asset_path=str(POSE_LANDMARKER_PATH)),
                        num_poses=1,
                        min_pose_detection_confidence=0.45,
                        min_pose_presence_confidence=0.45,
                    )
                )
            except Exception as e:
                self.load_errors.append(f"PoseLandmarker: {e}")
                logger.warning("PoseLandmarker unavailable (continuing without pose): %s", e)

        self.models_loaded = (
            self.object_detector is not None
            and self.face_detector is not None
            and self.face_landmarker is not None
        )

    @staticmethod
    def _compute_face_quality(
        face_bgr: np.ndarray,
        det_conf: float,
        landmarks: list,
    ) -> Tuple[float, bool, str]:
        """Validate anatomical landmark order, resolution, sharpness, and lighting."""
        fh, fw = face_bgr.shape[:2]
        if fh < 24 or fw < 24:
            return 0.10, False, "Face crop too small"

        # Check anatomical vertical ordering: eyes (33, 263) < nose (1) < mouth (13)
        if len(landmarks) < 468:
            return 0.15, False, "Incomplete facial landmarks"

        le_y = landmarks[33].y
        re_y = landmarks[263].y
        nose_y = landmarks[1].y
        mouth_y = landmarks[13].y
        eye_y = (le_y + re_y) / 2.0

        if not (eye_y < nose_y < mouth_y):
            return 0.15, False, "Invalid landmark vertical order"

        inter_ocular_px = abs(landmarks[263].x - landmarks[33].x) * fw
        if inter_ocular_px < 10.0:
            return 0.18, False, "Inter-ocular distance too small"

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        if brightness < 18.0 or brightness > 246.0:
            return 0.20, False, f"Extreme lighting (mean={brightness:.0f})"

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = min(1.0, lap_var / 120.0)
        size_score = min(1.0, (fw * fh) / (110.0 * 110.0))
        lighting_score = 1.0 - min(1.0, abs(brightness - 128.0) / 128.0) * 0.4

        quality = (
            0.35 * min(1.0, det_conf)
            + 0.30 * size_score
            + 0.20 * max(0.2, sharpness_score)
            + 0.15 * lighting_score
        )
        quality = round(float(max(0.10, min(1.0, quality))), 3)
        return quality, True, "OK"

    @staticmethod
    def _box_enclosed_by_any(
        inner: Tuple[float, float, float, float],
        outers: List[Tuple[float, float, float, float]],
        ratio_thresh: float = 0.55,
    ) -> bool:
        ix1, iy1, ix2, iy2 = inner
        i_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
        icx = (ix1 + ix2) / 2.0
        icy = (iy1 + iy2) / 2.0
        for ox1, oy1, ox2, oy2 in outers:
            if not (ox1 <= icx <= ox2 and oy1 <= icy <= oy2):
                continue
            xa = max(ix1, ox1)
            ya = max(iy1, oy1)
            xb = min(ix2, ox2)
            yb = min(iy2, oy2)
            inter = max(0.0, xb - xa) * max(0.0, yb - ya)
            if (inter / i_area) >= ratio_thresh:
                return True
        return False

    def detect_frame(
        self,
        frame_bgr: np.ndarray,
        run_pose: bool = True,
    ) -> Tuple[List[DetectionCandidate], List[Tuple[str, Tuple[float, float, float, float], float]]]:
        """
        Run hierarchical detection on a BGR frame.

        Returns:
            candidates: Confirmed person detections with validated face/pose/landmarks.
            rejected_debug: Rejected candidate info [(reason, bbox, conf)] for telemetry.
        """
        if not self.models_loaded or frame_bgr is None or frame_bgr.size == 0:
            return [], []

        img_h, img_w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Stage 1: Detect persons + potential display/book distractors
        obj_res = self.object_detector.detect(mp_image)
        raw_persons: List[Tuple[float, float, float, float]] = []
        raw_person_scores: List[float] = []
        screen_or_book_boxes: List[Tuple[float, float, float, float]] = []

        for det in obj_res.detections:
            if not det.categories:
                continue
            cat = det.categories[0]
            name = (cat.category_name or "").lower()
            score = float(cat.score)
            bb = det.bounding_box
            x1 = float(max(0, bb.origin_x))
            y1 = float(max(0, bb.origin_y))
            x2 = float(min(img_w, bb.origin_x + bb.width))
            y2 = float(min(img_h, bb.origin_y + bb.height))
            bw, bh = x2 - x1, y2 - y1

            if name == "person":
                if score < 0.30:
                    continue
                if bw < self.min_person_w or bh < self.min_person_h:
                    continue
                aspect = bw / max(1.0, bh)
                if not (0.18 <= aspect <= 2.3):
                    continue
                raw_persons.append((x1, y1, x2, y2))
                raw_person_scores.append(score)
            elif name in ("tv", "laptop", "cell phone", "book") and score >= 0.42:
                # Only treat as distractor screen if it's not huge background covering whole frame
                if (bw * bh) < 0.85 * (img_w * img_h):
                    screen_or_book_boxes.append((x1, y1, x2, y2))

        keep_idx = _nms_boxes(raw_persons, raw_person_scores, iou_thresh=0.50)
        candidates: List[DetectionCandidate] = []
        rejected_debug: List[Tuple[str, Tuple[float, float, float, float], float]] = []

        # Stage 2: Inside each confirmed person ROI, run Pose + Face detection + Landmarks
        for idx in keep_idx:
            px1, py1, px2, py2 = raw_persons[idx]
            p_conf = raw_person_scores[idx]
            pw = px2 - px1
            ph = py2 - py1

            # Extract padded person crop for PoseLandmarker
            pad_x = int(pw * 0.06)
            pad_y = int(ph * 0.06)
            cx1 = max(0, int(px1) - pad_x)
            cy1 = max(0, int(py1) - pad_y)
            cx2 = min(img_w, int(px2) + pad_x)
            cy2 = min(img_h, int(py2) + pad_y)
            person_crop_bgr = frame_bgr[cy1:cy2, cx1:cx2]
            if person_crop_bgr.size == 0:
                continue

            pose_lms = None
            shoulder_y_abs: Optional[float] = None
            if run_pose and self.pose_landmarker is not None:
                try:
                    p_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
                    p_res = self.pose_landmarker.detect(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=p_rgb)
                    )
                    if p_res.pose_landmarks:
                        pose_lms = p_res.pose_landmarks[0]
                        ls, rs = pose_lms[11], pose_lms[12]
                        if getattr(ls, "visibility", 1.0) >= 0.40 and getattr(rs, "visibility", 1.0) >= 0.40:
                            shoulder_y_abs = cy1 + ((ls.y + rs.y) / 2.0) * (cy2 - cy1)
                except Exception as e:
                    logger.debug("Pose detection error in ROI: %s", e)

            # Constrain face search vertically inside person box:
            # For full-body standing/tall boxes (h/w >= 1.4), head is in upper 50% of box.
            # For close-up webcam upper-body boxes, head is in upper 78% of box.
            upper_frac = 0.52 if (ph / max(1.0, pw)) >= 1.40 else 0.80
            search_y2 = min(img_h, int(py1 + ph * upper_frac))
            search_y1 = max(0, int(py1 - ph * 0.04))
            search_x1 = max(0, int(px1 - pw * 0.04))
            search_x2 = min(img_w, int(px2 + pw * 0.04))
            roi_bgr = frame_bgr[search_y1:search_y2, search_x1:search_x2]
            rh, rw = roi_bgr.shape[:2]

            best_face: Optional[Tuple[Tuple[float, float, float, float], float, float, object, Dict[str, float]]] = None
            face_status = "NO_FACE_IN_PERSON"
            rej_reason = ""

            if rh >= self.min_face_size and rw >= self.min_face_size:
                self.face_detector.setInputSize((rw, rh))
                _, faces = self.face_detector.detect(roi_bgr)

                if faces is not None and len(faces) > 0:
                    # Evaluate candidate faces in upper person region, pick highest anatomical valid head
                    scored_faces = []
                    for f in faces:
                        fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                        f_conf = float(f[-1])
                        if f_conf < self.face_conf_thresh:
                            continue
                        if fw < self.min_face_size or fh < self.min_face_size:
                            continue
                        f_aspect = fw / max(1.0, fh)
                        if not (0.52 <= f_aspect <= 1.40):
                            continue

                        abs_box = (
                            search_x1 + fx,
                            search_y1 + fy,
                            search_x1 + fx + fw,
                            search_y1 + fy + fh,
                        )
                        face_cy = (abs_box[1] + abs_box[3]) / 2.0

                        # Anti-spoof 1: Reject if enclosed inside a detected TV/laptop/phone/book
                        if self._box_enclosed_by_any(abs_box, screen_or_book_boxes):
                            rejected_debug.append(("REJECTED_DISPLAY_SCREEN", abs_box, f_conf))
                            face_status = "REJECTED_SPOOF_OR_UNCERTAIN"
                            rej_reason = "Face enclosed within detected screen/book object"
                            continue

                        # Anti-spoof 2: Reject chest/torso-level printed faces using pose shoulder line
                        if shoulder_y_abs is not None and face_cy > (shoulder_y_abs + 0.12 * fh):
                            rejected_debug.append(("REJECTED_TORSO_TSHIRT", abs_box, f_conf))
                            face_status = "REJECTED_SPOOF_OR_UNCERTAIN"
                            rej_reason = "Candidate face below shoulder line (torso/clothing print)"
                            continue

                        # Priority score favors higher anatomical head position + detection confidence
                        rel_y = (face_cy - py1) / max(1.0, ph)
                        anat_score = f_conf - (0.35 * rel_y)
                        scored_faces.append((anat_score, abs_box, f_conf))

                    scored_faces.sort(key=lambda item: item[0], reverse=True)

                    for _, abs_box, f_conf in scored_faces:
                        # Stage 3: Run FaceLandmarker on padded face crop for 478-pt validation
                        fx1, fy1, fx2, fy2 = abs_box
                        fw, fh = fx2 - fx1, fy2 - fy1
                        mpad_x, mpad_y = int(fw * 0.25), int(fh * 0.25)
                        lx1 = max(0, int(fx1) - mpad_x)
                        ly1 = max(0, int(fy1) - mpad_y)
                        lx2 = min(img_w, int(fx2) + mpad_x)
                        ly2 = min(img_h, int(fy2) + mpad_y)
                        face_crop_bgr = frame_bgr[ly1:ly2, lx1:lx2]
                        if face_crop_bgr.size == 0:
                            continue

                        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                        fl_res = self.face_landmarker.detect(
                            mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
                        )
                        if not fl_res.face_landmarks:
                            rejected_debug.append(("REJECTED_NO_LANDMARKS", abs_box, f_conf))
                            face_status = "REJECTED_SPOOF_OR_UNCERTAIN"
                            rej_reason = "No 3D facial landmarks verified"
                            continue

                        lms = fl_res.face_landmarks[0]
                        # Remap normalized landmarks from padded crop to original face box for quality check
                        tight_face_bgr = frame_bgr[max(0, int(fy1)):min(img_h, int(fy2)), max(0, int(fx1)):min(img_w, int(fx2))]
                        if tight_face_bgr.size == 0:
                            continue
                        quality, is_valid_lm, q_reason = self._compute_face_quality(tight_face_bgr, f_conf, lms)
                        if not is_valid_lm:
                            rejected_debug.append((f"REJECTED_{q_reason}", abs_box, f_conf))
                            face_status = "OCCLUDED_OR_UNAVAILABLE"
                            rej_reason = q_reason
                            continue

                        bs_dict: Dict[str, float] = {}
                        if fl_res.face_blendshapes:
                            for b in fl_res.face_blendshapes[0]:
                                bs_dict[b.category_name] = float(b.score)

                        # Remap landmarks to crop-normalized coordinates relative to full frame
                        best_face = (abs_box, f_conf, quality, (lms, (lx1, ly1, lx2 - lx1, ly2 - ly1), fl_res.facial_transformation_matrixes), bs_dict)
                        face_status = "VALID"
                        rej_reason = ""
                        break

            candidates.append(
                DetectionCandidate(
                    person_bbox=(px1, py1, px2, py2),
                    person_conf=p_conf,
                    face_bbox=best_face[0] if best_face else None,
                    face_conf=best_face[1] if best_face else 0.0,
                    face_quality=best_face[2] if best_face else 0.0,
                    face_status=face_status,
                    rejection_reason=rej_reason,
                    raw_landmarks=best_face[3] if best_face else None,
                    blendshapes=best_face[4] if best_face else None,
                    pose_landmarks=pose_lms,
                )
            )

        return candidates, rejected_debug

