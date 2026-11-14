
"""
EduSense ByteTrack-Style Multi-Person Tracker
=============================================
Implements Kalman-predicted, two-stage (high/low confidence) Hungarian assignment
tracking with monotonic track IDs, tentative-to-confirmed hysteresis, and
explicit occlusion/lost-track handling.

Guarantees:
- Monotonic unique track IDs (never recycled after track termination).
- Separate high-confidence and low-confidence association stages (ByteTrack).
- Explicit `visible_this_frame` flag so stale boxes are never rendered when lost.
- Configurable confidence thresholds, confirmation hits, and expiration buffer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Compute Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(1.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return float(inter / (area_a + area_b - inter + 1e-6))


class SimpleKalmanBoxTracker:
    """Constant-velocity Kalman filter on [cx, cy, area, aspect_ratio]."""

    def __init__(self, bbox: Tuple[float, float, float, float]):
        cx, cy, area, ratio = self._bbox_to_z(bbox)
        # State: [cx, cy, area, ratio, vx, vy, va]
        self.x = np.array([cx, cy, area, ratio, 0.0, 0.0, 0.0], dtype=np.float64)
        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0

        self.H = np.zeros((4, 7), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

        self.P = np.diag([10.0, 10.0, 100.0, 1.0, 100.0, 100.0, 100.0])
        self.Q = np.diag([1.0, 1.0, 10.0, 0.05, 5.0, 5.0, 10.0])
        self.R = np.diag([4.0, 4.0, 25.0, 0.2])

    @staticmethod
    def _bbox_to_z(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        return x1 + w / 2.0, y1 + h / 2.0, w * h, w / h

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> Tuple[float, float, float, float]:
        cx, cy, area, ratio = z[0], z[1], max(1.0, z[2]), max(0.1, z[3])
        w = math.sqrt(area * ratio)
        h = area / max(1e-3, w)
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def predict(self) -> Tuple[float, float, float, float]:
        if self.x[2] + self.x[6] <= 1.0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._z_to_bbox(self.x[:4])

    def update(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        z = np.array(self._bbox_to_z(bbox), dtype=np.float64)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(7, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return self._z_to_bbox(self.x[:4])


@dataclass
class DetectionCandidate:
    """Single person detection with optional validated face & metadata."""
    person_bbox: Tuple[float, float, float, float]
    person_conf: float
    face_bbox: Optional[Tuple[float, float, float, float]] = None
    face_conf: float = 0.0
    face_quality: float = 0.0
    face_status: str = "NO_FACE_IN_PERSON"  # VALID, OCCLUDED_OR_UNAVAILABLE, REJECTED_SPOOF_OR_UNCERTAIN, NO_FACE_IN_PERSON
    rejection_reason: str = ""
    raw_landmarks: Optional[object] = None
    blendshapes: Optional[Dict[str, float]] = None
    pose_landmarks: Optional[object] = None


@dataclass
class PersonTrack:
    track_id: int
    person_bbox: Tuple[float, float, float, float]
    person_conf: float
    kf: SimpleKalmanBoxTracker
    hits: int = 1
    age: int = 1
    missing_frames: int = 0
    confirmed: bool = False
    visible_this_frame: bool = True
    smoothed_person_bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    latest_detection: Optional[DetectionCandidate] = None

    def __post_init__(self):
        self.smoothed_person_bbox = self.person_bbox

    @property
    def student_id(self) -> str:
        return f"Student {self.track_id:02d}"


class ByteTrackManager:
    """Two-stage Kalman + IoU/distance ByteTrack manager with monotonic student IDs."""

    def __init__(
        self,
        high_conf_thresh: float = 0.50,
        low_conf_thresh: float = 0.30,
        iou_thresh: float = 0.20,
        max_missing_frames: int = 15,
        min_hits_to_confirm: int = 2,
        bbox_ema_alpha: float = 0.35,
        deadband_px: float = 3.0,
    ):
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.iou_thresh = iou_thresh
        self.max_missing_frames = max_missing_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.bbox_ema_alpha = bbox_ema_alpha
        self.deadband_px = deadband_px

        self.tracks: Dict[int, PersonTrack] = {}
        self._next_track_id: int = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_track_id = 1

    def _smooth_bbox(
        self,
        prev_box: Tuple[float, float, float, float],
        new_box: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        a = self.bbox_ema_alpha
        out = []
        for p, n in zip(prev_box, new_box):
            out.append((a * n + (1.0 - a) * p) if abs(n - p) > self.deadband_px else p)
        return (out[0], out[1], out[2], out[3])

    def _match(
        self,
        track_ids: List[int],
        detections: List[DetectionCandidate],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not track_ids or not detections:
            return [], list(track_ids), list(range(len(detections)))

        cost = np.ones((len(track_ids), len(detections)), dtype=np.float64) * 10.0
        for r, tid in enumerate(track_ids):
            t_box = self.tracks[tid].person_bbox
            tcx = (t_box[0] + t_box[2]) / 2.0
            tcy = (t_box[1] + t_box[3]) / 2.0
            t_diag = max(1.0, math.hypot(t_box[2] - t_box[0], t_box[3] - t_box[1]))

            for c, det in enumerate(detections):
                d_box = det.person_bbox
                iou = compute_iou(t_box, d_box)
                dcx = (d_box[0] + d_box[2]) / 2.0
                dcy = (d_box[1] + d_box[3]) / 2.0
                norm_dist = math.hypot(tcx - dcx, tcy - dcy) / t_diag

                if iou >= self.iou_thresh or norm_dist <= 0.65:
                    # Lower cost is better
                    cost[r, c] = (1.0 - iou) + 0.35 * norm_dist

        row_idx, col_idx = linear_sum_assignment(cost)
        matched: List[Tuple[int, int]] = []
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < 1.5:
                tid = track_ids[r]
                matched.append((tid, c))
                unmatched_tracks.discard(tid)
                unmatched_dets.discard(c)

        return matched, sorted(unmatched_tracks), sorted(unmatched_dets)

    def update(self, detections: List[DetectionCandidate]) -> List[PersonTrack]:
        """Advance tracker by one frame and return active confirmed tracks."""
        # 1. Predict all existing tracks forward
        for trk in self.tracks.values():
            trk.age += 1
            trk.visible_this_frame = False
            trk.latest_detection = None
            trk.person_bbox = trk.kf.predict()

        high_dets = [d for d in detections if d.person_conf >= self.high_conf_thresh]
        low_dets = [d for d in detections if self.low_conf_thresh <= d.person_conf < self.high_conf_thresh]

        all_track_ids = list(self.tracks.keys())
        # Stage 1: High-confidence detections matched against all tracks
        matched_1, unmatched_tids_1, unmatched_high_idx = self._match(all_track_ids, high_dets)

        for tid, det_idx in matched_1:
            self._apply_match(tid, high_dets[det_idx])

        # Stage 2: Low-confidence detections matched against remaining active tracks (ByteTrack)
        matched_2, unmatched_tids_2, _ = self._match(unmatched_tids_1, low_dets)
        for tid, det_idx in matched_2:
            self._apply_match(tid, low_dets[det_idx])

        # Increment missing count on unmatched tracks
        for tid in unmatched_tids_2:
            self.tracks[tid].missing_frames += 1

        # Create new tracks for unmatched high-confidence detections only
        for det_idx in unmatched_high_idx:
            det = high_dets[det_idx]
            tid = self._next_track_id
            self._next_track_id += 1
            trk = PersonTrack(
                track_id=tid,
                person_bbox=det.person_bbox,
                person_conf=det.person_conf,
                kf=SimpleKalmanBoxTracker(det.person_bbox),
                hits=1,
                confirmed=(self.min_hits_to_confirm <= 1),
                visible_this_frame=True,
                latest_detection=det,
            )
            self.tracks[tid] = trk

        # Purge expired tracks
        expired = [
            tid for tid, trk in self.tracks.items()
            if trk.missing_frames > self.max_missing_frames or (not trk.confirmed and trk.missing_frames >= 2)
        ]
        for tid in expired:
            del self.tracks[tid]

        return sorted(
            [t for t in self.tracks.values() if t.confirmed],
            key=lambda t: t.track_id,
        )

    def _apply_match(self, tid: int, det: DetectionCandidate) -> None:
        trk = self.tracks[tid]
        kf_box = trk.kf.update(det.person_bbox)
        # Blend raw detection with Kalman output for tight yet jitter-free box
        blended = tuple(0.65 * d + 0.35 * k for d, k in zip(det.person_bbox, kf_box))
        trk.person_bbox = det.person_bbox
        trk.smoothed_person_bbox = self._smooth_bbox(trk.smoothed_person_bbox, blended)
        trk.person_conf = det.person_conf
        trk.hits += 1
        trk.missing_frames = 0
        trk.visible_this_frame = True
        trk.latest_detection = det
        if trk.hits >= self.min_hits_to_confirm:
            trk.confirmed = True

