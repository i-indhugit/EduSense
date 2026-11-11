"""
EduSense-X Uncertainty Estimation & Quality Thresholding Engine
================================================================
Calculates overall prediction uncertainty using:
1. Model Softmax Entropy & Confidence Margin
2. Input Feature & Bounding Box Quality
3. Modality Availability & Reliability Ratio
4. Temporal Variance & Prediction Disagreement

Forces explicit human-readable states when evidence is insufficient:
- "NO FACE DETECTED"
- "UNABLE TO DETERMINE"
- "INSUFFICIENT DATA"
- "STUDENT NOT VISIBLE"
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional


class UncertaintyEstimationEngine:
    def __init__(self, uncertainty_threshold: float = 0.45, min_reliability_threshold: float = 0.25):
        self.uncertainty_threshold = uncertainty_threshold
        self.min_reliability_threshold = min_reliability_threshold

    def calculate_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate normalized Shannon entropy H(p) / log(N) ∈ [0, 1]."""
        p = np.clip(probabilities, 1e-9, 1.0)
        p = p / np.sum(p)
        n = len(p)
        if n <= 1:
            return 0.0
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(n)
        return float(entropy / max_entropy)

    def estimate_uncertainty(self,
                             face_detected: bool,
                             face_quality: float,
                             modality_reliabilities: Dict[str, float],
                             model_probabilities: Optional[np.ndarray] = None,
                             temporal_variance: float = 0.0) -> Tuple[float, str, bool]:
        """
        Estimates total uncertainty U ∈ [0.0, 1.0] and determines if evidence is valid.

        Returns:
            uncertainty_score: float ∈ [0, 1]
            status_reason: str (Human-readable explanation if invalid)
            is_valid: bool (True if prediction is reliable enough to output)
        """
        # 1. Immediate invalid state checks
        if not face_detected:
            return 1.0, "NO FACE DETECTED", False

        if face_quality < 0.20:
            return 0.95, "UNABLE TO DETERMINE (Face Quality Too Low)", False

        # 2. Modality Reliability Ratio
        reliabilities = list(modality_reliabilities.values())
        if not reliabilities:
            return 1.0, "INSUFFICIENT DATA", False

        avg_reliability = float(np.mean(reliabilities))
        if avg_reliability < self.min_reliability_threshold:
            return 0.90, "UNABLE TO DETERMINE (Low Modality Reliability)", False

        # 3. Model Entropy Uncertainty (if probabilities provided)
        if model_probabilities is not None and len(model_probabilities) > 0:
            entropy_u = self.calculate_entropy(model_probabilities)
        else:
            entropy_u = 0.3  # Default nominal entropy

        # 4. Multi-Factor Uncertainty Fusion
        # U = w1*(1 - face_quality) + w2*(1 - avg_reliability) + w3*entropy_u + w4*temporal_variance
        u_face = max(0.0, 1.0 - face_quality)
        u_rel = max(0.0, 1.0 - avg_reliability)
        u_temp = min(1.0, temporal_variance / 25.0)

        total_uncertainty = (0.30 * u_face +
                             0.35 * u_rel +
                             0.25 * entropy_u +
                             0.10 * u_temp)
        total_uncertainty = round(float(np.clip(total_uncertainty, 0.0, 1.0)), 3)

        # 5. Validation Threshold Decision
        if total_uncertainty > self.uncertainty_threshold:
            return total_uncertainty, f"UNABLE TO DETERMINE (Uncertainty: {total_uncertainty*100:.1f}%)", False

        return total_uncertainty, "VALID PREDICTION", True


def validate_evidence_or_fallback(raw_score: float,
                                  raw_status: str,
                                  face_detected: bool,
                                  face_quality: float,
                                  modality_reliabilities: Dict[str, float]) -> Dict[str, Any]:
    """
    Validates prediction evidence and wraps result with explicit uncertainty and fallback handling.
    """
    engine = UncertaintyEstimationEngine()
    u_score, reason, is_valid = engine.estimate_uncertainty(
        face_detected=face_detected,
        face_quality=face_quality,
        modality_reliabilities=modality_reliabilities
    )

    if not is_valid:
        return {
            "score": 0.0,
            "status": reason if not face_detected else "UNABLE TO DETERMINE",
            "reason": reason,
            "is_valid": False,
            "uncertainty": u_score,
            "confidence_pct": round(max(0.0, (1.0 - u_score) * 100.0), 1)
        }

    confidence_pct = round(max(0.0, min(100.0, (1.0 - u_score) * 100.0)), 1)
    return {
        "score": raw_score,
        "status": raw_status,
        "reason": "Sufficient Multimodal Evidence",
        "is_valid": True,
        "uncertainty": u_score,
        "confidence_pct": confidence_pct
    }
