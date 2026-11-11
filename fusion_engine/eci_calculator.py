"""
EduSense Classroom Index (ECI) & Confusion Solver Modules
===========================================================
Calculates classroom-wide index (ECI) and multi-modal student confusion scores.
"""

from typing import List, Dict

class ECICalculator:
    @staticmethod
    def compute_eci(student_esi_list: List[float]) -> dict:
        """
        Computes EduSense Classroom Index (ECI):
            ECI = (1 / N) * sum(ESI_i)
        """
        if not student_esi_list:
            return {"eci_score": 0.0, "status": "Poor", "total_students": 0}

        avg_eci = sum(student_esi_list) / len(student_esi_list)
        eci_score = round(avg_eci, 1)

        if eci_score >= 85.0:
            status = "Excellent"
        elif eci_score >= 70.0:
            status = "Good"
        elif eci_score >= 55.0:
            status = "Average"
        else:
            status = "Poor"

        return {
            "eci_score": eci_score,
            "status": status,
            "total_students": len(student_esi_list)
        }

class ConfusionSolver:
    @staticmethod
    def compute_confusion_index(attention_pct: float, emotion: str, 
                                eye_distance_ratio: float = 1.0, 
                                hand_raised: bool = False) -> Dict:
        """
        Estimates Confusion Index (0 - 100) based on facial furrowing,
        attention volatility, emotion state, and non-verbal cues.
        """
        base_confusion = 10.0

        if emotion == "Confused":
            base_confusion += 55.0
        elif emotion == "Frustrated":
            base_confusion += 45.0
        elif emotion == "Bored":
            base_confusion += 25.0

        # Eyebrow contraction / furrowing multiplier (MediaPipe landmark ratio < 0.85)
        if eye_distance_ratio < 0.85:
            base_confusion += 20.0

        # Attention drop correlation
        if attention_pct < 50.0:
            base_confusion += 15.0

        # If student raised hand, clarification interest reduces perceived negative confusion
        if hand_raised:
            base_confusion -= 10.0

        confusion_score = round(max(0.0, min(100.0, base_confusion)), 1)

        if confusion_score >= 60.0:
            level = "High"
        elif confusion_score >= 35.0:
            level = "Medium"
        else:
            level = "Low"

        return {
            "confusion_index": confusion_score,
            "level": level
        }
