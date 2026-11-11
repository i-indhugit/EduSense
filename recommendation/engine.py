"""
EduSense Teacher Recommendation Engine
=======================================
Rule-based heuristic analytics engine that monitors temporal classroom trends
and generates actionable pedagogical recommendations for teachers in real time.
"""

from typing import List, Dict
from datetime import datetime

class TeacherRecommendationEngine:
    def __init__(self):
        self.history_window: List[dict] = []

    def evaluate_classroom_state(self, classroom_metrics: dict) -> List[Dict]:
        """
        Evaluates current frame metrics & trend history to produce recommendations.
        """
        recommendations = []
        now_str = datetime.now().strftime("%H:%M:%S")

        avg_attention = classroom_metrics.get("avg_attention", 80.0)
        avg_confusion = classroom_metrics.get("avg_confusion", 15.0)
        avg_participation = classroom_metrics.get("avg_participation", 70.0)
        drowsy_count = classroom_metrics.get("drowsy_count", 0)
        students = classroom_metrics.get("students", [])

        # 1. Attention Drop Alert
        if avg_attention < 65.0:
            recommendations.append({
                "id": "rec_att_drop",
                "timestamp": now_str,
                "type": "ATTENTION_DROP",
                "severity": "HIGH" if avg_attention < 50.0 else "MEDIUM",
                "title": "Attention Drop Detected",
                "message": f"Average classroom attention has dropped to {avg_attention}%.",
                "action": "Recommend pausing the lecture to ask a quick check question or change speaking pace."
            })

        # 2. Confusion Spike Alert
        confused_students = [s["student_id"] for s in students if s.get("confusion_level") == "High" or s.get("emotion") in ["Confused", "Frustrated"]]
        if len(confused_students) >= 2 or avg_confusion >= 35.0:
            student_str = ", ".join(confused_students[:3]) if confused_students else "Multiple students"
            recommendations.append({
                "id": "rec_conf_spike",
                "timestamp": now_str,
                "type": "CONFUSION_SPIKE",
                "severity": "HIGH",
                "title": "Concept Confusion Alert",
                "message": f"{len(confused_students)} students ({student_str}) appear confused.",
                "action": "Recommend repeating or clarifying the current topic with a visual example."
            })

        # 3. Low Interaction / Participation Alert
        if avg_participation < 55.0:
            recommendations.append({
                "id": "rec_low_part",
                "timestamp": now_str,
                "type": "LOW_PARTICIPATION",
                "severity": "MEDIUM",
                "title": "Low Classroom Interaction",
                "message": f"Participation score is currently at {avg_participation}%.",
                "action": "Recommend asking an interactive poll or calling on students for quick feedback."
            })

        # 4. Drowsiness / Fatigue Alert
        if drowsy_count >= 1:
            recommendations.append({
                "id": "rec_drowsy",
                "timestamp": now_str,
                "type": "DROWSINESS_CLUSTER",
                "severity": "HIGH" if drowsy_count >= 2 else "MEDIUM",
                "title": "Student Fatigue / Drowsiness",
                "message": f"Detected eye closure/yawning in {drowsy_count} student(s).",
                "action": "Recommend a 2-minute stretch break or initiating a classroom discussion."
            })

        # Default positive feedback if everything is performing well
        if not recommendations:
            recommendations.append({
                "id": "rec_optimal",
                "timestamp": now_str,
                "type": "OPTIMAL_ENGAGEMENT",
                "severity": "INFO",
                "title": "Optimal Classroom Engagement",
                "message": f"Classroom ECI index is strong ({classroom_metrics.get('eci_score', 85.0)}). High attention maintained.",
                "action": "Continue current instructional pace."
            })

        return recommendations
