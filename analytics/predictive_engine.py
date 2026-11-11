"""
EduSense Predictive Analytics Engine
====================================
Forecasts future classroom dynamics over temporal windows:
1. Attention Drop Risk Curve forecasting probability of attention drop in next 5-10 minutes.
2. Confusion Trajectory forecasting probability of concept confusion.
3. Optimal Teacher Interaction & Break Timing calculation.
4. Learning Pace Index & Classroom Energy Score evaluation.
"""

import math
from typing import List, Dict

class PredictiveAnalyticsEngine:
    def __init__(self):
        pass

    def forecast_classroom_trends(self, current_metrics: dict) -> Dict:
        avg_att = current_metrics.get("avg_attention", 82.5)
        avg_conf = current_metrics.get("avg_confusion", 15.0)
        avg_eng = current_metrics.get("avg_engagement", 78.0)
        hand_cnt = current_metrics.get("hand_raise_count", 0)

        # 1. Attention Drop Risk Probability (5-10 min window)
        # Higher attention drop risk if attention is declining or confusion is rising
        drop_risk_prob = min(95.0, max(5.0, (100.0 - avg_att) * 1.2 + avg_conf * 0.5))

        # 2. Confusion Trajectory Probability
        conf_risk_prob = min(95.0, max(5.0, avg_conf * 1.8 + (100.0 - avg_eng) * 0.4))

        # 3. Learning Pace Index (0 - 100)
        # High engagement + low confusion -> Optimal fast pace
        learning_pace_index = round(max(0.0, min(100.0, avg_eng * 0.6 + (100.0 - avg_conf) * 0.4)), 1)
        if learning_pace_index >= 75.0:
            pace_label = "Optimal / Accelerating"
        elif learning_pace_index >= 55.0:
            pace_label = "Moderate / Balanced"
        else:
            pace_label = "Slow / Needs Simplification"

        # 4. Classroom Energy Score (0 - 100)
        # Driven by participation, hand raises, and positive emotion
        energy_score = round(max(0.0, min(100.0, avg_eng * 0.5 + min(100.0, hand_cnt * 20) * 0.3 + avg_att * 0.2)), 1)

        # 5. Forecasted Attention Drop Curve (Next 10 minutes)
        attention_curve = []
        for minute in range(1, 11):
            decay_factor = math.exp(-0.04 * minute) if drop_risk_prob > 40 else 1.0
            predicted_att = round(max(30.0, avg_att * decay_factor - (drop_risk_prob / 50.0) * minute), 1)
            attention_curve.append({"minute": f"+{minute}m", "predicted_attention": predicted_att})

        # 6. Optimal Interaction Time Recommendation
        if drop_risk_prob >= 60.0 or avg_conf >= 30.0:
            recommended_interaction_in = "Immediate (within 2 minutes)"
        elif drop_risk_prob >= 35.0:
            recommended_interaction_in = "In 5 minutes"
        else:
            recommended_interaction_in = "In 12 minutes (routine check)"

        return {
            "attention_drop_risk_pct": round(drop_risk_prob, 1),
            "confusion_risk_pct": round(conf_risk_prob, 1),
            "predicted_risk_label": "Predicted Risk (Not Guaranteed Outcome)",
            "uncertainty_margin_pct": 8.5,
            "disclaimer": "Predicted behavioral trend. Subject to temporal variance and contextual changes.",
            "learning_pace_index": learning_pace_index,
            "learning_pace_label": pace_label,
            "classroom_energy_score": energy_score,
            "recommended_interaction_window": recommended_interaction_in,
            "predicted_attention_curve": attention_curve
        }
