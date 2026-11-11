"""
EduSense Student Index (ESI) Calculator
========================================
Implements the novel mathematical multi-modal fusion formulation for individual student assessment:

    ESI_i = w_a * Attention_i + w_e * Emotion_i + w_p * Participation_i + w_s * Posture_i - lambda_d * Drowsiness_i

Where all sub-scores are normalized to [0, 100].
Output ESI_i is bounded in [0, 100].
"""

class ESICalculator:
    def __init__(self, w_a=0.35, w_e=0.25, w_p=0.20, w_s=0.20, lambda_d=15.0):
        self.w_a = w_a
        self.w_e = w_e
        self.w_p = w_p
        self.w_s = w_s
        self.lambda_d = lambda_d

        # Emotion score mapping [0, 100]
        self.emotion_weights = {
            "Happy": 95.0,
            "Neutral": 80.0,
            "Confused": 55.0,
            "Bored": 40.0,
            "Frustrated": 30.0,
            "Sleepy": 20.0
        }

        # Posture score mapping [0, 100]
        self.posture_weights = {
            "Upright": 100.0,
            "Leaning": 75.0,
            "Slouching": 40.0
        }

    def compute_esi(self, attention_pct: float, emotion: str, participation_pct: float, 
                    posture: str, is_drowsy: bool = False, is_yawning: bool = False) -> dict:
        """
        Computes ESI score and assigns anonymous student status.
        """
        att = max(0.0, min(100.0, attention_pct))
        emo = self.emotion_weights.get(emotion, 70.0)
        part = max(0.0, min(100.0, participation_pct))
        pos = self.posture_weights.get(posture, 75.0)

        drowsiness_penalty = 0.0
        if is_drowsy:
            drowsiness_penalty += self.lambda_d
        if is_yawning:
            drowsiness_penalty += (self.lambda_d * 0.5)

        raw_esi = (self.w_a * att) + (self.w_e * emo) + (self.w_p * part) + (self.w_s * pos) - drowsiness_penalty
        esi_score = round(max(0.0, min(100.0, raw_esi)), 1)

        if esi_score >= 85.0:
            status = "Excellent"
        elif esi_score >= 65.0:
            status = "Good"
        else:
            status = "Needs Attention"

        return {
            "esi_score": esi_score,
            "status": status,
            "components": {
                "attention_contribution": round(self.w_a * att, 2),
                "emotion_contribution": round(self.w_e * emo, 2),
                "participation_contribution": round(self.w_p * part, 2),
                "posture_contribution": round(self.w_s * pos, 2),
                "drowsiness_penalty": round(drowsiness_penalty, 2)
            }
        }
