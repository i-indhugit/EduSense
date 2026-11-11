"""
EduSense Feature Fusion Engine
==============================
Master Feature Fusion Engine unifying computer vision inputs (Gaze, Head Pose, Emotion, Drowsiness, Hand Raise, Posture),
temporal EMA smoothing, human-readable attention status classification, Learnable EduSense Intelligence Engine (EIE),
Explainable AI (XAI) Engine, and Predictive Analytics.
"""

from fusion_engine.esi_calculator import ESICalculator
from fusion_engine.eci_calculator import ECICalculator, ConfusionSolver
from fusion_engine.cross_attention_fusion import create_numpy_reliability_fusion
from explainability.xai_engine import ExplainableAIEngine
from analytics.predictive_engine import PredictiveAnalyticsEngine
from fusion_engine.esi_calculator import ESICalculator
from fusion_engine.eci_calculator import ECICalculator, ConfusionSolver
from fusion_engine.cross_attention_fusion import create_numpy_reliability_fusion
from explainability.xai_engine import ExplainableAIEngine
from analytics.predictive_engine import PredictiveAnalyticsEngine
from utils.attention_utils import compute_attention_status, compute_final_attention_state
from typing import Dict, List

import logging

logger = logging.getLogger("EduSenseFusion")

class FeatureFusionEngine:
    def __init__(self):
        self.esi_calc = ESICalculator()
        self.confusion_solver = ConfusionSolver()
        self.xai_engine = ExplainableAIEngine()
        self.predictive_engine = PredictiveAnalyticsEngine()
        self.smoothed_scores: Dict[str, float] = {}
        self.smoothed_statuses: Dict[str, str] = {}

    def process_student_features(self, features: dict) -> dict:
        student_id = features.get("student_id", "Student 01")
        raw_att = features.get("attention_pct", 85.0)
        conf_pct = features.get("confidence_pct", 96.5)
        gaze = features.get("gaze_direction", "Looking at Board")
        head_pose = features.get("head_pose", "Upright (0°)")
        emotion = features.get("emotion", "Neutral")
        is_drowsy = features.get("is_drowsy", False)
        is_yawning = features.get("is_yawning", False)
        hand_raised = features.get("hand_raised", False)
        posture = features.get("posture", "Upright")
        eye_ratio = features.get("eye_distance_ratio", 1.0)
        ear_val = features.get("ear_value", 0.28)
        debug_info = features.get("debug_info", None)

        # Log actual observed features for this student ID
        logger.info(
            f"[{student_id}] Observations -> Gaze: '{gaze}', HeadPose: '{head_pose}', "
            f"Posture: '{posture}', Drowsy: {is_drowsy}, Hand: {hand_raised} | "
            f"RawAtt: {raw_att:.1f}%, Conf: {conf_pct:.1f}%"
        )

        # Handle unavailable face or low confidence
        if gaze == "Unable to Determine" or conf_pct < 50.0:
            att = 0.0
            att_status = "UNABLE TO DETERMINE"
            reason = "Insufficient face quality or confidence threshold"
        elif features.get("final_state") or features.get("attention_status"):
            att = raw_att
            att_status = features.get("final_state") or features.get("attention_status")
            reason = features.get("reason_for_state", "Sustained temporal observation")
        else:
            # Apply Independent Temporal Exponential Moving Average (EMA) score smoothing per student_id
            alpha = 0.20
            prev_att = self.smoothed_scores.get(student_id, raw_att)
            att = round(alpha * raw_att + (1.0 - alpha) * prev_att, 1)
            self.smoothed_scores[student_id] = att

            # Compute Human-Readable Attention Status with Hysteresis
            prev_status = self.smoothed_statuses.get(student_id, None)
            att_status, reason = compute_final_attention_state(
                score=att,
                is_drowsy_consistent=is_drowsy,
                face_detected=True,
                is_valid_evidence=True,
                current_status=prev_status
            )
            self.smoothed_statuses[student_id] = att_status

        # 1. Participation % computation
        participation_score = 40.0
        if hand_raised:
            participation_score += 50.0
        if posture == "Upright":
            participation_score += 10.0
        participation_pct = min(100.0, participation_score)

        # 2. Confusion evaluation
        confusion_res = self.confusion_solver.compute_confusion_index(
            attention_pct=att,
            emotion=emotion,
            eye_distance_ratio=eye_ratio,
            hand_raised=hand_raised
        )

        # 3. Engagement Score computation
        emo_weights = {"Happy": 90, "Neutral": 80, "Confused": 50, "Bored": 30, "Sleepy": 20, "Frustrated": 25}
        emo_val = emo_weights.get(emotion, 70)
        posture_val = 100 if posture == "Upright" else (70 if posture == "Leaning" else 40)
        
        raw_eng = (0.40 * att) + (0.30 * emo_val) + (0.20 * participation_pct) + (0.10 * posture_val) - (0.15 * confusion_res["confusion_index"])
        engagement_pct = round(max(0.0, min(100.0, raw_eng)), 1)

        # 4. EduSense Student Index (ESI) computation
        esi_res = self.esi_calc.compute_esi(
            attention_pct=att,
            emotion=emotion,
            participation_pct=participation_pct,
            posture=posture,
            is_drowsy=is_drowsy,
            is_yawning=is_yawning
        )

        # 5. Reliability-Aware Cross-Attention Fusion
        modality_features = {
            "emotion": [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9] if emotion == "Neutral" else [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.2],
            "gaze": [0.0, 0.0, 0.0, 0.0] if gaze == "Looking at Board" else [5.0, 5.0, 0.0, 0.0],
            "head_pose": [0.0, 0.0, 0.0] if posture == "Upright" else [15.0, 0.0, 0.0],
            "posture": [0.0, 0.0, 1.0, 1.0] if posture == "Upright" else [15.0, 15.0, 0.7, 0.6],
            "hand": [1.0, 1.0] if hand_raised else [0.0, 0.0],
            "drowsiness": [ear_val, 0.2, 0.0, 1.0]
        }
        modality_reliabilities = {
            "emotion": 0.90 if emotion != "Unable to Determine" else 0.0,
            "gaze": 0.95 if gaze != "Unable to Determine" else 0.0,
            "head_pose": 0.85 if posture != "Unable to Determine" else 0.0,
            "posture": 0.80 if posture != "Unable to Determine" else 0.0,
            "hand": 0.90,
            "drowsiness": 0.95
        }

        cross_attn_res = create_numpy_reliability_fusion(modality_features, modality_reliabilities)

        # 6. Explainable AI (XAI) & Counterfactual Rationale Generation
        att_xai = self.xai_engine.explain_attention(gaze_direction=gaze, eyes_open=not is_drowsy)
        eng_xai = self.xai_engine.explain_engagement(att_pct=att, emotion=emotion, posture=posture, hand_raised=hand_raised)
        conf_xai = self.xai_engine.explain_confusion(emotion=emotion, eye_ratio=eye_ratio)
        drowsy_xai = self.xai_engine.explain_drowsiness(ear_val=ear_val, is_drowsy=is_drowsy, is_yawning=is_yawning)
        counterfactual_xai = self.xai_engine.compute_counterfactual_explanation(
            original_prediction="HIGH" if att >= 70.0 else ("MODERATE" if att >= 40.0 else "LOW"),
            modality_scores=cross_attn_res["modality_scores"],
            norm_weights=cross_attn_res["norm_weights"]
        )

        return {
            "student_id": student_id,
            "confidence_pct": conf_pct,
            "attention_pct": att,
            "attention_status": att_status,
            "final_state": att_status,
            "reason_for_state": reason,
            "gaze_direction": gaze,
            "head_pose": head_pose,
            "emotion": emotion,
            "is_drowsy": is_drowsy,
            "is_yawning": is_yawning,
            "hand_raised": hand_raised,
            "posture": posture,
            "participation_pct": participation_pct,
            "engagement_pct": engagement_pct,
            "confusion_index": confusion_res["confusion_index"],
            "confusion_level": confusion_res["level"],
            "esi_score": esi_res["esi_score"],
            "status": esi_res["status"],
            "components": esi_res["components"],
            "cross_modal_fusion": cross_attn_res,
            "debug_info": debug_info,
            "xai": {
                "attention": att_xai,
                "engagement": eng_xai,
                "confusion": conf_xai,
                "drowsiness": drowsy_xai,
                "counterfactual": counterfactual_xai
            }
        }

    def process_classroom_frame(self, students_feature_list: list) -> dict:
        if isinstance(students_feature_list, dict):
            return students_feature_list

        processed_students = [self.process_student_features(s) for s in students_feature_list if isinstance(s, dict)]

        live_detected_persons = 0
        active_tracks = 0
        if students_feature_list and isinstance(students_feature_list[0], dict):
            live_detected_persons = students_feature_list[0].get("_meta_live_detected_persons", len(processed_students))
            active_tracks = students_feature_list[0].get("_meta_active_tracks", len(processed_students))

        if not processed_students:
            # Clear previous smoothing buffers when detection lost
            self.smoothed_scores.clear()
            self.smoothed_statuses.clear()

            return {
                "eci_score": 0.0,
                "eci_status": "No Face Detected",
                "avg_attention": 0.0,
                "avg_engagement": 0.0,
                "avg_participation": 0.0,
                "avg_confusion": 0.0,
                "drowsy_count": 0,
                "hand_raise_count": 0,
                "live_detected_persons": 0,
                "active_tracks": 0,
                "predictive": {},
                "students": []
            }

        esi_list = [s["esi_score"] for s in processed_students]
        eci_res = ECICalculator.compute_eci(esi_list)

        avg_att = round(sum(s["attention_pct"] for s in processed_students) / len(processed_students), 1)
        avg_eng = round(sum(s["engagement_pct"] for s in processed_students) / len(processed_students), 1)
        avg_part = round(sum(s["participation_pct"] for s in processed_students) / len(processed_students), 1)
        avg_conf = round(sum(s["confusion_index"] for s in processed_students) / len(processed_students), 1)

        drowsy_cnt = sum(1 for s in processed_students if s["is_drowsy"])
        hand_cnt = sum(1 for s in processed_students if s["hand_raised"])

        current_summary = {
            "avg_attention": avg_att,
            "avg_engagement": avg_eng,
            "avg_participation": avg_part,
            "avg_confusion": avg_conf,
            "hand_raise_count": hand_cnt
        }

        # Run Predictive Analytics Engine Forecast
        predictive_res = self.predictive_engine.forecast_classroom_trends(current_summary)

        return {
            "eci_score": eci_res["eci_score"],
            "eci_status": eci_res["status"],
            "avg_attention": avg_att,
            "avg_engagement": avg_eng,
            "avg_participation": avg_part,
            "avg_confusion": avg_conf,
            "drowsy_count": drowsy_cnt,
            "hand_raise_count": hand_cnt,
            "live_detected_persons": live_detected_persons,
            "active_tracks": active_tracks,
            "predictive": predictive_res,
            "students": processed_students
        }
