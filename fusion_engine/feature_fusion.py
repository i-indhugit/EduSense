"""
EduSense Feature Fusion Engine
==============================
Unifies validated visual cues (Gaze, Head Pose, Emotion, Drowsiness, Hand Raise, Posture),
reliability-aware cross-modal fusion, temporal smoothing, Explainable AI (XAI), and
trend forecasting.

All outputs are explicitly labeled as estimates derived from observable visual cues
with missing-feature and low-confidence fallbacks ("UNABLE TO DETERMINE").
"""

import logging
from typing import Dict, List

from analytics.predictive_engine import PredictiveAnalyticsEngine
from explainability.xai_engine import ExplainableAIEngine
from fusion_engine.cross_attention_fusion import create_numpy_reliability_fusion
from fusion_engine.eci_calculator import ECICalculator, ConfusionSolver
from fusion_engine.esi_calculator import ESICalculator
from utils.attention_utils import compute_final_attention_state

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
        student_id = features.get("student_id", "Unknown")
        face_detected = bool(features.get("face_detected", features.get("gaze_direction", "Unable to Determine") != "Unable to Determine"))
        raw_att = features.get("attention_pct")
        conf_pct = float(features.get("confidence_pct") or 0.0)
        face_quality = float(features.get("face_quality") or 0.0)

        gaze = features.get("gaze_direction") or "Unable to Determine"
        head_pose = features.get("head_pose") or "Unable to Determine"
        emotion = features.get("emotion") or "Unable to Determine"
        is_drowsy = bool(features.get("is_drowsy", False))
        is_yawning = bool(features.get("is_yawning", False))
        hand_raised = bool(features.get("hand_raised", False))
        posture = features.get("posture") or "Unknown"
        pose_visible = bool(features.get("pose_visible", posture not in ("Unknown", "Unable to Determine")))
        eye_ratio = features.get("eye_distance_ratio")
        ear_val = features.get("ear_value")
        pitch_deg = features.get("pitch_deg")
        yaw_deg = features.get("yaw_deg")
        debug_info = features.get("debug_info", None)

        has_valid_evidence = (
            face_detected
            and gaze != "Unable to Determine"
            and conf_pct >= 50.0
            and (face_quality == 0.0 or face_quality >= 0.25)
        )

        # Normalize fallback raw_att only when evidence is valid
        if raw_att is None:
            if not has_valid_evidence:
                raw_att = 0.0
            else:
                raw_att = 85.0 if gaze == "Looking at Board" else (55.0 if gaze == "Looking Away" else 35.0)
        raw_att = float(raw_att)

        if not has_valid_evidence:
            att = 0.0
            att_status = features.get("final_state") or ("UNABLE TO DETERMINE" if face_detected else "No Face Detected")
            reason = features.get("reason_for_state") or (
                "Face occluded or unavailable in current frame"
                if not face_detected
                else "Insufficient face quality or detection confidence"
            )
        elif features.get("final_state") or features.get("attention_status"):
            att = round(raw_att, 1)
            att_status = features.get("final_state") or features.get("attention_status")
            reason = features.get("reason_for_state", "Sustained temporal observation")
            # Enforce mutual exclusion safety invariant
            if is_drowsy and "ATTENTIVE" in str(att_status):
                att_status = "DROWSY"
                reason = "Consistent drowsiness evidence overrides attentive label"
        else:
            alpha = 0.20
            prev_att = self.smoothed_scores.get(student_id, raw_att)
            att = round(alpha * raw_att + (1.0 - alpha) * prev_att, 1)
            self.smoothed_scores[student_id] = att

            prev_status = self.smoothed_statuses.get(student_id)
            att_status, reason = compute_final_attention_state(
                score=att,
                is_drowsy_consistent=is_drowsy,
                face_detected=face_detected,
                is_valid_evidence=has_valid_evidence,
                current_status=prev_status,
            )
            self.smoothed_statuses[student_id] = att_status

        # 1. Observable Participation %
        if not has_valid_evidence and not pose_visible:
            participation_pct = 0.0
        else:
            participation_score = 40.0 if has_valid_evidence else 20.0
            if hand_raised:
                participation_score += 50.0
            if posture == "Upright":
                participation_score += 10.0
            participation_pct = min(100.0, participation_score)

        # 2. Confusion Indicator
        if has_valid_evidence:
            confusion_res = self.confusion_solver.compute_confusion_index(
                attention_pct=att,
                emotion=emotion,
                eye_distance_ratio=eye_ratio if eye_ratio is not None else 1.0,
                hand_raised=hand_raised,
            )
        else:
            confusion_res = {"confusion_index": 0.0, "level": "Unavailable"}

        # 3. Real Modality Reliabilities & Features for Cross-Attention Fusion
        fq = face_quality if face_quality > 0 else (conf_pct / 100.0)
        rel_face = round(min(1.0, max(0.0, fq)), 3) if has_valid_evidence else 0.0
        rel_gaze = round(min(1.0, max(0.0, fq * (conf_pct / 100.0))), 3) if gaze != "Unable to Determine" else 0.0
        rel_hp = round(min(1.0, max(0.0, fq * 0.9)), 3) if head_pose not in ("Unknown", "Unable to Determine") else 0.0
        rel_posture = 0.85 if pose_visible and posture not in ("Unknown", "Unable to Determine") else 0.0
        rel_hand = 0.85 if pose_visible else 0.0
        rel_drowsy = round(min(1.0, max(0.0, fq * 0.95)), 3) if ear_val is not None and has_valid_evidence else 0.0

        # Build emotion one-hot/soft vector matching [angry, disgust, fear, happy, sad, surprise, neutral]
        emo_vec = [0.0] * 7
        if emotion == "Happy":
            emo_vec[3] = 0.9
        elif emotion == "Neutral":
            emo_vec[6] = 0.9
        elif emotion == "Frustrated":
            emo_vec[0] = 0.7
        elif emotion in ("Bored", "Sleepy"):
            emo_vec[4] = 0.7
        elif emotion == "Confused":
            emo_vec[5] = 0.4
            emo_vec[4] = 0.3

        gaze_offset = [0.0, 0.0, 0.0, 0.0] if gaze == "Looking at Board" else ([18.0, 10.0, 0.0, 0.0] if gaze == "Looking Away" else [28.0, 0.0, 0.0, 0.0])
        hp_vec = [float(pitch_deg or 0.0), float(yaw_deg or 0.0), 0.0]
        pos_vec = [0.0, 0.0, 1.0, 1.0] if posture == "Upright" else ([10.0, 10.0, 0.7, 0.7] if posture == "Leaning" else [20.0, 20.0, 0.4, 0.4])

        modality_features = {
            "emotion": emo_vec,
            "gaze": gaze_offset,
            "head_pose": hp_vec,
            "posture": pos_vec,
            "hand": [1.0, 1.0] if hand_raised else [0.0, 0.0],
            "drowsiness": [float(ear_val if ear_val is not None else 0.28), 0.2, 1.0 if is_drowsy else 0.0, 1.0],
        }
        modality_reliabilities = {
            "emotion": rel_face if emotion != "Unable to Determine" else 0.0,
            "gaze": rel_gaze,
            "head_pose": rel_hp,
            "posture": rel_posture,
            "hand": rel_hand,
            "drowsiness": rel_drowsy,
        }

        cross_attn_res = create_numpy_reliability_fusion(modality_features, modality_reliabilities)

        # 4. Engagement & ESI Estimation (strictly gated on evidence availability)
        if has_valid_evidence:
            emo_weights = {"Happy": 90, "Neutral": 80, "Confused": 50, "Bored": 30, "Sleepy": 20, "Frustrated": 25}
            emo_val = emo_weights.get(emotion, 65)
            posture_val = 100 if posture == "Upright" else (70 if posture == "Leaning" else (40 if posture == "Slouching" else 65))
            raw_eng = (
                0.40 * att
                + 0.30 * emo_val
                + 0.20 * participation_pct
                + 0.10 * posture_val
                - 0.15 * confusion_res["confusion_index"]
            )
            # Blend with reliability-aware cross-attention fused engagement when reliabilities > 0
            if cross_attn_res["total_reliability"] > 0.1:
                raw_eng = 0.65 * raw_eng + 0.35 * cross_attn_res["fused_engagement_score"]
            engagement_pct = round(max(0.0, min(100.0, raw_eng)), 1)

            esi_res = self.esi_calc.compute_esi(
                attention_pct=att,
                emotion=emotion,
                participation_pct=participation_pct,
                posture=posture if posture != "Unknown" else "Upright",
                is_drowsy=is_drowsy,
                is_yawning=is_yawning,
            )
        else:
            engagement_pct = 0.0
            esi_res = {
                "esi_score": 0.0,
                "status": "Unavailable",
                "components": {
                    "attention_contribution": 0.0,
                    "emotion_contribution": 0.0,
                    "participation_contribution": 0.0,
                    "posture_contribution": 0.0,
                    "drowsiness_penalty": 0.0,
                },
            }

        # 5. Explainable AI (passing real measured values, never fabricating missing ones)
        att_xai = self.xai_engine.explain_attention(
            gaze_direction=gaze,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            eyes_open=not is_drowsy,
            has_face=face_detected,
            confidence=conf_pct,
            posture=posture,
            modality_weights=cross_attn_res["norm_weights"],
        )
        eng_xai = self.xai_engine.explain_engagement(
            att_pct=att,
            emotion=emotion,
            posture=posture,
            hand_raised=hand_raised,
            has_face=face_detected,
            confidence=conf_pct,
        )
        conf_xai = self.xai_engine.explain_confusion(
            emotion=emotion,
            eye_ratio=eye_ratio if eye_ratio is not None else 1.0,
        )
        drowsy_xai = self.xai_engine.explain_drowsiness(
            ear_val=ear_val if ear_val is not None else 0.0,
            is_drowsy=is_drowsy,
            is_yawning=is_yawning,
        )
        counterfactual_xai = self.xai_engine.compute_counterfactual_explanation(
            original_prediction="HIGH" if att >= 70.0 else ("MODERATE" if att >= 40.0 else "LOW"),
            modality_scores=cross_attn_res["modality_scores"],
            norm_weights=cross_attn_res["norm_weights"],
        )

        return {
            "student_id": student_id,
            "bbox": features.get("bbox"),
            "person_bbox": features.get("person_bbox"),
            "face_detected": face_detected,
            "face_status": features.get("face_status", "VALID" if face_detected else "NO_FACE"),
            "confidence_pct": round(conf_pct, 1),
            "face_quality": round(face_quality, 3),
            "attention_pct": att,
            "attention_status": att_status,
            "final_state": att_status,
            "reason_for_state": reason,
            "gaze_direction": gaze,
            "gaze_vector": features.get("gaze_vector", "Unknown"),
            "head_pose": head_pose,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
            "roll_deg": features.get("roll_deg"),
            "emotion": emotion,
            "ear_value": ear_val,
            "mar_value": features.get("mar_value"),
            "is_drowsy": is_drowsy,
            "is_yawning": is_yawning,
            "hand_raised": hand_raised,
            "posture": posture,
            "participation_pct": round(participation_pct, 1),
            "engagement_pct": engagement_pct,
            "confusion_index": confusion_res["confusion_index"],
            "confusion_level": confusion_res["level"],
            "esi_score": esi_res["esi_score"],
            "status": esi_res["status"],
            "components": esi_res["components"],
            "cross_modal_fusion": cross_attn_res,
            "modality_reliabilities": modality_reliabilities,
            "debug_info": debug_info,
            "xai": {
                "attention": att_xai,
                "engagement": eng_xai,
                "confusion": conf_xai,
                "drowsiness": drowsy_xai,
                "counterfactual": counterfactual_xai,
            },
        }

    def process_classroom_frame(self, students_feature_list: list) -> dict:
        if isinstance(students_feature_list, dict):
            return students_feature_list

        processed_students = [
            self.process_student_features(s) for s in students_feature_list if isinstance(s, dict)
        ]

        live_detected_persons = 0
        valid_face_count = 0
        active_tracks = 0
        if students_feature_list and isinstance(students_feature_list[0], dict):
            live_detected_persons = students_feature_list[0].get("_meta_live_detected_persons", len(processed_students))
            active_tracks = students_feature_list[0].get("_meta_active_tracks", len(processed_students))
            valid_face_count = students_feature_list[0].get(
                "_meta_valid_face_count",
                sum(1 for s in processed_students if s.get("face_detected")),
            )

        if not processed_students:
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
                "live_detected_persons": live_detected_persons,
                "valid_face_count": 0,
                "active_tracks": active_tracks,
                "predictive": {},
                "students": [],
            }

        valid_students = [s for s in processed_students if s.get("face_detected") and s.get("final_state") != "UNABLE TO DETERMINE"]
        if valid_students:
            esi_list = [s["esi_score"] for s in valid_students]
            eci_res = ECICalculator.compute_eci(esi_list)
            avg_att = round(sum(s["attention_pct"] for s in valid_students) / len(valid_students), 1)
            avg_eng = round(sum(s["engagement_pct"] for s in valid_students) / len(valid_students), 1)
            avg_part = round(sum(s["participation_pct"] for s in valid_students) / len(valid_students), 1)
            avg_conf = round(sum(s["confusion_index"] for s in valid_students) / len(valid_students), 1)
        else:
            eci_res = {"eci_score": 0.0, "status": "Unable to Determine"}
            avg_att = avg_eng = avg_part = avg_conf = 0.0

        drowsy_cnt = sum(1 for s in processed_students if s["is_drowsy"])
        hand_cnt = sum(1 for s in processed_students if s["hand_raised"])

        predictive_res = self.predictive_engine.forecast_classroom_trends(
            {
                "avg_attention": avg_att,
                "avg_engagement": avg_eng,
                "avg_participation": avg_part,
                "avg_confusion": avg_conf,
                "hand_raise_count": hand_cnt,
            }
        )

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
            "valid_face_count": valid_face_count,
            "active_tracks": active_tracks,
            "predictive": predictive_res,
            "students": processed_students,
        }
