"""
EduSense Multi-Modal Feature Fusion & Mathematics Unit Tests
============================================================
Tests the accuracy, boundary constraints, status classifications,
and score calculations of ESI, ECI, Confusion Solver, and Feature Fusion Engine.
"""

import pytest
from fusion_engine.esi_calculator import ESICalculator
from fusion_engine.eci_calculator import ECICalculator, ConfusionSolver
from fusion_engine.feature_fusion import FeatureFusionEngine
from recommendation.engine import TeacherRecommendationEngine

def test_esi_calculator_excellent():
    calc = ESICalculator()
    res = calc.compute_esi(
        attention_pct=95.0,
        emotion="Happy",
        participation_pct=90.0,
        posture="Upright",
        is_drowsy=False
    )
    assert res["esi_score"] >= 85.0
    assert res["status"] == "Excellent"

def test_esi_calculator_drowsy_penalty():
    calc = ESICalculator()
    res_normal = calc.compute_esi(attention_pct=80.0, emotion="Neutral", participation_pct=70.0, posture="Upright", is_drowsy=False)
    res_drowsy = calc.compute_esi(attention_pct=80.0, emotion="Neutral", participation_pct=70.0, posture="Upright", is_drowsy=True)
    assert res_drowsy["esi_score"] < res_normal["esi_score"]
    assert res_drowsy["components"]["drowsiness_penalty"] > 0

def test_eci_calculator():
    scores = [90.0, 85.0, 80.0, 95.0]
    res = ECICalculator.compute_eci(scores)
    assert res["eci_score"] == 87.5
    assert res["status"] == "Excellent"
    assert res["total_students"] == 4

def test_confusion_solver():
    res = ConfusionSolver.compute_confusion_index(attention_pct=40.0, emotion="Confused", eye_distance_ratio=0.75)
    assert res["confusion_index"] >= 60.0
    assert res["level"] == "High"

def test_teacher_recommendation_engine():
    rec_engine = TeacherRecommendationEngine()
    metrics = {
        "avg_attention": 50.0,
        "avg_confusion": 40.0,
        "avg_participation": 45.0,
        "drowsy_count": 2,
        "students": [{"student_id": "Student 01", "confusion_level": "High", "emotion": "Confused"}]
    }
    recs = rec_engine.evaluate_classroom_state(metrics)
    assert len(recs) >= 3
    types = [r["type"] for r in recs]
    assert "ATTENTION_DROP" in types
    assert "CONFUSION_SPIKE" in types
    assert "DROWSINESS_CLUSTER" in types

def test_feature_fusion_engine_pipeline():
    engine = FeatureFusionEngine()
    raw_students = [
        {"student_id": "Student 01", "attention_pct": 90.0, "emotion": "Happy", "hand_raised": True, "posture": "Upright"},
        {"student_id": "Student 02", "attention_pct": 40.0, "emotion": "Confused", "hand_raised": False, "posture": "Slouching"}
    ]
    res = engine.process_classroom_frame(raw_students)
    assert len(res["students"]) == 2
    assert res["hand_raise_count"] == 1
    assert "eci_score" in res


def test_drowsy_and_attentive_mutual_exclusion():
    """Verify that DROWSY and ATTENTIVE states can NEVER occur simultaneously."""
    from utils.attention_utils import compute_final_attention_state
    state, reason = compute_final_attention_state(score=95.0, is_drowsy_consistent=True)
    assert state == "DROWSY"
    assert "ATTENTIVE" not in state


def test_single_final_attention_state_taxonomy():
    """Verify final state belongs strictly to the allowed 6-state taxonomy."""
    from utils.attention_utils import compute_final_attention_state
    allowed = {"HIGHLY ATTENTIVE", "ATTENTIVE", "MODERATE", "LOW ATTENTION", "DROWSY", "UNABLE TO DETERMINE", "No Face Detected"}
    
    s1, _ = compute_final_attention_state(score=92.0, is_drowsy_consistent=False)
    s2, _ = compute_final_attention_state(score=75.0, is_drowsy_consistent=False)
    s3, _ = compute_final_attention_state(score=58.0, is_drowsy_consistent=False)
    s4, _ = compute_final_attention_state(score=30.0, is_drowsy_consistent=False)
    s5, _ = compute_final_attention_state(score=80.0, is_drowsy_consistent=True)
    s6, _ = compute_final_attention_state(score=80.0, is_valid_evidence=False)
    s7, _ = compute_final_attention_state(score=0.0, face_detected=False)

    assert s1 in allowed
    assert s2 in allowed
    assert s3 in allowed
    assert s4 in allowed
    assert s5 in allowed
    assert s6 in allowed
    assert s7 in allowed

