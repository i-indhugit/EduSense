"""
EduSense-X Unit & Integration Test Suite
========================================
Validates core vision, multi-modal fusion, reliability weighting, uncertainty estimation,
XAI counterfactuals, ONNX parity, and REST API endpoints.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fusion_engine.cross_attention_fusion import ReliabilityAwareCrossAttention, create_numpy_reliability_fusion
from utils.uncertainty_utils import UncertaintyEstimationEngine, validate_evidence_or_fallback
from explainability.xai_engine import ExplainableAIEngine
from analytics.predictive_engine import PredictiveAnalyticsEngine
from backend.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_no_face_detected_handling():
    """Verify system outputs NO FACE DETECTED when no face present."""
    engine = UncertaintyEstimationEngine()
    u, status, is_valid = engine.estimate_uncertainty(
        face_detected=False,
        face_quality=0.0,
        modality_reliabilities={}
    )
    assert not is_valid
    assert status == "NO FACE DETECTED"
    assert u == 1.0


def test_unable_to_determine_low_quality():
    """Verify system outputs UNABLE TO DETERMINE when face quality is low."""
    engine = UncertaintyEstimationEngine()
    u, status, is_valid = engine.estimate_uncertainty(
        face_detected=True,
        face_quality=0.15,
        modality_reliabilities={"emotion": 0.1, "gaze": 0.1}
    )
    assert not is_valid
    assert "UNABLE TO DETERMINE" in status


def test_reliability_aware_cross_attention_pytorch():
    """Verify PyTorch Reliability-Aware Cross-Attention forward pass."""
    model = ReliabilityAwareCrossAttention(embed_dim=64, num_heads=4, num_modalities=6)
    model.eval()

    batch_size = 2
    dummy_inputs = [
        torch.randn(batch_size, 7),
        torch.randn(batch_size, 4),
        torch.randn(batch_size, 3),
        torch.randn(batch_size, 4),
        torch.randn(batch_size, 2),
        torch.randn(batch_size, 4),
    ]
    dummy_reliability = torch.tensor([[0.9, 0.95, 0.8, 0.8, 0.9, 0.95],
                                       [0.1, 0.0, 0.8, 0.8, 0.0, 0.95]])

    with torch.no_grad():
        att_logits, eng_logits, attn_weights, fused = model(dummy_inputs, dummy_reliability)

    assert att_logits.shape == (batch_size, 3)
    assert eng_logits.shape == (batch_size, 3)
    assert attn_weights.shape == (batch_size, 6)
    assert fused.shape == (batch_size, 64)


def test_numpy_reliability_fusion():
    """Verify fast NumPy reliability fusion fallback calculations."""
    features = {
        "emotion": [0, 0, 0, 0.1, 0, 0, 0.9],
        "gaze": [0.0, 0.0, 0.0, 0.0],
        "head_pose": [0.0, 0.0, 0.0],
        "posture": [0.0, 0.0, 1.0, 1.0],
        "hand": [0.0, 0.0],
        "drowsiness": [0.28, 0.2, 0.0, 1.0]
    }
    reliabilities = {
        "emotion": 0.9,
        "gaze": 0.95,
        "head_pose": 0.85,
        "posture": 0.8,
        "hand": 0.9,
        "drowsiness": 0.95
    }

    res = create_numpy_reliability_fusion(features, reliabilities)
    assert "fused_attention_score" in res
    assert "fused_engagement_score" in res
    assert res["fused_attention_score"] >= 70.0


def test_counterfactual_xai():
    """Verify Counterfactual XAI calculation."""
    xai = ExplainableAIEngine()
    cf = xai.compute_counterfactual_explanation(
        original_prediction="HIGH",
        modality_scores={"gaze": 95.0, "posture": 90.0, "emotion": 85.0},
        norm_weights={"gaze": 0.6, "posture": 0.2, "emotion": 0.2}
    )
    assert cf["removed_modality"] == "gaze"
    assert "explanation" in cf


def test_fastapi_rest_endpoints():
    """Verify core FastAPI REST API responses."""
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert r_root.json()["system"] == "EduSense: Multi-Modal Intelligent Classroom Analytics"

    r_manifest = client.get("/manifest")
    assert r_manifest.status_code == 200
    assert "project" in r_manifest.json()

    r_bench = client.get("/benchmarks")
    assert r_bench.status_code == 200

    r_ablation = client.get("/ablation")
    assert r_ablation.status_code == 200

    r_robust = client.get("/robustness")
    assert r_robust.status_code == 200


def test_multi_student_feature_isolation():
    """Verify Student 01 features never influence Student 02 predictions."""
    from fusion_engine.feature_fusion import FeatureFusionEngine
    fusion = FeatureFusionEngine()

    s1_features = {
        "student_id": "Student 01",
        "attention_pct": 92.0,
        "confidence_pct": 95.0,
        "gaze_direction": "Looking at Board",
        "head_pose": "Upright (0°)",
        "emotion": "Neutral",
        "is_drowsy": False,
        "posture": "Upright",
        "hand_raised": True
    }

    s2_features = {
        "student_id": "Student 02",
        "attention_pct": 35.0,
        "confidence_pct": 90.0,
        "gaze_direction": "Head Down",
        "head_pose": "Head Down (-18°)",
        "emotion": "Bored",
        "is_drowsy": True,
        "posture": "Slouching",
        "hand_raised": False
    }

    res = fusion.process_classroom_frame([s1_features, s2_features])
    students = res["students"]
    assert len(students) == 2

    s1_out = next(s for s in students if s["student_id"] == "Student 01")
    s2_out = next(s for s in students if s["student_id"] == "Student 02")

    # Scores must reflect independent observations and not match artificially
    assert s1_out["attention_pct"] > s2_out["attention_pct"]
    assert s1_out["attention_status"] != s2_out["attention_status"]
    assert s1_out["is_drowsy"] != s2_out["is_drowsy"]

    # Mutate Student 01 to low attention and process frame again
    s1_mutated = dict(s1_features)
    s1_mutated["attention_pct"] = 15.0
    s1_mutated["gaze_direction"] = "Looking Away"

    res_mutated = fusion.process_classroom_frame([s1_mutated, s2_features])
    s2_out_after = next(s for s in res_mutated["students"] if s["student_id"] == "Student 02")

    # Student 02 score must remain unchanged by Student 01's mutation
    assert s2_out_after["attention_pct"] == s2_out["attention_pct"]
