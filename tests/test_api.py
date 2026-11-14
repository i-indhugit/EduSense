"""
Tests for EduSense Lightweight Metadata & Aggregated Analytics Backend
"""

import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.main import app

client = TestClient(app)


def test_read_root():
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "Online"
    assert data["inference_location"] == "browser"


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["inference_location"] == "browser"


def test_system_status():
    resp = client.get("/system/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["system"] == "EduSense"
    assert "memory_mb" in data
    assert "cpu_percent" in data


def test_create_session_and_ingest_analytics():
    sess_resp = client.post("/api/sessions", json={"title": "CS101 Lecture", "course_code": "CS101"})
    assert sess_resp.status_code == 200
    session_id = sess_resp.json()["session_id"]
    assert session_id.startswith("session-")

    payload = {
        "session_id": session_id,
        "eci_score": 81.5,
        "eci_status": "High Engagement Behavior",
        "avg_attention": 84.0,
        "avg_engagement": 80.5,
        "avg_participation": 75.0,
        "avg_confusion": 14.0,
        "drowsy_count": 0,
        "hand_raise_count": 1,
        "live_detected_persons": 2,
        "valid_face_count": 2,
        "active_tracks": 2,
        "fps": 10.0,
        "latency_ms": 28.4,
    }
    ingest = client.post("/api/analytics", json=payload)
    assert ingest.status_code == 200
    assert ingest.json()["accepted"] is True

    latest = client.get("/api/analytics/latest").json()
    assert latest["latest"]["eci_score"] == 81.5
    assert latest["latest"]["valid_face_count"] == 2


def test_classroom_and_dashboard_read_endpoints():
    c_resp = client.get("/classroom")
    assert c_resp.status_code == 200
    assert "eci_score" in c_resp.json()

    d_resp = client.get("/dashboard")
    assert d_resp.status_code == 200
    assert d_resp.json()["inference_location"] == "browser"


def test_gemini_report_endpoint_and_fallback():
    # Test fallback & schema validation when GEMINI_API_KEY is unset/mocked
    payload = {
        "total_students": 20,
        "valid_observations": 16,
        "looking_toward_front": 12,
        "looking_away": 4,
        "hand_raised": 2,
        "insufficient_evidence": 4,
        "drowsy_count": 1,
        "phone_distracted": 0,
        "avg_attention_score": 78.5,
        "avg_confusion_index": 22.0,
        "user_query": "How is student participation today?",
    }
    resp = client.post("/api/gemini/report", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "headline" in data
    assert "observations_summary" in data
    assert isinstance(data["pedagogical_recommendations"], list)
    assert len(data["pedagogical_recommendations"]) > 0
    assert "limitations_and_uncertainty" in data
    assert "visual cues" in data["limitations_and_uncertainty"].lower()

    # Test latest report getter
    latest_rep = client.get("/api/gemini/latest-report")
    assert latest_rep.status_code == 200
    assert "observations_summary" in latest_rep.json()


def test_gemini_anti_hallucination_guardrails():
    from backend.gemini_service import (
        AggregateClassroomInput,
        extract_valid_student_ids,
        generate_fallback_report,
        generate_gemini_report,
        sanitize_anti_hallucination,
    )

    # 1. Zero students guardrail
    zero_input = AggregateClassroomInput(total_students=0)
    zero_rep = generate_gemini_report(zero_input)
    assert zero_rep.headline == "No Students Detected"
    assert "zero students" in zero_rep.observations_summary.lower()

    # 2. Extract valid student IDs
    summaries = [
        "Student 01: state=ATTENTIVE (att=85%)",
        "Student 02: state=MODERATE (att=55%)",
    ]
    valid_ids = extract_valid_student_ids(summaries)
    assert valid_ids == {"Student 01", "Student 02"}

    # 3. Sanitize hallucinated Student IDs (e.g. Student 09 not in telemetry)
    hallucinated_text = "Check in with Student 09 and acknowledge Student 01."
    sanitized = sanitize_anti_hallucination(hallucinated_text, valid_ids, drowsy_count=0, phone_count=0)
    assert "Student 09" not in sanitized
    assert "Student 01" in sanitized


def test_security_and_privacy_hardening():
    from backend.gemini_service import _validate_api_key, _prune_cache, _cache
    from backend.main import _sessions

    # 1. API key format validation (reject command injection / malformed keys)
    assert _validate_api_key("AIzaSyValidKey1234567890_-") is not None
    assert _validate_api_key("; rm -rf /; evil_key") is None
    assert _validate_api_key("<script>alert(1)</script>") is None

    # 2. Session storage bounded anti-DoS
    for i in range(60):
        client.post("/api/sessions", json={"title": f"Session {i}"})
    assert len(_sessions) <= 50

    # 3. Cache pruning bounded
    import time
    now = time.time()
    for i in range(95):
        _cache[f"key_{i}"] = (now, {"sample": i})
    assert len(_cache) >= 95
    _prune_cache(now)
    assert len(_cache) <= 80


