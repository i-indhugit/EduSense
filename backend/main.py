"""
EduSense Lightweight Metadata & Aggregated Analytics Backend
=============================================================
Designed for Render Free Web Service (<60 MB RAM, zero GPU, zero local storage).
All computer vision inference (Person, Face, Landmarks, Pose, Tracking) runs
entirely in the user's browser via MediaPipe Tasks.

This service handles ONLY:
- Health checks (/health)
- Session metadata (/api/sessions)
- Aggregated, anonymized classroom telemetry (/api/analytics)
- Never receives raw video frames, face images, or biometric templates.
"""

import logging
import os
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.gemini_service import (
    AggregateClassroomInput,
    GeminiReportResponse,
    generate_gemini_report,
)

logging.basicConfig(
    level=os.getenv("EDUSENSE_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("EduSenseAPI")

CORS_ORIGINS_RAW = os.getenv("EDUSENSE_CORS_ORIGINS", "*")
CORS_ORIGINS = (
    [o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()]
    if CORS_ORIGINS_RAW != "*"
    else ["*"]
)


def _get_process_memory_mb() -> float:
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return 0.0


def _get_process_cpu_percent() -> float:
    try:
        import psutil
        return round(psutil.Process(os.getpid()).cpu_percent(interval=None), 1)
    except Exception:
        return 0.0


class SessionCreateRequest(BaseModel):
    title: str = Field(default="Live Classroom Session", max_length=120)
    course_code: Optional[str] = Field(default=None, max_length=40)


class AggregatedAnalyticsPayload(BaseModel):
    session_id: str = Field(..., max_length=80)
    timestamp: Optional[str] = None
    eci_score: float = Field(default=0.0, ge=0.0, le=100.0)
    eci_status: str = Field(default="No Face Detected", max_length=80)
    avg_attention: float = Field(default=0.0, ge=0.0, le=100.0)
    avg_engagement: float = Field(default=0.0, ge=0.0, le=100.0)
    avg_participation: float = Field(default=0.0, ge=0.0, le=100.0)
    avg_confusion: float = Field(default=0.0, ge=0.0, le=100.0)
    drowsy_count: int = Field(default=0, ge=0)
    hand_raise_count: int = Field(default=0, ge=0)
    live_detected_persons: int = Field(default=0, ge=0)
    valid_face_count: int = Field(default=0, ge=0)
    active_tracks: int = Field(default=0, ge=0)
    fps: float = Field(default=0.0, ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)


# Bounded in-memory telemetry store (ephemeral, zero biometric/image storage)
_sessions: Dict[str, dict] = {}
_analytics_history: Deque[dict] = deque(maxlen=120)
_latest_analytics: dict = {
    "session_id": None,
    "timestamp": None,
    "eci_score": 0.0,
    "eci_status": "No Face Detected",
    "avg_attention": 0.0,
    "avg_engagement": 0.0,
    "avg_participation": 0.0,
    "avg_confusion": 0.0,
    "drowsy_count": 0,
    "hand_raise_count": 0,
    "live_detected_persons": 0,
    "valid_face_count": 0,
    "active_tracks": 0,
    "fps": 0.0,
    "latency_ms": 0.0,
}

app = FastAPI(
    title="EduSense Metadata & Analytics API",
    description="Lightweight Render-ready backend for browser-side AI classroom analytics",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "system": "EduSense Browser-AI Metadata Backend",
        "status": "Online",
        "version": "3.0.0",
        "inference_location": "browser",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
def health_check():
    """Render Web Service health check endpoint."""
    return {
        "status": "healthy",
        "inference_location": "browser",
        "memory_mb": _get_process_memory_mb(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/system/status")
def get_system_status():
    return {
        "system": "EduSense",
        "inference_location": "browser",
        "active_sessions": len(_sessions),
        "history_points": len(_analytics_history),
        "memory_mb": _get_process_memory_mb(),
        "cpu_percent": _get_process_cpu_percent(),
        "latest_eci_score": _latest_analytics.get("eci_score", 0.0),
        "live_detected_persons": _latest_analytics.get("live_detected_persons", 0),
        "valid_face_count": _latest_analytics.get("valid_face_count", 0),
        "active_tracks": _latest_analytics.get("active_tracks", 0),
    }


@app.post("/api/sessions")
def create_session(payload: SessionCreateRequest):
    global _sessions
    if len(_sessions) >= 50:
        oldest_key = next(iter(_sessions))
        _sessions.pop(oldest_key, None)

    session_id = f"session-{uuid.uuid4().hex[:10]}"
    record = {
        "session_id": session_id,
        "title": payload.title,
        "course_code": payload.course_code,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _sessions[session_id] = record
    return record


@app.post("/api/analytics")
def ingest_aggregated_analytics(payload: AggregatedAnalyticsPayload):
    """Ingest aggregated, anonymized classroom metrics computed in the browser."""
    global _latest_analytics
    entry = payload.model_dump()
    if not entry.get("timestamp"):
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    _latest_analytics = entry
    _analytics_history.append(entry)
    return {"accepted": True, "history_count": len(_analytics_history)}


@app.get("/api/analytics/latest")
def get_latest_aggregated_analytics():
    return {
        "latest": _latest_analytics,
        "history": list(_analytics_history),
    }


@app.get("/classroom")
def get_classroom_metrics():
    return {
        "eci_score": _latest_analytics.get("eci_score", 0.0),
        "eci_status": _latest_analytics.get("eci_status", "No Face Detected"),
        "avg_attention": _latest_analytics.get("avg_attention", 0.0),
        "avg_engagement": _latest_analytics.get("avg_engagement", 0.0),
        "avg_participation": _latest_analytics.get("avg_participation", 0.0),
        "avg_confusion": _latest_analytics.get("avg_confusion", 0.0),
        "drowsy_count": _latest_analytics.get("drowsy_count", 0),
        "hand_raise_count": _latest_analytics.get("hand_raise_count", 0),
        "live_detected_persons": _latest_analytics.get("live_detected_persons", 0),
        "valid_face_count": _latest_analytics.get("valid_face_count", 0),
        "active_tracks": _latest_analytics.get("active_tracks", 0),
    }


@app.get("/analytics")
def get_analytics_history():
    return {
        "emotion_distribution": {},
        "heatmap": [],
        "live_detected_persons": _latest_analytics.get("live_detected_persons", 0),
        "valid_face_count": _latest_analytics.get("valid_face_count", 0),
        "active_tracks": _latest_analytics.get("active_tracks", 0),
        "timeline": list(_analytics_history),
    }


@app.get("/dashboard")
def get_dashboard_summary():
    return {
        "classroom": get_classroom_metrics(),
        "students": [],
        "recommendations": [],
        "inference_location": "browser",
    }


@app.post("/api/gemini/report", response_model=GeminiReportResponse)
def get_gemini_classroom_report(payload: AggregateClassroomInput):
    """
    Generate an explainable pedagogical report and activity recommendations via Google Gemini.
    Strict privacy: only minimal aggregated, anonymized counts are processed.
    """
    return generate_gemini_report(payload)


@app.get("/api/gemini/latest-report", response_model=GeminiReportResponse)
def get_gemini_latest_report():
    """Generates a Gemini summary from the latest ingested telemetry."""
    active_tracks = _latest_analytics.get("active_tracks", 0)
    valid_faces = _latest_analytics.get("valid_face_count", 0)
    drowsy = _latest_analytics.get("drowsy_count", 0)
    hands = _latest_analytics.get("hand_raise_count", 0)
    avg_att = _latest_analytics.get("avg_attention", 0.0)
    avg_conf = _latest_analytics.get("avg_confusion", 0.0)

    looking_front = max(0, valid_faces - drowsy)
    insufficient = max(0, active_tracks - valid_faces)

    agg = AggregateClassroomInput(
        total_students=active_tracks,
        valid_observations=valid_faces,
        looking_toward_front=looking_front,
        looking_away=0,
        hand_raised=hands,
        insufficient_evidence=insufficient,
        drowsy_count=drowsy,
        phone_distracted=0,
        avg_attention_score=avg_att,
        avg_confusion_index=avg_conf,
    )
    return generate_gemini_report(agg)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
