"""
EduSense FastAPI Endpoint & Integration Unit Tests
==================================================
Tests REST API endpoints: /students, /classroom, /analytics, /dashboard, /student/{id}, /camera/start, /camera/stop.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["system"] == "EduSense: Multi-Modal Intelligent Classroom Analytics"

def test_get_students():
    response = client.get("/students")
    assert response.status_code == 200
    json_data = response.json()
    assert "total_count" in json_data
    assert "students" in json_data

def test_get_classroom():
    response = client.get("/classroom")
    assert response.status_code == 200
    json_data = response.json()
    assert "eci_score" in json_data
    assert "avg_attention" in json_data

def test_get_analytics():
    response = client.get("/analytics")
    assert response.status_code == 200
    json_data = response.json()
    assert "emotion_distribution" in json_data
    assert "heatmap" in json_data
    assert "timeline" in json_data

def test_get_dashboard():
    response = client.get("/dashboard")
    assert response.status_code == 200
    json_data = response.json()
    assert "frame" in json_data
    assert "classroom" in json_data
    assert "students" in json_data
    assert "recommendations" in json_data

def test_get_individual_student():
    response = client.get("/student/Student%2001")
    assert response.status_code == 200
    json_data = response.json()
    assert "student" in json_data
    assert json_data["student"]["student_id"] == "Student 01"

def test_camera_controls():
    resp_start = client.post("/camera/start?mode=synthetic")
    assert resp_start.status_code == 200
    assert resp_start.json()["status"] == "started"

    resp_stop = client.post("/camera/stop")
    assert resp_stop.status_code == 200
    assert resp_stop.json()["status"] == "stopped"
