# EduSense: API & WebSocket Specification

> **FastAPI REST Endpoints & Real-Time WebSocket Interface**

This document details all REST API endpoints and WebSocket streaming interfaces provided by the **EduSense** backend server.

---

## 🌐 Server Base URL

- **REST API**: `http://127.0.0.1:8000`
- **WebSocket Stream**: `ws://127.0.0.1:8000/ws/classroom`

---

## 📡 WebSocket Streaming Endpoint

### `GET /ws/classroom`
- **Protocol**: WebSocket (`ws://`)
- **Frequency**: ~20 FPS (every 50ms)
- **Description**: Streams base64 encoded video frame along with complete real-time classroom analytics, student lists, confidence scores, attention statuses, and teacher recommendations.

#### Example Payload JSON:
```json
{
  "timestamp": "2026-07-23T21:57:01.260129",
  "frame": "data:image/jpeg;base64,...",
  "analytics": {
    "eci_score": 85.0,
    "eci_status": "Good",
    "avg_attention": 82.5,
    "avg_engagement": 78.0,
    "avg_participation": 70.0,
    "avg_confusion": 15.0,
    "drowsy_count": 0,
    "hand_raise_count": 0,
    "students": [
      {
        "student_id": "Student 01",
        "confidence_pct": 97.6,
        "attention_pct": 60.8,
        "attention_status": "Moderately Attentive",
        "gaze_direction": "Looking Away",
        "head_pose": "Leaning (P:5°, Y:28°)",
        "emotion": "Bored",
        "is_drowsy": false,
        "is_yawning": false,
        "hand_raised": false,
        "posture": "Leaning",
        "esi_score": 54.3,
        "status": "Needs Attention"
      }
    ],
    "recommendations": []
  }
}
```

---

## 🛠️ REST API Endpoints

### 1. Root System Status
- **Method**: `GET /`
- **Response**:
```json
{
  "system": "EduSense: Multi-Modal Intelligent Classroom Analytics",
  "status": "Online",
  "version": "1.0.0",
  "timestamp": "2026-07-23T21:57:01.260129"
}
```

### 2. Active Students Summary
- **Method**: `GET /students`
- **Response**: Returns list of all active detected students and total count.

### 3. Classroom High-Level Metrics
- **Method**: `GET /classroom`
- **Response**: Returns `eci_score`, `avg_attention`, `avg_engagement`, `avg_participation`, `avg_confusion`, `drowsy_count`, `hand_raise_count`.

### 4. Time-Series Analytics & Seating Heatmap
- **Method**: `GET /analytics`
- **Response**: Returns emotion distributions, 2D seating heatmap grid, and 30-minute metric trends.

### 5. Aggregate Dashboard Summary
- **Method**: `GET /dashboard`
- **Response**: Combined real-time frame, classroom metrics, student cards, and teacher recommendations.

### 6. Individual Student Deep-Dive
- **Method**: `GET /student/{student_id}`
- **Path Parameter**: `student_id` (e.g. `Student 01`)
- **Response**: Deep-dive individual analytics, ESI component breakdown, XAI explanations, and 15-minute history timeline.

### 7. Camera Controls
- **Method**: `POST /camera/start?mode={synthetic|webcam}`
- **Method**: `POST /camera/stop`
- **Response**: `{"status": "started", "mode": "synthetic"}` or `{"status": "stopped"}`.
