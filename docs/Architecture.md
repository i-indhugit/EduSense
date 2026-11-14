# EduSense: System Architecture & Sub-Engine Pipeline Specification

> **Multi-Modal Intelligent Classroom Analytics System**

EduSense is built as a highly modular, decoupled, and scalable research-grade AI system designed for real-time classroom analytics.

---

## 1. High-Level System Architecture

```
[ Classroom Camera / Video Stream ]
                │
                ▼
┌────────────────────────────────────────────────────────┐
│             EduSense Vision & Detection Engine         │
│  - Face Detection & 3D Mesh (RetinaFace / MediaPipe)   │
│  - Student Tracking & Occlusion Handling (ByteTrack)   │
│  - 3D Head Pose SolvePnP & 2D/3D Eye Gaze Direction    │
│  - Facial Emotion Classification (MobileNetV3-Small)   │
│  - Hand Raise & Gesture Recognition (MediaPipe Hands)  │
│  - Drowsiness PERCLOS & Blink Rate Estimation          │
└───────────────────────────┬────────────────────────────┘
                            │ Raw Student Features
                            ▼
┌────────────────────────────────────────────────────────┐
│        EduSense Intelligence Engine (EIE Fusion)       │
│  - Multi-Head Cross-Attention Feature Weighting        │
│  - EduSense Student Index (ESI) Formulation            │
│  - EduSense Classroom Index (ECI) Aggregator           │
│  - Temporal EMA Smoothing & Hysteresis Status Locking  │
│  - Explainable AI (XAI) Feature Attribution Engine     │
│  - Predictive Analytics & Attention Drop Curve Engine  │
└───────────────────────────┬────────────────────────────┘
                            │ Analytics & Stream Payload
                            ▼
┌────────────────────────────────────────────────────────┐
│          FastAPI Web Server & SQLite Storage           │
│  - Real-Time WebSocket Stream (`/ws/classroom`)        │
│  - REST API Endpoints (`/students`, `/dashboard`, etc) │
│  - SQLite Transactional Database (`edusense.db`)       │
└───────────────────────────┬────────────────────────────┘
                            │ JSON & Frame Stream
                            ▼
┌────────────────────────────────────────────────────────┐
│            React + TypeScript + Tailwind UI            │
│  - Real-Time Camera Stream & AI Overlays               │
│  - Key Performance Indicators (KPI) Metrics Grid        │
│  - Spatial Classroom Seating Heatmap                   │
│  - Teacher Recommendation Alert Feed                   │
│  - Student Detail Deep-Dive Modal                      │
└────────────────────────────────────────────────────────┘
```

---

## 2. Core Architectural Sub-Engines

### A. Vision & Feature Extraction Engine (`inference/`)
- Orchestrates camera frame capture, synthetic simulation fallback, face detection, multi-student tracking, and feature extraction.
- **Key Modules**:
  - `inference/pipeline.py`: Main vision pipeline controller.
  - `inference/synthetic_simulator.py`: Synthetic multi-student video generator with temporal EMA filter and multi-line overlay graphics.

### B. Multi-Modal Feature Fusion Engine (`fusion_engine/`)
- Unifies individual feature streams into normalized scores ($0-100$).
- **Key Modules**:
  - `fusion_engine/feature_fusion.py`: Master fusion processor with temporal EMA smoothing ($\alpha = 0.20$) and hysteresis status locking ($\Delta = 2.5\%$).
  - `fusion_engine/esi_calculator.py`: Computes EduSense Student Index (ESI).
  - `fusion_engine/eci_calculator.py`: Computes EduSense Classroom Index (ECI) & Confusion Index.

### C. Explainable AI Engine (`explainability/`)
- Generates human-interpretable natural language rationales and percentage attributions for every prediction.
- **Key Module**: `explainability/xai_engine.py`.

### D. Predictive Analytics Engine (`analytics/`)
- Forecasts 5-10 minute attention drop curves, confusion trajectories, learning pace indices, and classroom energy scores using sliding temporal windows.
- **Key Module**: `analytics/predictive_engine.py`.

### E. Neural Architecture Models (`models/`)
- `models/emotion_classifier.py`: Candidate facial emotion classifiers (MobileNetV3, ResNet18, ViT-Tiny).
- `models/fusion_attention.py`: Learnable Cross-Attention Network (EIE).
- `models/temporal_models.py`: Temporal sequence modeling suite (Temporal LSTM, GRU, TCN, Transformer Encoder).
