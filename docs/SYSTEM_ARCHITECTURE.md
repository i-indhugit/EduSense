# EduSense-X System Architecture Specification

## Overview
EduSense-X is a real-time, reliability-aware multimodal intelligence framework for classroom attention, engagement, and behavioral analysis.

## End-to-End Processing Architecture
```
LIVE VIDEO (Webcam/RTSP)
  │
  ▼
STUDENT DETECTION (YOLOv11s / MediaPipe BBox)
  │
  ▼
STUDENT TRACKING (ByteTrack / IoU Centroid Association)
  │
  ▼
INDIVIDUAL STUDENT CROP & FACE DETECTION (RetinaFace / SCRFD)
  │
  ▼
FACE & LANDMARK QUALITY CHECK (Confidence >= 0.50, Size >= 50x50, Quality >= 0.20)
  ├── Invalid Face / Low Quality ──► Output: "NO FACE DETECTED" / "UNABLE TO DETERMINE"
  └── Valid Face Quality ──────────►
        │
        ▼
  MULTIMODAL FEATURE EXTRACTION
    ├── Facial Emotion (MobileNetV3 / ConvNeXt-Tiny)
    ├── Eye Gaze & Iris Vector (MediaPipe Face Mesh 468 Keypoints)
    ├── Head Pose (3D SolvePnP Pitch/Yaw/Roll)
    ├── Body Posture (MediaPipe Pose Keypoints & Torso Tilt)
    ├── Hand Gesture & Participation (MediaPipe Hands / HaGRID)
    └── Drowsiness (EAR, MAR, PERCLOS, Temporal Window)
        │
        ▼
  FEATURE RELIABILITY ESTIMATION (r_m ∈ [0, 1] for each modality)
        │
        ▼
  TEMPORAL MODEL (Sliding Window EMA & Hysteresis Filtering)
        │
        ▼
  RELIABILITY-AWARE CROSS-ATTENTION FUSION
    └── Attention_Weights = softmax( (Q * K^T)/sqrt(d_k) + log(r_m + 1e-6) )
        │
        ▼
  UNCERTAINTY ESTIMATION ENGINE
    └── U = 0.30(1 - q_face) + 0.35(1 - r_bar) + 0.25 H(p) + 0.10 σ_temp
        ├── U > 0.45 ─────────────► Output: "UNABLE TO DETERMINE"
        └── U <= 0.45 ────────────► Output: Validated Attention / Engagement Prediction
              │
              ▼
        EXPLAINABLE AI (XAI) & COUNTERFACTUAL ENGINE
              │
              ▼
        REAL-TIME DASHBOARD (FastAPI + React Academic UI)
```

## Core Modules & Responsibilities
- `inference/pipeline.py`: Real-time vision pipeline executing live camera capture, IoU tracking, feature extraction, and overlay drawing.
- `fusion_engine/cross_attention_fusion.py`: PyTorch and NumPy Reliability-Aware Cross-Attention fusion modules.
- `utils/uncertainty_utils.py`: Multi-factor uncertainty estimation and fallback thresholding engine.
- `explainability/xai_engine.py`: Natural language rationale generation, modality attributions, and counterfactual perturbation analysis.
- `analytics/predictive_engine.py`: Temporal forecasting for attention drop risk and learning pace index.
- `backend/main.py`: FastAPI server serving WebSocket streams and REST research endpoints.
