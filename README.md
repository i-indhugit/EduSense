# EduSense: Multi-Modal Intelligent Classroom Analytics System
> **Academic Final-Year Engineering Project — Local Execution Specification**

EduSense is an AI-powered real-time classroom analytics platform designed to evaluate student engagement, attention, emotions, pose, drowsiness, and participation using non-intrusive Computer Vision and Deep Learning.

This project runs **100% locally** on a standard student laptop/desktop without cloud services, Docker, containers, authentication overhead, or external SaaS dependencies.

---

## 🛠️ Technology Stack (Local & Lightweight)

- **Backend**: Python 3.11+ | FastAPI | Uvicorn
- **Frontend**: React | TypeScript | Tailwind CSS | Vite
- **Database**: SQLite (`edusense.db`)
- **AI & Analytics**: PyTorch | OpenCV | MediaPipe | Scikit-Learn | NumPy | Pandas

---

## 🚀 Running EduSense Locally

### 1. Start the FastAPI Backend Server
From the project root folder:
```bash
python backend/main.py
```
*(Backend runs locally at `http://127.0.0.1:8000` with WebSocket real-time analytics streaming at `ws://127.0.0.1:8000/ws`)*

### 2. Start the React Frontend Dashboard
In a second terminal window:
```bash
cd frontend
npm install
npm run dev
```
*(Frontend UI launches locally at `http://localhost:5173`)*

---

## 📁 Project Directory Structure

```
EduSense/
├── backend/                  # FastAPI Application & SQLite DB Models
│   ├── main.py               # Main API entry point (uvicorn runner)
│   └── database/             # SQLite models & initialization
├── frontend/                 # React + TypeScript + Tailwind UI Dashboard
│   ├── src/                  # Components, Hooks, API & WebSocket client
│   ├── package.json          # Node dependencies
│   └── vite.config.ts        # Vite configuration
├── datasets/                 # 9 Public Datasets Storage
│   ├── DAiSEE/               # Engagement & Affective States
│   ├── FER2013/              # Baseline Facial Emotion Recognition
│   ├── AffectNet/            # Deep Emotion & Valence-Arousal Mapping
│   ├── GazeCapture/          # Eye Gaze Estimation & Attention Tracking
│   ├── BIWI/                 # 3D Head Pose Angles (Pitch, Yaw, Roll)
│   ├── HaGRID/               # Hand Raise & Gesture Recognition
│   ├── NTHU/                 # Drowsiness, Yawning & Microsleep Detection
│   ├── WIDER_FACE/           # Multi-Student Face Detection
│   ├── COCO/                 # Student Person Detection & Bounding Boxes
│   ├── label_mappings.json   # Unified class mapping definitions
│   └── dataset_index.json    # Global dataset registry index
├── models/                   # Neural Architecture Definitions
│   ├── emotion_classifier.py # MobileNetV3 / ResNet / ViT Emotion Models
│   ├── fusion_attention.py   # Multi-Head Cross-Attention Fusion Engine (EIE)
│   └── temporal_models.py    # Temporal Sequence LSTM / GRU Models
├── inference/                # Vision Pipeline & Synthetic Simulator
│   ├── pipeline.py           # Real-Time Frame Processor
│   └── synthetic_simulator.py# Synthetic Classroom Generator
├── fusion_engine/            # Feature Fusion, ESI & ECI Calculators
├── explainability/           # Explainable AI (XAI) Attribution Engine
├── analytics/                # Predictive Analytics Engine
├── recommendation/           # Teacher Recommendation Feed Engine
├── scripts/                  # Dataset Management Scripts
│   ├── download_datasets.py  # Automated downloader for public direct mirrors
│   ├── verify_datasets.py    # Integrity & corruption diagnostic engine
│   ├── prepare_datasets.py   # PyTorch split manifests & metadata generator
│   └── dataset_statistics.py # Dataset profiling & class imbalance calculator
├── logs/                     # Local diagnostic logs & JSON/CSV outputs
├── docs/                     # Documentation & dataset statistics charts
├── dataset_report.md         # Full research paper specs, licenses & citations
├── MANUAL_DOWNLOAD.md        # Step-by-step download guide for manual datasets
└── README.md                 # Project documentation & local execution guide
```

---

## 📊 Dataset Pipeline Commands

Run the local dataset management scripts in order:

```bash
# 1. Download direct public datasets (COCO, BIWI, HaGRID, WIDER FACE)
python scripts/download_datasets.py

# 2. Inspect MANUAL_DOWNLOAD.md for datasets requiring academic forms/EULAs

# 3. Verify dataset folder integrity & corruption status
python scripts/verify_datasets.py

# 4. Generate PyTorch train/val/test manifests & metadata
python scripts/prepare_datasets.py

# 5. Compute sample statistics, resolutions & class imbalance metrics
python scripts/dataset_statistics.py
```

---

## 📜 Project Guidelines Compliance

- ❌ No Docker / Kubernetes / Containers
- ❌ No AWS / Azure / GCP / Cloud Deployment
- ❌ No CI/CD Pipelines / DevOps / Microservices
- ❌ No Redis / Complex Authentication / Multi-User Login
- ✅ 100% Local Execution (FastAPI + React + SQLite + PyTorch + OpenCV)
