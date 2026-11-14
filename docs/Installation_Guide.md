# EduSense: Local Installation & System Setup Guide

> **Multi-Modal Intelligent Classroom Analytics System**

This guide provides step-by-step instructions to set up, configure, and launch **EduSense** locally on Windows, Linux, or macOS.

---

## 📋 System Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS Monterey+
- **Python**: Python 3.11+
- **Node.js**: Node.js 18+ & npm
- **RAM**: Minimum 8 GB (16 GB recommended)
- **GPU**: Optional (CPU execution fully supported)

---

## 🛠️ Step 1: Environment Setup

Clone or extract the project workspace:
```bash
cd EduSense
```

Create and activate a Python virtual environment (optional but recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

---

## 📦 Step 2: Install Python Dependencies

Install required Python packages:
```bash
pip install -r requirements.txt
```

Verified core dependencies:
- `fastapi>=0.104.0`
- `uvicorn>=0.24.0`
- `websockets>=12.0`
- `sqlalchemy>=2.0.0`
- `torch>=2.0.0`
- `opencv-python>=4.8.0`
- `mediapipe>=0.10.0`
- `numpy>=1.24.0`
- `pytest>=7.4.0`

---

## 💻 Step 3: Install Frontend Dependencies

Navigate to the `frontend/` directory and install Node modules:
```bash
cd frontend
npm install
cd ..
```

---

## 📊 Step 4: Prepare Datasets (Optional for Training)

Execute the automated dataset pipeline scripts:
```bash
# 1. Download direct public datasets (COCO, BIWI, HaGRID, WIDER FACE)
python scripts/download_datasets.py

# 2. Verify dataset folder integrity
python scripts/verify_datasets.py

# 3. Generate PyTorch train/val/test split manifests
python scripts/prepare_datasets.py

# 4. Generate dataset statistics and charts
python scripts/dataset_statistics.py
```

---

## 🚀 Step 5: Launch the EduSense System

Start the application using two terminal windows:

### Terminal 1: Start FastAPI Backend Server
```bash
python backend/main.py
```
*(Backend runs at `http://127.0.0.1:8000` with WebSocket stream at `ws://127.0.0.1:8000/ws/classroom`)*

### Terminal 2: Start React Frontend Dashboard
```bash
cd frontend
npm run dev
```
*(Frontend UI launches at `http://localhost:3000`)*

---

## 🧪 Step 6: Verify System Health

Run automated unit and integration tests:
```bash
python -m pytest tests/
```
All 13 unit tests will execute and pass cleanly (`13 passed`).
