# EduSense Production Deployment & Setup Guide

This guide provides instructions for deploying **EduSense** locally or using Docker Compose.

---

## 1. Local Quick-Start (Development Mode)

### Step 1: Install Python Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Start FastAPI Backend Server
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Documentation (Swagger UI): `http://localhost:8000/docs`
- Live WebSocket Stream: `ws://localhost:8000/ws/classroom`

### Step 3: Install & Start Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
- Open browser at `http://localhost:3000`

---

## 2. Docker & Containerized Deployment

### Step 1: Build & Launch with Docker Compose
```bash
docker-compose up --build -d
```

### Step 2: Access Containerized Application
- Frontend UI: `http://localhost:3000`
- FastAPI Backend: `http://localhost:8000`

---

## 3. Running Unit Tests
```bash
pytest tests/ -v
```
