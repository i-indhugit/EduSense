# EduSense — Render Free Tier Deployment Guide

This guide details how to deploy **EduSense** to Render using the **100% Free Tier** configuration:
- **Frontend**: Render Free Static Site (Vite production build, browser-side AI inference).
- **Backend**: Render Free Web Service (FastAPI metadata & Google Gemini reporting, zero GPU, <60 MB RAM).

---

## Architecture Overview

```
                      +---------------------------------------+
                      |       User's Web Browser              |
                      |  - Camera Access (getUserMedia)       |
                      |  - MediaPipe Tasks WebAssembly & GPU  |
                      |  - ObjectDetector (4.39 MB)           |
                      |  - FaceLandmarker (3.58 MB)           |
                      |  - PoseLandmarker (5.51 MB)           |
                      |  - ByteTrack & Reliability Fusion     |
                      +-------------------+-------------------+
                                          |
                        Aggregated Telemetry (JSON only)
                        (Zero video / zero face uploads)
                                          |
                                          v
                      +---------------------------------------+
                      |     Render Free Web Service           |
                      |     (Python 3.11 + FastAPI)           |
                      |  - Health checks (/health)            |
                      |  - Session metadata (/api/sessions)   |
                      |  - Gemini Reporting Engine            |
                      +-------------------+-------------------+
                                          |
                      Aggregated Anonymized Counts (JSON only)
                                          |
                                          v
                      +---------------------------------------+
                      |         Google Gemini API             |
                      |         (gemini-2.5-flash)            |
                      |  - Pedagogical recommendations        |
                      |  - Classroom observation summaries    |
                      +---------------------------------------+
```

---

## 1. Prerequisites

1. A GitHub account containing the EduSense repository.
2. A free [Render.com](https://render.com) account.
3. (Optional but recommended) A Google Gemini API key from [Google AI Studio](https://aistudio.google.com). If omitted, EduSense automatically operates with its built-in rule-based fallback engine.

---

## 2. One-Click Blueprint Deployment (`render.yaml`)

The repository includes a root `render.yaml` blueprint defining both services:

1. Log into your **Render Dashboard**.
2. Click **New +** and select **Blueprint**.
3. Connect your **EduSense** GitHub repository.
4. Render automatically parses `render.yaml` and sets up:
   - **`edusense-api`** (Web Service, Python 3.11, Free plan)
   - **`edusense-dashboard`** (Static Site, Free plan)
5. Under `edusense-api` Environment Variables, supply:
   - `GEMINI_API_KEY`: Your Google Gemini API Key (`sync: false` ensures it is entered securely in the Render UI and never committed to Git).
   - `GEMINI_MODEL`: `gemini-2.5-flash`
   - `EDUSENSE_CORS_ORIGINS`: `*` (or your static site domain)
6. Click **Apply**.

---

## 3. Manual Service Configuration (Alternative to Blueprint)

### A. Backend Service (`edusense-api`)
- **Type**: Web Service
- **Runtime**: Python 3
- **Plan**: Free
- **Build Command**: `pip install --upgrade pip && pip install -r requirements-render.txt`
- **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path**: `/health`
- **Environment Variables**:
  - `PYTHON_VERSION`: `3.11.9`
  - `EDUSENSE_CORS_ORIGINS`: `*`
  - `GEMINI_API_KEY`: `<YOUR_GEMINI_API_KEY>`
  - `GEMINI_MODEL`: `gemini-2.5-flash`

### B. Frontend Static Site (`edusense-dashboard`)
- **Type**: Static Site
- **Build Command**: `cd frontend && npm install && npm run build`
- **Publish Directory**: `frontend/dist`
- **Environment Variables**:
  - `VITE_API_BASE_URL`: URL of your deployed `edusense-api` service (e.g., `https://edusense-api.onrender.com`)
- **Rewrite Rule**:
  - Source: `/*`
  - Destination: `/index.html`

---

## 4. Free-Tier Cold Starts & Sleeping Resilience

Render Free Web Services spin down after 15 minutes of inactivity:
- **Cold Start Duration**: Typically 30–50 seconds upon the first incoming request.
- **Client Resilience**: The EduSense frontend runs 100% of its computer vision inference locally in the user's browser. Even while the Render backend wakes up:
  - Detection, tracking, posture, drowsiness, and continuous attention analytics continue uninterrupted in real time.
  - The dashboard displays a friendly notice (`"Metadata backend is sleeping or waking up..."`) alongside a one-click **Ping Backend Health** button.
  - Once awake, aggregated telemetry resumes syncing automatically every 4 seconds.
