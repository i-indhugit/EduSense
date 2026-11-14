# EduSense — AI-Based Classroom Attention & Engagement Analysis System

EduSense is an explainable, privacy-preserving, and evidence-based classroom analytics application designed for **100% browser-side computer vision inference** and **Render Free-Tier hosting**.

> **Crucial Disclaimer**: Visual cues (head orientation, gaze direction, eye blinks, posture) represent observable classroom behaviors only. They do not constitute direct proof of psychological attention, cognitive understanding, or learning comprehension. EduSense explicitly separates observable behavioral cues from cautious engagement estimates.

---

## 1. Core Architecture

- **Frontend (`frontend/`)**: React 18 + TypeScript + Vite + Tailwind CSS running on Render Static Site.
  - **Computer Vision**: `@mediapipe/tasks-vision` via WebAssembly & WebGL.
  - **Local Models (`frontend/public/models/`)**:
    - `efficientdet_lite0.tflite` (4.39 MB) — Person & distractor object detection (cell phone, laptop, book, tv).
    - `blaze_face_short_range.tflite` (0.22 MB) — Fast face confidence boost.
    - `face_landmarker.task` (3.58 MB) — 478 3D landmarks, iris geometry, 52 ARKit blendshapes, 4×4 rigid transformation matrix.
    - `pose_landmarker_lite.task` (5.51 MB, throttled every 2 frames) — 33-point body posture (standing, upright, leaning, slouching) & wrist-above-shoulder hand-raising.
  - **Multi-Object Tracker**: `BrowserByteTracker` with two-stage matching, monotonic `Student XX` IDs, rapid noise expiration, and zero stale face boxes.
  - **Multimodal Fusion**: Reliability-aware dynamic weighting with contradiction prevention (`DROWSY` overrides `ATTENTIVE`), rolling 30-frame PERCLOS, 60s blink rate, note-taking vs. micro-sleep distinction, and phone distraction isolation.
- **Backend (`backend/`)**: Python FastAPI running on Render Free Web Service (<60 MB RAM, zero GPU).
  - Handles `/health`, `/system/status`, `/api/sessions`, and `/api/analytics`.
  - **Zero video, zero face images, and zero biometric templates** are transmitted to or stored on the backend.
- **Gemini Intelligence Module (`backend/gemini_service.py`)**:
  - Uses the official `google-genai` SDK (`gemini-2.5-flash`).
  - Processes strictly anonymized, aggregated classroom observation counts (`total_students`, `valid_observations`, `looking_toward_front`, `looking_away`, `hand_raised`, `insufficient_evidence`, `drowsy_count`, `phone_distracted`, `avg_attention_score`, `avg_confusion_index`).
  - Generates explainable pedagogical summaries, activity recommendations, and answers teacher queries with an automatic rule-based fallback when offline.

---

## 2. Local Setup & Running Instructions

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+ (Python 3.11 recommended)

### Backend Setup
```bash
cd backend
# Create virtual environment (if not already created)
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate

# Install lightweight dependencies
pip install -r ../requirements-render.txt

# (Optional) Copy environment template and add your Gemini API key
cp .env.example .env

# Run FastAPI backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend Setup
```bash
cd frontend
npm install

# Start Vite development server
npm run dev
```
Open `http://localhost:3000/` in your browser.

---

## 3. Running Verification Tests

### Frontend Unit & Accuracy Benchmarks (32 Tests)
```bash
cd frontend
npm test
```
*Executes `src/ai/__tests__/browserAi.test.ts` (11 unit tests), `src/ai/__tests__/accuracyBenchmark.test.ts` (10 scenario benchmarks), and `src/ai/__tests__/stateAccuracyAndAiSuggestions.test.ts` (11 targeted state accuracy, anti-hallucination, and student-level suggestion tests).*

### Backend Tests (48 Tests)
```bash
pytest -v
```
*Validates root, health, telemetry ingestion, session creation, anti-hallucination guardrails, and Gemini reporting with fallback redirection.*

### Production Build
```bash
cd frontend
npm run build
```
*Compiles TypeScript and bundles production assets into `frontend/dist` with zero errors.*

---

## 4. Render Free-Tier Deployment Checklist

- [x] Push code to GitHub repository.
- [x] Log into [Render.com](https://render.com) and click **New + -> Blueprint**.
- [x] Select the EduSense repository to apply root `render.yaml`.
- [x] In the Render dashboard under `edusense-api`, add `GEMINI_API_KEY` (secret).
- [x] Verify backend health at `https://<your-api>.onrender.com/health`.
- [x] Open static site at `https://<your-dashboard>.onrender.com`.

---

## 5. Known Limitations

1. **Client Hardware**: Running 4 MediaPipe vision models simultaneously in the browser requires a modern browser supporting WebAssembly and WebGL. On older low-tier mobile devices, the internal throttler caps inference to ~10 FPS to ensure the UI thread remains responsive.
2. **Extreme Occlusion**: If a student is turned completely away (>90° yaw) or occluded for more than 15 consecutive frames (~1.5s), their track state transitions to occluded and eventually expires to prevent ghost tracks.
3. **Anti-Spoofing**: EduSense rejects faces detected on T-shirts (below shoulder line) and inside secondary screens (laptops/tvs), but cannot guarantee 100% defense against physical printed photographs without active liveness depth hardware.
4. **Render Free-Tier Cold Starts**: The backend spins down after 15 minutes of inactivity. When waking up, requests take 30–50 seconds. However, **all local camera inference in the browser continues running uninterrupted**.
