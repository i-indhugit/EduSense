# EduSense: Troubleshooting & Diagnostic Guide

> **Common Issues, Diagnostic Steps, and Resolution Protocols**

This document provides troubleshooting solutions for common runtime, dependency, network, or hardware issues when running **EduSense**.

---

## 🔍 Diagnostic Checklist

If experiencing any issue, run the diagnostic script:
```bash
python scripts/verify_datasets.py
```

---

## ❓ Frequently Encountered Issues & Solutions

### 1. `ModuleNotFoundError: No module named 'fastapi'` (or `torch`, `cv2`)
- **Cause**: Active Python environment lacks installed project dependencies.
- **Solution**: Run pip install from root:
  ```bash
  pip install -r requirements.txt
  ```

### 2. `UnicodeEncodeError: 'charmap' codec can't encode character...`
- **Cause**: Windows terminal stdout default encoding (CP1252) failing to print non-ASCII Unicode characters.
- **Solution**: The backend `main.py` uses sanitized `[OK]` ASCII output formatting and UTF-8 reconfiguration.

### 3. Frontend Displays `WebSocket Error` or Falls Back to `REST Polling`
- **Cause**: FastAPI backend server is not running or port 8000 is occupied.
- **Solution**:
  1. Verify backend server is started: `python backend/main.py`.
  2. Test API root endpoint in browser: `http://127.0.0.1:8000/`.
  3. If port 8000 is in use, terminate conflicting process or restart terminal.

### 4. `No Face Detected` Status Displayed in UI
- **Cause**: Live camera feed or simulator has no visible face bounding boxes.
- **Solution**:
  1. Ensure subject is facing the camera under adequate lighting.
  2. If using synthetic mode, switch camera mode toggle in header to `Synthetic Simulator`.

### 5. `HTTP Error 403 / 404` During Dataset Downloads
- **Cause**: Direct download links for academic/EULA datasets (DAiSEE, AffectNet, GazeCapture, NTHU) require registration forms or sign-in.
- **Solution**: Open 📄 **[MANUAL_DOWNLOAD.md](file:///c:/Users/INDU/Desktop/EduSense/MANUAL_DOWNLOAD.md)** and follow step-by-step download guidance.
