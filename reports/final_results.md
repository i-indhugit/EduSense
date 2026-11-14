# EduSense Real-Time AI Classroom Attention & Learning Difficulty Analyzer — Final System Report

> **Multi-Modal Intelligent Classroom Analytics System**

EduSense is an end-to-end research-grade AI platform built for real-time classroom analytics using **100% REAL KAGGLE DATASETS** and **LIVE WEBCAM INFERENCE**.

---

## 1. System Compliance & Principles Verification

- [x] **Zero Synthetic / Fake Data**: All models trained strictly on real Kaggle datasets (`FER-2013`, `Drowsiness`, `Student Engagement`, `Hand Gesture`). Synthetic simulator excluded from live inference stream.
- [x] **Live Webcam Source**: The live camera is the sole input source (`cv2.VideoCapture(0)` / `/ws/classroom`).
- [x] **Strict Fallback Handling**:
  - No face in frame $\to$ Displays `"No Face Detected"`.
  - Low confidence or partial face $\to$ Displays `"Unable to Determine"`.
- [x] **Research & Terminology Safety**:
  - Phrases indicators strictly as `"Possible Learning Difficulty Indicator"`.
  - Never diagnoses learning disabilities.
- [x] **Empirical Metrics**: All reported performance numbers are experimentally measured on dataset splits.

---

## 2. Model & Pipeline Architecture

```
[ Live Webcam Frame ] ──► [ YOLOv11s Student Detector ] ──► [ ByteTrack Multi-Person Tracker ]
                                                                      │
                                                                      ▼
[ Real-Time Analytics Dashboard ] ◄── [ XAI Engine ] ◄── [ Trained PyTorch Model Ensemble ]
  - Live Stream Overlay                  - Feature Attributions    - Emotion (MobileNetV3)
  - Student Cards                        - Natural Rationale       - Drowsiness (EAR/MAR)
  - Classroom KPIs                                                 - Engagement (ResNet18)
                                                                   - Hand Activity (Gesture)
```

---

## 3. Real-Time Performance & Fallback Matrix

| Input Condition | System Action | Student View Output | Classroom View Output |
| :--- | :--- | :--- | :--- |
| **Clear Visible Face** | Full inference & temporal smoothing | Active Student Card + Features | Aggregated KPI Stats |
| **No Face Detected** | Halts inference calculations | `"No Face Detected"` | Classroom status: `"No Students Detected"` |
| **Low Confidence / Partial Face**| Suppresses unverified predictions | `"Unable to Determine"` | Excluded from KPI averages |
| **Persistent Confusion + Gaze Shift**| Evaluates 10s temporal window | `"Possible Learning Difficulty Indicator"`| Alert Feed notification |

---

## 4. Test Suite Execution Summary

- **Total Test Cases**: 17 / 17 Passed Cleanly
- **Test File**: [`tests/test_edusense_full.py`](file:///c:/Users/INDU/Desktop/EduSense/tests/test_edusense_full.py)
- **Scenarios Tested**:
  1. Face detected
  2. No face detected
  3. Partial/poor face
  4. Low confidence
  5. Multiple students
  6. Student entering frame
  7. Student leaving frame
  8. Occluded face
  9. Poor lighting
  10. Emotion prediction
  11. Drowsiness prediction
  12. Hand activity
  13. Engagement prediction
  14. Temporal smoothing
  15. Invalid model output
  16. WebSocket/API communication
  17. Dashboard rendering payload
