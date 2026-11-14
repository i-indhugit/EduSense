# EduSense: User & Operating Guide

> **Teacher Operating Guide for Real-Time Classroom Analytics**

This document provides a comprehensive operational guide for educators, researchers, and project evaluators using **EduSense**.

---

## 🖥️ Dashboard Interface Layout

The EduSense Dashboard is divided into 5 interactive zones:

1. **Header Bar**:
   - Live EduSense Classroom Index (ECI) Score & Status badge (`Excellent`, `Good`, `Average`, `Poor`).
   - Mode Toggle Selector (`Synthetic Simulator` vs `Live Webcam Stream`).
   - WebSocket Connection Status indicator (`LIVE WS` vs `REST POLLING`).

2. **Key Performance Indicators (KPI) Metrics Grid**:
   - **Average Attention %**: Classroom mean sustained attention score ($0-100\%$).
   - **Average Engagement %**: Multi-factor engagement metric ($0-100\%$).
   - **Average Participation %**: Hand raises & upright body posture contribution.
   - **Average Confusion Index**: Eyebrow furrowing and facial confusion index ($0-100$).
   - **Drowsiness Counter**: Total active students flagged with eye closure / yawns.
   - **Hand Raise Counter**: Active student hand raises.

3. **Live Camera Stream & Overlay View**:
   - High-definition video feed with real-time AI bounding box overlays.
   - Multi-line face badge displaying:
     - Student ID & Model Confidence (`Student 01 | Conf: 97.6%`).
     - Attention Score & Status (`Att: 60.8% | Moderately Attentive`).
     - Facial Emotion, Head Pose, & Gaze Direction (`Bored | Leaning | Looking Away`).

4. **Teacher Recommendation Alert Feed**:
   - Priority-ranked actionable feedback cards (`HIGH`, `MEDIUM`, `LOW`, `INFO`).
   - Alerts for attention drops, confusion spikes, drowsiness clusters, or low participation.

5. **Spatial Seating Heatmap & Deep-Dive Modal**:
   - 2D grid matrix mapping classroom seating positions.
   - Click any student card to open the **Explainable AI (XAI) Deep-Dive Modal** displaying feature attributions, natural language rationales, and 15-minute metric trend timelines.

---

## 🎯 Operating Modes

### Mode 1: Multi-Student Classroom Mode
- Activated when multiple students are present in the frame.
- Displays classroom-wide aggregate metrics (ECI score, seating heatmap, teacher recommendations, and timeline charts).

### Mode 2: Single Student Deep-Dive Mode
- Activated when only one student is visible or selected.
- Displays individual student detection status, confidence %, attention score, human-readable status, emotion, eye gaze, head pose, drowsiness status, hand activity, and Explainable AI reasoning card.

### Mode 3: Zero Face Detection / Low Visibility Handling
- **No Face Detected**: When no face is visible, displays `"No Face Detected"` banner and clears all predictions immediately. No stale or fabricated scores are displayed.
- **Face Not Clear**: When a face is heavily blurred or occluded, displays `"Face Not Clear"` and skips analysis.
- **Unable to Determine**: When model confidence is $< 60\%$, displays `"Unable to Determine"` to prevent false predictions.
