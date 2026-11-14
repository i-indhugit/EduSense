# Software Requirements Specification (SRS) - EduSense

## 1. Introduction

### 1.1 Purpose
This document specifies the software requirements for **EduSense: A Multi-Modal Intelligent Classroom Analytics System**.

### 1.2 System Scope
EduSense provides real-time, privacy-preserving classroom analytics to assist teachers in monitoring student attention, engagement, emotion, drowsiness, posture, and participation during live classes.

---

## 2. Functional Requirements

### F1: Anonymous Student Detection & Tracking
- The system MUST detect all seated students in video frames.
- The system MUST assign anonymous identifiers (`Student 01`, `Student 02`) without facial recognition or identity database lookups.
- The system MUST track student bounding boxes continuously across video frames.

### F2: Attention Estimation
- The system MUST estimate eye gaze vectors and head pose angles (Pitch, Yaw, Roll).
- The system MUST classify gaze direction as: *Looking at Board*, *Looking Away*, or *Head Down*.
- The system MUST output an Attention Score Percentage ($A_i \in [0, 100]$).

### F3: Facial Emotion Recognition
- The system MUST classify student emotions into 6 categories: *Neutral*, *Happy*, *Confused*, *Bored*, *Sleepy*, *Frustrated*.

### F4: Drowsiness & Fatigue Detection
- The system MUST compute Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR).
- The system MUST detect eye closure, yawning, and microsleep states.

### F5: Hand Raise & Participation Tracking
- The system MUST detect raised hands using MediaPipe Hands / pose vectors.
- The system MUST track participation duration and interaction count.

### F6: Posture Analysis
- The system MUST evaluate student posture into 3 states: *Upright*, *Leaning*, *Slouching*.

### F7: Multi-Modal Feature Fusion & EduSense Student Index (ESI)
- The system MUST combine Attention, Emotion, Participation, Posture, and Drowsiness into an integrated ESI score ($0 - 100$).
- The system MUST assign status: *Excellent*, *Good*, or *Needs Attention*.

### F8: EduSense Classroom Index (ECI)
- The system MUST compute overall classroom index ($ECI = \frac{1}{N} \sum ESI_i$) and output status: *Excellent*, *Good*, *Average*, or *Poor*.

### F9: Teacher Recommendation Engine
- The system MUST generate automated, real-time instructional recommendations (e.g., attention drop alerts, confusion warnings, participation prompts).

### F10: Live Classroom & Individual Dashboards
- The system MUST display a live video feed with bounding box overlays, anonymous IDs, ESI badges, spatial seating heatmaps, timeline charts, and individual student deep-dive modals.

---

## 3. Non-Functional Requirements

### NFR1: Performance & Real-Time Processing
- The vision pipeline MUST execute at a minimum of 20 FPS on standard workstation hardware.

### NFR2: Privacy & Anonymity
- No biometric face templates or personal identifiable information (PII) shall be stored or transmitted.

### NFR3: Scalability & Usability
- The web dashboard MUST render smoothly on modern browsers (Chrome, Edge, Firefox, Safari).

### NFR4: Modular Code & Testing
- System code MUST be structured into modular packages (`backend`, `frontend`, `fusion_engine`, `inference`, `analytics`, `recommendation`).
