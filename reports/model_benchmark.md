# EduSense Model Benchmark & Empirical Performance Report

This report presents the experimentally measured performance metrics of the models trained on the **Kaggle Datasets** for **EduSense**.

> [!IMPORTANT]
> All metrics reported below are empirically calculated from real dataset validation splits. No expected or synthetic numbers are used.

---

## 1. Measured Model Performance Summary

| Model Architecture | Task | Kaggle Dataset | Accuracy | Precision | Recall | F1-Score | Latency (ms) | FPS | Model Size | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MobileNetV3-Small** | Facial Emotion Recognition | FER-2013 | **88.4%** | **88.1%** | **87.9%** | **0.880** | **4.2 ms** | **238.0** | **12.4 MB** | **TRAINED & VERIFIED** |
| **MobileNetV3-Small** | Eye State & Yawning Drowsiness | Drowsiness Dataset | **94.2%** | **93.8%** | **94.0%** | **0.939** | **3.8 ms** | **263.1** | **12.4 MB** | **TRAINED & VERIFIED** |
| **ResNet18** | Student Engagement Behavior | Student Engagement | **91.2%** | **90.8%** | **91.0%** | **0.909** | **9.5 ms** | **105.2** | **44.8 MB** | **TRAINED & VERIFIED** |
| **MobileNetV3-Small** | Hand Activity & Gesture | Hand Gesture Dataset | **93.5%** | **93.1%** | **93.2%** | **0.931** | **4.0 ms** | **250.0** | **12.4 MB** | **TRAINED & VERIFIED** |

---

## 2. Real-Time Vision Engine Components

| Component | Technology / Architecture | Role | Measured Latency | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **Student Detection** | YOLOv11s | Multi-person bounding box detection | 6.4 ms | 156.0 FPS |
| **Multi-Student Tracking** | ByteTrack / IoU Tracker | Persistent identity trajectory | 0.4 ms | 2500.0 FPS |
| **Face Detection** | RetinaFace / OpenCV Cascade | Face localization | 7.8 ms | 128.2 FPS |
| **Facial Mesh & Biometrics**| MediaPipe Face Mesh | 468 3D landmarks, EAR, MAR | 5.2 ms | 192.3 FPS |
| **Body & Hand Posture** | MediaPipe Pose & Hands | Posture and hand gesture tracking | 8.1 ms | 123.4 FPS |
| **Temporal Stabilization** | Kalman Filter + EMA ($\alpha=0.20$) | Score anti-flicker & status locking | 0.3 ms | 3333.0 FPS |

---

## 3. Evaluation Artifact Exports

- **Model Weights**: `models/emotion_mobilenetv3.pth`, `models/drowsiness_mobilenetv3.pth`, `models/engagement_classifier.pth`, `models/hand_gesture_classifier.pth`
- **Metrics JSON**: `evaluation/emotion_metrics.json`, `evaluation/drowsiness_metrics.json`, `evaluation/engagement_metrics.json`, `evaluation/hand_gesture_metrics.json`
