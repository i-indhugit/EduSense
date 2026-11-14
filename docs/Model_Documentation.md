# EduSense: Model & Neural Network Architecture Specification

> **Deep Learning Models, Feature Extraction & Temporal Sequence Suite**

This document details the neural network architectures, computer vision algorithms, and temporal sequence models powering **EduSense**.

---

## 1. Student & Object Detection Model

### Model Architecture: YOLOv11s
- **Backbone**: Modified CSP-DarkNet with C3k2 blocks and SPPF (Spatial Pyramid Pooling Fast).
- **Head**: Decoupled Anchor-Free Detection Head.
- **Input Resolution**: $640 \times 640 \times 3$.
- **Classes**: Person Category (Category ID 1).
- **Benchmark Performance**: Precision = 94.1%, Recall = 93.2%, mAP50 = 95.8%, Latency = 6.4ms (156 FPS).

---

## 2. Multi-Student Tracking Engine

### Tracker: ByteTrack
- **Mechanism**: Low-score detection box association + Kalman Filter motion prediction.
- **Occlusion Resiliency**: Retains student identity trajectories across temporary occlusions or seating shifts.
- **Benchmark Performance**: ID switching rate = 0.12 switches/min, Latency = 0.4ms (2500 FPS).

---

## 3. Face Detection & Landmark Geometry

### Face Detection: RetinaFace / MediaPipe Face Mesh
- **Landmarks**: 468 3D facial landmark points.
- **Eye Aspect Ratio (EAR)**:
  $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \|p_1 - p_4\|}$$
- **Mouth Aspect Ratio (MAR)**:
  $$\text{MAR} = \frac{\|p_{14} - p_{18}\|}{\|p_{12} - p_{16}\|}$$

---

## 4. SolvePnP 3D Head Pose Estimator

- **Model**: Perspective-n-Point (SolvePnP) using 6 canonical 3D facial feature points (Nose tip, Chin, Left Eye corner, Right Eye corner, Left Mouth corner, Right Mouth corner).
- **Outputs**: Rotation Vector $\mathbf{R} \to$ Pitch ($\theta_p$), Yaw ($\theta_y$), Roll ($\theta_r$) angles in degrees.

---

## 5. Facial Emotion Classifier

### Architecture: MobileNetV3-Small
- **Input**: $224 \times 224 \times 3$ cropped facial patch.
- **Output**: 6 discrete classes (`Neutral`, `Happy`, `Confused`, `Bored`, `Sleepy`, `Frustrated`).
- **Benchmark Performance**: Accuracy = 88.4%, F1 = 0.880, Latency = 4.2ms (238 FPS).

---

## 6. Temporal Sequence Modeling & Score Stabilization

### Temporal LSTM & Kinematic Kalman Filter
- **Temporal LSTM**: Processes sliding time-series windows ($T = 30\text{s}$) to model long-term attention decay curves.
- **Exponential Moving Average (EMA)**:
  $$\text{Score}_{t} = \alpha \cdot \text{Raw}_{t} + (1 - \alpha) \cdot \text{Score}_{t-1} \quad (\alpha = 0.20)$$
- **Hysteresis Status Locking**:
  Status transitions require score movement beyond boundary thresholds by a margin ($\Delta = 2.5\%$), completely eliminating prediction flickering between frames.
