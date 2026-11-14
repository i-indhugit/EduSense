# EduSense System Architecture & Data Flow Specification

## Overview
EduSense employs a decoupled, multi-tier micro-service architecture separating real-time vision processing, multi-modal feature fusion, backend API streaming, and reactive frontend visualization.

## System Architecture Diagram (Mermaid)

```mermaid
flowchart TD
    subgraph Data Sources & Vision Capture
        A1[Live Webcam Feed]
        A2[Pre-recorded Classroom Video]
        A3[Synthetic Frame Simulator]
    end

    subgraph Vision & Detection Pipeline
        B1[YOLOv11 / OpenCV Frame Acquisition]
        B2[ByteTrack Anonymous ID Tracker]
        B3[MediaPipe 3D Face Mesh - 468 Landmark Points]
        B4[MediaPipe Body Pose & Hands Extraction]
        
        B1 --> B2 --> B3 --> B4
    end

    subgraph Multi-Modal Feature Extractors
        C1[Gaze Vector & Head Pose Pitch/Yaw/Roll]
        C2[Facial Emotion Classifier - FER2013/DAiSEE]
        C3[Drowsiness & EAR / PERCLOS Evaluator]
        C4[Hand Raise & Posture Evaluator]
        
        B3 --> C1
        B3 --> C2
        B3 --> C3
        B4 --> C4
    end

    subgraph Feature Fusion & Analytics Engine
        D1[Confusion Index Solver]
        D2[EduSense Student Index - ESI Solver]
        D3[EduSense Classroom Index - ECI Solver]
        D4[Teacher Recommendation Engine]

        C1 & C2 & C3 & C4 --> D1
        C1 & C2 & C3 & C4 --> D2
        D2 --> D3
        D3 --> D4
    end

    subgraph FastAPI Backend & Streaming Services
        E1[WebSocket Frame & Payload Broadcaster]
        E2[REST APIs - /classroom, /student, /camera]
        E3[SQLAlchemy ORM & Database]

        D3 & D4 --> E1
        D3 & D4 --> E2
        E2 --> E3
    end

    subgraph Web Frontend Dashboard
        F1[Live Video Canvas & Student Overlays]
        F2[Classroom KPI Metrics Grid]
        F3[Temporal Recharts Analytics]
        F4[2D Spatial Seating Heatmap]
        F5[Teacher Recommendations Feed]
        F6[Individual Student Detail Modal]

        E1 & E2 --> F1 & F2 & F3 & F4 & F5 & F6
    end

    A1 & A2 & A3 --> B1
```
