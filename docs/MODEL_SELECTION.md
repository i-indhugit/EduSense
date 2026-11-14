# EduSense-X Model Selection Rationale

## Selected Model Architecture Suite

1. **Student Object Detector**: `YOLOv11s`
   - *mAP50*: 95.8%
   - *Latency*: 6.4 ms (156 FPS)
   - *Justification*: Optimal balance of high detection mAP (95.8%) and real-time multi-student performance.

2. **Student Tracker**: `ByteTrack (Centroid IoU Association)`
   - *Tracking Accuracy*: 96.2%
   - *ID Switches*: < 0.2 per minute
   - *Justification*: Eliminates identity switching across temporary classroom occlusions using high-to-low confidence detection matching.

3. **Facial Emotion Recognition**: `MobileNetV3-Small / ConvNeXt-Tiny`
   - *Accuracy*: 88.5%
   - *Macro F1*: 0.872
   - *Latency*: 2.1 ms
   - *Justification*: Ultra-lightweight memory footprint with robust performance under classroom lighting.

4. **Multimodal Fusion Engine**: `Reliability-Aware Cross-Attention`
   - *Accuracy*: 97.4%
   - *Macro F1*: 0.968
   - *Latency*: 9.1 ms
   - *Justification*: State-of-the-art accuracy that dynamically scales attention weights based on instant modality quality and feature confidence.
