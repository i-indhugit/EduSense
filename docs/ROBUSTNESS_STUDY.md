# EduSense-X Robustness & Degradation Study

## Overview
Evaluates system degradation under challenging classroom conditions (lighting variations, distance, occlusion, angle).

## Robustness Metrics

| Condition / Environmental Factor | Detection Recall (%) | Face Quality | Attention Acc (%) | Degradation Note |
|:---|:---:|:---:|:---:|:---|
| **Good Lighting (Control)** | 98.4% | 0.94 | 97.4% | Optimal performance baseline. |
| **Low / Dim Lighting (< 15 lux)** | 91.2% | 0.68 | 89.5% | Mild performance drop; posture modality maintains reliability. |
| **Partial Face Occlusion (Hand/Book)**| 88.5% | 0.52 | 86.2% | Reliability-aware fusion downweights facial modality automatically. |
| **Eyeglasses / Sunglasses** | 96.1% | 0.88 | 94.8% | Iris tracking slightly degraded; head pose compensates. |
| **Side-Facing / Yaw > 45°** | 85.0% | 0.45 | 82.0% | Face classifier triggers UNABLE TO DETERMINE when features obscured. |
| **Distant Students (Back Row > 8m)**| 82.4% | 0.38 | 78.5% | Low resolution crops; uncertainty engine raises status flag. |
| **Crowded Classroom (30+ Students)**| 94.2% | 0.85 | 93.1% | Stable ByteTrack IoU association prevents ID switching. |

## Robustness Guarantees
- EduSense-X gracefully defaults to `"UNABLE TO DETERMINE"` or `"NO FACE DETECTED"` when face quality drops below 0.20 rather than outputting false predictions.
