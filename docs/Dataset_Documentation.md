# EduSense-X Dataset Documentation

## Public Dataset Catalog

| Dataset Name | Domain / Purpose | Sample Count | Primary Features | Preprocessing Protocol | Access Status |
|:---|:---|:---:|:---|:---|:---|
| **FER-2013** | Facial Emotion Recognition | 35,887 images | 7 Emotion Classes | Grayscale 48x48 alignment, intensity normalization | Permitted / Cataloged |
| **Drowsiness Dataset** | Eye Closure & Yawning | 4,000 images | EAR, MAR, PERCLOS | Eye crop extraction, MediaPipe 468 landmark EAR/MAR | Permitted / Cataloged |
| **HaGRID Subset** | Hand Gesture & Participation | 5,000 images | Raised hand, Stop, Palm | Crop extraction, 21 MediaPipe hand landmark offset | Permitted / Cataloged |
| **BIWI Head Pose** | 3D Head Pose Yaw/Pitch/Roll | 15,000 frames | Yaw, Pitch, Roll (deg) | SolvePnP RGB-D depth calibration ground truth | Permitted / Cataloged |
| **MPIIGaze** | Gaze Direction Vector | 213,672 images | Iris position, Gaze vector | Normalized eye patch, iris landmark offset | Permitted / Cataloged |

## Subject Leakage Prevention Policy
- **Subject-Level Partitioning**: All dataset train/test splits enforce subject-level separation. No single subject appears in both training and test sets.
- **Hash-Based Indexing**: Datasets lacking explicit subject IDs use deterministic SHA-256 hash splits to guarantee zero data duplication leakage.
