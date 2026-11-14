# EduSense: Comprehensive Dataset Specification Report

This document provides a technical specification of all 9 public datasets integrated into the **EduSense** Multi-Modal Intelligent Classroom Analytics System.

---

## Dataset Specifications

### 1. DAiSEE (Dataset for Affective States in E-Learning Environments)

- **Dataset Name**: DAiSEE
- **Purpose in EduSense**: Benchmarking and training student engagement, boredom, confusion, and frustration models in e-learning / classroom environments.
- **Number of Classes**: 4 Affective States (Boredom, Engagement, Confusion, Frustration), each rated on a 4-point intensity scale (0: None, 1: Low, 2: Medium, 3: High).
- **Number of Samples**: 9,068 video snippets (2,738,400 frames) captured from 112 subjects.
- **File Format**: `.mp4` video files + `.csv` ground truth label tables.
- **Folder Location**: `datasets/DAiSEE/`
- **License**: Non-Commercial Academic Research License.
- **Citation**:
  ```bibtex
  @inproceedings{gupta2016daisee,
    title={DAiSEE: Towards User Engagement Recognition in the Wild},
    author={Gupta, Abhinav and D'Cunha, Anushree and Awasthi, Kshitij and Balasubramanian, Vineeth},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    year={2016}
  }
  ```

---

### 2. FER2013 (Facial Expression Recognition 2013)

- **Dataset Name**: FER2013
- **Purpose in EduSense**: Baseline real-time facial expression and emotion classification (Neutral, Happy, Confused, Bored, Sleepy, Frustrated).
- **Number of Classes**: 7 emotion categories (0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral).
- **Number of Samples**: 35,887 grayscale images (48×48 pixels).
  - Training: 28,709 images
  - Public Test: 3,589 images
  - Private Test: 3,589 images
- **File Format**: `.csv` (pixel intensity string per row) or `.png`/`.jpg` extracted images.
- **Folder Location**: `datasets/FER2013/`
- **License**: Creative Commons Attribution 4.0 International / Kaggle Competition Terms.
- **Citation**:
  ```bibtex
  @article{goodfellow2013fer2013,
    title={Challenges in representation learning: A report on three machine learning contests},
    author={Goodfellow, Ian J and Erhan, Dumitru and Carrier, Luc Carrier and Courville, Aaron and Mirza, Mehdi and Hamner, Ben and Cukierski, Will and Tang, Yichuan and Thaler, David and Lee, Dong-Hyun and others},
    journal={Neural Networks},
    volume={64},
    pages={59--63},
    year={2015}
  }
  ```

---

### 3. AffectNet

- **Dataset Name**: AffectNet
- **Purpose in EduSense**: Fine-tuning deep facial expression recognition models and continuous valence-arousal mapping for fine-grained emotion estimation.
- **Number of Classes**: 8 discrete expression categories (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt) + continuous Valence & Arousal values [-1.0, 1.0].
- **Number of Samples**: 440,000+ manually annotated facial images (over 1,000,000 images total collected from web).
- **File Format**: `.jpg` facial images + `.csv`/`.npy` annotations.
- **Folder Location**: `datasets/AffectNet/`
- **License**: Academic Research End User License Agreement (EULA).
- **Citation**:
  ```bibtex
  @article{mollahosseini2017affectnet,
    title={AffectNet: A database for facial expression, valence, and arousal computing in the wild},
    author={Mollahosseini, Ali and Hasani, Behzad and Mahoor, Mohammad H},
    journal={IEEE Transactions on Affective Computing},
    volume={10},
    number={1},
    pages={18--31},
    year={2017}
  }
  ```

---

### 4. GazeCapture

- **Dataset Name**: GazeCapture
- **Purpose in EduSense**: Training 2D/3D eye gaze estimation and attention tracking models to determine whether a student is looking at the board, teacher, screen, or looking away.
- **Number of Classes**: Continuous (X, Y) 2D screen gaze point coordinates + 3D eye gaze vectors.
- **Number of Samples**: 2,445,504 frame images collected from 1,450 subjects.
- **File Format**: `.jpg` camera frames + `.json` coordinate metadata per subject.
- **Folder Location**: `datasets/GazeCapture/`
- **License**: Academic Research License (MIT CSAIL).
- **Citation**:
  ```bibtex
  @inproceedings{krafka2016gazecapture,
    title={Eye Tracking for Everyone},
    author={Krafka, Kyle and Khosla, Aditya and Nishimura, Pablo and Figueiredo, Gabriel and Torralba, Antonio},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages={2176--2184},
    year={2016}
  }
  ```

---

### 5. BIWI Head Pose Dataset

- **Dataset Name**: BIWI Head Pose
- **Purpose in EduSense**: 3D Head Pose Estimation (Pitch, Yaw, Roll) to determine head orientation (Looking Forward, Head Down, Looking Away, Leaning).
- **Number of Classes**: Continuous 3D rotation angles (Pitch $\in [-60^\circ, +60^\circ]$, Yaw $\in [-75^\circ, +75^\circ]$, Roll $\in [-50^\circ, +50^\circ]$) + 3D translation coordinates.
- **Number of Samples**: 15,000+ RGB-D frames across 24 sequences of 20 unique subjects (14 male, 6 female).
- **File Format**: `.png` RGB images + `.txt` pose transformation matrices & ground truth vectors.
- **Folder Location**: `datasets/BIWI/`
- **License**: Free for Non-Commercial Academic Research (ETH Zurich).
- **Citation**:
  ```bibtex
  @article{fanelli2013biwi,
    title={Random forests for real time 3d face analysis},
    author={Fanelli, Gabriele and Dantone, Matthias and Gall, Juergen and Fossati, Andrea and Van Gool, Luc},
    journal={International Journal of Computer Vision},
    volume={101},
    number={3},
    pages={437--458},
    year={2013}
  }
  ```

---

### 6. HaGRID (Hand Gesture Recognition Image Dataset)

- **Dataset Name**: HaGRID
- **Purpose in EduSense**: Hand raise detection and gesture-based student participation measuring.
- **Number of Classes**: 18 gesture classes (including `raised_hand`, `stop`, `one`, `two`, `three`, `like`, `dislike`, `fist`, etc.).
- **Number of Samples**: 552,992 high-resolution RGB images with bounding box & keypoint annotations.
- **File Format**: `.jpg` images + `.json` bounding box annotations.
- **Folder Location**: `datasets/HaGRID/`
- **License**: Apache 2.0 License.
- **Citation**:
  ```bibtex
  @article{kapitanov2024hagrid,
    title={HaGRID--HAnd Gesture Recognition Image Dataset},
    author={Kapitanov, Alexander and Makhlyarchuk, Karina and Kvanchiani, Alexander},
    journal={Pattern Recognition Letters},
    year={2024}
  }
  ```

---

### 7. NTHU Driver Drowsiness Detection Dataset

- **Dataset Name**: NTHU Drowsiness
- **Purpose in EduSense**: Drowsiness, microsleep, and yawn detection (Eye Aspect Ratio thresholds, Eye Closure duration, Yawning frequency).
- **Number of Classes**: 5 state categories (Normal Driving/Sitting, Slow Blinking, Yawning, Falling Asleep / Microsleep, Head Nodding).
- **Number of Samples**: 36 subjects recorded under diverse illumination conditions (day/night, with/without glasses).
- **File Format**: `.avi` video sequences + `.txt` temporal state annotations.
- **Folder Location**: `datasets/NTHU/`
- **License**: Academic Research License (NTHU Computer Vision Lab).
- **Citation**:
  ```bibtex
  @inproceedings{weng2016nthu,
    title={Driver Drowsiness Detection via Continuous Conditional Random Fields},
    author={Weng, Ching-Hua and Lai, Yi-Hing and Lai, Shang-Hong},
    booktitle={IEEE International Conference on Image Processing (ICIP)},
    year={2016}
  }
  ```

---

### 8. WIDER FACE

- **Dataset Name**: WIDER FACE
- **Purpose in EduSense**: Fine-tuning YOLOv11 and face detectors for multi-student face detection under severe occlusions, extreme poses, and variable lighting in classrooms.
- **Number of Classes**: 1 object class (`face`) with attributes (scale, posture, occlusion, illumination).
- **Number of Samples**: 32,203 images containing 393,703 annotated face bounding boxes.
- **File Format**: `.jpg` images + `.txt` bounding box coordinates.
- **Folder Location**: `datasets/WIDER_FACE/`
- **License**: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0).
- **Citation**:
  ```bibtex
  @inproceedings{yang2016wider,
    title={WIDER FACE: A Face Detection Benchmark},
    author={Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2016}
  }
  ```

---

### 9. COCO (Common Objects in Context - Person Subset)

- **Dataset Name**: COCO (val2017)
- **Purpose in EduSense**: Pre-training and validating YOLOv11 person detection & ByteTrack tracking for seated student detection in classroom video feeds.
- **Number of Classes**: 80 object categories (specifically Category ID 1: `person`).
- **Number of Samples**: 5,000 validation images (`val2017.zip`), 118,287 training images (`train2017.zip`).
- **File Format**: `.jpg` images + `.json` COCO format bounding box and keypoint annotations (`instances_val2017.json`).
- **Folder Location**: `datasets/COCO/`
- **License**: Creative Commons Attribution 4.0 (CC BY 4.0).
- **Citation**:
  ```bibtex
  @inproceedings{lin2014coco,
    title={Microsoft COCO: Common Objects in Context},
    author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Doll{\'a}r, Piotr and Zitnick, C Lawrence},
    booktitle={European Conference on Computer Vision (ECCV)},
    pages={740--755},
    year={2014}
  }
  ```

---

## Dataset Summary Table

| Dataset | Focus Domain | Samples Count | Format | Primary Metric / Output | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DAiSEE** | Student Engagement | 9,068 videos | MP4 + CSV | Engagement / Boredom / Confusion / Frustration (0-3) | Academic |
| **FER2013** | Facial Emotion | 35,887 images | CSV / PNG | 7 Discrete Emotions (Happy, Sad, Neutral...) | CC BY 4.0 |
| **AffectNet** | Emotion & Valence | 440,000+ images | JPG + CSV | 8 Discrete Emotions + Valence/Arousal | EULA |
| **GazeCapture**| Eye Gaze / Attention | 2.4M frames | JPG + JSON | 2D Screen Gaze (X, Y) / 3D Vector | Academic |
| **BIWI** | Head Pose | 15,000+ frames | PNG + TXT | Pitch, Yaw, Roll Angles ($\mathbf{R}_{3\times3}$) | Academic |
| **HaGRID** | Hand Gestures | 552,992 images | JPG + JSON | 18 Hand Gestures (`raised_hand`...) | Apache 2.0 |
| **NTHU** | Drowsiness / Yawn | 36 video sets | AVI + TXT | Drowsiness / Yawn / Eye Closure / Microsleep | Academic |
| **WIDER FACE**| Multi-Face BBox | 32,203 images | JPG + TXT | Face Bounding Boxes ($x, y, w, h$) | CC BY-NC-ND |
| **COCO** | Person Detection | 5,000+ images | JPG + JSON | Person Bounding Boxes & Keypoints | CC BY 4.0 |
