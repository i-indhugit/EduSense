# EduSense: Manual Dataset Download Guide

Some datasets required for **EduSense** cannot be downloaded automatically via script because they require academic end-user license agreements (EULA), university approval, Kaggle authentication, or manual form registration.

This guide provides official sources, exact URLs, reasons for manual access, and step-by-step instructions to obtain and place each dataset into the proper folder.

---

## Summary of Manual Datasets

| Dataset | Primary Role in EduSense | Official Host | Access Requirement | Target Folder |
| :--- | :--- | :--- | :--- | :--- |
| **DAiSEE** | Student Engagement & Affective State | IIT Mandi | Google Form / Academic Request | `datasets/DAiSEE/` |
| **AffectNet** | Emotion Recognition (Facial Expressions) | University of Denver | Signed EULA & Request Email | `datasets/AffectNet/` |
| **GazeCapture** | Eye Gaze Estimation & Attention | MIT CSAIL | Academic Registration Form | `datasets/GazeCapture/` |
| **NTHU Drowsiness** | Drowsiness & Yawn Detection | NTHU Computer Vision Lab | Signed End-User Agreement | `datasets/NTHU/` |
| **FER2013** | Baseline Emotion Classification | Kaggle Competitions | Kaggle Account Login | `datasets/FER2013/` |
| **WIDER FACE (Full)** | Dense Multi-Student Face Detection | CUHK | Google Drive / Baidu Verification | `datasets/WIDER_FACE/` |

---

## Detailed Download Instructions

### 1. DAiSEE (Dataset for Affective States in E-Learning Environments)

- **Official Website**: [https://people.iitmandi.ac.in/~tagore/daisee/daisee_main.html](https://people.iitmandi.ac.in/~tagore/daisee/daisee_main.html)
- **Download URL**: [https://people.iitmandi.ac.in/~tagore/daisee/daisee_main.html#download](https://people.iitmandi.ac.in/~tagore/daisee/daisee_main.html#download) or Kaggle Dataset Mirror.
- **Why Manual Download is Required**: IIT Mandi requires researchers to fill out an academic request form to agree to non-commercial academic research usage terms.
- **Steps to Obtain Access**:
  1. Visit the official website or Kaggle mirror page.
  2. Fill in your academic email, institution name, and project title (*EduSense: Multi-Modal Intelligent Classroom Analytics*).
  3. Upon approval or downloading the dataset (`DAiSEE.zip`), extract the contents.
  4. Place the video files and labels (`Labels/` CSV files) directly into:
     ```
     EduSense/datasets/DAiSEE/
     ├── DataSet/
     │   ├── Train/
     │   ├── Validation/
     │   └── Test/
     └── Labels/
         ├── TrainLabels.csv
         ├── ValidationLabels.csv
         └── TestLabels.csv
     ```

---

### 2. AffectNet

- **Official Website**: [http://mohammadmahoor.com/affectnet/](http://mohammadmahoor.com/affectnet/)
- **Download URL**: [http://mohammadmahoor.com/affectnet/](http://mohammadmahoor.com/affectnet/)
- **Why Manual Download is Required**: Requires a signed End User License Agreement (EULA) sent to Prof. Mohammad Mahoor at University of Denver.
- **Steps to Obtain Access**:
  1. Download the AffectNet EULA PDF from the official website.
  2. Print, sign, and scan the agreement form with your academic institution details.
  3. Email the signed PDF to the dataset maintainers as specified on the website.
  4. Once credentials/download links are received, download the images and annotations.
  5. Extract and place files into:
     ```
     EduSense/datasets/AffectNet/
     ├── Manually_Annotated_Images/
     ├── Automatically_Annotated_Images/
     └── labels.csv
     ```

---

### 3. GazeCapture

- **Official Website**: [http://gazecapture.csail.mit.edu/](http://gazecapture.csail.mit.edu/)
- **Download URL**: [http://gazecapture.csail.mit.edu/download.php](http://gazecapture.csail.mit.edu/download.php)
- **Why Manual Download is Required**: Hosted by MIT CSAIL and requires registration with a valid academic email address.
- **Steps to Obtain Access**:
  1. Go to the download portal and submit your name, institution email, and research scope.
  2. Click the verification link sent to your email to unlock the download links (JSON metadata + frame archives).
  3. Download and extract the dataset parts.
  4. Place extracted subject directories into:
     ```
     EduSense/datasets/GazeCapture/
     ├── 00002/
     ├── 00003/
     ├── ...
     └── gaze_capture_meta.json
     ```

---

### 4. NTHU Driver Drowsiness Detection Dataset

- **Official Website**: [http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/](http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/)
- **Download URL**: Contact NTHU CV Lab maintainers via website request portal.
- **Why Manual Download is Required**: Requires academic user agreement signature to protect subject privacy.
- **Steps to Obtain Access**:
  1. Download the NTHU DDD License Agreement document.
  2. Complete the form with supervisor signature / project summary.
  3. Submit the document to the NTHU Computer Vision Laboratory.
  4. Receive access links, download scenario videos (BareFace, Glasses, Night, etc.), and extract to:
     ```
     EduSense/datasets/NTHU/
     ├── Training_Evaluation_Dataset/
     └── Testing_Dataset/
     ```

---

### 5. FER2013 (Facial Expression Recognition 2013)

- **Official Website**: [https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- **Download URL**: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Why Manual Download is Required**: Hosted on Kaggle; requires user authentication (Kaggle API key `kaggle.json` or standard browser login).
- **Steps to Obtain Access**:
  - **Option A (Manual Browser)**:
    1. Log in to Kaggle and visit the FER2013 dataset page.
    2. Click **Download** to obtain `fer2013.tar.gz` or `fer2013.csv`.
    3. Extract the archive into:
       ```
       EduSense/datasets/FER2013/
       ├── fer2013.csv
       ├── train/
       └── test/
       ```
  - **Option B (Kaggle CLI)**:
    ```bash
    kaggle datasets download -d msambare/fer2013 -p EduSense/datasets/FER2013 --unzip
    ```

---

### 6. WIDER FACE (Full Dataset)

- **Official Website**: [http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/)
- **Download URL**: [http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/)
- **Why Manual Download is Required**: The primary training partitions (~3.5 GB) are mirrored on Google Drive and Baidu Cloud, requiring browser verification or OAuth token.
- **Steps to Obtain Access**:
  1. Visit the official website download links for WIDER Face Training, Validation, and Annotations.
  2. Download `WIDER_train.zip`, `WIDER_val.zip`, and `wider_face_split.zip`.
  3. Extract all archives into:
     ```
     EduSense/datasets/WIDER_FACE/
     ├── WIDER_train/
     ├── WIDER_val/
     └── wider_face_split/
     ```

---

## Verification After Manual Downloads

Once you have downloaded and placed any manual dataset in its respective folder, run:

```bash
python verify_datasets.py
```

The script will scan all 9 dataset folders, detect extracted files/labels, count samples, and report readiness status for model training.
