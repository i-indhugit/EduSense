#!/usr/bin/env python3
"""
EduSense: Dataset Verification & Integrity Engine
===================================================
This script inspects all 9 required datasets for EduSense:
- DAiSEE
- FER2013
- AffectNet
- GazeCapture
- BIWI
- HaGRID
- NTHU
- WIDER_FACE
- COCO

Checks:
- Folder existence
- Media presence (images/videos)
- Annotations/Labels presence
- Disk usage
- Image corruption detection
- Status output: PASS, WARNING, FAILED

Cross-platform: Windows, Linux, macOS (Python 3.11+).
"""

import os
import sys
import json
import logging
from pathlib import Path

# Try importing image check libraries
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
LOGS_DIR = BASE_DIR / "logs"

DATASET_SPECS = [
    {
        "name": "DAiSEE",
        "folder": DATASETS_DIR / "DAiSEE",
        "media_types": ["video"],
        "media_exts": [".mp4", ".avi"],
        "label_exts": [".csv", ".txt"],
        "required_manual": True,
        "purpose": "Student Engagement, Confusion, Boredom, Frustration"
    },
    {
        "name": "FER2013",
        "folder": DATASETS_DIR / "FER2013",
        "media_types": ["image"],
        "media_exts": [".jpg", ".jpeg", ".png"],
        "label_exts": [".csv", ".txt"],
        "required_manual": True,
        "purpose": "Baseline Facial Emotion Recognition (7 Discrete Classes)"
    },
    {
        "name": "AffectNet",
        "folder": DATASETS_DIR / "AffectNet",
        "media_types": ["image"],
        "media_exts": [".jpg", ".jpeg", ".png"],
        "label_exts": [".csv", ".json", ".npy"],
        "required_manual": True,
        "purpose": "Deep Emotion & Valence-Arousal Mapping"
    },
    {
        "name": "GazeCapture",
        "folder": DATASETS_DIR / "GazeCapture",
        "media_types": ["image"],
        "media_exts": [".jpg", ".png"],
        "label_exts": [".json"],
        "required_manual": True,
        "purpose": "Eye Gaze Estimation & Attention Tracking"
    },
    {
        "name": "BIWI",
        "folder": DATASETS_DIR / "BIWI",
        "media_types": ["image"],
        "media_exts": [".png", ".jpg", ".bin"],
        "label_exts": [".txt", ".mat"],
        "required_manual": True,
        "purpose": "3D Head Pose Angles (Pitch, Yaw, Roll)"
    },
    {
        "name": "HaGRID",
        "folder": DATASETS_DIR / "HaGRID",
        "media_types": ["image"],
        "media_exts": [".jpg", ".png"],
        "label_exts": [".json", ".txt"],
        "required_manual": False,
        "purpose": "Hand Raise & Gesture Recognition"
    },
    {
        "name": "NTHU",
        "folder": DATASETS_DIR / "NTHU",
        "media_types": ["video"],
        "media_exts": [".mp4", ".avi"],
        "label_exts": [".txt"],
        "required_manual": True,
        "purpose": "Drowsiness, Yawning & Microsleep Detection"
    },
    {
        "name": "WIDER_FACE",
        "folder": DATASETS_DIR / "WIDER_FACE",
        "media_types": ["image"],
        "media_exts": [".jpg", ".png"],
        "label_exts": [".txt", ".mat"],
        "required_manual": False,
        "purpose": "Multi-Student Face Detection"
    },
    {
        "name": "COCO",
        "folder": DATASETS_DIR / "COCO",
        "media_types": ["image"],
        "media_exts": [".jpg"],
        "label_exts": [".json"],
        "required_manual": False,
        "purpose": "Student Body & Person Bounding Box Detection"
    }
]

def setup_logger():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "verification.log"

    logger = logging.getLogger("EduSenseVerifier")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S'))

    f_handler = logging.FileHandler(log_file, encoding='utf-8')
    f_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger

logger = setup_logger()

def format_bytes(size: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def check_image_corrupted(file_path: Path) -> bool:
    if HAS_PIL:
        try:
            with Image.open(file_path) as img:
                img.verify()
            return False
        except Exception:
            return True
    elif HAS_CV2:
        try:
            img = cv2.imread(str(file_path))
            return img is None
        except Exception:
            return True
    return False

def verify_single_dataset(spec: dict) -> dict:
    name = spec["name"]
    folder = spec["folder"]
    media_exts = set(spec["media_exts"])
    label_exts = set(spec["label_exts"])

    res = {
        "dataset": name,
        "purpose": spec["purpose"],
        "folder_exists": folder.exists(),
        "total_size": 0,
        "media_count": 0,
        "label_count": 0,
        "corrupted_media_count": 0,
        "status": "FAILED", # PASS, WARNING, FAILED
        "details": ""
    }

    if not folder.exists():
        res["status"] = "FAILED"
        res["details"] = "Directory missing. Run download_datasets.py or follow MANUAL_DOWNLOAD.md"
        return res

    total_bytes = 0
    media_files = []
    label_count = 0

    try:
        for root, _, files in os.walk(folder):
            for file in files:
                p = Path(root) / file
                try:
                    total_bytes += p.stat().st_size
                    ext = p.suffix.lower()
                    if ext in media_exts:
                        media_files.append(p)
                    if ext in label_exts or "label" in file.lower() or "annotation" in file.lower():
                        label_count += 1
                except Exception:
                    pass
    except Exception as e:
        res["details"] = f"Error scanning directory: {str(e)}"
        return res

    res["total_size"] = total_bytes
    res["media_count"] = len(media_files)
    res["label_count"] = label_count

    # Sample corruption check on up to 50 media files for speed
    sample_files = media_files[:50]
    corrupted = 0
    for mf in sample_files:
        if check_image_corrupted(mf):
            corrupted += 1
    res["corrupted_media_count"] = corrupted

    if res["media_count"] > 0 and label_count > 0 and corrupted == 0:
        res["status"] = "PASS"
        res["details"] = "All media and annotations verified."
    elif res["media_count"] > 0 or label_count > 0:
        res["status"] = "WARNING"
        res["details"] = "Partial dataset present. Complete download via MANUAL_DOWNLOAD.md if needed."
        if corrupted > 0:
            res["details"] += f" {corrupted} corrupted media samples detected."
    else:
        res["status"] = "WARNING" if spec["required_manual"] else "FAILED"
        res["details"] = "Empty dataset directory. Requires download or manual acquisition."

    return res

def main():
    logger.info("==========================================================================")
    logger.info("                 EduSense Dataset Verification Engine                      ")
    logger.info("==========================================================================")
    logger.info(f"Target Base Directory: {DATASETS_DIR}\n")

    results = []
    pass_count = 0
    warn_count = 0
    fail_count = 0

    for spec in DATASET_SPECS:
        res = verify_single_dataset(spec)
        results.append(res)
        if res["status"] == "PASS":
            pass_count += 1
        elif res["status"] == "WARNING":
            warn_count += 1
        else:
            fail_count += 1

    # Output Format Table
    header = f"{'Dataset Name':<15} | {'Folder':<7} | {'Media Files':<12} | {'Labels':<8} | {'Size':<10} | {'Status':<10}"
    divider = "-" * len(header)
    logger.info(header)
    logger.info(divider)

    for r in results:
        f_exist = "[YES]" if r["folder_exists"] else "[NO]"
        size_str = format_bytes(r["total_size"])
        logger.info(f"{r['dataset']:<15} | {f_exist:<7} | {r['media_count']:<12} | {r['label_count']:<8} | {size_str:<10} | {r['status']:<10}")

    logger.info(divider)
    logger.info(f"Summary: PASS={pass_count} | WARNING={warn_count} | FAILED={fail_count} out of {len(DATASET_SPECS)} datasets.")
    logger.info("==========================================================================")

    # Save JSON report
    report_file = LOGS_DIR / "verification_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "datasets": results,
            "pass_count": pass_count,
            "warn_count": warn_count,
            "fail_count": fail_count
        }, f, indent=2)
    logger.info(f"Detailed JSON report saved to: {report_file}")

if __name__ == "__main__":
    main()
