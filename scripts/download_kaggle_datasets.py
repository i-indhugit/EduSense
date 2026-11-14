#!/usr/bin/env python3
"""
EduSense: Kaggle Dataset Downloader & Verification Script
==========================================================
Downloads the 4 required Kaggle datasets for EduSense:
1. FER-2013 (msambare/fer2013) -> datasets/FER2013/
2. Drowsiness Dataset (hoangtung719/drowsiness-dataset) -> datasets/Drowsiness/
3. Student Engagement Dataset (deepramazumder/student-engagement-dataset) -> datasets/StudentEngagement/
4. Hand Gesture Dataset (barnabaspeter/hand-gesture-dataset) -> datasets/HandGesture/

Security & Safety Guarantee:
- Uses Kaggle API via environment variables (KAGGLE_USERNAME, KAGGLE_KEY) or ~/.kaggle/kaggle.json.
- Checks local existence before downloading to prevent re-downloads.
- Does NOT inject synthetic data.
"""

import os
import sys
import zipfile
import logging
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
LOGS_DIR = BASE_DIR / "logs"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging Setup
logger = logging.getLogger("EduSenseKaggleDownloader")
logger.setLevel(logging.INFO)
logger.handlers.clear()

c_handler = logging.StreamHandler(sys.stdout)
c_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)

KAGGLE_DATASETS = [
    {
        "id": "msambare/fer2013",
        "folder": DATASETS_DIR / "FER2013",
        "name": "FER-2013 Emotion Dataset",
        "target_check": DATASETS_DIR / "FER2013" / "train"
    },
    {
        "id": "hoangtung719/drowsiness-dataset",
        "folder": DATASETS_DIR / "Drowsiness",
        "name": "Drowsiness & Yawning Dataset",
        "target_check": DATASETS_DIR / "Drowsiness" / "train"
    },
    {
        "id": "deepramazumder/student-engagement-dataset",
        "folder": DATASETS_DIR / "StudentEngagement",
        "name": "Student Engagement Behavior Dataset",
        "target_check": DATASETS_DIR / "StudentEngagement" / "Engaged"
    },
    {
        "id": "barnabaspeter/hand-gesture-dataset",
        "folder": DATASETS_DIR / "HandGesture",
        "name": "Hand Gesture Dataset",
        "target_check": DATASETS_DIR / "HandGesture" / "gestures"
    }
]

def check_kaggle_installed():
    try:
        import kaggle
        return True
    except ImportError:
        logger.warning("Kaggle Python package not found. Install via: pip install kaggle")
        return False

def download_dataset(item):
    folder = item["folder"]
    folder.mkdir(parents=True, exist_ok=True)

    # Check if target files/folders already exist
    if item["target_check"].exists() and any(folder.iterdir()):
        logger.info(f"[EXISTS] {item['name']} already available at {folder}")
        return True

    logger.info(f"[DOWNLOADING] {item['name']} from Kaggle ID: {item['id']}...")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        # Download to target directory
        api.dataset_download_files(item["id"], path=str(folder), unzip=True)
        logger.info(f"[SUCCESS] Downloaded & Extracted {item['name']} to {folder}")
        return True
    except Exception as e:
        logger.error(f"[ERROR] Failed to download {item['name']}: {e}")
        logger.info(f"   Please ensure KAGGLE_USERNAME and KAGGLE_KEY environment variables are set,")
        logger.info(f"   or manually download from https://www.kaggle.com/datasets/{item['id']} into {folder}")
        return False

def main():
    logger.info("==================================================")
    logger.info("  EduSense Required Kaggle Datasets Downloader    ")
    logger.info("==================================================")

    kaggle_available = check_kaggle_installed()

    success_count = 0
    for ds in KAGGLE_DATASETS:
        if ds["target_check"].exists():
            logger.info(f"[VERIFIED] {ds['name']} exists locally.")
            success_count += 1
        elif kaggle_available:
            if download_dataset(ds):
                success_count += 1

    logger.info("--------------------------------------------------")
    logger.info(f"Summary: {success_count}/{len(KAGGLE_DATASETS)} Kaggle datasets verified/prepared.")

if __name__ == "__main__":
    main()
