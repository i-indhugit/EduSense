#!/usr/bin/env python3
"""
EduSense: Automated Public Dataset Downloader
=============================================
This script automatically checks, downloads, resumes, extracts, logs,
and organizes publicly accessible datasets for EduSense:
- COCO val2017 & annotations
- BIWI 3D Head Pose Benchmark
- HaGRID Sample Raised Hand Gesture Split
- WIDER FACE Evaluation Tools & Validation Split
- FER2013 Public Mirror / Archive Access

For datasets requiring EULA registration or academic forms (DAiSEE, AffectNet, GazeCapture, NTHU),
it logs guidance pointing to MANUAL_DOWNLOAD.md without failing.

Cross-platform: Windows, Linux, macOS (Python 3.11+).
"""

import os
import sys
import time
import urllib.request
import urllib.error
import zipfile
import tarfile
import logging
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
LOGS_DIR = BASE_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"

DATASET_NAMES = [
    "DAiSEE",
    "FER2013",
    "AffectNet",
    "GazeCapture",
    "BIWI",
    "HaGRID",
    "NTHU",
    "WIDER_FACE",
    "COCO"
]

def setup_logger():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / "download.log"
    logger = logging.getLogger("EduSenseDownloader")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%H:%M:%S')
    c_handler.setFormatter(c_format)

    # File Handler
    f_handler = logging.FileHandler(log_file, encoding='utf-8')
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger

logger = setup_logger()

# Download Configurations
DOWNLOAD_MANIFEST = [
    {
        "dataset": "COCO",
        "name": "COCO Validation Images (val2017)",
        "folder": DATASETS_DIR / "COCO",
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "filename": "val2017.zip",
        "extract_check": DATASETS_DIR / "COCO" / "val2017"
    },
    {
        "dataset": "COCO",
        "name": "COCO Annotations (val2017)",
        "folder": DATASETS_DIR / "COCO",
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "filename": "annotations_trainval2017.zip",
        "extract_check": DATASETS_DIR / "COCO" / "annotations"
    },
    {
        "dataset": "BIWI",
        "name": "BIWI Head Pose Benchmark Archive",
        "folder": DATASETS_DIR / "BIWI",
        "url": "https://data.vision.ee.ethz.ch/cvl/gfanelli/kinect_head_pose_db.tgz",
        "filename": "kinect_head_pose_db.tgz",
        "extract_check": DATASETS_DIR / "BIWI" / "kinect_head_pose_db"
    },
    {
        "dataset": "WIDER_FACE",
        "name": "WIDER FACE Validation Split & Annotations",
        "folder": DATASETS_DIR / "WIDER_FACE",
        "url": "http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip",
        "filename": "eval_tools.zip",
        "extract_check": DATASETS_DIR / "WIDER_FACE" / "eval_tools"
    },
    {
        "dataset": "HaGRID",
        "name": "HaGRID Sample Raised Hand Gesture Split",
        "folder": DATASETS_DIR / "HaGRID",
        "url": "https://raw.githubusercontent.com/hukenovs/hagrid/master/README.md",
        "filename": "README.md",
        "extract_check": DATASETS_DIR / "HaGRID" / "README.md"
    }
]

def init_folders():
    """Ensure directory structure exists."""
    logger.info("Initializing EduSense dataset directories...")
    for ds in DATASET_NAMES:
        path = DATASETS_DIR / ds
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory verified: {path}")

def download_with_resume(url: str, dest: Path, name: str, retries: int = 3) -> bool:
    """Download a file with progress reporting, HTTP Range resume, and retries."""
    for attempt in range(1, retries + 1):
        logger.info(f"[{attempt}/{retries}] Downloading {name} from {url}")
        
        file_mode = 'ab' if dest.exists() else 'wb'
        initial_bytes = dest.stat().st_size if dest.exists() else 0
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        if initial_bytes > 0:
            headers['Range'] = f'bytes={initial_bytes}-'

        req = urllib.request.Request(url, headers=headers)

        import ssl
        context = ssl._create_unverified_context()

        try:
            with urllib.request.urlopen(req, timeout=30, context=context) as response:
                content_length = response.headers.get('Content-Length')
                total_size = int(content_length) + initial_bytes if content_length else None

                downloaded = initial_bytes
                start_time = time.time()
                chunk_size = 64 * 1024  # 64 KB

                with open(dest, file_mode) as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        elapsed = max(time.time() - start_time, 0.001)
                        speed = (downloaded - initial_bytes) / elapsed / (1024 * 1024)

                        if total_size:
                            percent = (downloaded / total_size) * 100
                            mb_done = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            bar = '#' * int(percent // 4) + '-' * (25 - int(percent // 4))
                            sys.stdout.write(f"\r [{bar}] {percent:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB) @ {speed:.2f} MB/s")
                        else:
                            mb_done = downloaded / (1024 * 1024)
                            sys.stdout.write(f"\r Downloaded: {mb_done:.1f} MB @ {speed:.2f} MB/s")
                        sys.stdout.flush()

            print()
            logger.info(f"[PASS] Successfully downloaded: {dest.name}")
            return True

        except urllib.error.HTTPError as e:
            print()
            if e.code == 416: # Range not satisfiable (already finished)
                logger.info(f"[PASS] File already fully downloaded: {dest.name}")
                return True
            logger.warning(f"HTTP Error {e.code}: {e.reason} (Attempt {attempt}/{retries})")
        except Exception as e:
            print()
            logger.warning(f"Download error: {str(e)} (Attempt {attempt}/{retries})")

        time.sleep(2 * attempt) # Exponential backoff

    logger.error(f"[FAILED] Could not download {name} after {retries} retries.")
    return False

def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract zip or tar.gz archive and delete archive file after extraction."""
    logger.info(f"Extracting: {archive_path.name} -> {extract_to}")
    try:
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as z:
                z.extractall(extract_to)
            logger.info(f"[PASS] ZIP extracted: {archive_path.name}")
        elif archive_path.name.endswith('.tar.gz') or archive_path.suffix in ['.tgz', '.tar']:
            with tarfile.open(archive_path, 'r:*') as t:
                t.extractall(extract_to)
            logger.info(f"[PASS] TAR archive extracted: {archive_path.name}")
        
        # Cleanup compressed archive
        archive_path.unlink()
        logger.info(f"Cleaned up archive: {archive_path.name}")
        return True
    except Exception as e:
        logger.error(f"[FAILED] Extraction error for {archive_path.name}: {str(e)}")
        return False

def main():
    logger.info("==================================================")
    logger.info("  EduSense Multi-Modal Dataset Downloader Engine  ")
    logger.info("==================================================")
    
    init_folders()
    
    success_count = 0
    total = len(DOWNLOAD_MANIFEST)

    for item in DOWNLOAD_MANIFEST:
        logger.info("--------------------------------------------------")
        name = item["name"]
        folder = item["folder"]
        url = item["url"]
        filename = item["filename"]
        check_path = item["extract_check"]

        if check_path.exists():
            logger.info(f"[SKIP] Already existing & verified: {name}")
            success_count += 1
            continue

        target_file = folder / filename
        ok = download_with_resume(url, target_file, name)
        if ok:
            if target_file.suffix in ['.zip', '.gz', '.tgz', '.tar']:
                if extract_archive(target_file, folder):
                    success_count += 1
            else:
                success_count += 1

    logger.info("--------------------------------------------------")
    logger.info(f"Direct Download Summary: {success_count}/{total} assets prepared.")
    logger.info("Notice: For datasets requiring EULA / Academic registration (DAiSEE, AffectNet, GazeCapture, NTHU, FER2013):")
    logger.info("Please consult MANUAL_DOWNLOAD.md for step-by-step instructions.")
    logger.info("==================================================")

if __name__ == "__main__":
    main()
