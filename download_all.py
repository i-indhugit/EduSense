#!/usr/bin/env python3
"""
EduSense: Top-Level Automated Dataset Downloader Wrapper
========================================================
Delegates execution to scripts/download_datasets.py.
"""

import sys
from pathlib import Path

# Add project root and scripts directory to path
BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_datasets import main

if __name__ == "__main__":
    main()
