#!/usr/bin/env python3
"""
EduSense: Top-Level Dataset Verification Wrapper
================================================
Delegates execution to scripts/verify_datasets.py.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = BASE_DIR / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from verify_datasets import main

if __name__ == "__main__":
    main()
