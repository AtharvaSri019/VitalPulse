#!/usr/bin/env python3
"""Wrapper script to run the training pipeline from the project root."""

import sys
from pathlib import Path

# Add the src directory to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Now import and run the training script
from train import main

if __name__ == "__main__":
    main()