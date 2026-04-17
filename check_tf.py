"""
TensorFlow availability checker
This script verifies TensorFlow installation and auto-installs if missing.
"""
import sys

# Attempt to import TensorFlow and check version
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print("TensorFlow is installed!")
except ImportError as e:
    # TensorFlow not found, proceed with installation
    print(f"TensorFlow not found: {e}")
    print("Installing TensorFlow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    print("TensorFlow installed!")
  