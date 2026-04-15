import sys
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print("TensorFlow is installed!")
except ImportError as e:
    print(f"TensorFlow not found: {e}")
    print("Installing TensorFlow...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    print("TensorFlow installed!")