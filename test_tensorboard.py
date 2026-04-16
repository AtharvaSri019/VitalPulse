#!/usr/bin/env python3
"""Test script to verify TensorBoard installation."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_tensorboard():
    """Test TensorBoard functionality."""
    try:
        import tensorflow as tf
        from tensorflow import keras

        # Test basic TensorBoard functionality
        log_dir = "test_logs"
        writer = tf.summary.create_file_writer(log_dir)

        with writer.as_default():
            tf.summary.scalar("test_metric", 0.5, step=1)

        print("✓ TensorBoard is working correctly")
        print(f"✓ Test logs written to: {log_dir}")

        # Clean up
        import shutil
        if Path(log_dir).exists():
            shutil.rmtree(log_dir)

        return True

    except Exception as e:
        print(f"❌ TensorBoard test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_tensorboard()
    if success:
        print("\n🎉 TensorBoard is ready! You can now run the training script:")
        print("python run_training.py")
    else:
        print("\n❌ TensorBoard setup failed. Please check your TensorFlow installation.")
    sys.exit(0 if success else 1)