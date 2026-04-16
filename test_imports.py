#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test all the key imports."""
    try:
        print("Testing imports...")

        # Test TensorFlow
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__}")

        # Test other dependencies
        import numpy as np
        import scipy
        import shap
        import pywt
        print("✓ All dependencies imported successfully")

        # Test project modules
        from src.preprocessing.signal_cleaner import PPGProcessor
        from src.features.hrv_metrics import HRVMetrics
        from src.models.classifier import create_hybrid_classifier
        from src.api.stream_handler import PPGStreamHandler
        print("✓ All project modules imported successfully")

        # Test model creation
        model = create_hybrid_classifier()
        print("✓ Hybrid model created successfully")

        # Test stream handler
        handler = PPGStreamHandler(model)
        print("✓ Stream handler created successfully")

        print("\n🎉 All tests passed! The project is ready to use.")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)