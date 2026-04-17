"""Demo script for real-time PPG streaming with periodic predictions."""

import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path so absolute imports resolve correctly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.api.stream_handler import PPGStreamHandler
    from src.models.classifier import create_hybrid_classifier
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("  pip install tensorflow shap PyWavelets")
    sys.exit(1)


def generate_synthetic_ppg_chunk(
    duration_seconds: float,
    sample_rate: float,
    heart_rate_hz: float = 1.2,
    noise_level: float = 0.05,
) -> np.ndarray:
    """Generate a synthetic PPG chunk for simulation."""
    length = int(duration_seconds * sample_rate)
    t = np.arange(length) / sample_rate
    signal = np.sin(2 * np.pi * heart_rate_hz * t)
    noise = noise_level * np.random.randn(length)
    return (signal + noise).astype(np.float32)


def run_demo() -> None:
    """Run the real-time streaming demo."""
    sample_rate = 100.0
    chunk_seconds = 1.0
    model = create_hybrid_classifier(compile=True)
    stream_handler = PPGStreamHandler(
        model=model,
        sample_rate=sample_rate,
        window_seconds=10.0,
        prediction_interval_seconds=5.0,
    )

    print("Starting real-time PPG streaming demo...")
    print("A prediction will be emitted every 5 seconds once the buffer is full.")

    total_duration_seconds = 30
    num_chunks = int(total_duration_seconds / chunk_seconds)

    for chunk_index in range(num_chunks):
        chunk = generate_synthetic_ppg_chunk(
            duration_seconds=chunk_seconds,
            sample_rate=sample_rate,
            heart_rate_hz=1.2,
            noise_level=0.08,
        )

        results = stream_handler.add_chunk(chunk)
        if results:
            for result in results:
                print("Prediction at {:.1f}s:".format(result["timestamp_seconds"]))
                print("  Probability:", f"{result['probability']:.4f}")
                print("  Label:", result["label"])
                print("  HRV Features:")
                for name, value in result["hrv_features"].items():
                    print(f"    {name}: {value:.4f}")
                print("---")

    print("Demo complete.")


if __name__ == "__main__":
    run_demo()
