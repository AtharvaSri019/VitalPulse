"""Real-time stream handler for PPG signal chunks."""

from typing import Optional, List, Dict, Tuple, Any

import numpy as np
from tensorflow.keras import Model

from src.preprocessing.signal_cleaner import PPGProcessor
from src.features.hrv_metrics import HRVMetrics


class PPGStreamHandler:
    """Handle streaming PPG input and generate periodic model predictions."""

    def __init__(
        self,
        model: Model,
        sample_rate: float = 100.0,
        window_seconds: float = 10.0,
        prediction_interval_seconds: float = 5.0,
        hrv_feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the streaming handler.

        Args:
            model: A compiled Keras hybrid model expecting [ppg_signal, hrv_features].
            sample_rate: Sampling rate of the PPG signal in Hz.
            window_seconds: Sliding window duration in seconds.
            prediction_interval_seconds: How often to issue predictions in seconds.
            hrv_feature_names: Optional list of HRV feature names.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if prediction_interval_seconds <= 0:
            raise ValueError("prediction_interval_seconds must be positive")
        if prediction_interval_seconds > window_seconds:
            raise ValueError("prediction_interval_seconds must be <= window_seconds")

        self.model = model
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.prediction_interval_seconds = prediction_interval_seconds
        self.buffer_length = int(window_seconds * sample_rate)
        self.prediction_interval = int(prediction_interval_seconds * sample_rate)
        self.buffer = np.zeros(self.buffer_length, dtype=np.float32)
        self.write_index = 0
        self.total_samples = 0
        self.samples_since_last_prediction = 0
        self.ppg_processor = PPGProcessor(sample_rate=sample_rate)
        self.hrv_extractor = HRVMetrics(sample_rate=sample_rate)
        self.hrv_feature_names = hrv_feature_names or [
            "SDNN_ms",
            "RMSSD_ms",
            "mean_HR_bpm",
            "LF",
            "HF",
            "LF_HF_ratio",
            "LF_norm",
            "HF_norm",
        ]

    def add_chunk(self, chunk: np.ndarray) -> List[Dict[str, Any]]:
        """
        Add a new chunk of PPG samples to the circular buffer.

        Args:
            chunk: 1D numpy array of raw PPG samples.

        Returns:
            A list of prediction results produced by this chunk.
        """
        if not isinstance(chunk, np.ndarray):
            raise TypeError("chunk must be a numpy array")
        if chunk.ndim != 1:
            raise ValueError("chunk must be a 1D array of PPG samples")

        results: List[Dict[str, Any]] = []
        chunk_length = len(chunk)

        for sample in chunk.astype(np.float32):
            self.buffer[self.write_index] = sample
            self.write_index = (self.write_index + 1) % self.buffer_length

        self.total_samples += chunk_length
        self.samples_since_last_prediction += chunk_length

        if self.total_samples >= self.buffer_length and self.samples_since_last_prediction >= self.prediction_interval:
            result = self._predict_from_buffer()
            results.append(result)
            self.samples_since_last_prediction %= self.prediction_interval

        return results

    def _get_current_window(self) -> np.ndarray:
        """Return the most recent sliding window of PPG samples."""
        if self.total_samples < self.buffer_length:
            return self.buffer[: self.write_index].copy()

        indices = (np.arange(self.buffer_length) + self.write_index) % self.buffer_length
        return self.buffer[indices].copy()

    def _predict_from_buffer(self) -> Dict[str, Any]:
        """Process the current buffer and generate a prediction."""
        window = self._get_current_window()
        cleaned_signal, _ = self.ppg_processor.process_signal(
            window, artifact_method="moving_average", apply_filter=True
        )

        hrv_metrics = self.hrv_extractor.extract_all_hrv_features(cleaned_signal)
        hrv_vector = np.array(
            [
                hrv_metrics[feature_name]
                for feature_name in self.hrv_feature_names
            ],
            dtype=np.float32,
        )

        ppg_input = cleaned_signal.reshape(1, -1, 1)
        hrv_input = hrv_vector.reshape(1, -1)
        probability = float(self.model.predict([ppg_input, hrv_input], verbose=0)[0, 0])
        label = int(probability >= 0.5)

        return {
            "timestamp_seconds": float(self.total_samples / self.sample_rate),
            "window_seconds": self.window_seconds,
            "probability": probability,
            "label": label,
            "hrv_features": hrv_metrics,
        }

    def reset(self) -> None:
        """Reset the internal buffer and prediction state."""
        self.buffer.fill(0.0)
        self.write_index = 0
        self.total_samples = 0
        self.samples_since_last_prediction = 0

    def is_ready(self) -> bool:
        """Return True when enough samples have been received to make predictions."""
        return self.total_samples >= self.buffer_length
