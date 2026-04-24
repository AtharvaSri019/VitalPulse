"""PPG signal processing and cleaning module.

This module provides functionality for cleaning photoplethysmography (PPG) signals,
including bandpass filtering and motion artifact detection.
"""

from typing import Tuple, Optional, Literal
import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft


class PPGProcessor:
    """
    Processor for cleaning and preprocessing PPG (photoplethysmography) signals.

    This class implements a fourth-order Butterworth bandpass filter and provides
    multiple methods for detecting and removing motion artifacts from PPG signals.

    Attributes:
        sample_rate (float): Sampling rate of the signal in Hz.
        lowcut (float): Low cutoff frequency in Hz (default: 0.5 Hz).
        highcut (float): High cutoff frequency in Hz (default: 4.0 Hz).
        order (int): Order of the Butterworth filter (default: 4).
    """

    def __init__(
        self,
        sample_rate: float,
        lowcut: float = 0.5,
        highcut: float = 4.0,
        order: int = 4,
    ) -> None:
        """
        Initialize the PPGProcessor.

        Args:
            sample_rate: Sampling rate of the signal in Hz.
            lowcut: Low cutoff frequency in Hz. Default is 0.5 Hz.
            highcut: High cutoff frequency in Hz. Default is 4.0 Hz.
            order: Order of the Butterworth filter. Default is 4.

        Raises:
            ValueError: If sample_rate <= 0, or if lowcut >= highcut.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if lowcut >= highcut:
            raise ValueError("lowcut must be less than highcut")

        self.sample_rate = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order

        # Design Butterworth bandpass filter
        self._design_filter()

    def _design_filter(self) -> None:
        """Design the Butterworth bandpass filter."""
        nyquist_freq = self.sample_rate / 2
        low = self.lowcut / nyquist_freq
        high = self.highcut / nyquist_freq

        if low <= 0 or high >= 1:
            raise ValueError(
                f"Cutoff frequencies must be between 0 and Nyquist frequency "
                f"({nyquist_freq} Hz)"
            )

        self.sos = signal.butter(
            self.order, [low, high], btype="band", output="sos"
        )

    def apply_bandpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Apply fourth-order Butterworth bandpass filter to the signal.

        Args:
            signal_data: Input signal as a 1D numpy array.

        Returns:
            Filtered signal as a 1D numpy array.

        Raises:
            TypeError: If signal_data is not a numpy array.
            ValueError: If signal_data is not 1D.
        """
        if not isinstance(signal_data, np.ndarray):
            raise TypeError("signal_data must be a numpy array")
        if signal_data.ndim != 1:
            raise ValueError("signal_data must be 1D")

        filtered_signal = signal.sosfilt(self.sos, signal_data)
        return filtered_signal

    def detect_motion_artifacts_moving_average(
        self,
        signal_data: np.ndarray,
        window_size: int = 5,
        threshold_std: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect motion artifacts using a moving average threshold approach.

        This method detects artifacts by identifying regions where the signal
        deviates significantly from a moving average baseline.

        Args:
            signal_data: Input signal as a 1D numpy array.
            window_size: Size of the moving average window. Default is 5.
            threshold_std: Number of standard deviations for artifact threshold.
                          Default is 3.0.

        Returns:
            Tuple containing:
            - cleaned_signal: Signal with detected artifacts removed/interpolated.
            - artifact_mask: Boolean array where True indicates artifact regions.

        Raises:
            ValueError: If window_size is too large or <= 0.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if window_size > len(signal_data):
            raise ValueError("window_size cannot exceed signal length")

        # Calculate moving average
        moving_avg = np.convolve(
            signal_data, np.ones(window_size) / window_size, mode="same"
        )

        # Calculate deviation from moving average
        deviation = np.abs(signal_data - moving_avg)

        # Calculate threshold based on standard deviation
        std_deviation = np.std(deviation)
        artifact_threshold = threshold_std * std_deviation

        # Detect artifacts
        artifact_mask = deviation > artifact_threshold

        # Clean signal by interpolating artifact regions
        cleaned_signal = signal_data.copy()
        if np.any(artifact_mask):
            # Linear interpolation across artifact regions
            valid_indices = np.where(~artifact_mask)[0]
            if len(valid_indices) > 1:
                cleaned_signal[artifact_mask] = np.interp(
                    np.where(artifact_mask)[0],
                    valid_indices,
                    signal_data[valid_indices],
                )

        return cleaned_signal, artifact_mask

    def detect_motion_artifacts_wavelet(
        self,
        signal_data: np.ndarray,
        wavelet: str = "db4",
        threshold_scale: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect motion artifacts using Wavelet Transform approach.

        This method uses wavelet decomposition to identify high-frequency
        motion artifacts in the PPG signal.

        Args:
            signal_data: Input signal as a 1D numpy array.
            wavelet: Wavelet type to use. Default is "db4" (Daubechies-4).
            threshold_scale: Scaling factor for artifact detection threshold.
                            Default is 2.0.

        Returns:
            Tuple containing:
            - cleaned_signal: Signal with detected artifacts removed/interpolated.
            - artifact_mask: Boolean array where True indicates artifact regions.

        Raises:
            ValueError: If signal length is too short for wavelet decomposition.
        """
        if len(signal_data) < 4:
            raise ValueError("Signal must have at least 4 samples for wavelet analysis")

        try:
            import pywt
        except ImportError:
            raise ImportError(
                "PyWavelets is required for wavelet-based artifact detection. "
                "Install it with: pip install PyWavelets"
            )

        # Perform 1-level wavelet decomposition
        coeffs = pywt.dwt(signal_data, wavelet)
        cA, cD = coeffs

        # Calculate threshold for detail coefficients
        sigma = np.median(np.abs(cD)) / 0.6745  # Robust standard deviation estimate
        threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal_data)))

        # Soft thresholding on detail coefficients
        cD_thresholded = pywt.threshold(cD, threshold, mode="soft")

        # Reconstruct signal
        cleaned_signal = pywt.idwt(cA, cD_thresholded, wavelet)

        # Create artifact mask by comparing original and cleaned signals
        # Artifacts show large differences between original and cleaned
        difference = np.abs(signal_data - cleaned_signal)
        artifact_threshold = np.mean(difference) + threshold_scale * np.std(difference)
        artifact_mask = difference > artifact_threshold

        return cleaned_signal, artifact_mask

    def process_signal(
        self,
        signal_data: np.ndarray,
        artifact_method: Literal["moving_average", "wavelet"] = "moving_average",
        apply_filter: bool = True,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete signal processing pipeline: bandpass filtering and artifact removal.

        Args:
            signal_data: Input signal as a 1D numpy array.
            artifact_method: Method for artifact detection.
                            Options: "moving_average" or "wavelet".
                            Default is "moving_average".
            apply_filter: Whether to apply bandpass filter. Default is True.
            **kwargs: Additional keyword arguments passed to the artifact detection method.

        Returns:
            Tuple containing:
            - cleaned_signal: Processed and cleaned signal.
            - artifact_mask: Boolean array indicating detected artifact regions.

        Raises:
            ValueError: If artifact_method is not recognized.
        """
        # Apply bandpass filter
        if apply_filter:
            signal_data = self.apply_bandpass_filter(signal_data)

        # Detect and remove artifacts
        if artifact_method == "moving_average":
            cleaned_signal, artifact_mask = (
                self.detect_motion_artifacts_moving_average(signal_data, **kwargs)
            )
        elif artifact_method == "wavelet":
            cleaned_signal, artifact_mask = self.detect_motion_artifacts_wavelet(
                signal_data, **kwargs
            )
        else:
            raise ValueError(
                f"artifact_method must be 'moving_average' or 'wavelet', "
                f"got '{artifact_method}'"
            )

        return cleaned_signal, artifact_mask

    def get_filter_info(self) -> dict:
        """
        Get information about the configured filter.

        Returns:
            Dictionary containing filter specifications.
        """
        return {
            "type": "Butterworth",
            "order": self.order,
            "lowcut_hz": self.lowcut,
            "highcut_hz": self.highcut,
            "sample_rate_hz": self.sample_rate,
            "nyquist_freq_hz": self.sample_rate / 2,
        }
