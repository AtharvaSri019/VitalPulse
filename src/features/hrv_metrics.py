"""HRV (Heart Rate Variability) metrics extraction module.

This module provides functionality to extract time-domain and frequency-domain
HRV features from PPG signals and RR intervals.
"""

from typing import Tuple, Dict, Optional
import numpy as np
from scipy import signal


class HRVMetrics:
    """
    Extract Heart Rate Variability metrics from PPG signals and RR intervals.

    This class provides methods for calculating time-domain and frequency-domain
    HRV features commonly used in cardiac analysis and health assessment.

    Attributes:
        sample_rate (float): Sampling rate of the signal in Hz.
        lf_band (Tuple[float, float]): Low frequency band range in Hz (default: (0.04, 0.15)).
        hf_band (Tuple[float, float]): High frequency band range in Hz (default: (0.15, 0.4)).
    """

    def __init__(
        self,
        sample_rate: float,
        lf_band: Tuple[float, float] = (0.04, 0.15),
        hf_band: Tuple[float, float] = (0.15, 0.4),
    ) -> None:
        """
        Initialize the HRVMetrics calculator.

        Args:
            sample_rate: Sampling rate of the PPG signal in Hz.
            lf_band: Low frequency band range as (low, high) in Hz.
                    Default is (0.04, 0.15) Hz.
            hf_band: High frequency band range as (low, high) in Hz.
                    Default is (0.15, 0.4) Hz.

        Raises:
            ValueError: If sample_rate <= 0 or if frequency bands are invalid.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.sample_rate = sample_rate
        self.lf_band = lf_band
        self.hf_band = hf_band

        # Validate frequency bands
        if lf_band[0] >= lf_band[1]:
            raise ValueError("lf_band low must be less than lf_band high")
        if hf_band[0] >= hf_band[1]:
            raise ValueError("hf_band low must be less than hf_band high")

    def detect_peaks(
        self, signal_data: np.ndarray, prominence: float = 0.5
    ) -> np.ndarray:
        """
        Detect R-peaks or systolic peaks in the PPG signal.

        Uses scipy.signal.find_peaks with prominence-based detection.

        Args:
            signal_data: Input PPG signal as a 1D numpy array.
            prominence: Prominence threshold for peak detection. Default is 0.5.

        Returns:
            Array of indices where peaks are detected.

        Raises:
            TypeError: If signal_data is not a numpy array.
            ValueError: If signal_data is not 1D.
        """
        if not isinstance(signal_data, np.ndarray):
            raise TypeError("signal_data must be a numpy array")
        if signal_data.ndim != 1:
            raise ValueError("signal_data must be 1D")

        # Normalize signal for robust peak detection
        normalized_signal = (signal_data - np.mean(signal_data)) / np.std(signal_data)

        # Detect peaks with prominence constraint
        peaks, _ = signal.find_peaks(
            normalized_signal, prominence=prominence, distance=int(self.sample_rate * 0.4)
        )

        return peaks

    def extract_rr_intervals(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Extract RR intervals (time between consecutive beats) in milliseconds.

        Args:
            signal_data: Input PPG signal as a 1D numpy array.

        Returns:
            Array of RR intervals in milliseconds.

        Raises:
            ValueError: If fewer than 2 peaks are detected.
        """
        peaks = self.detect_peaks(signal_data)

        if len(peaks) < 2:
            raise ValueError("Not enough peaks detected to compute RR intervals")

        # Convert peak indices to time intervals (in milliseconds)
        peak_times = peaks / self.sample_rate * 1000  # Convert to ms
        rr_intervals = np.diff(peak_times)

        return rr_intervals

    def calculate_sdnn(self, rr_intervals: np.ndarray) -> float:
        """
        Calculate SDNN (Standard Deviation of NN intervals).

        SDNN represents the standard deviation of the time intervals between
        consecutive normal heartbeats. It reflects overall HRV.

        Args:
            rr_intervals: Array of RR intervals in milliseconds.

        Returns:
            SDNN value in milliseconds.

        Raises:
            ValueError: If rr_intervals has fewer than 2 samples.
            TypeError: If rr_intervals is not a numpy array.
        """
        if not isinstance(rr_intervals, np.ndarray):
            raise TypeError("rr_intervals must be a numpy array")
        if len(rr_intervals) < 2:
            raise ValueError("rr_intervals must have at least 2 samples")

        sdnn = np.std(rr_intervals, ddof=1)
        return float(sdnn)

    def calculate_rmssd(self, rr_intervals: np.ndarray) -> float:
        """
        Calculate RMSSD (Root Mean Square of Successive Differences).

        RMSSD = sqrt(1/(N-1) * sum((RR_i - RR_{i+1})^2))

        This metric reflects short-term HRV and is related to parasympathetic
        nervous system activity.

        Args:
            rr_intervals: Array of RR intervals in milliseconds.

        Returns:
            RMSSD value in milliseconds.

        Raises:
            ValueError: If rr_intervals has fewer than 2 samples.
            TypeError: If rr_intervals is not a numpy array.
        """
        if not isinstance(rr_intervals, np.ndarray):
            raise TypeError("rr_intervals must be a numpy array")
        if len(rr_intervals) < 2:
            raise ValueError("rr_intervals must have at least 2 samples")

        # Calculate successive differences
        successive_diffs = np.diff(rr_intervals)

        # Apply the RMSSD formula
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))

        return float(rmssd)

    def calculate_psd_welch(
        self, rr_intervals: np.ndarray, nperseg: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Power Spectral Density using Welch's method.

        Args:
            rr_intervals: Array of RR intervals in milliseconds.
            nperseg: Length of each segment for Welch's method.
                    If None, automatically set to length of signal.

        Returns:
            Tuple of (frequencies, power_spectral_density).

        Raises:
            ValueError: If rr_intervals has fewer than 4 samples.
            TypeError: If rr_intervals is not a numpy array.
        """
        if not isinstance(rr_intervals, np.ndarray):
            raise TypeError("rr_intervals must be a numpy array")
        if len(rr_intervals) < 4:
            raise ValueError("rr_intervals must have at least 4 samples for PSD estimation")

        # Set default nperseg if not provided
        if nperseg is None:
            nperseg = len(rr_intervals)

        # Ensure nperseg doesn't exceed signal length
        nperseg = min(nperseg, len(rr_intervals))

        # Interpolate RR intervals to uniform sampling
        # This is necessary because RR intervals are unevenly spaced
        time_original = np.cumsum(rr_intervals)
        time_original = np.insert(time_original, 0, 0)
        rr_interpolated = np.interp(
            np.linspace(0, time_original[-1], len(rr_intervals) * 4),
            time_original,
            np.concatenate([[rr_intervals[0]], rr_intervals]),
        )

        # Calculate PSD using Welch's method
        # Convert to Hz (RR intervals are in ms, so divide by 1000)
        sampling_freq = 1000.0 / np.mean(rr_intervals)  # Hz

        frequencies, psd = signal.welch(
            rr_interpolated,
            fs=sampling_freq,
            nperseg=nperseg,
            scaling="density",
        )

        return frequencies, psd

    def calculate_lf_hf_ratio(
        self, rr_intervals: np.ndarray, nperseg: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate LF/HF ratio and component powers using Welch's method.

        LF (Low Frequency, 0.04-0.15 Hz): Reflects sympathetic and parasympathetic modulation.
        HF (High Frequency, 0.15-0.4 Hz): Reflects parasympathetic modulation.
        LF/HF Ratio: Indicates sympathovagal balance.

        Args:
            rr_intervals: Array of RR intervals in milliseconds.
            nperseg: Length of each segment for Welch's method.

        Returns:
            Dictionary containing:
            - 'LF': Low frequency power
            - 'HF': High frequency power
            - 'LF_HF_ratio': Ratio of LF to HF power
            - 'LF_norm': Normalized LF power
            - 'HF_norm': Normalized HF power
            - 'VLF': Very low frequency power (optional)

        Raises:
            ValueError: If rr_intervals is invalid.
        """
        frequencies, psd = self.calculate_psd_welch(rr_intervals, nperseg)

        # Extract power in different bands
        lf_mask = (frequencies >= self.lf_band[0]) & (frequencies < self.lf_band[1])
        hf_mask = (frequencies >= self.hf_band[0]) & (frequencies < self.hf_band[1])

        lf_power = np.trapz(psd[lf_mask], frequencies[lf_mask])
        hf_power = np.trapz(psd[hf_mask], frequencies[hf_mask])

        # Avoid division by zero
        if hf_power == 0:
            lf_hf_ratio = 0.0
            hf_norm = 0.0
            lf_norm = 0.0
        else:
            lf_hf_ratio = lf_power / hf_power
            total_power = lf_power + hf_power
            hf_norm = (hf_power / total_power) * 100 if total_power > 0 else 0.0
            lf_norm = (lf_power / total_power) * 100 if total_power > 0 else 0.0

        return {
            "LF": float(lf_power),
            "HF": float(hf_power),
            "LF_HF_ratio": float(lf_hf_ratio),
            "LF_norm": float(lf_norm),
            "HF_norm": float(hf_norm),
        }

    def extract_all_hrv_features(
        self, signal_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract all HRV features (time and frequency domain) from a PPG signal.

        Args:
            signal_data: Input PPG signal (should be cleaned) as a 1D numpy array.

        Returns:
            Dictionary containing all HRV metrics:
            - Time-domain: SDNN, RMSSD, mean_hr
            - Frequency-domain: LF, HF, LF_HF_ratio, LF_norm, HF_norm

        Raises:
            ValueError: If signal is invalid or too short.
        """
        # Extract RR intervals
        rr_intervals = self.extract_rr_intervals(signal_data)

        # Calculate time-domain metrics
        sdnn = self.calculate_sdnn(rr_intervals)
        rmssd = self.calculate_rmssd(rr_intervals)
        mean_hr = 60000 / np.mean(rr_intervals)  # Convert to beats per minute

        # Calculate frequency-domain metrics
        freq_metrics = self.calculate_lf_hf_ratio(rr_intervals)

        # Combine all metrics
        all_metrics = {
            "SDNN_ms": sdnn,
            "RMSSD_ms": rmssd,
            "mean_HR_bpm": float(mean_hr),
            **freq_metrics,
        }

        return all_metrics

    def get_hrv_interpretation(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Provide clinical interpretation of HRV metrics.

        Args:
            metrics: Dictionary of HRV metrics from extract_all_hrv_features().

        Returns:
            Dictionary with interpretations of each metric.
        """
        interpretations = {}

        # SDNN interpretation (in ms)
        sdnn = metrics.get("SDNN_ms", 0)
        if sdnn < 50:
            interpretations["SDNN"] = "Very low - Possible health concern"
        elif sdnn < 100:
            interpretations["SDNN"] = "Low - Reduced HRV"
        elif sdnn < 200:
            interpretations["SDNN"] = "Normal"
        else:
            interpretations["SDNN"] = "High - Excellent HRV"

        # RMSSD interpretation (in ms)
        rmssd = metrics.get("RMSSD_ms", 0)
        if rmssd < 20:
            interpretations["RMSSD"] = "Low - Reduced parasympathetic activity"
        elif rmssd < 50:
            interpretations["RMSSD"] = "Moderate"
        else:
            interpretations["RMSSD"] = "High - Good parasympathetic tone"

        # LF/HF ratio interpretation
        lf_hf = metrics.get("LF_HF_ratio", 0)
        if lf_hf < 1.0:
            interpretations["LF_HF_ratio"] = "Parasympathetic dominance"
        elif lf_hf < 2.0:
            interpretations["LF_HF_ratio"] = "Balanced sympathovagal tone"
        else:
            interpretations["LF_HF_ratio"] = "Sympathetic dominance"

        return interpretations
