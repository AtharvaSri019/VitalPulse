"""Unit tests for PPG signal preprocessing module.

Tests the PPGProcessor class functionality including bandpass filtering,
artifact detection, and signal processing pipeline.
"""

import numpy as np
import pytest
from scipy import signal
from scipy.fftpack import fft, fftfreq

from src.preprocessing.signal_cleaner import PPGProcessor


class TestPPGProcessor:
    """Test suite for PPGProcessor class."""

    @pytest.fixture
    def sample_rate(self):
        """Sample rate for testing."""
        return 100.0

    @pytest.fixture
    def ppg_processor(self, sample_rate):
        """Create PPGProcessor instance for testing."""
        return PPGProcessor(
            sample_rate=sample_rate,
            lowcut=0.5,
            highcut=4.0,
            order=4
        )

    @pytest.fixture
    def synthetic_signal(self, sample_rate):
        """Generate synthetic PPG-like signal with noise."""
        # Create time array
        duration = 10.0  # 10 seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        # Generate base sine wave (simulating heart rate ~1 Hz)
        base_freq = 1.2  # 1.2 Hz (72 BPM)
        base_signal = np.sin(2 * np.pi * base_freq * t)

        # Add respiratory modulation (0.3 Hz)
        resp_freq = 0.3
        resp_modulation = 0.2 * np.sin(2 * np.pi * resp_freq * t)
        base_signal += resp_modulation

        # Add high-frequency noise (above 4 Hz)
        noise_freq1 = 10.0  # 10 Hz noise
        noise_freq2 = 20.0  # 20 Hz noise
        noise1 = 0.3 * np.sin(2 * np.pi * noise_freq1 * t)
        noise2 = 0.2 * np.sin(2 * np.pi * noise_freq2 * t)
        noisy_signal = base_signal + noise1 + noise2

        return t, base_signal, noisy_signal

    def test_initialization(self, sample_rate):
        """Test PPGProcessor initialization."""
        processor = PPGProcessor(sample_rate=sample_rate)

        assert processor.sample_rate == sample_rate
        assert processor.lowcut == 0.5
        assert processor.highcut == 4.0
        assert processor.order == 4

    def test_initialization_invalid_sample_rate(self):
        """Test initialization with invalid sample rate."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            PPGProcessor(sample_rate=0)

        with pytest.raises(ValueError, match="sample_rate must be positive"):
            PPGProcessor(sample_rate=-1)

    def test_initialization_invalid_cutoff_frequencies(self, sample_rate):
        """Test initialization with invalid cutoff frequencies."""
        with pytest.raises(ValueError, match="lowcut must be less than highcut"):
            PPGProcessor(sample_rate=sample_rate, lowcut=4.0, highcut=0.5)

    def test_bandpass_filter_design(self, ppg_processor):
        """Test that Butterworth filter is properly designed."""
        # Check that sos (second-order sections) is created
        assert hasattr(ppg_processor, 'sos')
        assert ppg_processor.sos.shape[0] == ppg_processor.order  # 4 sections for 4th order
        assert ppg_processor.sos.shape[1] == 6  # SOS format: [b0, b1, b2, a0, a1, a2]

    def test_apply_bandpass_filter_length_preservation(self, ppg_processor, synthetic_signal):
        """Test that bandpass filter preserves signal length."""
        _, _, noisy_signal = synthetic_signal

        filtered_signal = ppg_processor.apply_bandpass_filter(noisy_signal)

        assert len(filtered_signal) == len(noisy_signal)
        assert filtered_signal.shape == noisy_signal.shape

    def test_apply_bandpass_filter_noise_reduction(self, ppg_processor, synthetic_signal):
        """Test that bandpass filter effectively reduces high-frequency noise."""
        t, clean_signal, noisy_signal = synthetic_signal

        # Apply filter
        filtered_signal = ppg_processor.apply_bandpass_filter(noisy_signal)

        # Calculate power spectral density before and after filtering
        def calculate_psd(signal_data):
            """Calculate power spectral density."""
            n = len(signal_data)
            freqs = fftfreq(n, d=1/ppg_processor.sample_rate)
            fft_vals = fft(signal_data)
            psd = np.abs(fft_vals)**2 / n

            # Only positive frequencies
            pos_mask = freqs > 0
            return freqs[pos_mask], psd[pos_mask]

        # Get PSDs
        freqs_noisy, psd_noisy = calculate_psd(noisy_signal)
        freqs_filtered, psd_filtered = calculate_psd(filtered_signal)

        # Calculate power in different frequency bands
        low_band_mask = (freqs_noisy >= 0.5) & (freqs_noisy <= 4.0)  # Passband
        high_noise_mask = (freqs_noisy > 4.0) & (freqs_noisy <= 25.0)  # High freq noise

        # Power in passband should be preserved or enhanced
        passband_power_noisy = np.sum(psd_noisy[low_band_mask])
        passband_power_filtered = np.sum(psd_filtered[low_band_mask])

        # Power in high-frequency noise should be significantly reduced
        noise_power_noisy = np.sum(psd_noisy[high_noise_mask])
        noise_power_filtered = np.sum(psd_filtered[high_noise_mask])

        # Assert noise reduction (should be reduced by at least 50%)
        noise_reduction_ratio = noise_power_filtered / noise_power_noisy
        assert noise_reduction_ratio < 0.5, f"Noise reduction insufficient: {noise_reduction_ratio}"

        # Assert passband power is reasonably preserved (within 20% of original)
        passband_preservation_ratio = passband_power_filtered / passband_power_noisy
        assert 0.8 <= passband_preservation_ratio <= 1.2, \
            f"Passband power not preserved: {passband_preservation_ratio}"

    def test_apply_bandpass_filter_input_validation(self, ppg_processor):
        """Test input validation for bandpass filter."""
        # Test with non-numpy array
        with pytest.raises(TypeError, match="signal_data must be a numpy array"):
            ppg_processor.apply_bandpass_filter([1, 2, 3])

        # Test with 2D array
        with pytest.raises(ValueError, match="signal_data must be 1D"):
            ppg_processor.apply_bandpass_filter(np.array([[1, 2], [3, 4]]))

    def test_detect_motion_artifacts_moving_average(self, ppg_processor, synthetic_signal):
        """Test moving average motion artifact detection."""
        _, _, test_signal = synthetic_signal

        # Add artificial motion artifact (sudden spike)
        artifact_signal = test_signal.copy()
        artifact_signal[500:520] += 5.0  # Add large spike

        cleaned_signal, artifact_mask = ppg_processor.detect_motion_artifacts_moving_average(
            artifact_signal, window_size=10, threshold_std=2.0
        )

        # Check that artifacts were detected
        assert artifact_mask.sum() > 0, "No artifacts detected"

        # Check that signal length is preserved
        assert len(cleaned_signal) == len(artifact_signal)
        assert len(artifact_mask) == len(artifact_signal)

        # Check that artifact regions are interpolated
        artifact_indices = np.where(artifact_mask)[0]
        if len(artifact_indices) > 0:
            # Signal should be different in artifact regions
            assert not np.array_equal(
                cleaned_signal[artifact_indices],
                artifact_signal[artifact_indices]
            )

    def test_detect_motion_artifacts_moving_average_validation(self, ppg_processor, synthetic_signal):
        """Test input validation for moving average artifact detection."""
        _, _, test_signal = synthetic_signal

        # Test invalid window size
        with pytest.raises(ValueError, match="window_size must be positive"):
            ppg_processor.detect_motion_artifacts_moving_average(test_signal, window_size=0)

        # Test window size larger than signal
        with pytest.raises(ValueError, match="window_size cannot exceed signal length"):
            ppg_processor.detect_motion_artifacts_moving_average(test_signal, window_size=len(test_signal) + 1)

    def test_detect_motion_artifacts_wavelet(self, ppg_processor, synthetic_signal):
        """Test wavelet-based motion artifact detection."""
        _, _, test_signal = synthetic_signal

        # Add artificial motion artifact
        artifact_signal = test_signal.copy()
        artifact_signal[500:520] += 3.0

        try:
            cleaned_signal, artifact_mask = ppg_processor.detect_motion_artifacts_wavelet(
                artifact_signal, threshold_scale=1.5
            )

            # Check that signal length is preserved
            assert len(cleaned_signal) == len(artifact_signal)
            assert len(artifact_mask) == len(artifact_signal)

        except ImportError:
            # PyWavelets not installed, test should pass
            pytest.skip("PyWavelets not installed, skipping wavelet test")

    def test_detect_motion_artifacts_wavelet_validation(self, ppg_processor, synthetic_signal):
        """Test input validation for wavelet artifact detection."""
        _, _, test_signal = synthetic_signal

        # Test with too short signal
        short_signal = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Signal must have at least 4 samples"):
            ppg_processor.detect_motion_artifacts_wavelet(short_signal)

    def test_process_signal_complete_pipeline(self, ppg_processor, synthetic_signal):
        """Test complete signal processing pipeline."""
        _, _, test_signal = synthetic_signal

        # Test with moving average artifact detection
        cleaned_signal, artifact_mask = ppg_processor.process_signal(
            test_signal,
            artifact_method="moving_average",
            apply_filter=True
        )

        assert len(cleaned_signal) == len(test_signal)
        assert len(artifact_mask) == len(test_signal)

        # Test with wavelet artifact detection
        try:
            cleaned_signal, artifact_mask = ppg_processor.process_signal(
                test_signal,
                artifact_method="wavelet",
                apply_filter=True
            )

            assert len(cleaned_signal) == len(test_signal)
            assert len(artifact_mask) == len(test_signal)

        except ImportError:
            pytest.skip("PyWavelets not installed, skipping wavelet pipeline test")

    def test_process_signal_invalid_method(self, ppg_processor, synthetic_signal):
        """Test process_signal with invalid artifact method."""
        _, _, test_signal = synthetic_signal

        with pytest.raises(ValueError, match="artifact_method must be 'moving_average' or 'wavelet'"):
            ppg_processor.process_signal(test_signal, artifact_method="invalid")

    def test_get_filter_info(self, ppg_processor):
        """Test filter information retrieval."""
        info = ppg_processor.get_filter_info()

        required_keys = ["type", "order", "lowcut_hz", "highcut_hz", "sample_rate_hz", "nyquist_freq_hz"]
        for key in required_keys:
            assert key in info

        assert info["type"] == "Butterworth"
        assert info["order"] == ppg_processor.order
        assert info["lowcut_hz"] == ppg_processor.lowcut
        assert info["highcut_hz"] == ppg_processor.highcut
        assert info["sample_rate_hz"] == ppg_processor.sample_rate
        assert info["nyquist_freq_hz"] == ppg_processor.sample_rate / 2

    def test_synthetic_signal_properties(self, synthetic_signal):
        """Test properties of synthetic signal generation."""
        t, clean_signal, noisy_signal = synthetic_signal

        # Check time array
        assert len(t) == 1000  # 10 seconds * 100 Hz
        assert t[0] == 0.0
        assert t[-1] == 9.99  # Almost 10 seconds

        # Check signal lengths
        assert len(clean_signal) == len(t)
        assert len(noisy_signal) == len(t)

        # Check that noisy signal has higher variance than clean signal
        assert np.var(noisy_signal) > np.var(clean_signal)

    @pytest.mark.parametrize("lowcut,highcut", [
        (0.5, 4.0),
        (0.8, 3.5),
        (1.0, 5.0),
    ])
    def test_different_filter_parameters(self, sample_rate, lowcut, highcut):
        """Test PPGProcessor with different filter parameters."""
        processor = PPGProcessor(
            sample_rate=sample_rate,
            lowcut=lowcut,
            highcut=highcut,
            order=4
        )

        # Generate test signal
        t = np.linspace(0, 1, int(sample_rate), endpoint=False)
        test_signal = np.sin(2 * np.pi * 2.0 * t)  # 2 Hz signal

        filtered = processor.apply_bandpass_filter(test_signal)

        # Signal length should be preserved
        assert len(filtered) == len(test_signal)

        # Filter should be designed
        assert hasattr(processor, 'sos')

        # Check filter info
        info = processor.get_filter_info()
        assert info["lowcut_hz"] == lowcut
        assert info["highcut_hz"] == highcut
