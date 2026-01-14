"""
Simulated RTL-SDR module for testing without hardware
"""
import numpy as np


class RtlSdr:
    """Simulated RTL-SDR dongle"""

    def __init__(self):
        self.sample_rate = 2.4e6
        self.center_freq = 1420e6
        self.gain = 49.6
        self.bias_tee = False

    def read_samples(self, num_samples):
        """
        Return simulated IQ samples with some noise and signal components
        Signals are placed at fixed absolute frequencies, so they remain
        at the same frequencies even when the center frequency is changed.
        """
        # Generate time array
        t = np.arange(num_samples) / self.sample_rate

        # Create complex IQ samples with:
        # 1. White Gaussian noise
        # 2. Some simulated signals at fixed absolute frequencies
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)

        # Define signals at fixed absolute frequencies
        # These remain constant regardless of center_freq
        abs_freq_1 = 1420e6 + 100e3
        abs_freq_2 = 1420e6 + 150e3
        abs_freq_3 = 1420e6 - 100e3

        # Convert absolute frequencies to baseband (relative to current center_freq)
        signal_freq_1 = abs_freq_1 - self.center_freq
        signal_freq_2 = abs_freq_2 - self.center_freq
        signal_freq_3 = abs_freq_3 - self.center_freq

        # Generate signals at baseband frequencies
        signal = 0.5 * np.exp(2j * np.pi * signal_freq_1 * t)
        signal += 0.3 * np.exp(2j * np.pi * signal_freq_2 * t)
        signal += 0.2 * np.exp(2j * np.pi * signal_freq_3 * t)

        # Combine noise and signal
        samples = noise + signal

        # Scale to simulate ADC output
        samples = (samples * 100).astype(np.complex64)

        return samples

    def set_bias_tee(self, enabled):
        """Enable/disable bias tee"""
        self.bias_tee = enabled

    def close(self):
        """Close the connection (no-op for simulated)"""
        pass
