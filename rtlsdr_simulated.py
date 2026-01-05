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
        """
        # Generate time array
        t = np.arange(num_samples) / self.sample_rate

        # Create complex IQ samples with:
        # 1. White Gaussian noise
        # 2. Some simulated signals at different frequencies
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)

        # Add a simulated signal near center frequency (relative offset)
        signal_freq = 100e3  # 100 kHz offset from center
        signal = 0.5 * np.exp(2j * np.pi * signal_freq * t)

        # Add some additional spectral features
        signal += 0.3 * np.exp(2j * np.pi * (signal_freq + 50e3) * t)
        signal += 0.2 * np.exp(2j * np.pi * (-signal_freq) * t)

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
