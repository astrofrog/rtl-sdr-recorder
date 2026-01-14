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
        Return simulated IQ samples with noise and a Gaussian HI line.
        The HI line is at 1420.405 MHz with 0.1 MHz FWHM.
        """
        # Frequency array (absolute frequencies)
        freqs = np.fft.fftfreq(num_samples, 1 / self.sample_rate) + self.center_freq

        # Flat noise spectrum with random phase
        spectrum = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)

        # HI line parameters
        hi_center = 1420.405e6  # Hydrogen line frequency
        hi_fwhm = 0.1e6  # 0.1 MHz width
        hi_sigma = hi_fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

        # Add Gaussian HI line with random phase
        hi_amplitude = 5.0
        gaussian = np.exp(-(freqs - hi_center) ** 2 / (2 * hi_sigma ** 2))
        spectrum += hi_amplitude * gaussian * np.exp(2j * np.pi * np.random.rand(num_samples))

        # IFFT to time domain
        samples = np.fft.ifft(spectrum)
        samples = (samples * 100).astype(np.complex64)

        return samples

    def set_bias_tee(self, enabled):
        """Enable/disable bias tee"""
        self.bias_tee = enabled

    def close(self):
        """Close the connection (no-op for simulated)"""
        pass
