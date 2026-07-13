"""
Simulated RTL-SDR dongle for testing without hardware.
"""

import numpy as np

__all__ = ["R820T_GAINS", "SimulatedRtlSdr"]

# Gains (in dB) supported by the common R820T/R820T2 tuner, also used as the
# fallback when no real dongle is connected to query
R820T_GAINS = [0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6,
               19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2,
               38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6]


class SimulatedRtlSdr:
    """Simulated RTL-SDR dongle with the same interface as `rtlsdr.RtlSdr`."""

    valid_gains_db = R820T_GAINS

    def __init__(self):
        self.sample_rate = 2.4e6
        self.center_freq = 1420e6
        self.gain = 49.6
        self.bias_tee = False

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        # Like the real dongle, snap to the nearest supported gain
        self._gain = min(self.valid_gains_db, key=lambda g: abs(g - value))

    def read_samples(self, num_samples):
        """
        Return simulated IQ samples with noise, bandpass shape, and a Gaussian HI line.
        The HI line is at 1420.405 MHz with 0.1 MHz FWHM.
        """
        # Frequency array (absolute frequencies)
        freqs = np.fft.fftfreq(num_samples, 1 / self.sample_rate) + self.center_freq

        # Bandpass shape: flat top with sharp roll-off at edges
        # Model as product of two tanh functions for left and right edges
        edge_steepness = 10e-6  # Controls roll-off sharpness (smaller = sharper)
        left_edge = self.center_freq - 0.9e6
        right_edge = self.center_freq + 0.9e6
        bandpass = 0.5 * (1 + np.tanh((freqs - left_edge) / (edge_steepness * self.center_freq)))
        bandpass *= 0.5 * (1 + np.tanh((right_edge - freqs) / (edge_steepness * self.center_freq)))

        # Add some ripple in the passband
        ripple = 1 + 0.05 * np.sin(2 * np.pi * (freqs - self.center_freq) / 0.3e6)
        bandpass *= ripple

        # Base noise spectrum shaped by bandpass
        noise = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        spectrum = noise * (0.1 + bandpass * 10)  # Floor + bandpass-shaped noise

        # HI line parameters
        hi_center = 1420.405e6  # Hydrogen line frequency
        hi_fwhm = 0.1e6  # 0.1 MHz width
        hi_sigma = hi_fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma

        # Add Gaussian HI line with random phase
        hi_amplitude = 3.0
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
