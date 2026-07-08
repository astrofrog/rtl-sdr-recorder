"""
Analysis of recorded on/off spectrum pairs: RFI cleaning, downsampling,
averaging, plotting, and export to FITS.
"""

import glob
import os

import numpy as np
from astropy import units as u
from astropy.nddata import block_reduce
from astropy.stats import sigma_clip
from specutils import Spectrum

from rtlsdr_recorder.recorder import (
    DEFAULT_CENTER_FREQ,
    DEFAULT_OFFSET_FREQ,
    DEFAULT_SAMPLE_RATE,
    frequency_array,
)

__all__ = [
    "load_spectrum_pairs",
    "clean_spectrum",
    "downsample_spectrum",
    "average_spectra",
    "plot_spectrum_pair",
    "to_spectrum",
    "write_fits",
]


def load_spectrum_pairs(directory):
    """
    Load all matched on/off spectra from a directory of ``spectrum_*_{on,off}.npy``
    files, returning ``(spectra_on, spectra_off)`` as two lists of arrays.
    """
    spectra_on = []
    spectra_off = []
    for filename_on in sorted(glob.glob(os.path.join(directory, "*_on.npy"))):
        filename_off = filename_on.replace("_on.npy", "_off.npy")
        if os.path.exists(filename_off):
            spectra_on.append(np.load(filename_on))
            spectra_off.append(np.load(filename_off))
    return spectra_on, spectra_off


def clean_spectrum(spectrum, sigma=3, center_width=39):
    """
    Mask RFI spikes by sigma clipping, and always mask the ``center_width``
    channels around the DC spike at the center of the band. Returns a masked
    array.
    """
    cleaned = sigma_clip(spectrum, sigma=sigma)
    n = len(spectrum)
    half = center_width // 2
    cleaned.mask[n // 2 - half:n // 2 + half + 1] = True
    return cleaned


def downsample_spectrum(spectrum, factor):
    """
    Spectrally downsample by block-averaging ``factor`` channels together.
    Also works on frequency arrays and masked arrays.
    """
    return block_reduce(spectrum, factor, np.mean)


def average_spectra(spectra):
    """
    Average a list of (possibly masked) spectra channel by channel, ignoring
    masked values.
    """
    return np.nanmean(np.ma.array(spectra).filled(np.nan), axis=0)


def plot_spectrum_pair(spectrum_on, spectrum_off, frequencies=None,
                       frequencies_off=None, center_freq=DEFAULT_CENTER_FREQ,
                       offset_freq=DEFAULT_OFFSET_FREQ,
                       sample_rate=DEFAULT_SAMPLE_RATE, **kwargs):
    """
    Plot an on spectrum, an off spectrum, and their difference in three
    panels, returning the figure. Frequencies (in MHz) are computed from
    ``center_freq``/``offset_freq`` unless given explicitly (e.g. for
    downsampled spectra). Extra keyword arguments are passed to ``plot``.
    """
    import matplotlib.pyplot as plt

    if frequencies is None:
        frequencies = frequency_array(center_freq, sample_rate, len(spectrum_on))
    if frequencies_off is None:
        frequencies_off = frequencies - (center_freq - offset_freq) / 1e6

    fig, axes = plt.subplots(3, 1, sharex=False)
    for ax, freq, spec, label in [
        (axes[0], frequencies, spectrum_on, "On"),
        (axes[1], frequencies_off, spectrum_off, "Off"),
        (axes[2], frequencies, spectrum_on - spectrum_off, "Difference"),
    ]:
        ax.plot(freq, spec, **kwargs)
        ax.set_ylabel(label)
        if ax is not axes[-1]:
            ax.set_xticks([])
    axes[-1].set_xlabel("Frequency (MHz)")
    return fig


def to_spectrum(values, frequencies=None, center_freq=DEFAULT_CENTER_FREQ,
                sample_rate=DEFAULT_SAMPLE_RATE, unit=u.count):
    """
    Convert a spectrum to a `specutils.Spectrum` with a spectral axis in MHz.
    Masked channels become NaN. Frequencies are computed from ``center_freq``
    unless given explicitly. The flux unit defaults to counts since the power
    values are uncalibrated.
    """
    if frequencies is None:
        frequencies = frequency_array(center_freq, sample_rate, len(values))
    flux = np.ma.filled(np.ma.asarray(values).astype(float), np.nan)
    return Spectrum(flux=flux * unit,
                    spectral_axis=np.asarray(frequencies) * u.MHz)


def write_fits(values, filename, frequencies=None, overwrite=False, **kwargs):
    """
    Export a spectrum to a FITS file (tabular format). Accepts the same
    keyword arguments as `to_spectrum`.
    """
    spectrum = to_spectrum(values, frequencies=frequencies, **kwargs)
    spectrum.write(filename, format="tabular-fits", overwrite=overwrite)
    return spectrum
