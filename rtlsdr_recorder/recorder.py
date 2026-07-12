"""
Functional API for acquiring spectra with an RTL-SDR dongle.
"""

import json
import os
import time
from datetime import datetime
from typing import NamedTuple

import numpy as np

from rtlsdr_recorder.simulate import SimulatedRtlSdr

__all__ = [
    "DEFAULT_CENTER_FREQ",
    "DEFAULT_OFFSET_FREQ",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_GAIN",
    "DEFAULT_FFT_LEN",
    "RecordingError",
    "SpectrumPair",
    "open_sdr",
    "set_bias_tee",
    "frequency_array",
    "compute_averaged_spectrum",
    "capture_spectrum",
    "capture_spectrum_pair",
    "save_spectrum_pair",
    "save_settings",
    "load_settings",
    "timestamped_output_dir",
    "record",
]

DEFAULT_CENTER_FREQ = 1420e6
DEFAULT_OFFSET_FREQ = 1416e6  # 4 MHz below, no overlap with on-frequency band
DEFAULT_SAMPLE_RATE = 2.4e6
DEFAULT_GAIN = 49.6
DEFAULT_FFT_LEN = 4096

SETTINGS_FILENAME = "settings.json"


class RecordingError(RuntimeError):
    """Raised when recording fails and reconnection attempts are exhausted."""


class SpectrumPair(NamedTuple):
    """An on/off frequency-switched observation."""

    timestamp: str
    spectrum_on: np.ndarray
    spectrum_off: np.ndarray
    spectrum_diff: np.ndarray


def open_sdr(simulated=False, sample_rate=DEFAULT_SAMPLE_RATE,
             center_freq=DEFAULT_CENTER_FREQ, gain=DEFAULT_GAIN):
    """
    Open an RTL-SDR dongle (or a simulated one) and read a burn-in chunk.
    """
    if simulated:
        RtlSdr = SimulatedRtlSdr
    else:
        from rtlsdr import RtlSdr
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.center_freq = center_freq
    sdr.gain = gain
    sdr.read_samples(int(sample_rate))
    return sdr


def set_bias_tee(sdr, enabled):
    """Enable or disable the bias tee on an open dongle."""
    sdr.set_bias_tee(enabled)


def frequency_array(center_freq=DEFAULT_CENTER_FREQ, sample_rate=DEFAULT_SAMPLE_RATE,
                    fft_len=DEFAULT_FFT_LEN):
    """
    Return the frequencies (in MHz) of the channels in a spectrum.
    """
    return (np.fft.fftshift(np.fft.fftfreq(fft_len, 1 / sample_rate)) + center_freq) / 1e6


def compute_averaged_spectrum(samples, fft_len=DEFAULT_FFT_LEN):
    """
    Compute the power spectrum averaged over all complete FFT windows.
    """
    n_ffts = len(samples) // fft_len
    chunks = samples[:n_ffts * fft_len].reshape(n_ffts, fft_len)
    spectra = np.abs(np.fft.fftshift(np.fft.fft(chunks, axis=1), axes=1)) ** 2
    return np.mean(spectra, axis=0)


def capture_spectrum(sdr, center_freq=None, fft_len=DEFAULT_FFT_LEN, num_samples=None):
    """
    Capture an averaged power spectrum, optionally retuning first.
    """
    if center_freq is not None:
        sdr.center_freq = center_freq
    if num_samples is None:
        num_samples = int(sdr.sample_rate)
    samples = sdr.read_samples(num_samples)
    return compute_averaged_spectrum(samples, fft_len=fft_len)


def capture_spectrum_pair(sdr, center_freq=DEFAULT_CENTER_FREQ,
                          offset_freq=DEFAULT_OFFSET_FREQ, fft_len=DEFAULT_FFT_LEN):
    """
    Capture an on-frequency and an off-frequency spectrum and their difference.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    spectrum_on = capture_spectrum(sdr, center_freq, fft_len=fft_len)
    spectrum_off = capture_spectrum(sdr, offset_freq, fft_len=fft_len)
    return SpectrumPair(timestamp, spectrum_on, spectrum_off, spectrum_on - spectrum_off)


def save_spectrum_pair(pair, output_dir="raw"):
    """
    Save the on, off, and difference spectra of a pair to ``.npy`` files.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for suffix, spectrum in [("on", pair.spectrum_on), ("off", pair.spectrum_off),
                             ("diff", pair.spectrum_diff)]:
        path = os.path.join(output_dir, f"spectrum_{pair.timestamp}_{suffix}.npy")
        np.save(path, spectrum)
        paths.append(path)
    return paths


def save_settings(output_dir, settings):
    """
    Save recording settings alongside the recorded spectra. If a settings
    file already exists with different values, raise an error rather than
    mix data recorded with different settings in one directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, SETTINGS_FILENAME)
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
        if existing != settings:
            raise ValueError(f"{path} contains different settings from the "
                             "current recording; use a different output directory")
    else:
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
    return path


def load_settings(directory):
    """
    Return the recording settings for a directory of recorded spectra. Data
    recorded before settings files existed used the defaults, so if there is
    no settings file the defaults are returned.
    """
    settings = {
        "center_freq": DEFAULT_CENTER_FREQ,
        "offset_freq": DEFAULT_OFFSET_FREQ,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "gain": DEFAULT_GAIN,
        "fft_len": DEFAULT_FFT_LEN,
    }
    path = os.path.join(directory, SETTINGS_FILENAME)
    if os.path.exists(path):
        with open(path) as f:
            settings.update(json.load(f))
    return settings


def timestamped_output_dir():
    """Return an output directory name based on the current time."""
    return datetime.now().strftime("raw-%Y-%m-%d-%H-%M-%S")


def _reopen_sdr(sdr, simulated):
    try:
        sdr.close()
    except Exception:
        pass
    time.sleep(1)
    return open_sdr(simulated=simulated, sample_rate=sdr.sample_rate, gain=sdr.gain)


def record(sdr=None, output_dir="auto", center_freq=DEFAULT_CENTER_FREQ,
           offset_freq=DEFAULT_OFFSET_FREQ, fft_len=DEFAULT_FFT_LEN,
           sample_rate=DEFAULT_SAMPLE_RATE, gain=DEFAULT_GAIN, simulated=False,
           count=None, max_retries=3, on_reconnect=None):
    """
    Record on/off spectrum pairs, yielding a `SpectrumPair` after each capture.

    If ``sdr`` is None, a dongle is opened (and closed again at the end);
    otherwise the given one is used. Pairs are saved to ``output_dir``: the
    default "auto" uses a new ``raw-YYYY-MM-DD-HH-MM-SS`` directory named
    after the recording start time, and None disables saving. Recording
    continues until ``count`` pairs have been captured
    (forever if None) or the consumer stops iterating. On capture errors the
    dongle is reopened, calling ``on_reconnect(new_sdr)`` if given so that the
    caller can track the new handle; after ``max_retries`` consecutive
    failures, or if reconnecting fails, a `RecordingError` is raised.
    """
    if sdr is None:
        sdr = open_sdr(simulated=simulated, sample_rate=sample_rate,
                       center_freq=center_freq, gain=gain)
        owns_sdr = True
    else:
        simulated = isinstance(sdr, SimulatedRtlSdr)
        owns_sdr = False

    if output_dir == "auto":
        output_dir = timestamped_output_dir()

    retries = 0
    captured = 0
    try:
        if output_dir is not None:
            save_settings(output_dir, {"center_freq": center_freq,
                                       "offset_freq": offset_freq,
                                       "sample_rate": sdr.sample_rate,
                                       "gain": sdr.gain,
                                       "fft_len": fft_len})
        while count is None or captured < count:
            try:
                pair = capture_spectrum_pair(sdr, center_freq=center_freq,
                                             offset_freq=offset_freq, fft_len=fft_len)
            except Exception as exc:
                retries += 1
                if retries > max_retries:
                    raise RecordingError("Max retries exceeded while recording") from exc
                try:
                    sdr = _reopen_sdr(sdr, simulated)
                except Exception as reconnect_exc:
                    raise RecordingError("Could not reconnect to dongle") from reconnect_exc
                if on_reconnect is not None:
                    on_reconnect(sdr)
                continue
            retries = 0
            captured += 1
            if output_dir is not None:
                save_spectrum_pair(pair, output_dir)
            yield pair
    finally:
        if owns_sdr:
            sdr.close()
