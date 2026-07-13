import re

import numpy as np
import pytest

from rtlsdr_recorder import recorder
from rtlsdr_recorder.recorder import (
    DEFAULT_CENTER_FREQ,
    DEFAULT_FFT_LEN,
    DEFAULT_GAIN,
    DEFAULT_OFFSET_FREQ,
    DEFAULT_SAMPLE_RATE,
    RecordingError,
    capture_spectrum_pair,
    frequency_array,
    load_settings,
    open_sdr,
    record,
    set_bias_tee,
)
from rtlsdr_recorder.simulate import SimulatedRtlSdr


def test_open_sdr_simulated():
    sdr = open_sdr(simulated=True)
    assert isinstance(sdr, SimulatedRtlSdr)
    assert sdr.center_freq == DEFAULT_CENTER_FREQ
    set_bias_tee(sdr, True)
    assert sdr.bias_tee
    sdr.close()


def test_frequency_array():
    freq = frequency_array()
    assert len(freq) == DEFAULT_FFT_LEN
    assert freq[0] == pytest.approx(1418.8, abs=0.01)
    assert freq[DEFAULT_FFT_LEN // 2] == pytest.approx(1420.0)


def test_capture_spectrum_pair():
    sdr = open_sdr(simulated=True)
    pair = capture_spectrum_pair(sdr)
    assert pair.spectrum_on.shape == (DEFAULT_FFT_LEN,)
    assert pair.spectrum_off.shape == (DEFAULT_FFT_LEN,)
    np.testing.assert_allclose(pair.spectrum_diff,
                               pair.spectrum_on - pair.spectrum_off)
    # The simulated HI line at 1420.405 MHz should show up in the difference
    freq = frequency_array()
    line = np.abs(freq - 1420.405) < 0.05
    assert pair.spectrum_diff[line].mean() > pair.spectrum_diff[~line].mean()


def test_record_saves_files(tmp_path):
    pairs = list(record(simulated=True, output_dir=str(tmp_path), count=2))
    assert len(pairs) == 2
    assert len(list(tmp_path.glob("*_on.npy"))) == 2
    assert len(list(tmp_path.glob("*_off.npy"))) == 2
    assert len(list(tmp_path.glob("*_diff.npy"))) == 2
    loaded = np.load(sorted(tmp_path.glob("*_on.npy"))[0])
    np.testing.assert_allclose(loaded, pairs[0].spectrum_on)


def test_record_saves_settings(tmp_path):
    list(record(simulated=True, output_dir=str(tmp_path), count=1,
                center_freq=1421e6))
    settings = load_settings(str(tmp_path))
    assert settings == {"center_freq": 1421e6,
                        "offset_freq": DEFAULT_OFFSET_FREQ,
                        "sample_rate": DEFAULT_SAMPLE_RATE,
                        "gain": DEFAULT_GAIN,
                        "fft_len": DEFAULT_FFT_LEN}
    # Recording again with the same settings is fine, different ones are not
    list(record(simulated=True, output_dir=str(tmp_path), count=1,
                center_freq=1421e6))
    with pytest.raises(ValueError, match="different settings"):
        list(record(simulated=True, output_dir=str(tmp_path), count=1))


def test_load_settings_missing_file(tmp_path):
    assert load_settings(str(tmp_path)) == {"center_freq": DEFAULT_CENTER_FREQ,
                                            "offset_freq": DEFAULT_OFFSET_FREQ,
                                            "sample_rate": DEFAULT_SAMPLE_RATE,
                                            "gain": DEFAULT_GAIN,
                                            "fft_len": DEFAULT_FFT_LEN}


def test_record_auto_output_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    list(record(simulated=True, count=1))
    directories = list(tmp_path.iterdir())
    assert len(directories) == 1
    assert re.fullmatch(r"raw-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}",
                        directories[0].name)
    assert len(list(directories[0].glob("*.npy"))) == 3


def test_record_output_dir_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    list(record(simulated=True, count=1, output_dir="galplane-<date>"))
    directories = list(tmp_path.iterdir())
    assert len(directories) == 1
    assert re.fullmatch(r"galplane-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}",
                        directories[0].name)


def test_record_without_saving(tmp_path):
    pairs = list(record(simulated=True, output_dir=None, count=1))
    assert len(pairs) == 1
    assert list(tmp_path.iterdir()) == []


def test_record_reconnects(monkeypatch):
    monkeypatch.setattr(recorder.time, "sleep", lambda seconds: None)

    sdr = open_sdr(simulated=True)
    original_read = sdr.read_samples
    failures = iter([True, False])

    def flaky_read(num_samples):
        if next(failures, False):
            raise IOError("device lost")
        return original_read(num_samples)

    sdr.read_samples = flaky_read
    reconnected = []
    pairs = list(record(sdr=sdr, output_dir=None, count=2,
                        on_reconnect=reconnected.append))
    assert len(pairs) == 2
    assert len(reconnected) == 1
    assert isinstance(reconnected[0], SimulatedRtlSdr)


def test_record_reconnect_failure(monkeypatch):
    monkeypatch.setattr(recorder.time, "sleep", lambda seconds: None)

    sdr = open_sdr(simulated=True)

    def broken_read(num_samples):
        raise IOError("device lost")

    sdr.read_samples = broken_read
    monkeypatch.setattr(recorder, "open_sdr",
                        lambda *args, **kwargs: (_ for _ in ()).throw(IOError("no device")))
    with pytest.raises(RecordingError, match="Could not reconnect"):
        list(record(sdr=sdr, output_dir=None, count=1))


def test_record_max_retries(monkeypatch):
    monkeypatch.setattr(recorder.time, "sleep", lambda seconds: None)

    def broken_read(num_samples):
        raise IOError("device lost")

    def make_broken_sdr(*args, **kwargs):
        sdr = SimulatedRtlSdr()
        sdr.read_samples = broken_read
        return sdr

    monkeypatch.setattr(recorder, "open_sdr", make_broken_sdr)
    with pytest.raises(RecordingError, match="Max retries"):
        list(record(sdr=make_broken_sdr(), output_dir=None, count=1, max_retries=2))
