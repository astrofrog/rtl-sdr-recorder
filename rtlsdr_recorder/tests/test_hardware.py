"""
Tests that require a real RTL-SDR dongle to be plugged in. Excluded by
default; run with ``pytest -m hardware``.
"""

import numpy as np
import pytest

from rtlsdr_recorder.recorder import (
    DEFAULT_FFT_LEN,
    capture_spectrum_pair,
    open_sdr,
    record,
)

pytestmark = pytest.mark.hardware


@pytest.fixture
def sdr():
    sdr = open_sdr()
    yield sdr
    sdr.close()


def test_open_and_capture(sdr):
    assert sdr.sample_rate == pytest.approx(2.4e6, rel=1e-3)
    pair = capture_spectrum_pair(sdr)
    assert pair.spectrum_on.shape == (DEFAULT_FFT_LEN,)
    assert np.all(np.isfinite(pair.spectrum_on))
    assert pair.spectrum_on.max() > 0


def test_record_to_disk(sdr, tmp_path):
    pairs = list(record(sdr=sdr, output_dir=str(tmp_path), count=1))
    assert len(pairs) == 1
    assert len(list(tmp_path.glob("*.npy"))) == 3
