import numpy as np
import pytest
from astropy import units as u
from specutils import Spectrum

from rtlsdr_recorder.analysis import (
    average_spectra,
    clean_spectrum,
    downsample_spectrum,
    load_spectrum_pairs,
    plot_spectrum_pair,
    reduce_spectra,
    to_spectrum,
    write_fits,
)
from rtlsdr_recorder.recorder import DEFAULT_FFT_LEN, frequency_array, record


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw")
    list(record(simulated=True, output_dir=str(path), count=3))
    return path


def test_load_spectrum_pairs(raw_dir):
    spectra_on, spectra_off = load_spectrum_pairs(str(raw_dir))
    assert len(spectra_on) == len(spectra_off) == 3
    assert all(spec.shape == (DEFAULT_FFT_LEN,) for spec in spectra_on)


def test_load_spectrum_pairs_skips_unmatched(raw_dir):
    (raw_dir / "spectrum_99999999_000000_000_on.npy").write_bytes(b"")
    spectra_on, spectra_off = load_spectrum_pairs(str(raw_dir))
    assert len(spectra_on) == 3


def test_clean_spectrum():
    spectrum = np.ones(DEFAULT_FFT_LEN)
    spectrum[100] = 1e6
    cleaned = clean_spectrum(spectrum)
    assert cleaned.mask[100]
    # Center DC spike always masked over 39 channels
    center = DEFAULT_FFT_LEN // 2
    assert cleaned.mask[center - 19:center + 20].all()
    assert not cleaned.mask[center - 20]
    assert not cleaned.mask[center + 20]


def test_downsample_spectrum():
    spectrum = np.arange(100, dtype=float)
    down = downsample_spectrum(spectrum, 10)
    assert down.shape == (10,)
    assert down[0] == pytest.approx(4.5)


def test_average_spectra():
    spec1 = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
    spec2 = np.ma.array([3.0, 4.0, 5.0], mask=[False, False, False])
    average = average_spectra([spec1, spec2])
    np.testing.assert_allclose(average, [2.0, 4.0, 4.0])


def test_average_spectra_fully_masked_channel(recwarn):
    spec1 = np.ma.array([1.0, 2.0], mask=[False, True])
    spec2 = np.ma.array([3.0, 4.0], mask=[False, True])
    average = average_spectra([spec1, spec2])
    assert average[0] == pytest.approx(2.0)
    assert np.isnan(average[1])
    assert not any(issubclass(warning.category, RuntimeWarning)
                   for warning in recwarn.list)


def test_reduce_spectra(raw_dir):
    reduced = reduce_spectra(str(raw_dir))
    assert (len(reduced.frequencies) == len(reduced.spectrum_on)
            == len(reduced.spectrum_off) == len(reduced.spectrum_diff) == 409)
    np.testing.assert_allclose(reduced.spectrum_diff,
                               reduced.spectrum_on - reduced.spectrum_off)
    assert np.isnan(reduced.spectrum_diff[204])  # masked DC channels
    assert np.isfinite(reduced.spectrum_diff[100])
    # The simulated HI line should be visible in the difference spectrum
    line = np.abs(reduced.frequencies - 1420.405) < 0.05
    with np.errstate(invalid="ignore"):
        assert (np.nanmean(reduced.spectrum_diff[line])
                > np.nanmean(reduced.spectrum_diff[~line]))


def test_reduce_spectra_clip_difference(raw_dir):
    reduced = reduce_spectra(str(raw_dir), clip_difference=True)
    assert len(reduced.frequencies) == len(reduced.spectrum_diff) == 409
    assert np.isnan(reduced.spectrum_diff[204])
    assert np.isfinite(reduced.spectrum_diff[100])


def test_reduce_spectra_empty(tmp_path):
    with pytest.raises(ValueError, match="No matched on/off spectra"):
        reduce_spectra(str(tmp_path))


def test_reduce_spectra_uses_saved_settings(tmp_path):
    list(record(simulated=True, output_dir=str(tmp_path), count=1,
                center_freq=1421e6, offset_freq=1417e6))
    reduced = reduce_spectra(str(tmp_path))
    assert reduced.frequencies[len(reduced.frequencies) // 2] == pytest.approx(1421.0, abs=0.01)
    # Without the settings file, fall back to the default frequencies
    (tmp_path / "settings.json").unlink()
    reduced = reduce_spectra(str(tmp_path))
    assert reduced.frequencies[len(reduced.frequencies) // 2] == pytest.approx(1420.0, abs=0.01)


def test_plot_spectrum_pair(raw_dir):
    spectra_on, spectra_off = load_spectrum_pairs(str(raw_dir))
    fig = plot_spectrum_pair(spectra_on[0], spectra_off[0], lw=0.5)
    assert len(fig.axes) == 3


def test_plot_spectrum_pair_downsampled(raw_dir):
    spectra_on, spectra_off = load_spectrum_pairs(str(raw_dir))
    freq_down = downsample_spectrum(frequency_array(), 8)
    fig = plot_spectrum_pair(downsample_spectrum(spectra_on[0], 8),
                             downsample_spectrum(spectra_off[0], 8),
                             frequencies=freq_down)
    assert len(fig.axes) == 3


def test_to_spectrum():
    values = np.ma.array(np.ones(DEFAULT_FFT_LEN), mask=False)
    values.mask = np.zeros(DEFAULT_FFT_LEN, dtype=bool)
    values.mask[5] = True
    spectrum = to_spectrum(values)
    assert isinstance(spectrum, Spectrum)
    assert spectrum.spectral_axis.unit == u.MHz
    assert spectrum.flux.unit == u.count
    assert np.isnan(spectrum.flux.value[5])
    assert spectrum.spectral_axis.value[DEFAULT_FFT_LEN // 2] == pytest.approx(1420.0)


def test_write_fits_roundtrip(tmp_path):
    values = np.random.default_rng(42).random(128)
    filename = str(tmp_path / "spectrum.fits")
    write_fits(values, filename, frequencies=np.linspace(1419, 1421, 128))
    loaded = Spectrum.read(filename, format="tabular-fits")
    np.testing.assert_allclose(loaded.flux.value, values)
    assert loaded.spectral_axis.unit.physical_type == "frequency"
