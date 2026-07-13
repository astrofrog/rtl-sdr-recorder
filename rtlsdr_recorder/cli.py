import click
from astropy import units as u

from rtlsdr_recorder.recorder import DEFAULT_GAIN, DEFAULT_FFT_LEN, parse_frequency


class FrequencyType(click.ParamType):
    """A frequency given with a unit (e.g. 1420.2MHz); plain numbers are Hz."""

    name = "frequency"

    def convert(self, value, param, ctx):
        try:
            return parse_frequency(value)
        except u.UnitConversionError:
            self.fail(f"{value!r} is not a frequency", param, ctx)
        except Exception:
            self.fail(f"{value!r} is not a valid frequency", param, ctx)


FREQUENCY = FrequencyType()


@click.command()
@click.option("--simulated", is_flag=True, help="Use a simulated RTL-SDR dongle.")
@click.option("--center-freq", type=FREQUENCY, default="1420MHz", show_default=True,
              help="On-frequency center frequency (plain numbers are Hz).")
@click.option("--offset-freq", type=FREQUENCY, default="1416MHz", show_default=True,
              help="Off-frequency center frequency (plain numbers are Hz).")
@click.option("--sample-rate", type=FREQUENCY, default="2.4MHz", show_default=True,
              help="Sample rate (plain numbers are Hz).")
@click.option("--gain", default=DEFAULT_GAIN, show_default=True,
              help="Tuner gain in dB.")
@click.option("--fft-len", default=DEFAULT_FFT_LEN, show_default=True,
              help="Number of channels in each spectrum.")
@click.option("--output-dir", default="auto",
              help="Directory in which to save recorded spectra; by default "
                   "each session uses a new raw-YYYY-MM-DD-HH-MM-SS directory.")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=5000, show_default=True)
def main(simulated, center_freq, offset_freq, sample_rate, gain, fft_len,
         output_dir, host, port):
    """Start the web app for recording HI spectra with an RTL-SDR dongle."""
    from rtlsdr_recorder.web import create_app

    app = create_app(simulated=simulated, center_freq=center_freq,
                     offset_freq=offset_freq, sample_rate=sample_rate, gain=gain,
                     fft_len=fft_len, output_dir=output_dir)
    try:
        app.run(host=host, port=port)
    finally:
        app.config["RECORDER"].disconnect()
