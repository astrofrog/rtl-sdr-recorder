import threading
import webbrowser

import click
from astropy import units as u

from rtlsdr_recorder.recorder import DEFAULT_GAIN, DEFAULT_FFT_LEN, parse_frequency


def _schedule_browser_open(url):
    # Open the browser shortly after starting so the server is listening by
    # the time it connects
    threading.Timer(1, webbrowser.open, args=[url]).start()


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
              help="Sample rate, equal to the observed bandwidth "
                   "(plain numbers are Hz).")
@click.option("--gain", default=DEFAULT_GAIN, show_default=True,
              help="Tuner gain in dB.")
@click.option("--fft-len", default=DEFAULT_FFT_LEN, show_default=True,
              help="Number of channels in each spectrum.")
@click.option("--output-dir", default="raw-<date>", show_default=True,
              help="Directory in which to save recorded spectra; <date> is "
                   "replaced by the time each recording session starts.")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=5000, show_default=True)
@click.option("--no-browser", is_flag=True,
              help="Do not open the web app in a browser.")
def main(simulated, center_freq, offset_freq, sample_rate, gain, fft_len,
         output_dir, host, port, no_browser):
    """Start the web app for recording HI spectra with an RTL-SDR dongle."""
    from rtlsdr_recorder.web import create_app

    app = create_app(simulated=simulated, center_freq=center_freq,
                     offset_freq=offset_freq, sample_rate=sample_rate, gain=gain,
                     fft_len=fft_len, output_dir=output_dir)
    if not no_browser:
        browser_host = "127.0.0.1" if host == "0.0.0.0" else host
        _schedule_browser_open(f"http://{browser_host}:{port}")
    try:
        app.run(host=host, port=port)
    finally:
        app.config["RECORDER"].disconnect()
