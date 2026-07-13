"""
Web app for controlling the dongle and monitoring recording, built on top of
the functional API in `rtlsdr_recorder.recorder`.
"""

import threading

import numpy as np
from flask import Flask, jsonify, render_template, request

from rtlsdr_recorder.analysis import reduce_spectrum_pairs
from rtlsdr_recorder.recorder import (
    DEFAULT_CENTER_FREQ,
    DEFAULT_OFFSET_FREQ,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_GAIN,
    DEFAULT_FFT_LEN,
    frequency_array,
    open_sdr,
    parse_frequency,
    record,
    recording_settings,
    save_settings,
    set_bias_tee,
    timestamped_output_dir,
)

__all__ = ["WebRecorder", "create_app"]


class WebRecorder:
    """Holds the dongle handle and recording state for the web app."""

    def __init__(self, simulated=False, center_freq=DEFAULT_CENTER_FREQ,
                 offset_freq=DEFAULT_OFFSET_FREQ, sample_rate=DEFAULT_SAMPLE_RATE,
                 gain=DEFAULT_GAIN, fft_len=DEFAULT_FFT_LEN, output_dir="auto",
                 downsample=10):
        self.simulated = simulated
        self.center_freq = center_freq
        self.offset_freq = offset_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.fft_len = fft_len
        self.output_dir = output_dir
        self.session_dir = None
        self.downsample = downsample
        self.frequencies = frequency_array(center_freq, sample_rate, fft_len)
        self.off_frequencies = frequency_array(offset_freq, sample_rate, fft_len)

        self.sdr = None
        self.connected = False
        self.bias_tee_enabled = False
        self.recording = False
        self.stop_requested = False
        self.recording_thread = None
        self.spectrum_lock = threading.Lock()
        self.last_pair = None
        self.on_buffer = []
        self.off_buffer = []

    def connect(self):
        try:
            self.sdr = open_sdr(simulated=self.simulated, sample_rate=self.sample_rate,
                                center_freq=self.center_freq, gain=self.gain)
            self.connected = True
            return True, "Connected to RTL-SDR dongle"
        except Exception as e:
            return False, f"Error connecting to dongle: {str(e)}"

    def disconnect(self):
        try:
            if self.recording:
                self.stop_recording()
            if self.sdr is not None:
                self.sdr.close()
                self.sdr = None
            self.connected = False
            self.bias_tee_enabled = False
            return True, "Disconnected from RTL-SDR dongle"
        except Exception as e:
            return False, f"Error disconnecting: {str(e)}"

    def set_bias_tee(self, enabled):
        try:
            if self.sdr is None:
                return False, "Dongle not connected"
            set_bias_tee(self.sdr, enabled)
            self.bias_tee_enabled = enabled
            return True, f"Bias tee {'enabled' if enabled else 'disabled'}"
        except Exception as e:
            return False, f"Error setting bias tee: {str(e)}"

    def start_recording(self):
        if self.recording:
            return False, "Already recording"
        if not self.connected:
            return False, "Dongle not connected"
        if self.session_dir is None:
            self.session_dir = (timestamped_output_dir() if self.output_dir == "auto"
                                else self.output_dir)
        try:
            # Fail here rather than in the recording thread if the directory
            # was already used with different settings
            save_settings(self.session_dir,
                          recording_settings(self.sdr, self.center_freq,
                                             self.offset_freq, self.fft_len))
        except ValueError as e:
            return False, str(e)
        self.recording = True
        self.stop_requested = False
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()
        return True, f"Recording to {self.session_dir}"

    def stop_recording(self):
        self.stop_requested = True
        self.recording = False
        if self.recording_thread is not None:
            self.recording_thread.join(timeout=2)

    def get_settings(self):
        return {
            "center_freq": self.center_freq,
            "offset_freq": self.offset_freq,
            "sample_rate": self.sample_rate,
            "gain": self.gain,
            "fft_len": self.fft_len,
            "downsample": self.downsample,
        }

    def apply_settings(self, settings):
        """Apply new settings, stopping recording and resetting the session
        since a directory's settings file must describe all its data."""
        try:
            center_freq = parse_frequency(settings.get("center_freq", self.center_freq))
            offset_freq = parse_frequency(settings.get("offset_freq", self.offset_freq))
            sample_rate = parse_frequency(settings.get("sample_rate", self.sample_rate))
            gain = float(settings.get("gain", self.gain))
            fft_len = int(settings.get("fft_len", self.fft_len))
            downsample = int(settings.get("downsample", self.downsample))
            if sample_rate <= 0 or fft_len < 2 or not 1 <= downsample <= fft_len:
                raise ValueError("sample rate, FFT length, and downsample "
                                 "factor must be positive (and the downsample "
                                 "factor no larger than the FFT length)")
        except Exception as e:
            return False, f"Invalid settings: {e}"

        self.reset_session()
        self.center_freq = center_freq
        self.offset_freq = offset_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.fft_len = fft_len
        self.downsample = downsample
        self.frequencies = frequency_array(center_freq, sample_rate, fft_len)
        self.off_frequencies = frequency_array(offset_freq, sample_rate, fft_len)

        if self.sdr is not None:
            try:
                self.sdr.sample_rate = sample_rate
                self.sdr.center_freq = center_freq
                self.sdr.gain = gain
            except Exception as e:
                return False, f"Settings applied but retuning the dongle failed: {e}"
        return True, "Settings applied; session reset"

    def reset_session(self):
        """Stop recording if active and start a fresh session: clear the
        accumulated spectra and use a new output directory next time."""
        if self.recording:
            self.stop_recording()
        with self.spectrum_lock:
            self.last_pair = None
            self.on_buffer = []
            self.off_buffer = []
        self.session_dir = None
        return True, "Session reset; recording will use a new output folder"

    def _recording_loop(self):
        def track_reconnect(sdr):
            self.sdr = sdr

        try:
            for pair in record(sdr=self.sdr, output_dir=self.session_dir,
                               center_freq=self.center_freq, offset_freq=self.offset_freq,
                               fft_len=self.fft_len, on_reconnect=track_reconnect):
                with self.spectrum_lock:
                    self.last_pair = pair
                    self.on_buffer.append(pair.spectrum_on)
                    self.off_buffer.append(pair.spectrum_off)
                print(f"Saved spectrum: {pair.timestamp}")
                if self.stop_requested:
                    break
        except Exception as e:
            print(f"Error in recording thread: {str(e)}")
            self.recording = False
            self.connected = False

    def get_plot_data(self):
        with self.spectrum_lock:
            if self.last_pair is None:
                return None
            pair = self.last_pair
            spectra_on = list(self.on_buffer)
            spectra_off = list(self.off_buffer)

        reduce_kwargs = dict(downsample=self.downsample,
                             center_freq=self.center_freq,
                             sample_rate=self.sample_rate)
        accumulated = reduce_spectrum_pairs(spectra_on, spectra_off, **reduce_kwargs)
        last = reduce_spectrum_pairs([pair.spectrum_on], [pair.spectrum_off],
                                     **reduce_kwargs)

        return {
            "on": (10 * np.log10(pair.spectrum_on + 1e-12)).tolist(),
            "off": (10 * np.log10(pair.spectrum_off + 1e-12)).tolist(),
            "diff": _nan_to_none(last.spectrum_diff),
            "accumulated": _nan_to_none(accumulated.spectrum_diff),
            "frequencies": self.frequencies.tolist(),
            "off_frequencies": self.off_frequencies.tolist(),
            "reduced_frequencies": accumulated.frequencies.tolist(),
        }


def _nan_to_none(values):
    """Channels masked in every spectrum average to NaN, which is not valid JSON."""
    return [None if np.isnan(value) else float(value) for value in values]


def create_app(**recorder_kwargs):
    """Create the Flask app; keyword arguments are passed to `WebRecorder`."""
    app = Flask(__name__)
    recorder = WebRecorder(**recorder_kwargs)
    app.config["RECORDER"] = recorder

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/status")
    def get_status():
        return jsonify({
            "connected": recorder.connected,
            "bias_tee_enabled": recorder.bias_tee_enabled,
            "recording": recorder.recording,
            "simulated": recorder.simulated,
            "output_dir": recorder.session_dir,
        })

    @app.route("/api/connect", methods=["POST"])
    def connect():
        if recorder.connected:
            return jsonify({"success": False, "message": "Already connected"}), 400
        success, message = recorder.connect()
        return jsonify({"success": success, "message": message})

    @app.route("/api/disconnect", methods=["POST"])
    def disconnect():
        if not recorder.connected:
            return jsonify({"success": False, "message": "Not connected"}), 400
        success, message = recorder.disconnect()
        return jsonify({"success": success, "message": message})

    @app.route("/api/bias-tee", methods=["POST"])
    def toggle_bias_tee():
        if not recorder.connected:
            return jsonify({"success": False, "message": "Dongle not connected"}), 400
        enabled = request.get_json().get("enabled", False)
        success, message = recorder.set_bias_tee(enabled)
        return jsonify({"success": success, "message": message})

    @app.route("/api/recording/start", methods=["POST"])
    def start_recording():
        success, message = recorder.start_recording()
        if not success:
            return jsonify({"success": False, "message": message}), 400
        return jsonify({"success": True, "message": message})

    @app.route("/api/recording/stop", methods=["POST"])
    def stop_recording():
        if not recorder.recording:
            return jsonify({"success": False, "message": "Not recording"}), 400
        recorder.stop_recording()
        return jsonify({"success": True, "message": "Recording paused"})

    @app.route("/api/recording/reset", methods=["POST"])
    def reset_recording():
        success, message = recorder.reset_session()
        return jsonify({"success": success, "message": message})

    @app.route("/api/settings")
    def get_settings():
        return jsonify(recorder.get_settings())

    @app.route("/api/settings", methods=["POST"])
    def set_settings():
        success, message = recorder.apply_settings(request.get_json() or {})
        if not success:
            return jsonify({"success": False, "message": message}), 400
        return jsonify({"success": True, "message": message})

    @app.route("/api/spectrum/plot")
    def get_spectrum_plot():
        data = recorder.get_plot_data()
        if data is None:
            return jsonify({"success": False, "message": "No spectrum data available"}), 400
        return jsonify({"success": True, "data": data})

    return app
