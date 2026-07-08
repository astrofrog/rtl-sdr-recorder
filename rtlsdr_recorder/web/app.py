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
    record,
    set_bias_tee,
)

__all__ = ["WebRecorder", "create_app"]


class WebRecorder:
    """Holds the dongle handle and recording state for the web app."""

    def __init__(self, simulated=False, center_freq=DEFAULT_CENTER_FREQ,
                 offset_freq=DEFAULT_OFFSET_FREQ, sample_rate=DEFAULT_SAMPLE_RATE,
                 gain=DEFAULT_GAIN, fft_len=DEFAULT_FFT_LEN, output_dir="raw",
                 downsample=10):
        self.simulated = simulated
        self.center_freq = center_freq
        self.offset_freq = offset_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.fft_len = fft_len
        self.output_dir = output_dir
        self.downsample = downsample
        self.frequencies = frequency_array(center_freq, sample_rate, fft_len)

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
        self.recording = True
        self.stop_requested = False
        self.on_buffer = []
        self.off_buffer = []
        self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self.recording_thread.start()
        return True, "Recording started"

    def stop_recording(self):
        self.stop_requested = True
        self.recording = False
        if self.recording_thread is not None:
            self.recording_thread.join(timeout=2)

    def _recording_loop(self):
        def track_reconnect(sdr):
            self.sdr = sdr

        try:
            for pair in record(sdr=self.sdr, output_dir=self.output_dir,
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
        return jsonify({"success": True, "message": "Recording stopped"})

    @app.route("/api/spectrum/plot")
    def get_spectrum_plot():
        data = recorder.get_plot_data()
        if data is None:
            return jsonify({"success": False, "message": "No spectrum data available"}), 400
        return jsonify({"success": True, "data": data})

    return app
