import numpy as np
from rtlsdr import RtlSdr
import threading
import os
from datetime import datetime
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Configuration
CENTER_FREQ = 1420.2e6
SAMPLE_RATE = 2.4e6
CHUNK_SAMPLES = int(SAMPLE_RATE)
FFT_LEN = 4096
OUTPUT_DIR = "raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pre-compute frequency array (in MHz)
FREQUENCIES = (np.fft.fftshift(np.fft.fftfreq(FFT_LEN, 1 / SAMPLE_RATE)) + CENTER_FREQ) / 1e6


class RadioRecorder:
    """Manages RTL-SDR dongle and recording operations"""

    def __init__(self):
        self.sdr = None
        self.connected = False
        self.bias_tee_enabled = False
        self.recording = False
        self.last_spectrum = None
        self.spectrum_lock = threading.Lock()
        self.stop_recording = False
        self.recording_thread = None
        self.median_spectrum = None
        self.spectrum_buffer = []

    def connect(self):
        """Initialize RTL-SDR dongle"""
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = SAMPLE_RATE
            self.sdr.center_freq = CENTER_FREQ
            self.sdr.gain = 49.6
            # Burn in
            self.sdr.read_samples(CHUNK_SAMPLES)
            self.connected = True
            return True, "Connected to RTL-SDR dongle"
        except Exception as e:
            return False, f"Error connecting to dongle: {str(e)}"

    def disconnect(self):
        """Close RTL-SDR dongle"""
        try:
            # Stop recording if active
            if self.recording:
                self.stop_recording_internal()

            if self.sdr is not None:
                self.sdr.close()
                self.sdr = None
            self.connected = False
            self.bias_tee_enabled = False
            return True, "Disconnected from RTL-SDR dongle"
        except Exception as e:
            return False, f"Error disconnecting: {str(e)}"

    def set_bias_tee(self, enabled):
        """Enable/disable bias tee"""
        try:
            if self.sdr is None:
                return False, "Dongle not connected"
            self.sdr.set_bias_tee(enabled)
            self.bias_tee_enabled = enabled
            return True, f"Bias tee {'enabled' if enabled else 'disabled'}"
        except Exception as e:
            return False, f"Error setting bias tee: {str(e)}"

    def start_recording(self):
        """Start recording spectra in background thread"""
        if self.recording:
            return False, "Already recording"
        if not self.connected:
            return False, "Dongle not connected"

        self.recording = True
        self.stop_recording = False
        self.spectrum_buffer = []
        self.median_spectrum = None
        self.recording_thread = threading.Thread(target=self._recording_thread, daemon=True)
        self.recording_thread.start()
        return True, "Recording started"

    def stop_recording_internal(self):
        """Internal method to stop recording"""
        self.stop_recording = True
        self.recording = False
        if self.recording_thread is not None:
            self.recording_thread.join(timeout=2)

    def _recording_thread(self):
        """Background thread for recording spectra"""
        while self.recording and not self.stop_recording:
            try:
                # Read samples and compute FFT
                samples = self.sdr.read_samples(CHUNK_SAMPLES)
                spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples, n=FFT_LEN))) ** 2

                # Store current spectrum (linear) and add to buffer for median
                with self.spectrum_lock:
                    self.last_spectrum = spectrum
                    self.spectrum_buffer.append(spectrum)
                    # Compute median spectrum from buffer (in linear space)
                    if len(self.spectrum_buffer) > 0:
                        self.median_spectrum = np.median(self.spectrum_buffer, axis=0)

                # Save to file with timestamp (raw linear spectrum)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = os.path.join(OUTPUT_DIR, f"spectrum_{timestamp}.npy")
                np.save(filename, spectrum)

                print(f"Saved spectrum: {filename}")

            except Exception as e:
                print(f"Error in recording thread: {str(e)}")
                self.recording = False
                break

    def get_spectrum_plot(self):
        """Return current and median spectrum data with frequency labels (converted to dB for display)"""
        with self.spectrum_lock:
            if self.last_spectrum is None:
                return None

            current = self.last_spectrum.copy()
            median = self.median_spectrum.copy() if self.median_spectrum is not None else None

        # Convert to dB for display
        current_db = 10 * np.log10(current + 1e-12)
        median_db = 10 * np.log10(median + 1e-12) if median is not None else None

        return {
            "current": current_db.tolist(),
            "median": median_db.tolist() if median_db is not None else None,
            "frequencies": FREQUENCIES.tolist(),
        }


# Global recorder instance
recorder = RadioRecorder()


@app.route("/")
def index():
    """Serve the main page"""
    return render_template("index.html")


@app.route("/api/status")
def get_status():
    """Get current status"""
    return jsonify(
        {
            "connected": recorder.connected,
            "bias_tee_enabled": recorder.bias_tee_enabled,
            "recording": recorder.recording,
        }
    )


@app.route("/api/connect", methods=["POST"])
def connect():
    """Connect to RTL-SDR dongle"""
    if recorder.connected:
        return jsonify({"success": False, "message": "Already connected"}), 400

    success, message = recorder.connect()
    return jsonify({"success": success, "message": message})


@app.route("/api/disconnect", methods=["POST"])
def disconnect():
    """Disconnect from RTL-SDR dongle"""
    if not recorder.connected:
        return jsonify({"success": False, "message": "Not connected"}), 400

    success, message = recorder.disconnect()
    return jsonify({"success": success, "message": message})


@app.route("/api/bias-tee", methods=["POST"])
def toggle_bias_tee():
    """Toggle bias tee"""
    if not recorder.connected:
        return jsonify({"success": False, "message": "Dongle not connected"}), 400

    data = request.get_json()
    enabled = data.get("enabled", False)

    success, message = recorder.set_bias_tee(enabled)
    return jsonify({"success": success, "message": message})


@app.route("/api/recording/start", methods=["POST"])
def start_recording():
    """Start recording"""
    success, message = recorder.start_recording()
    if not success:
        return jsonify({"success": False, "message": message}), 400
    return jsonify({"success": True, "message": message})


@app.route("/api/recording/stop", methods=["POST"])
def stop_recording():
    """Stop recording"""
    if not recorder.recording:
        return jsonify({"success": False, "message": "Not recording"}), 400

    recorder.stop_recording_internal()
    return jsonify({"success": True, "message": "Recording stopped"})


@app.route("/api/spectrum/plot")
def get_spectrum_plot():
    """Get current spectrum data as JSON"""
    spectrum_data = recorder.get_spectrum_plot()

    if spectrum_data is None:
        return jsonify({"success": False, "message": "No spectrum data available"}), 400

    return jsonify({"success": True, "data": spectrum_data})


if __name__ == "__main__":
    try:
        app.run(debug=True, host="127.0.0.1", port=5000)
    finally:
        recorder.disconnect()
