import numpy as np
import argparse
import threading
import os
from datetime import datetime
from flask import Flask, render_template, jsonify, request

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--simulated', action='store_true', help='Use simulated RTL-SDR dongle')
args = parser.parse_args()

if args.simulated:
    from rtlsdr_simulated import RtlSdr
else:
    from rtlsdr import RtlSdr

app = Flask(__name__)

# Configuration
CENTER_FREQ = 1420e6
OFFSET_FREQ = 1416e6  # 4 MHz below, no overlap with on-frequency band
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
        self.spectrum_lock = threading.Lock()
        self.stop_recording = False
        self.recording_thread = None
        # Spectrum data
        self.last_on = None
        self.last_off = None
        self.last_diff = None
        self.accumulated_diff = None
        self.diff_buffer = []

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
        self.diff_buffer = []
        self.accumulated_diff = None
        self.recording_thread = threading.Thread(target=self._recording_thread, daemon=True)
        self.recording_thread.start()
        return True, "Recording started"

    def stop_recording_internal(self):
        """Internal method to stop recording"""
        self.stop_recording = True
        self.recording = False
        if self.recording_thread is not None:
            self.recording_thread.join(timeout=2)

    def _compute_averaged_spectrum(self, samples):
        """Compute power spectrum averaged over all complete FFT windows."""
        n_ffts = len(samples) // FFT_LEN
        trimmed = samples[:n_ffts * FFT_LEN]
        chunks = trimmed.reshape(n_ffts, FFT_LEN)
        spectra = np.abs(np.fft.fftshift(np.fft.fft(chunks, axis=1), axes=1)) ** 2
        return np.mean(spectra, axis=0)

    def _save_spectrum(self, spectrum, timestamp, suffix):
        """Save spectrum to file."""
        filename = os.path.join(OUTPUT_DIR, f"spectrum_{timestamp}_{suffix}.npy")
        np.save(filename, spectrum)
        return filename

    def _reconnect(self):
        """Attempt to reconnect to the dongle"""
        print("Attempting to reconnect...")
        try:
            if self.sdr is not None:
                try:
                    self.sdr.close()
                except:
                    pass
                self.sdr = None

            import time
            time.sleep(1)

            self.sdr = RtlSdr()
            self.sdr.sample_rate = SAMPLE_RATE
            self.sdr.center_freq = CENTER_FREQ
            self.sdr.gain = 49.6
            self.sdr.read_samples(CHUNK_SAMPLES)  # Burn in
            print("Reconnected successfully")
            return True
        except Exception as e:
            print(f"Reconnection failed: {str(e)}")
            return False

    def _recording_thread(self):
        """Background thread for recording spectra"""
        retry_count = 0
        max_retries = 3

        while self.recording and not self.stop_recording:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                # Capture on-frequency spectrum
                self.sdr.center_freq = CENTER_FREQ
                samples = self.sdr.read_samples(CHUNK_SAMPLES)
                spectrum_on = self._compute_averaged_spectrum(samples)
                self._save_spectrum(spectrum_on, timestamp, "on")

                # Capture off-frequency spectrum
                self.sdr.center_freq = OFFSET_FREQ
                samples = self.sdr.read_samples(CHUNK_SAMPLES)
                spectrum_off = self._compute_averaged_spectrum(samples)
                self._save_spectrum(spectrum_off, timestamp, "off")

                # Compute and save difference
                spectrum_diff = spectrum_on - spectrum_off
                self._save_spectrum(spectrum_diff, timestamp, "diff")

                # Store spectra and update accumulated difference
                with self.spectrum_lock:
                    self.last_on = spectrum_on
                    self.last_off = spectrum_off
                    self.last_diff = spectrum_diff
                    self.diff_buffer.append(spectrum_diff)
                    self.accumulated_diff = np.median(self.diff_buffer, axis=0)

                print(f"Saved spectrum: {timestamp}")
                retry_count = 0  # Reset on success

            except Exception as e:
                print(f"Error in recording thread: {str(e)}")
                retry_count += 1

                if retry_count > max_retries:
                    print("Max retries exceeded, stopping recording")
                    self.recording = False
                    self.connected = False
                    break

                if self._reconnect():
                    continue
                else:
                    self.recording = False
                    self.connected = False
                    break

    def get_spectrum_plot(self):
        """Return spectrum data for all four plots"""
        with self.spectrum_lock:
            if self.last_on is None:
                return None

            on = self.last_on.copy()
            off = self.last_off.copy()
            diff = self.last_diff.copy()
            accumulated = self.accumulated_diff.copy() if self.accumulated_diff is not None else None

        # Convert on/off to dB for display
        on_db = 10 * np.log10(on + 1e-12)
        off_db = 10 * np.log10(off + 1e-12)

        return {
            "on": on_db.tolist(),
            "off": off_db.tolist(),
            "diff": diff.tolist(),
            "accumulated": accumulated.tolist() if accumulated is not None else None,
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
            "simulated": args.simulated,
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
        app.run(debug=False, host="127.0.0.1", port=5000)
    finally:
        recorder.disconnect()
