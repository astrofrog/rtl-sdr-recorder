import time

import pytest

import rtlsdr_recorder.web.app
from rtlsdr_recorder.web import create_app


def wait_for_spectrum(client):
    for _ in range(300):
        response = client.get("/api/spectrum/plot")
        if response.status_code == 200:
            return response
        time.sleep(0.1)
    raise TimeoutError("No spectrum data appeared")


@pytest.fixture
def client(tmp_path):
    app = create_app(simulated=True, output_dir=str(tmp_path))
    with app.test_client() as client:
        yield client
    app.config["RECORDER"].disconnect()


def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Radio Recorder" in response.data


def test_status_initially_disconnected(client):
    status = client.get("/api/status").get_json()
    assert status == {"connected": False, "bias_tee_enabled": False,
                      "recording": False, "simulated": True, "output_dir": None}


def test_requires_connection(client):
    assert client.post("/api/disconnect").status_code == 400
    assert client.post("/api/bias-tee", json={"enabled": True}).status_code == 400
    assert client.post("/api/recording/start").status_code == 400
    assert client.post("/api/recording/stop").status_code == 400
    assert client.get("/api/spectrum/plot").status_code == 400


def test_full_recording_cycle(client, tmp_path):
    assert client.post("/api/connect").get_json()["success"]
    assert client.post("/api/connect").status_code == 400  # already connected
    assert client.get("/api/status").get_json()["connected"]

    assert client.post("/api/bias-tee", json={"enabled": True}).get_json()["success"]
    assert client.get("/api/status").get_json()["bias_tee_enabled"]

    assert client.post("/api/recording/start").get_json()["success"]
    assert client.post("/api/recording/start").status_code == 400  # already recording

    data = wait_for_spectrum(client).get_json()["data"]
    assert len(data["on"]) == len(data["frequencies"]) == 4096
    assert len(data["off"]) == len(data["off_frequencies"]) == 4096
    # The off-frequency axis is centered on the offset frequency, 4 MHz below
    assert data["off_frequencies"][2048] == pytest.approx(1416.0)
    assert data["frequencies"][2048] == pytest.approx(1420.0)
    # Difference and accumulated spectra are cleaned and downsampled like in
    # the analysis API
    for key in ["diff", "accumulated"]:
        assert len(data[key]) == len(data["reduced_frequencies"]) == 409
        assert any(value is None for value in data[key])  # masked DC channels
        assert any(value is not None for value in data[key])

    assert client.post("/api/recording/stop").get_json()["success"]
    assert client.post("/api/disconnect").get_json()["success"]
    assert not client.get("/api/status").get_json()["connected"]
    assert list(tmp_path.glob("*_on.npy"))


def test_settings(client, tmp_path):
    settings = client.get("/api/settings").get_json()
    valid_gains = settings.pop("valid_gains")
    assert valid_gains[0] == 0.0
    assert valid_gains[-1] == 49.6
    assert settings == {"center_freq": 1420e6, "offset_freq": 1416e6,
                        "sample_rate": 2.4e6, "gain": 49.6, "fft_len": 4096,
                        "downsample": 10, "output_dir": str(tmp_path)}

    # The output folder can be customized, and blank means the default
    # timestamped scheme
    assert client.post("/api/settings",
                       json={"output_dir": "custom"}).get_json()["success"]
    assert client.get("/api/settings").get_json()["output_dir"] == "custom"
    assert client.post("/api/settings",
                       json={"output_dir": "  "}).get_json()["success"]
    assert client.get("/api/settings").get_json()["output_dir"] == "auto"

    # Gains snap to the nearest value supported by the dongle
    assert client.post("/api/settings", json={"gain": 30}).get_json()["success"]
    assert client.get("/api/settings").get_json()["gain"] == 29.7

    response = client.post("/api/settings", json={"offset_freq": "1417 MHz",
                                                  "downsample": 8})
    assert response.get_json()["success"]
    settings = client.get("/api/settings").get_json()
    assert settings["offset_freq"] == 1417e6
    assert settings["downsample"] == 8
    assert settings["center_freq"] == 1420e6  # unchanged

    assert client.post("/api/settings", json={"center_freq": "12 km"}).status_code == 400
    assert client.post("/api/settings", json={"downsample": 0}).status_code == 400
    assert client.post("/api/settings", json={"gain": "loud"}).status_code == 400


def test_settings_reset_session_and_retune(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app(simulated=True)
    with app.test_client() as client:
        assert client.post("/api/connect").get_json()["success"]
        assert client.post("/api/recording/start").get_json()["success"]
        wait_for_spectrum(client)

        response = client.post("/api/settings", json={"center_freq": "1421 MHz"})
        assert response.get_json()["success"]

        # Applying settings stopped the recording and reset the session
        status = client.get("/api/status").get_json()
        assert not status["recording"]
        assert status["output_dir"] is None
        assert client.get("/api/spectrum/plot").status_code == 400

        # The connected (simulated) dongle was retuned in place
        assert app.config["RECORDER"].sdr.center_freq == 1421e6

        # Recording again goes to a new folder and uses the new frequency axis
        assert client.post("/api/recording/start").get_json()["success"]
        data = wait_for_spectrum(client).get_json()["data"]
        assert data["frequencies"][2048] == pytest.approx(1421.0)
        assert len(list(tmp_path.iterdir())) == 2
    app.config["RECORDER"].disconnect()


def test_settings_change_with_fixed_output_dir(client):
    # With an explicit output directory, recording after a settings change
    # is refused at start rather than mixing settings in one directory
    assert client.post("/api/connect").get_json()["success"]
    assert client.post("/api/recording/start").get_json()["success"]
    assert client.post("/api/recording/stop").get_json()["success"]
    assert client.post("/api/settings",
                       json={"center_freq": "1421 MHz"}).get_json()["success"]
    response = client.post("/api/recording/start")
    assert response.status_code == 400
    assert "different settings" in response.get_json()["message"]


def test_sessions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_names = iter(["session-1", "session-2"])
    monkeypatch.setattr(rtlsdr_recorder.web.app, "timestamped_output_dir",
                        lambda: next(session_names))

    app = create_app(simulated=True)
    with app.test_client() as client:
        assert client.post("/api/connect").get_json()["success"]

        # First session starts in a new folder
        assert client.post("/api/recording/start").get_json()["success"]
        assert client.get("/api/status").get_json()["output_dir"] == "session-1"
        wait_for_spectrum(client)

        # Pausing and restarting keeps the folder and the accumulated spectra
        assert client.post("/api/recording/stop").get_json()["success"]
        assert client.get("/api/status").get_json()["output_dir"] == "session-1"
        assert client.get("/api/spectrum/plot").status_code == 200
        assert client.post("/api/recording/start").get_json()["success"]
        assert client.get("/api/status").get_json()["output_dir"] == "session-1"

        # Resetting stops recording, clears the spectra, and switches folder
        assert client.post("/api/recording/reset").get_json()["success"]
        status = client.get("/api/status").get_json()
        assert not status["recording"]
        assert status["output_dir"] is None
        assert client.get("/api/spectrum/plot").status_code == 400
        assert client.post("/api/recording/start").get_json()["success"]
        assert client.get("/api/status").get_json()["output_dir"] == "session-2"
        wait_for_spectrum(client)

        # A custom folder name can be set through the settings
        assert client.post("/api/settings",
                           json={"output_dir": "custom"}).get_json()["success"]
        assert client.post("/api/recording/start").get_json()["success"]
        assert client.get("/api/status").get_json()["output_dir"] == "custom"
        wait_for_spectrum(client)

    app.config["RECORDER"].disconnect()
    assert list((tmp_path / "session-1").glob("*_on.npy"))
    assert list((tmp_path / "session-2").glob("*_on.npy"))
    assert list((tmp_path / "custom").glob("*_on.npy"))
