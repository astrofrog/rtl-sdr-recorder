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

    app.config["RECORDER"].disconnect()
    assert list((tmp_path / "session-1").glob("*_on.npy"))
    assert list((tmp_path / "session-2").glob("*_on.npy"))
