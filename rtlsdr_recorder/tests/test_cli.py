from click.testing import CliRunner

import rtlsdr_recorder.web
from rtlsdr_recorder.cli import main


def test_help():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "--center-freq" in result.output


class FakeApp:
    def __init__(self):
        self.config = {"RECORDER": self}
        self.run_kwargs = None
        self.disconnected = False

    def run(self, **kwargs):
        self.run_kwargs = kwargs

    def disconnect(self):
        self.disconnected = True


def test_options(monkeypatch):
    created = {}

    def fake_create_app(**kwargs):
        created.update(kwargs)
        created["app"] = FakeApp()
        return created["app"]

    monkeypatch.setattr(rtlsdr_recorder.web, "create_app", fake_create_app)
    result = CliRunner().invoke(main, ["--simulated",
                                       "--center-freq", "1420.2MHz",
                                       "--sample-rate", "1e6",
                                       "--port", "8080"])
    assert result.exit_code == 0
    assert created["simulated"]
    assert created["center_freq"] == 1420.2e6
    assert created["offset_freq"] == 1416e6
    assert created["sample_rate"] == 1e6
    assert created["app"].run_kwargs == {"host": "127.0.0.1", "port": 8080}
    assert created["app"].disconnected


def test_frequency_with_units():
    result = CliRunner().invoke(main, ["--center-freq", "1.42something"])
    assert result.exit_code != 0
    assert "not a valid frequency" in result.output

    result = CliRunner().invoke(main, ["--center-freq", "1420km"])
    assert result.exit_code != 0
    assert "not a frequency" in result.output
