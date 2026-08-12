import json

import requests

import robot_bridge


def test_disabled_by_default_makes_no_network_call(monkeypatch):
    monkeypatch.setattr(robot_bridge.config, "ROBOT_REACTION_ENABLED", False)

    called = []
    monkeypatch.setattr(robot_bridge.requests, "post", lambda *a, **k: called.append(1))

    result = robot_bridge.send_reaction("HAPPINESS", 2)

    assert result == {"sent": False, "reason": "disabled"}
    assert called == []


def test_enabled_and_successful_post_returns_sent_true(monkeypatch):
    monkeypatch.setattr(robot_bridge.config, "ROBOT_REACTION_ENABLED", True)

    class FakeResponse:
        def raise_for_status(self):
            pass

    monkeypatch.setattr(robot_bridge.requests, "post", lambda *a, **k: FakeResponse())

    result = robot_bridge.send_reaction("ANGER", 3)

    assert result == {"sent": True, "reason": None}


def test_enabled_and_connection_error_does_not_raise(monkeypatch):
    monkeypatch.setattr(robot_bridge.config, "ROBOT_REACTION_ENABLED", True)

    def raise_connection_error(*a, **k):
        raise requests.exceptions.ConnectionError("Connection refused")

    monkeypatch.setattr(robot_bridge.requests, "post", raise_connection_error)

    result = robot_bridge.send_reaction("SADNESS", 1)

    assert result["sent"] is False
    assert "Connection refused" in result["reason"]


def test_enabled_sends_correct_payload_without_text(monkeypatch):
    monkeypatch.setattr(robot_bridge.config, "ROBOT_REACTION_ENABLED", True)

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(robot_bridge.requests, "post", fake_post)

    robot_bridge.send_reaction("INTEREST", 2)

    assert captured["url"] == robot_bridge.config.ROBOT_BRIDGE_URL
    assert json.loads(captured["data"]) == {"emotion": "INTEREST", "intensity": 2}
    assert captured["timeout"] == robot_bridge.config.ROBOT_BRIDGE_TIMEOUT
    assert "utf-8" in captured["headers"]["Content-Type"]


def test_enabled_sends_text_when_provided(monkeypatch):
    monkeypatch.setattr(robot_bridge.config, "ROBOT_REACTION_ENABLED", True)

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["data"] = data
        return FakeResponse()

    monkeypatch.setattr(robot_bridge.requests, "post", fake_post)

    robot_bridge.send_reaction("HAPPINESS", 3, text="תודה, זה נשמע כמו ניסיון רלוונטי ומוצק.")

    assert json.loads(captured["data"]) == {
        "emotion": "HAPPINESS",
        "intensity": 3,
        "text": "תודה, זה נשמע כמו ניסיון רלוונטי ומוצק.",
    }


def test_payload_sent_as_raw_utf8_not_unicode_escapes(monkeypatch):
    """RemoteControl.kt's regex-based parser can't decode \\uXXXX escapes —
    the Hebrew characters must appear literally in the request body."""
    monkeypatch.setattr(robot_bridge.config, "ROBOT_REACTION_ENABLED", True)

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

    def fake_post(url, data=None, headers=None, timeout=None):
        captured["data"] = data
        return FakeResponse()

    monkeypatch.setattr(robot_bridge.requests, "post", fake_post)

    robot_bridge.send_reaction("HAPPINESS", 2, text="תודה")

    assert b"\\u" not in captured["data"]
    assert "תודה".encode("utf-8") in captured["data"]
