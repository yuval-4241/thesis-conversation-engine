import config
import emotion_classifier
from emotion_classifier import classify_emotion, derive_valence, derive_arousal, CORE_EMOTIONS


def test_classify_emotion_parses_valid_llm_response(monkeypatch):
    monkeypatch.setattr(
        emotion_classifier.llm_client, "call_llm",
        lambda **k: '{"emotion": "FEAR", "intensity": 2}'
    )
    result = classify_emotion("Tell me about a challenge", "I was scared I'd miss the deadline.")
    assert result == {"emotion": "FEAR", "intensity": 2}


def test_classify_emotion_strips_code_fences(monkeypatch):
    monkeypatch.setattr(
        emotion_classifier.llm_client, "call_llm",
        lambda **k: '```json\n{"emotion": "HAPPINESS", "intensity": 3}\n```'
    )
    result = classify_emotion("Q", "A")
    assert result == {"emotion": "HAPPINESS", "intensity": 3}


def test_classify_emotion_discards_cue_field_from_return_value(monkeypatch):
    """The chain-of-thought "cue" field is a reasoning scratchpad the model
    fills in before naming the emotion -- it must never appear in
    classify_emotion()'s return value, only emotion/intensity do."""
    monkeypatch.setattr(
        emotion_classifier.llm_client, "call_llm",
        lambda **k: '{"cue": "no complaints, flat factual tone", "emotion": "SURPRISE", "intensity": 1}'
    )
    result = classify_emotion("Tell me about yourself", "I have 5 years of experience in QA.")
    assert result == {"emotion": "SURPRISE", "intensity": 1}
    assert "cue" not in result


def test_classify_emotion_uses_higher_temperature_for_variety(monkeypatch):
    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return '{"cue": "none", "emotion": "HAPPINESS", "intensity": 2}'

    monkeypatch.setattr(emotion_classifier.llm_client, "call_llm", fake_call_llm)
    classify_emotion("Q", "A")

    assert captured["temperature"] == 0.8


def test_classify_emotion_falls_back_on_non_json(monkeypatch):
    monkeypatch.setattr(emotion_classifier.llm_client, "call_llm", lambda **k: "not json at all")
    result = classify_emotion("Q", "A")
    assert result == {"emotion": "SURPRISE", "intensity": 2}


def test_classify_emotion_falls_back_on_unknown_emotion(monkeypatch):
    monkeypatch.setattr(
        emotion_classifier.llm_client, "call_llm",
        lambda **k: '{"emotion": "LOVE", "intensity": 2}'
    )
    result = classify_emotion("Q", "A")
    assert result == {"emotion": "SURPRISE", "intensity": 2}


def test_classify_emotion_falls_back_on_out_of_range_intensity(monkeypatch):
    monkeypatch.setattr(
        emotion_classifier.llm_client, "call_llm",
        lambda **k: '{"emotion": "SADNESS", "intensity": 7}'
    )
    result = classify_emotion("Q", "A")
    assert result == {"emotion": "SURPRISE", "intensity": 2}


def test_classify_emotion_falls_back_on_missing_keys(monkeypatch):
    monkeypatch.setattr(
        emotion_classifier.llm_client, "call_llm",
        lambda **k: '{"emotion": "SADNESS"}'
    )
    result = classify_emotion("Q", "A")
    assert result == {"emotion": "SURPRISE", "intensity": 2}


def test_every_core_emotion_maps_to_a_valid_valence():
    for emotion in CORE_EMOTIONS:
        assert derive_valence(emotion) in config.VALENCE_LEVELS


def test_derive_arousal_covers_all_three_intensities():
    assert derive_arousal(1) == "low"
    assert derive_arousal(2) == "medium"
    assert derive_arousal(3) == "high"


def test_happiness_maps_to_high_valence():
    assert derive_valence("HAPPINESS") == "high"


def test_surprise_maps_to_medium_valence():
    assert derive_valence("SURPRISE") == "medium"


def test_negative_emotions_map_to_low_valence():
    for emotion in ("ANGER", "CONTEMPT", "DISGUST", "FEAR", "SADNESS"):
        assert derive_valence(emotion) == "low"
