import pytest

import config
import emotion_bank
from emotion_bank import EmotionBank, pick_emotion_llm

VALID_EMOTIONS = {
    "HAPPINESS", "FEAR", "SADNESS", "ANGER", "SURPRISE",
    "DISGUST", "CONTEMPT", "LOVE", "SHAME", "INTEREST",
}


def test_get_returns_a_reaction_for_a_known_cell():
    bank = EmotionBank()
    reaction = bank.get("high", "high")
    assert "emotion" in reaction
    assert "intensity" in reaction


def test_get_raises_on_unknown_cell(tmp_path):
    empty_path = tmp_path / "empty_emotion_bank.json"
    empty_path.write_text("{}")
    bank = EmotionBank(path=str(empty_path))
    with pytest.raises(KeyError):
        bank.get("high", "high")


def test_all_nine_va_cells_are_present():
    bank = EmotionBank()
    for valence, arousal in config.va_cells():
        reaction = bank.get(valence, arousal)
        assert reaction["emotion"] in VALID_EMOTIONS, (
            f"{valence}_{arousal} has emotion {reaction['emotion']!r}, "
            f"not one of the values EmotionLibrary.getEmotion() implements"
        )
        assert reaction["intensity"] in (1, 2, 3), (
            f"{valence}_{arousal} has intensity {reaction['intensity']!r}, expected 1-3"
        )


def test_note_key_is_not_treated_as_a_va_cell():
    bank = EmotionBank()
    assert "_note" not in bank._data


def test_llm_selection_disabled_by_default_picks_first_candidate(monkeypatch):
    monkeypatch.setattr(config, "EMOTION_LLM_SELECTION_ENABLED", False)

    called = []
    monkeypatch.setattr(emotion_bank, "pick_emotion_llm", lambda *a, **k: called.append(1))

    bank = EmotionBank()
    reaction = bank.get("low", "medium", question="Q", answer_text="A", rationale="R")

    assert reaction == {"emotion": "SADNESS", "intensity": 2}  # candidates[0]
    assert called == []  # no LLM call made


def test_llm_selection_enabled_uses_pick_emotion_llm(monkeypatch):
    monkeypatch.setattr(config, "EMOTION_LLM_SELECTION_ENABLED", True)
    monkeypatch.setattr(emotion_bank, "pick_emotion_llm", lambda candidates, **k: "FEAR")

    bank = EmotionBank()
    reaction = bank.get("low", "medium", question="Q", answer_text="A", rationale="R")

    assert reaction == {"emotion": "FEAR", "intensity": 2}


def test_single_candidate_cell_never_calls_llm_even_when_enabled(monkeypatch):
    """high and medium valence each have only one valid candidate in the
    core-7-only design -- there's nothing to choose between, so the LLM
    must never be called for them regardless of the flag."""
    monkeypatch.setattr(config, "EMOTION_LLM_SELECTION_ENABLED", True)

    called = []
    monkeypatch.setattr(emotion_bank, "pick_emotion_llm", lambda *a, **k: called.append(1))

    bank = EmotionBank()
    high_reaction = bank.get("high", "medium")
    medium_reaction = bank.get("medium", "medium")

    assert high_reaction == {"emotion": "HAPPINESS", "intensity": 2}
    assert medium_reaction == {"emotion": "SURPRISE", "intensity": 2}
    assert called == []


def test_pick_emotion_llm_returns_llms_choice_when_valid(monkeypatch):
    monkeypatch.setattr(emotion_bank.llm_client, "call_llm", lambda **k: "FEAR")

    result = pick_emotion_llm(
        ["SADNESS", "FEAR"], question="Q", answer_text="A", rationale="R"
    )

    assert result == "FEAR"


def test_pick_emotion_llm_falls_back_to_first_candidate_on_bad_response(monkeypatch):
    """The LLM is asked to return exactly one of two words. If it returns
    anything else (extra text, hallucinated emotion, empty), fail closed
    to the deterministic default rather than propagating garbage into the
    VA-consistent contract."""
    monkeypatch.setattr(emotion_bank.llm_client, "call_llm", lambda **k: "I think SADNESS fits best")

    result = pick_emotion_llm(
        ["SADNESS", "FEAR"], question="Q", answer_text="A", rationale="R"
    )

    assert result == "SADNESS"  # candidates[0], not the garbled response
