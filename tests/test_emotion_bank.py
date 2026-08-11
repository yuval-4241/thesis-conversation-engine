import pytest

import config
from emotion_bank import EmotionBank

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
