from guardrail import ConsistencyResult, check_consistency


def test_bad_category_with_high_valence_is_inconsistent():
    result = check_consistency("BAD", "high")
    assert result.consistent is False
    assert result.note == "BAD category paired with high valence"


def test_good_category_with_low_valence_is_inconsistent():
    result = check_consistency("GOOD", "low")
    assert result.consistent is False
    assert result.note == "GOOD category paired with low valence"


def test_neutral_category_with_high_valence_is_inconsistent():
    result = check_consistency("NEUTRAL", "high")
    assert result.consistent is False


def test_neutral_category_with_low_valence_is_inconsistent():
    result = check_consistency("NEUTRAL", "low")
    assert result.consistent is False


def test_good_category_with_high_valence_is_consistent():
    result = check_consistency("GOOD", "high")
    assert result.consistent is True
    assert result.note is None


def test_bad_category_with_low_valence_is_consistent():
    result = check_consistency("BAD", "low")
    assert result.consistent is True


def test_medium_valence_never_flags_regardless_of_category():
    for category in ("GOOD", "NEUTRAL", "BAD"):
        result = check_consistency(category, "medium")
        assert result.consistent is True, f"{category}+medium should be consistent"


def test_as_dict_shape():
    result = check_consistency("BAD", "high")
    d = result.as_dict()
    assert d == {"consistent": False, "note": "BAD category paired with high valence"}
