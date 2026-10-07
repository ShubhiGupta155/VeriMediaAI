import pytest

from modules.lipsync.sync_score import (
    calculate_sync_score,
)


def test_good_sync_score():
    alignment_result = {
        "alignment_score": 1.0,
        "matched_intervals": 5,
        "mismatch_intervals": []
    }

    result = calculate_sync_score(
        alignment_result
    )

    assert result["sync_score"] == 100.0
    assert result["sync_status"] == "good"


def test_moderate_sync_score():
    alignment_result = {
        "alignment_score": 0.6,
        "matched_intervals": 3,
        "mismatch_intervals": [
            (4.0, 5.0),
            (7.0, 8.0)
        ]
    }

    result = calculate_sync_score(
        alignment_result
    )

    assert result["sync_score"] == 60.0
    assert result["sync_status"] == "moderate"


def test_poor_sync_score():
    alignment_result = {
        "alignment_score": 0.2,
        "matched_intervals": 1,
        "mismatch_intervals": [
            (2.0, 3.0)
        ]
    }

    result = calculate_sync_score(
        alignment_result
    )

    assert result["sync_score"] == 20.0
    assert result["sync_status"] == "poor"


def test_missing_required_key():
    alignment_result = {
        "alignment_score": 1.0
    }

    with pytest.raises(ValueError):
        calculate_sync_score(
            alignment_result
        )


def test_invalid_alignment_score():
    alignment_result = {
        "alignment_score": 1.5,
        "matched_intervals": 2,
        "mismatch_intervals": []
    }

    with pytest.raises(ValueError):
        calculate_sync_score(
            alignment_result
        )


def test_invalid_input_type():
    with pytest.raises(ValueError):
        calculate_sync_score([])


def test_zero_alignment_score():
    alignment_result = {
        "alignment_score": 0.0,
        "matched_intervals": 0,
        "mismatch_intervals": [
            (1.0, 2.0)
        ]
    }

    result = calculate_sync_score(
        alignment_result
    )

    assert result["sync_score"] == 0.0
    assert result["sync_status"] == "poor"