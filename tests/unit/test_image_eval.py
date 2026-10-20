
import json

import pytest

from evaluation.image_eval import (
    calculate_metrics,
    calculate_threshold_metrics,
    calculate_calibration_bins,
    calculate_expected_calibration_error,
    save_evaluation_results,
    evaluate_folder,
)


def make_result(expected, predicted, ai_probability, confidence):
    return {
        "filename": f"{expected}_{predicted}.jpg",
        "expected": expected,
        "predicted": predicted,
        "correct": expected == predicted,
        "real_probability": 1.0 - ai_probability,
        "ai_generated_probability": ai_probability,
        "confidence": confidence,
    }


@pytest.fixture
def sample_results():
    return [
        make_result("ai_generated", "ai_generated", 0.9, 0.9),
        make_result("real", "real", 0.1, 0.8),
        make_result("real", "ai_generated", 0.7, 0.7),
        make_result("ai_generated", "real", 0.3, 0.6),
    ]


def test_calculate_metrics_confusion_matrix_and_scores(sample_results):
    metrics = calculate_metrics(sample_results)

    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1_score"] == pytest.approx(0.5)


def test_calculate_metrics_empty_results():
    metrics = calculate_metrics([])

    assert metrics["true_positive"] == 0
    assert metrics["true_negative"] == 0
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0



def test_threshold_uses_ai_generated_probability():
    results = [
        make_result(
            "ai_generated",
            "real",
            ai_probability=0.8,
            confidence=0.2,
        ),
        make_result(
            "real",
            "real",
            ai_probability=0.6,
            confidence=0.9,
        ),
    ]

    metrics = calculate_threshold_metrics(results, 0.7)

    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0


def test_threshold_boundary_is_inclusive():
    results = [
        make_result("ai_generated", "ai_generated", 0.7, 0.7),
        make_result("real", "real", 0.2, 0.8),
    ]

    metrics = calculate_threshold_metrics(results, 0.7)

    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 0
    assert metrics["false_negative"] == 0


def test_threshold_with_empty_results():
    metrics = calculate_threshold_metrics([], 0.5)

    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0


def test_calibration_bins_include_accuracy_and_gap():
    results = [
        make_result("real", "real", 0.1, 0.85),
        make_result("ai_generated", "real", 0.4, 0.85),
    ]

    bins = calculate_calibration_bins(results, bin_count=2)

    high_bin = bins[1]

    assert high_bin["count"] == 2
    assert high_bin["average_confidence"] == pytest.approx(0.85)
    assert high_bin["accuracy"] == pytest.approx(0.5)
    assert high_bin["calibration_gap"] == pytest.approx(0.35)


def test_calibration_empty_results_return_none_for_empty_bin_metrics():
    bins = calculate_calibration_bins([], bin_count=4)

    assert len(bins) == 4

    for calibration_bin in bins:
        assert calibration_bin["count"] == 0
        assert calibration_bin["average_confidence"] is None
        assert calibration_bin["accuracy"] is None
        assert calibration_bin["calibration_gap"] is None


def test_calibration_includes_confidence_equal_to_one():
    results = [
        make_result("ai_generated", "ai_generated", 1.0, 1.0),
    ]

    bins = calculate_calibration_bins(results, bin_count=10)

    assert bins[-1]["count"] == 1
    assert bins[-1]["average_confidence"] == pytest.approx(1.0)
    assert bins[-1]["accuracy"] == pytest.approx(1.0)


def test_saved_json_contains_expected_structure(tmp_path, sample_results):
    metrics = calculate_metrics(sample_results)
    threshold_analysis = [
        calculate_threshold_metrics(sample_results, 0.5)
    ]
    calibration_analysis = calculate_calibration_bins(sample_results)

    output_path = tmp_path / "nested" / "evaluation.json"

    save_evaluation_results(
        output_path,
        sample_results,
        metrics,
        total=4,
        correct=2,
        errors=1,
        threshold_analysis=threshold_analysis,
        calibration_analysis=calibration_analysis,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["summary"] == {
        "total_images": 4,
        "correct_predictions": 2,
        "evaluation_errors": 1,
        "accuracy": 0.5,
    }
    assert set(saved["confusion_matrix"]) == {
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    }
    assert set(saved["classification_metrics"]) == {
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "false_negative_rate",
        "roc_auc",
    }
    assert saved["threshold_analysis"] == threshold_analysis
    assert saved["calibration_analysis"] == calibration_analysis
    assert saved["expected_calibration_error"] == pytest.approx(
        calculate_expected_calibration_error(sample_results)
    )
    assert saved["results"] == sample_results


def test_evaluate_folder_counts_failed_inference(tmp_path, monkeypatch):
    from evaluation import image_eval

    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"not a real image")

    def fail_inference(_path):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(
        image_eval,
        "detect_ai_generated",
        fail_inference,
    )

    total, correct, errors, results = evaluate_folder(
        tmp_path,
        "real",
    )

    assert total == 0
    assert correct == 0
    assert errors == 1
    assert results == []
from evaluation.image_eval import calculate_metrics


def test_error_rates_and_roc_auc():
    results = [
        {
            "expected": "ai_generated",
            "predicted": "ai_generated",
            "ai_generated_probability": 0.9,
        },
        {
            "expected": "ai_generated",
            "predicted": "real",
            "ai_generated_probability": 0.7,
        },
        {
            "expected": "real",
            "predicted": "ai_generated",
            "ai_generated_probability": 0.8,
        },
        {
            "expected": "real",
            "predicted": "real",
            "ai_generated_probability": 0.1,
        },
    ]

    metrics = calculate_metrics(results)

    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
    assert metrics["roc_auc"] == 0.75


def test_roc_auc_is_none_for_single_class():
    results = [
        {
            "expected": "real",
            "predicted": "real",
            "ai_generated_probability": 0.1,
        },
        {
            "expected": "real",
            "predicted": "real",
            "ai_generated_probability": 0.2,
        },
    ]

    metrics = calculate_metrics(results)

    assert metrics["roc_auc"] is None
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] is None

def test_empty_results_have_undefined_error_rates():
    metrics = calculate_metrics([])

    assert metrics["false_positive_rate"] is None
    assert metrics["false_negative_rate"] is None
    assert metrics["roc_auc"] is None

def test_false_positive_rate_is_none_when_no_real_samples():
    results = [
        {
            "expected": "ai_generated",
            "predicted": "ai_generated",
            "ai_generated_probability": 0.9,
        },
        {
            "expected": "ai_generated",
            "predicted": "real",
            "ai_generated_probability": 0.7,
        },
    ]

    metrics = calculate_metrics(results)

    assert metrics["false_positive_rate"] is None
    assert metrics["false_negative_rate"] == 0.5


def test_expected_calibration_error():
    results = [
        make_result("real", "real", 0.1, 0.8),
        make_result("ai_generated", "real", 0.4, 0.8),
        make_result("ai_generated", "ai_generated", 0.9, 0.9),
    ]

    ece = calculate_expected_calibration_error(
        results, bin_count=2
    )

    assert ece == pytest.approx(1 / 6)


def test_expected_calibration_error_empty_results():
    assert calculate_expected_calibration_error([]) is None


def test_expected_calibration_error_perfect_calibration():
    results = [
        make_result("real", "real", 0.1, 1.0),
        make_result("ai_generated", "ai_generated", 0.9, 1.0),
    ]

    assert calculate_expected_calibration_error(results) == 0.0
