from modules.image.metadata import extract_metadata
from modules.image.compression import analyze_compression
from modules.image.noise import analyze_noise
from modules.image.frequency import analyze_frequency
from modules.image.detector import detect_ai_generated
from modules.image.explainability import generate_forensic_heatmap
from modules.image.pipeline import analyze_image

def test_extract_metadata():
    image_path = "tests/test_image.jpg"

    metadata = extract_metadata(image_path)

    assert metadata["filename"] == "test_image.jpg"
    assert metadata["format"] == "JPEG"
    assert metadata["width"] > 0
    assert metadata["height"] > 0

    print("\nImage Metadata:")
    print(metadata)


def test_analyze_compression():
    image_path = "tests/test_image.jpg"

    result = analyze_compression(image_path)

    assert result["format"] == "JPEG"
    assert result["compression_type"] == "JPEG"
    assert result["jpeg_quantization_tables"] is True
    assert len(result["evidence"]) > 0

    print("\nCompression Analysis:")
    print(result)

def test_analyze_noise():
    image_path = "tests/test_image.jpg"

    result = analyze_noise(image_path)

    assert result["filename"] == "test_image.jpg"
    assert result["noise_std"] >= 0
    assert result["mean_absolute_residual"] >= 0
    assert len(result["regional_noise_std"]) == 4
    assert len(result["evidence"]) > 0

    print("\nNoise Analysis:")
    print(result)

def test_analyze_frequency():
    image_path = "tests/test_image.jpg"

    result = analyze_frequency(image_path)

    assert result["filename"] == "test_image.jpg"
    assert result["width"] > 0
    assert result["height"] > 0
    assert result["total_frequency_energy"] >= 0
    assert result["low_frequency_ratio"] >= 0
    assert result["high_frequency_ratio"] >= 0
    assert len(result["evidence"]) > 0

    print("\nFrequency Analysis:")
    print(result)

def test_detect_ai_generated():
    image_path = "tests/test_image.jpg"

    result = detect_ai_generated(image_path)

    assert result["filename"] == "test_image.jpg"
    assert result["predicted_label"] in {"real", "ai_generated"}

    assert 0.0 <= result["real_probability"] <= 1.0
    assert 0.0 <= result["ai_generated_probability"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0

    probability_sum = (
        result["real_probability"]
        + result["ai_generated_probability"]
    )

    assert abs(probability_sum - 1.0) < 0.001

    print("\nAI Detection:")
    print(result)

def test_generate_forensic_heatmap(tmp_path):
    image_path = "tests/test_image.jpg"
    output_path = tmp_path / "forensic_heatmap.jpg"

    result = generate_forensic_heatmap(
        image_path,
        output_path
    )

    assert result["filename"] == "test_image.jpg"
    assert result["image_width"] > 0
    assert result["image_height"] > 0
    assert 0.0 <= result["peak_score"] <= 1.0
    assert output_path.exists()

    print("\nExplainability:")
    print(result)

def test_analyze_image_pipeline(tmp_path):
    image_path = "tests/test_image.jpg"
    heatmap_path = tmp_path / "pipeline_heatmap.jpg"

    result = analyze_image(
        image_path,
        generate_heatmap=True,
        heatmap_path=heatmap_path,
    )

    assert result["image"]["filename"] == "test_image.jpg"

    assert result["metadata"] is not None
    assert result["compression"] is not None
    assert result["noise"] is not None
    assert result["frequency"] is not None
    assert result["ai_detection"] is not None
    assert result["explainability"] is not None

    assert result["ai_detection"]["predicted_label"] in {
        "real",
        "ai_generated",
    }

    assert heatmap_path.exists()

    print("\nComplete Image Forensics Pipeline:")
    print(result)