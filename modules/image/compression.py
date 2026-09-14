from pathlib import Path
from PIL import Image


def analyze_compression(image_path):
    """
    Analyze basic compression characteristics of an image.

    Returns forensic indicators such as:
    - image format
    - file size
    - dimensions
    - JPEG byte-density indicator when applicable
    - compression-related evidence
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    file_size = path.stat().st_size

    with Image.open(path) as image:
        image_format = image.format
        width = image.width
        height = image.height

        result = {
            "filename": path.name,
            "format": image_format,
            "file_size_bytes": file_size,
            "width": width,
            "height": height,
            "compression_type": "unknown",
            "jpeg_quantization_tables": False,
            "evidence": [],
        }

        if image_format == "JPEG":
            result["compression_type"] = "JPEG"

            # JPEG images normally contain quantization tables.
            if getattr(image, "quantization", None):
                result["jpeg_quantization_tables"] = True

                result["evidence"].append(
                "JPEG quantization tables are present; this is a normal file characteristic."
                )

            # Compare file size with image dimensions.
            pixel_count = width * height

            if pixel_count > 0:
                bytes_per_pixel = file_size / pixel_count

                result["bytes_per_pixel"] = round(bytes_per_pixel, 4)

                if bytes_per_pixel < 0.15:
                    result["evidence"].append(
                        "Relatively strong JPEG compression is indicated."
                    )
                elif bytes_per_pixel > 1.0:
                    result["evidence"].append(
                        "JPEG file size is relatively large for its dimensions."
                    )
                else:
                    result["evidence"].append(
                      "JPEG byte-density indicator falls within the calibrated analysis range."
                    )

        elif image_format == "PNG":
            result["compression_type"] = "PNG"

            result["evidence"].append(
                "PNG format detected; JPEG quantization analysis is not applicable."
            )

        else:
            result["evidence"].append(
                f"{image_format} format detected; JPEG compression analysis is not applicable."
            )

    return result