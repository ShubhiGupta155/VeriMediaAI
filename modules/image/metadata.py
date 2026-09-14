from pathlib import Path
from PIL import Image, ExifTags


def extract_metadata(image_path):
    """
    Extract basic image information and available EXIF metadata.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    result = {
        "filename": path.name,
        "file_size_bytes": path.stat().st_size,
        "format": None,
        "width": None,
        "height": None,
        "mode": None,
        "exif": {},
    }

    with Image.open(path) as image:
        result["format"] = image.format
        result["width"] = image.width
        result["height"] = image.height
        result["mode"] = image.mode

        exif_data = image.getexif()

        for tag_id, value in exif_data.items():
            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))

            try:
                result["exif"][tag_name] = str(value)
            except Exception:
                result["exif"][tag_name] = repr(value)

    return result