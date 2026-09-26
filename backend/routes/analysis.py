import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(tags=["analysis"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

SUPPORTED_EXTENSIONS = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".mp4": "video",
    ".mov": "video",
    ".avi": "video",
    ".mkv": "video",
    ".wav": "audio",
    ".mp3": "audio",
    ".m4a": "audio",
    ".flac": "audio",
    ".ogg": "audio",
    ".avif": "image",
}


@router.post("/analyses", status_code=202)
async def create_analysis(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required.")

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()
    media_type = SUPPORTED_EXTENSIONS.get(extension)

    if media_type is None:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension: {extension or '(none)'}",
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    return {
        "analysis_id": f"VM-{uuid.uuid4().hex[:8].upper()}",
        "filename": filename,
        "media_type": media_type,
        "sha256": hashlib.sha256(contents).hexdigest(),
        "bytes_received": len(contents),
        "status": "accepted",
        "message": "Upload validated. Detector orchestration is not connected yet.",
    }