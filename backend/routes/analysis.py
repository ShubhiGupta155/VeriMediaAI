import hashlib
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from backend.services.orchestration_service import process_uploaded_media
from database.connection import get_db
from database.models import AnalysisRecord

router = APIRouter(tags=["analysis"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024

SUPPORTED_EXTENSIONS = {
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".avif": "image",
    ".mp4": "video",
    ".mov": "video",
    ".avi": "video",
    ".mkv": "video",
    ".wav": "audio",
    ".mp3": "audio",
    ".m4a": "audio",
    ".flac": "audio",
    ".ogg": "audio",
}


@router.post("/analyses", status_code=202)
async def create_analysis(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

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
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    analysis_id = f"VM-{uuid.uuid4().hex[:8].upper()}"
    file_hash = hashlib.sha256(contents).hexdigest()

    result = await run_in_threadpool(
        process_uploaded_media,
        contents,
        filename,
        media_type,
        analysis_id,
    )

    response_body = {
        "analysis_id": analysis_id,
        "filename": filename,
        "media_type": media_type,
        "file_hash": file_hash,
        "bytes_received": len(contents),
        "status": result["status"],
        "module_results": result["module_results"],
        "fusion": result["fusion"],
        "risk_assessment": result["risk_assessment"],
    }

    record = AnalysisRecord(
        analysis_id=analysis_id,
        filename=filename,
        media_type=media_type,
        file_hash=file_hash,
        bytes_received=len(contents),
        status=result["status"],
        response_json=response_body,
    )

    try:
        db.add(record)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Analysis completed but could not be saved.",
        ) from exc

    return response_body


@router.get("/analyses/{analysis_id}")
def get_analysis(
    analysis_id: str,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    record = (
        db.query(AnalysisRecord)
        .filter(AnalysisRecord.analysis_id == analysis_id)
        .first()
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    return record.response_json