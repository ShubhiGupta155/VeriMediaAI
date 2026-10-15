import hashlib

from fastapi.testclient import TestClient
import pytest

from backend.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def fake_process_uploaded_media(contents, filename, media_type, analysis_id):
    return {
        "status": "completed",
        "module_results": [],
        "fusion": {},
        "risk_assessment": {},
    }


def test_accepts_avif_extension_without_testing_decoding(client, monkeypatch):
    # Checks extension acceptance only, not AVIF decoding.
    monkeypatch.setattr(
        "backend.routes.analysis.process_uploaded_media",
        fake_process_uploaded_media,
    )

    contents = b"dummy bytes for extension test"
    response = client.post(
        "/analyses",
        files={
            "file": (
                "source.avif",
                contents,
                "image/avif",
            )
        },
    )

    assert response.status_code == 202

    body = response.json()
    assert body["filename"] == "source.avif"
    assert body["media_type"] == "image"
    assert body["file_hash"] == hashlib.sha256(contents).hexdigest()
    assert body["status"] == "completed"


def test_rejects_unsupported_file_extension(client):
    response = client.post(
        "/analyses",
        files={
            "file": (
                "source.txt",
                b"not a supported media file",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
    assert "Unsupported file extension" in response.json()["detail"]


def test_rejects_empty_upload(client):
    response = client.post(
        "/analyses",
        files={
            "file": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."


def test_get_analysis_returns_saved_result(client, monkeypatch):
    monkeypatch.setattr(
        "backend.routes.analysis.process_uploaded_media",
        fake_process_uploaded_media,
    )

    upload = client.post(
        "/analyses",
        files={"file": ("sample.png", b"test image bytes", "image/png")},
    )
    assert upload.status_code == 202

    analysis_id = upload.json()["analysis_id"]
    fetched = client.get(f"/analyses/{analysis_id}")

    assert fetched.status_code == 200
    assert fetched.json()["analysis_id"] == analysis_id
    assert fetched.json()["file_hash"] == hashlib.sha256(b"test image bytes").hexdigest()
    assert fetched.json()["status"] == "completed"


def test_get_analysis_returns_404_for_unknown_id(client):
    response = client.get("/analyses/VM-DOES-NOT-EXIST")

    assert response.status_code == 404
    assert response.json()["detail"] == "Analysis not found."