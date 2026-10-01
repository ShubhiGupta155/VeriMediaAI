from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def fake_process_uploaded_media(contents, filename, media_type, analysis_id):
    return {
        "status": "completed",
        "module_results": [],
        "fusion": {"fused_score": 0.5},
        "risk_assessment": {"assessment": "Inconclusive"},
    }


def test_accepts_avif_extension_without_testing_decoding(monkeypatch):
    # Tests extension acceptance only; dummy bytes do not test AVIF decoding.
    monkeypatch.setattr(
        "backend.routes.analysis.process_uploaded_media",
        fake_process_uploaded_media,
    )

    response = client.post(
        "/analyses",
        files={
            "file": (
                "source.avif",
                b"dummy bytes for extension test",
                "image/avif",
            )
        },
    )

    assert response.status_code == 202

    body = response.json()
    assert body["filename"] == "source.avif"
    assert body["media_type"] == "image"
    assert body["status"] == "completed"
    assert len(body["sha256"]) == 64


def test_rejects_unsupported_file_extension():
    response = client.post(
        "/analyses",
        files={
            "file": (
                "document.txt",
                b"not a supported media file",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
    assert "Unsupported file extension" in response.json()["detail"]


def test_rejects_empty_upload():
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