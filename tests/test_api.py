from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_accepts_avif_upload():
    response = client.post(
        "/analyses",
        files={
            "file": (
                "source.avif",
                b"test AVIF upload bytes",
                "image/avif",
            )
        },
    )

    assert response.status_code == 202

    body = response.json()
    assert body["filename"] == "source.avif"
    assert body["media_type"] == "image"
    assert body["status"] == "accepted"
    assert len(body["sha256"]) == 64
    assert body["bytes_received"] == len(b"test AVIF upload bytes")


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