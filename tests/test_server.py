import io
import os
import time
import zlib

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from styleforge.cin import CINTransformer

CATALOG_YAML = """\
styles:
  - id: 0
    name: "style_a"
    display_name: "Style A"
    description: "test style a"
  - id: 1
    name: "style_b"
    display_name: "Style B"
    description: "test style b"
"""


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("styleforge_server")
    model = CINTransformer(num_styles=2)
    model_path = tmp / "cin.pth"
    torch.save(model.state_dict(), model_path)
    catalog_path = tmp / "catalog.yaml"
    catalog_path.write_text(CATALOG_YAML)

    os.environ["CIN_MODEL_PATH"] = str(model_path)
    os.environ["STYLES_CATALOG"] = str(catalog_path)

    from server.main import app

    with TestClient(app) as c:
        yield c


def _jpeg_bytes(size=(64, 64)):
    img = Image.fromarray(np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _png_with_dims(width, height):
    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="PNG")
    data = bytearray(buf.getvalue())
    data[16:20] = width.to_bytes(4, "big")
    data[20:24] = height.to_bytes(4, "big")
    crc = zlib.crc32(bytes(data[12:29]))
    data[29:33] = crc.to_bytes(4, "big")
    return bytes(data)


class TestHealth:
    def test_health(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["styles_available"] == 2


class TestStylize:
    def test_stylize_small_image(self, client):
        res = client.post(
            "/api/stylize",
            files={"image": ("t.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"style_id": "0"},
        )
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("image/jpeg")
        out = Image.open(io.BytesIO(res.content))
        assert out.size == (64, 64)

    def test_stylize_invalid_image(self, client):
        res = client.post(
            "/api/stylize",
            files={"image": ("t.jpg", b"not an image", "image/jpeg")},
        )
        assert res.status_code == 400

    def test_stylize_oversized_upload(self, client):
        os.environ["MAX_UPLOAD_BYTES"] = "1024"
        try:
            res = client.post(
                "/api/stylize",
                files={"image": ("t.jpg", _jpeg_bytes(size=(256, 256)), "image/jpeg")},
                data={"style_id": "0"},
            )
            assert res.status_code == 413
        finally:
            os.environ.pop("MAX_UPLOAD_BYTES", None)

    def test_stylize_pixel_bomb(self, client):
        res = client.post(
            "/api/stylize",
            files={"image": ("bomb.png", _png_with_dims(50000, 1000), "image/png")},
            data={"style_id": "0"},
        )
        assert res.status_code == 400

    def test_stylize_style_id_out_of_range(self, client):
        res = client.post(
            "/api/stylize",
            files={"image": ("t.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"style_id": "7"},
        )
        assert res.status_code == 400

    def test_stylize_interp_missing_style_b(self, client):
        res = client.post(
            "/api/stylize",
            files={"image": ("t.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"style_a": "0"},
        )
        assert res.status_code == 400

    def test_stylize_interp_equal_styles(self, client):
        res = client.post(
            "/api/stylize",
            files={"image": ("t.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"style_a": "0", "style_b": "0"},
        )
        assert res.status_code == 400

    def test_stylize_alpha_out_of_range(self, client):
        res = client.post(
            "/api/stylize",
            files={"image": ("t.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"style_a": "0", "style_b": "1", "alpha": "1.5"},
        )
        assert res.status_code == 400

    def test_stylize_bad_quality(self, client):
        res = client.post(
            "/api/stylize",
            files={"image": ("t.jpg", _jpeg_bytes(), "image/jpeg")},
            data={"style_id": "0", "quality": "ultra"},
        )
        assert res.status_code == 400

    def test_stylize_exif_orientation(self, client):
        img = Image.new("RGB", (104, 52), (120, 80, 40))
        exif = Image.Exif()
        exif[274] = 6
        buf = io.BytesIO()
        img.save(buf, format="JPEG", exif=exif)
        res = client.post(
            "/api/stylize",
            files={"image": ("rot.jpg", buf.getvalue(), "image/jpeg")},
            data={"style_id": "0"},
        )
        assert res.status_code == 200
        out = Image.open(io.BytesIO(res.content))
        assert out.size == (52, 104)


class TestAsyncJobs:
    def test_large_image_async_flow(self, client):
        img = Image.fromarray(np.random.randint(0, 255, (1400, 1400, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        payload = buf.getvalue()

        res = client.post(
            "/api/stylize",
            files={"image": ("big.jpg", payload, "image/jpeg")},
            data={"style_id": "0", "quality": "high_res"},
        )
        assert res.status_code == 200
        job = res.json()
        assert job["status"] == "queued"
        job_id = job["job_id"]

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            poll = client.get(f"/api/jobs/{job_id}")
            if poll.headers["content-type"].startswith("image/jpeg"):
                out = Image.open(io.BytesIO(poll.content))
                assert out.size == (1400, 1400)
                return
            assert poll.status_code == 200
            status = poll.json()["status"]
            assert status in ("queued", "processing", "error"), status
            time.sleep(0.25)
        pytest.fail("async job did not complete in time")

    def test_job_not_found(self, client):
        res = client.get("/api/jobs/nonexistent")
        assert res.status_code == 404
