import asyncio
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from styleforge.engine import InferenceEngine

engine: Optional[InferenceEngine] = None
styles_catalog: list[dict] = []
jobs: dict[str, dict] = {}
pending_tasks: set[asyncio.Task] = set()
job_semaphore = asyncio.Semaphore(2)
MAX_PIXELS_ASYNC = 1280 * 1280
MAX_JOBS = 100
JOB_TTL_SECONDS = 1800


def _upload_limit_bytes() -> int:
    return int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))


def _max_image_pixels() -> int:
    return int(os.getenv("MAX_IMAGE_PIXELS", str(33_000_000)))


def _purge_jobs() -> None:
    now = time.time()
    expired = [
        jid
        for jid, job in jobs.items()
        if job["status"] in ("complete", "error") and now - job["created_at"] > JOB_TTL_SECONDS
    ]
    for jid in expired:
        del jobs[jid]
    if len(jobs) > MAX_JOBS:
        active = {jid for jid, job in jobs.items() if job["status"] in ("queued", "processing")}
        candidates = sorted(
            (jid for jid in jobs if jid not in active), key=lambda jid: jobs[jid]["created_at"]
        )
        for jid in candidates[: len(jobs) - MAX_JOBS]:
            del jobs[jid]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, styles_catalog
    device = torch.device("cpu")
    engine = InferenceEngine(device=device)

    catalog_path = os.getenv("STYLES_CATALOG", "styles/catalog.yaml")
    if os.path.exists(catalog_path):
        with open(catalog_path) as f:
            data = yaml.safe_load(f)
        styles_catalog = data.get("styles", [])

    cin_model_path = os.getenv("CIN_MODEL_PATH", "models/multistyle.pth")
    if os.path.exists(cin_model_path):
        engine.load_cin("cin", cin_model_path, num_styles=len(styles_catalog))

    yield


app = FastAPI(title="StyleForge API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if os.getenv("ENV") != "production" else [],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StyleInfo(BaseModel):
    id: int
    name: str
    display_name: str
    description: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": engine is not None and len(engine._models) > 0,
        "styles_available": len(styles_catalog),
    }


@app.get("/api/styles", response_model=list[StyleInfo])
async def list_styles():
    return [
        StyleInfo(
            id=s["id"],
            name=s["name"],
            display_name=s["display_name"],
            description=s.get("description", ""),
        )
        for s in styles_catalog
    ]


@app.post("/api/stylize")
async def stylize(
    request: Request,
    image: UploadFile = File(...),
    style_id: Optional[int] = Form(None),
    style_a: Optional[int] = Form(None),
    style_b: Optional[int] = Form(None),
    alpha: Optional[float] = Form(None),
    quality: str = Form("standard"),
):
    if engine is None or not engine._models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    limit = _upload_limit_bytes()
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            too_big = int(content_length) > limit
        except ValueError:
            too_big = False
        if too_big:
            raise HTTPException(status_code=413, detail=f"Upload exceeds the {limit}-byte limit")
    content = await image.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"Upload exceeds the {limit}-byte limit")

    try:
        img = Image.open(io.BytesIO(content))
        if img.size[0] * img.size[1] > _max_image_pixels():
            raise HTTPException(status_code=400, detail="Image dimensions exceed the 33-megapixel limit")
        img = ImageOps.exif_transpose(img.convert("RGB"))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    num_styles = len(styles_catalog)
    if quality not in ("standard", "high_res"):
        raise HTTPException(status_code=400, detail="quality must be 'standard' or 'high_res'")
    if style_id is not None and (style_a is not None or style_b is not None):
        raise HTTPException(status_code=400, detail="Provide either style_id or style_a+style_b, not both")
    if style_a is not None or style_b is not None:
        if style_a is None or style_b is None:
            raise HTTPException(status_code=400, detail="Both style_a and style_b are required for interpolation")
        if style_a == style_b:
            raise HTTPException(status_code=400, detail="style_a and style_b must be different styles")
        if not (0 <= style_a < num_styles and 0 <= style_b < num_styles):
            raise HTTPException(status_code=400, detail=f"style_a/style_b must be in [0, {num_styles - 1}]")
        if alpha is None or not (0.0 <= alpha <= 1.0):
            raise HTTPException(status_code=400, detail="alpha must be in [0, 1]")
    elif style_id is not None:
        if not (0 <= style_id < num_styles):
            raise HTTPException(status_code=400, detail=f"style_id must be in [0, {num_styles - 1}]")

    w, h = img.size
    total_pixels = w * h

    if total_pixels > MAX_PIXELS_ASYNC:
        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "queued", "progress": 0.0, "created_at": time.time()}
        _purge_jobs()
        task = asyncio.create_task(
            _process_large_image(job_id, img, style_id, style_a, style_b, alpha, quality)
        )
        pending_tasks.add(task)
        task.add_done_callback(pending_tasks.discard)
        return JSONResponse({"job_id": job_id, "status": "queued"})

    result_img = _run_stylize(img, style_id, style_a, style_b, alpha, quality)
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


async def _process_large_image(job_id, img, style_id, style_a, style_b, alpha, quality):
    async with job_semaphore:
        try:
            jobs[job_id]["status"] = "processing"
            jobs[job_id]["progress"] = 0.5
            result = await asyncio.to_thread(_run_stylize, img, style_id, style_a, style_b, alpha, quality)
            buf = io.BytesIO()
            result.save(buf, format="JPEG", quality=92)
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["progress"] = 1.0
            jobs[job_id]["result"] = buf.getvalue()
        except Exception as e:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)


def _run_stylize(img, style_id, style_a, style_b, alpha, quality):
    if quality == "high_res":
        max_dim = 4096
    else:
        max_dim = 1280

    w, h = img.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)

    return engine.stylize(img, "cin", style_id=style_id or 0, style_a=style_a, style_b=style_b, alpha=alpha or 1.0)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] == "complete" and "result" in job:
        buf = io.BytesIO(job["result"])
        return StreamingResponse(buf, media_type="image/jpeg")
    return JSONResponse({"job_id": job_id, "status": job["status"], "progress": job.get("progress")})


if os.path.isdir("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
