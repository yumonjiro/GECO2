"""Standalone FastAPI service for AODC density-map inference.

Exposes a single POST /predict endpoint that accepts an image and returns
the top-1 density-peak location in original image pixel coordinates.

Usage:
    python aodc_server.py
    # or
    uvicorn aodc_server:app --host 0.0.0.0 --port 7861
"""

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import importlib.util
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Load aodc_wrapper directly to avoid triggering models/__init__.py (which
# requires hydra — a GECO2-only dependency not needed in this service).
_spec = importlib.util.spec_from_file_location(
    "aodc_wrapper",
    Path(__file__).parent / "models" / "aodc_wrapper.py",
)
_aodc_wrapper_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aodc_wrapper_mod)
AODCWrapper = _aodc_wrapper_mod.AODCWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AODC_CHECKPOINT = os.environ.get("AODC_CHECKPOINT", "aodc.pth")
AODC_PORT = int(os.environ.get("AODC_PORT", 7861))

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_aodc: Optional[AODCWrapper] = None
_device: Optional[torch.device] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load AODC model once at startup."""
    global _aodc, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading AODC from '%s' on %s …", AODC_CHECKPOINT, _device)
    _aodc = AODCWrapper(AODC_CHECKPOINT, _device)
    logger.info("AODC service ready.")
    yield
    logger.info("AODC service shutting down.")


app = FastAPI(
    title="AODC Service",
    description="Density-map based point extraction for zero-prompt object counting.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(_device)}


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
):
    """Extract top-1 density-map peak from image.

    **Request** (multipart/form-data):
    - `image`: image file

    **Response**:
    ```json
    {"x": 123.4, "y": 56.7}
    ```
    x, y are in original image pixel coordinates.
    """
    try:
        contents = await image.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    x, y = _aodc.run(image_np)
    return JSONResponse({"x": x, "y": y})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "aodc_server:app",
        host="0.0.0.0",
        port=AODC_PORT,
        log_level="info",
    )
