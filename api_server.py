"""FastAPI server for GECO2 (SAM2 + GECO2) point-to-count inference.

Accepts an image and point prompts via HTTP, runs SAM2 mask generation + GECO2 counting,
and returns detection results.

For fully automatic counting (no user points), this service delegates point extraction
to the AODC service configured via AODC_SERVICE_URL (default: http://aodc:7861).

Usage:
    python api_server.py
    # or
    uvicorn api_server:app --host 0.0.0.0 --port 7860
"""

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Tuple

import httpx
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from torchvision import transforms as T
import torchvision.ops as ops

from models.counter_infer import build_model
from models.point_to_count import PointToCountPipeline
from utils.arg_parser import get_argparser
from utils.data import resize_and_pad

# URL of the companion AODC service (override via env var for local dev)
AODC_SERVICE_URL = os.environ.get("AODC_SERVICE_URL", "http://aodc:7861")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (populated on startup)
# ---------------------------------------------------------------------------
_pipeline: Optional[PointToCountPipeline] = None
_device: Optional[torch.device] = None


def _load_pipeline(
    checkpoint: str = "CNTQG_multitrain_ca44.pth",
    iou_threshold: float = 0.7,
) -> Tuple[PointToCountPipeline, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_argparser().parse_args([])
    args.zero_shot = True
    cnt_model = build_model(args).to(device)

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in state["model"].items()}
    cnt_model.load_state_dict(state_dict, strict=False)
    cnt_model.eval()

    pipeline = PointToCountPipeline(cnt_model, iou_threshold=iou_threshold).to(device)
    pipeline.eval()
    return pipeline, device


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load SAM2+GECO2 pipeline once at startup."""
    global _pipeline, _device
    logger.info("Loading SAM2+GECO2 pipeline …")
    _pipeline, _device = _load_pipeline()
    logger.info("Pipeline ready on %s", _device)
    logger.info("Auto-count will call AODC service at %s", AODC_SERVICE_URL)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="GECO2 Point-to-Count API",
    description="Send an image + point prompts → receive object count and bounding boxes.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess_image(image: np.ndarray) -> Tuple[torch.Tensor, float]:
    """Normalize, resize-and-pad to [1, 3, 1024, 1024]."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
    dummy_bbox = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
    padded, _, scale = resize_and_pad(tensor, dummy_bbox, size=1024.0, zero_shot=True)
    return padded.unsqueeze(0).to(_device), scale


@torch.no_grad()
def _run_inference(
    image: np.ndarray,
    points: List[List[float]],
    labels: List[int],
    threshold: float = 0.33,
):
    """Core inference: image + points → boxes + count."""
    img_tensor, scale = _preprocess_image(image)
    point_coords = torch.tensor(points, dtype=torch.float32, device=_device) * scale
    point_labels = torch.tensor(labels, dtype=torch.int32, device=_device)

    # Single forward pass with FP16 mixed precision
    with torch.autocast(device_type=_device.type, dtype=torch.float16, enabled=(_device.type == 'cuda')):
        outputs, _, _, _, masks, exemplar_masks, exemplar_ious, exemplar_bboxes = \
            _pipeline.forward_with_exemplars(img_tensor, point_coords, point_labels)

    # Post-process
    pred_boxes = outputs[0]["pred_boxes"]
    box_v = outputs[0]["box_v"]
    if pred_boxes.dim() == 3:
        pred_boxes = pred_boxes[0]
    if box_v.dim() == 2:
        box_v = box_v[0]

    if box_v.numel() == 0:
        return [], 0, exemplar_bboxes.cpu().tolist()

    thr_inv = 1.0 / threshold
    sel = box_v > (box_v.max() / thr_inv)
    if sel.sum() == 0:
        return [], 0, exemplar_bboxes.cpu().tolist()

    keep = ops.nms(pred_boxes[sel], box_v[sel], 0.5)
    final_boxes = pred_boxes[sel][keep]
    final_boxes = torch.clamp(final_boxes, 0, 1)
    final_boxes = (final_boxes / scale * img_tensor.shape[-1]).cpu().tolist()

    return final_boxes, len(final_boxes), exemplar_bboxes.cpu().tolist()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(_device)}


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    points: str = Form(..., description='JSON array of [x,y] points, e.g. "[[100,200],[300,400]]"'),
    labels: str = Form(
        default=None,
        description='JSON array of labels (1=foreground, 0=background). Defaults to all foreground.',
    ),
    threshold: float = Form(default=0.33, description="Detection confidence threshold (0.05–0.95)"),
):
    """Run point-to-count inference.

    **Request** (multipart/form-data):
    - `image`: image file
    - `points`: JSON string `[[x1,y1],[x2,y2],...]` in original image pixel coordinates
    - `labels` *(optional)*: JSON string `[1,1,...]` — defaults to all foreground (1)
    - `threshold` *(optional)*: float, default 0.33

    **Response**:
    ```json
    {
      "count": 42,
      "pred_boxes": [[x1,y1,x2,y2], ...],
      "exemplar_boxes": [[x1,y1,x2,y2], ...]
    }
    ```
    """
    import json as _json

    # --- Parse points ---
    try:
        points_list: List[List[float]] = _json.loads(points)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid points JSON: {e}")

    if not points_list or not all(len(p) == 2 for p in points_list):
        raise HTTPException(status_code=400, detail="points must be a non-empty array of [x, y] pairs")

    # --- Parse labels ---
    if labels is not None:
        try:
            labels_list: List[int] = _json.loads(labels)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid labels JSON: {e}")
        if len(labels_list) != len(points_list):
            raise HTTPException(
                status_code=400,
                detail=f"labels length ({len(labels_list)}) must match points length ({len(points_list)})",
            )
    else:
        labels_list = [1] * len(points_list)

    # --- Read image ---
    try:
        contents = await image.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # --- Inference ---
    pred_boxes, count, exemplar_boxes = _run_inference(
        image_np, points_list, labels_list, threshold=threshold,
    )

    return JSONResponse({
        "count": count,
        "pred_boxes": pred_boxes,
        "exemplar_boxes": exemplar_boxes,
    })


@app.post("/predict_auto")
async def predict_auto(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    threshold: float = Form(default=0.33, description="Detection confidence threshold (0.05–0.95)"),
):
    """Run fully automatic counting — no point prompts needed.

    Delegates point extraction to the AODC service (AODC_SERVICE_URL), then
    runs the local SAM2+GECO2 pipeline with that seed point.

    **Request** (multipart/form-data):
    - `image`: image file
    - `threshold` *(optional)*: float, default 0.33

    **Response**:
    ```json
    {
      "count": 42,
      "pred_boxes": [[x1,y1,x2,y2], ...],
      "exemplar_boxes": [[x1,y1,x2,y2], ...]
    }
    ```
    """
    try:
        contents = await image.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # --- Call AODC service to get seed point ---
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{AODC_SERVICE_URL}/predict",
                files={"image": (image.filename or "image.jpg", contents, "image/jpeg")},
            )
            resp.raise_for_status()
            aodc_result = resp.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"AODC service unreachable at {AODC_SERVICE_URL}. "
                   "Is the aodc container running?",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"AODC service returned error: {e.response.status_code} {e.response.text}",
        )

    x, y = float(aodc_result["x"]), float(aodc_result["y"])

    # --- Run local SAM2+GECO2 pipeline with the AODC seed point ---
    pred_boxes, count, exemplar_boxes = _run_inference(
        image_np,
        points=[[x, y]],
        labels=[1],
        threshold=threshold,
    )

    return JSONResponse({
        "count": count,
        "pred_boxes": pred_boxes,
        "exemplar_boxes": exemplar_boxes,
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )
