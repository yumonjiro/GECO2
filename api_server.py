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
from PIL import Image, ImageDraw, ImageFont
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
    """Normalize, resize-and-pad to [1, 3, 1024, 1024] (zero-shot mode, no exemplar bboxes)."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
    dummy_bbox = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
    padded, _, scale = resize_and_pad(tensor, dummy_bbox, size=1024.0, zero_shot=True)
    return padded.unsqueeze(0).to(_device), scale


def _preprocess_image_with_bboxes(
    image: np.ndarray,
    bboxes: List[List[float]],
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Normalize, resize-and-pad with real exemplar bboxes (matches demo_gradio.py).

    Uses adaptive scaling (zero_shot=False) so that exemplar objects are ~80px.
    Returns bboxes in pixel coordinates of the padded 1024px image.
    """
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
    padded, bboxes_scaled, scale = resize_and_pad(tensor, bboxes_tensor, size=1024.0)
    img_tensor = padded.unsqueeze(0).to(_device)
    bboxes_scaled = bboxes_scaled.unsqueeze(0).to(_device)  # [1, M, 4] in pixel coords
    return img_tensor, bboxes_scaled, scale


@torch.no_grad()
def _run_inference(
    image: np.ndarray,
    points: List[List[float]],
    labels: List[int],
    threshold: float = 0.33,
):
    """Core inference: image + points → boxes + count (two-pass with adaptive scaling).

    Pass 1: zero-shot preprocessing → backbone → SAM2 point→mask→bbox (get exemplars).
    Pass 2: re-preprocess with adaptive scaling using those bboxes → backbone → forward_detect.

    This matches demo_gradio.py / /predict_bbox quality because the detection pass
    uses the same adaptive scaling (~80 px objects) that the model was trained with.
    """
    # --- Pass 1: SAM2 exemplar extraction at zero-shot scale ---
    img_tensor_zs, scale_zs = _preprocess_image(image)
    point_coords = torch.tensor(points, dtype=torch.float32, device=_device) * scale_zs
    point_labels = torch.tensor(labels, dtype=torch.int32, device=_device)

    feats_zs = _pipeline.cnt.forward_backbone(img_tensor_zs)
    exemplar_masks, exemplar_ious, exemplar_bboxes = \
        _pipeline.cnt.sam_mask.predict_masks_from_points(
            backbone_feats=feats_zs,
            point_coords=point_coords,
            point_labels=point_labels,
        )

    # Filter low-quality masks
    keep_mask = exemplar_ious >= _pipeline.iou_threshold
    if keep_mask.sum() == 0:
        keep_mask = exemplar_ious >= exemplar_ious.max() * 0.5
    exemplar_bboxes_px = exemplar_bboxes[keep_mask]  # pixel coords in the 1024-padded space

    if exemplar_bboxes_px.numel() == 0:
        return [], 0, exemplar_bboxes.cpu().tolist()

    # Convert SAM bboxes back to original image coordinates
    exemplar_bboxes_orig = (exemplar_bboxes_px / scale_zs).cpu().tolist()

    # --- Pass 2: adaptive re-preprocessing + detection ---
    img_tensor, bboxes_scaled, scale = _preprocess_image_with_bboxes(image, exemplar_bboxes_orig)
    image_size = float(img_tensor.shape[-1])  # 1024.0

    feats = _pipeline.cnt.forward_backbone(img_tensor)
    det_results = _pipeline.cnt.forward_detect(feats, bboxes_scaled, image_size=image_size)

    outputs = det_results[0]
    pred_boxes = outputs[0]["pred_boxes"]
    box_v = outputs[0]["box_v"]
    if pred_boxes.dim() == 3:
        pred_boxes = pred_boxes[0]
    if box_v.dim() == 2:
        box_v = box_v[0]

    if box_v.numel() == 0:
        return [], 0, exemplar_bboxes_orig

    thr_inv = 1.0 / threshold
    sel = box_v > (box_v.max() / thr_inv)
    if sel.sum() == 0:
        return [], 0, exemplar_bboxes_orig

    keep = ops.nms(pred_boxes[sel], box_v[sel], 0.5)
    final_boxes = pred_boxes[sel][keep]
    final_boxes = torch.clamp(final_boxes, 0, 1)
    final_boxes = (final_boxes / scale * image_size).cpu().tolist()

    return final_boxes, len(final_boxes), exemplar_bboxes_orig


@torch.no_grad()
def _run_inference_bbox(
    image: np.ndarray,
    bboxes: List[List[float]],
    threshold: float = 0.33,
):
    """Core inference with explicit exemplar bounding boxes (no SAM2 preprocessing).

    Matches demo_gradio.py: passes real bboxes to resize_and_pad for adaptive scaling,
    then feeds pixel-space bboxes directly to forward_detect (NOT normalized to [0,1]).
    """
    img_tensor, bboxes_scaled, scale = _preprocess_image_with_bboxes(image, bboxes)
    image_size = float(img_tensor.shape[-1])  # 1024.0

    feats = _pipeline.cnt.forward_backbone(img_tensor)
    det_results = _pipeline.cnt.forward_detect(feats, bboxes_scaled, image_size=image_size)

    outputs = det_results[0]
    pred_boxes_raw = outputs[0]["pred_boxes"]
    box_v = outputs[0]["box_v"]
    if pred_boxes_raw.dim() == 3:
        pred_boxes_raw = pred_boxes_raw[0]
    if box_v.dim() == 2:
        box_v = box_v[0]

    if box_v.numel() == 0:
        return [], 0, bboxes

    thr_inv = 1.0 / threshold
    sel = box_v > (box_v.max() / thr_inv)
    if sel.sum() == 0:
        return [], 0, bboxes

    keep = ops.nms(pred_boxes_raw[sel], box_v[sel], 0.5)
    final_boxes = pred_boxes_raw[sel][keep]
    final_boxes = torch.clamp(final_boxes, 0, 1)
    final_boxes = (final_boxes / scale * image_size).cpu().tolist()

    return final_boxes, len(final_boxes), bboxes


def _draw_boxes(
    pil_image: Image.Image,
    pred_boxes: List[List[float]],
    exemplar_boxes: List[List[float]],
    count: int,
) -> Image.Image:
    """Draw bounding boxes on a copy of the image and return it.

    pred_boxes    → green rectangles (detected objects)
    exemplar_boxes → red rectangles (exemplar / query objects)
    Count is rendered in the top-left corner.
    """
    img = pil_image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    for box in exemplar_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=2)

    for box in pred_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(60, 220, 60), width=2)

    # Count label
    label = f"Count: {count}"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=24)
    except OSError:
        font = ImageFont.load_default()
    margin = 6
    bbox = draw.textbbox((margin, margin), label, font=font)
    draw.rectangle([bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin],
                   fill=(0, 0, 0, 180))
    draw.text((margin, margin), label, fill=(255, 255, 255), font=font)

    return img


def _image_response(pil_image: Image.Image) -> StreamingResponse:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")




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


@app.post("/predict/image", response_class=StreamingResponse)
async def predict_image(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    points: str = Form(..., description='JSON array of [x,y] points, e.g. "[[100,200],[300,400]]"'),
    labels: str = Form(
        default=None,
        description='JSON array of labels (1=foreground, 0=background). Defaults to all foreground.',
    ),
    threshold: float = Form(default=0.33, description="Detection confidence threshold (0.05–0.95)"),
):
    """Same as /predict but returns a PNG image with bounding boxes drawn on it.

    - Green boxes: detected objects
    - Red boxes: exemplar (query) objects
    - Count label in top-left corner
    """
    import json as _json

    try:
        points_list: List[List[float]] = _json.loads(points)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid points JSON: {e}")
    if not points_list or not all(len(p) == 2 for p in points_list):
        raise HTTPException(status_code=400, detail="points must be a non-empty array of [x, y] pairs")

    if labels is not None:
        try:
            labels_list: List[int] = _json.loads(labels)
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid labels JSON: {e}")
        if len(labels_list) != len(points_list):
            raise HTTPException(status_code=400,
                detail=f"labels length ({len(labels_list)}) must match points length ({len(points_list)})")
    else:
        labels_list = [1] * len(points_list)

    try:
        contents = await image.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    pred_boxes, count, exemplar_boxes = _run_inference(
        image_np, points_list, labels_list, threshold=threshold,
    )
    return _image_response(_draw_boxes(pil, pred_boxes, exemplar_boxes, count))


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


@app.post("/predict_auto/image", response_class=StreamingResponse)
async def predict_auto_image(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    threshold: float = Form(default=0.33, description="Detection confidence threshold (0.05–0.95)"),
):
    """Same as /predict_auto but returns a PNG image with bounding boxes drawn on it.

    - Green boxes: detected objects
    - Red boxes: exemplar (query) objects
    - Count label in top-left corner
    """
    try:
        contents = await image.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

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
            detail=f"AODC service unreachable at {AODC_SERVICE_URL}. Is the aodc container running?",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"AODC service returned error: {e.response.status_code} {e.response.text}",
        )

    x, y = float(aodc_result["x"]), float(aodc_result["y"])
    pred_boxes, count, exemplar_boxes = _run_inference(
        image_np, points=[[x, y]], labels=[1], threshold=threshold,
    )
    return _image_response(_draw_boxes(pil, pred_boxes, exemplar_boxes, count))


@app.post("/predict_bbox")
async def predict_bbox(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    bboxes: str = Form(..., description='JSON array of exemplar bboxes [[x1,y1,x2,y2], ...] in original image pixel coordinates'),
    threshold: float = Form(default=0.33, description="Detection confidence threshold (0.05–0.95)"),
):
    """Run GECO2 counting with explicit exemplar bounding boxes (no SAM2, no AODC).

    Pass one or more bounding boxes that enclose a representative sample of the
    objects you want to count. GECO2 uses them directly as exemplars.

    **Request** (multipart/form-data):
    - `image`: image file
    - `bboxes`: JSON string `[[x1,y1,x2,y2], ...]` in original image pixel coordinates
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

    try:
        bboxes_list: List[List[float]] = _json.loads(bboxes)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid bboxes JSON: {e}")
    if not bboxes_list or not all(len(b) == 4 for b in bboxes_list):
        raise HTTPException(status_code=400, detail="bboxes must be a non-empty array of [x1, y1, x2, y2]")

    try:
        contents = await image.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    pred_boxes, count, exemplar_boxes = _run_inference_bbox(image_np, bboxes_list, threshold=threshold)
    return JSONResponse({"count": count, "pred_boxes": pred_boxes, "exemplar_boxes": exemplar_boxes})


@app.post("/predict_bbox/image", response_class=StreamingResponse)
async def predict_bbox_image(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    bboxes: str = Form(..., description='JSON array of exemplar bboxes [[x1,y1,x2,y2], ...] in original image pixel coordinates'),
    threshold: float = Form(default=0.33, description="Detection confidence threshold (0.05–0.95)"),
):
    """Same as /predict_bbox but returns a PNG image with bounding boxes drawn on it."""
    import json as _json

    try:
        bboxes_list: List[List[float]] = _json.loads(bboxes)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid bboxes JSON: {e}")
    if not bboxes_list or not all(len(b) == 4 for b in bboxes_list):
        raise HTTPException(status_code=400, detail="bboxes must be a non-empty array of [x1, y1, x2, y2]")

    try:
        contents = await image.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    pred_boxes, count, exemplar_boxes = _run_inference_bbox(image_np, bboxes_list, threshold=threshold)
    return _image_response(_draw_boxes(pil, pred_boxes, exemplar_boxes, count))


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
