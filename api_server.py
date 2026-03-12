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
from contextlib import asynccontextmanager, nullcontext
from typing import List, Optional, Tuple

import httpx
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont
import torchvision.ops as ops

from models.counter_infer import build_model
from utils.area_estimation import estimate_object_areas
from models.point_to_count import PointToCountPipeline
from utils.arg_parser import get_argparser
from utils.data import resize_and_pad

# URL of the companion AODC service (override via env var for local dev)
AODC_SERVICE_URL = os.environ.get("AODC_SERVICE_URL", "http://aodc:7861")
PNG_COMPRESS_LEVEL = max(0, min(9, int(os.environ.get("PNG_COMPRESS_LEVEL", "1"))))
DEFAULT_IMAGE_FORMAT = os.environ.get("DEFAULT_IMAGE_FORMAT", "jpeg").lower()
DEFAULT_IMAGE_MAX_SIDE = max(0, int(os.environ.get("DEFAULT_IMAGE_MAX_SIDE", "1280")))
DEFAULT_IMAGE_QUALITY = max(1, min(100, int(os.environ.get("DEFAULT_IMAGE_QUALITY", "85"))))
_SUPPORTED_IMAGE_FORMATS = {"png", "jpeg", "webp"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (populated on startup)
# ---------------------------------------------------------------------------
_pipeline: Optional[PointToCountPipeline] = None
_device: Optional[torch.device] = None
_aodc_client: Optional[httpx.AsyncClient] = None
_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_NORM_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _inference_autocast():
    """Use fp16 autocast on CUDA to unlock faster SDPA kernels (flash/mem-efficient)."""
    if _device is not None and _device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


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
    global _pipeline, _device, _aodc_client
    logger.info("Loading SAM2+GECO2 pipeline …")
    _pipeline, _device = _load_pipeline()
    _aodc_client = httpx.AsyncClient(timeout=30.0)
    if _device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    logger.info("Pipeline ready on %s", _device)
    logger.info("Auto-count will call AODC service at %s", AODC_SERVICE_URL)
    yield
    if _aodc_client is not None:
        await _aodc_client.aclose()
        _aodc_client = None
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

def _decode_aodc_areas(
    aodc_result: dict, orig_h: int, orig_w: int,
) -> Optional[List[float]]:
    """Decode AODC response and estimate per-peak object areas.

    Returns None if the response doesn't contain density map data
    (backward compatible with older AODC servers).
    """
    import base64

    den_b64 = aodc_result.get("density_map_b64")
    peak_indices = aodc_result.get("peak_indices")
    den_shape = aodc_result.get("den_shape")
    if not den_b64 or not peak_indices or not den_shape:
        return None

    den_bytes = base64.b64decode(den_b64)
    density_map = np.frombuffer(den_bytes, dtype=np.float32).reshape(den_shape)
    return estimate_object_areas(
        density_map,
        peak_indices=[(r, c) for r, c in peak_indices],
        orig_h=orig_h,
        orig_w=orig_w,
    )


def _filter_merged_exemplars(
    bboxes: List[List[float]],
    points: List[List[float]],
) -> List[List[float]]:
    """Remove exemplar BBs that contain multiple AODC seed points (merged objects).

    If a single SAM2 mask engulfs 2+ AODC peaks, the BB covers multiple objects
    and would mislead GECO2. Discard such BBs — the remaining single-object
    exemplars are sufficient.
    """
    if len(bboxes) <= 1 or len(points) <= 1:
        return bboxes

    keep = []
    for bb in bboxes:
        x1, y1, x2, y2 = bb
        count_inside = sum(1 for px, py in points if x1 <= px <= x2 and y1 <= py <= y2)
        if count_inside <= 1:
            keep.append(bb)

    return keep if keep else bboxes  # fallback: keep all if everything is merged


def _filter_exemplars_by_shape(
    bboxes: List[List[float]],
    max_aspect_dev: float = 2.0,
) -> List[List[float]]:
    """Remove exemplar BBs whose aspect ratio deviates from the median.

    Catches noise BBs (e.g. elongated text regions among round pills)
    by comparing each BB's aspect ratio to the group median.
    """
    if len(bboxes) <= 2:
        return bboxes

    aspects = []
    for b in bboxes:
        w = max(b[2] - b[0], 1e-6)
        h = max(b[3] - b[1], 1e-6)
        aspects.append(max(w, h) / min(w, h))  # always >= 1

    sorted_ar = sorted(aspects)
    median_ar = sorted_ar[len(sorted_ar) // 2]

    keep = []
    for i, ar in enumerate(aspects):
        ratio = max(ar, median_ar) / min(ar, median_ar)
        if ratio <= max_aspect_dev:
            keep.append(bboxes[i])

    return keep if keep else bboxes  # fallback: keep all if everything is outlier


def _filter_exemplars_by_area(
    bboxes: List[List[float]],
    area_ratio: float = 3.0,
) -> List[List[float]]:
    """Remove exemplar BBs that are much smaller than the rest (e.g. holes).

    When AODC picks peaks on both a "hole" and the surrounding object,
    SAM2 produces exemplars with very different sizes.  Holes are always
    smaller than the real objects, so we find the biggest area gap and
    keep the *larger-area* side.

    Algorithm:
        1. Sort BBs by area (ascending).
        2. Find the largest ratio gap between consecutive areas.
        3. If that gap exceeds *area_ratio*, split there and keep
           the larger-area group (everything after the gap).
        4. Otherwise keep all (sizes are uniform enough).
    """
    if len(bboxes) <= 1:
        return bboxes

    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
    indices = sorted(range(len(bboxes)), key=lambda i: areas[i])

    # Find the biggest ratio gap between consecutively-sorted areas
    max_gap_ratio = 0.0
    split_pos = -1  # index in sorted list *after* which to split
    for i in range(1, len(indices)):
        smaller = max(areas[indices[i - 1]], 1e-6)
        ratio = areas[indices[i]] / smaller
        if ratio > max_gap_ratio:
            max_gap_ratio = ratio
            split_pos = i

    if max_gap_ratio <= area_ratio:
        return bboxes  # all sizes are similar enough

    # Keep the larger-area group (everything from split_pos onward)
    keep = indices[split_pos:]
    return [bboxes[i] for i in keep]


def _filter_aodc_points_by_area(
    points: List[List[float]],
    expected_areas: List[float],
    area_ratio: float = 3.0,
) -> Tuple[List[List[float]], List[float]]:
    """Pre-filter AODC seed points by estimated area before SAM2.

    Uses the same biggest-gap logic as _filter_exemplars_by_area but operates
    on AODC density-map area estimates (not SAM2 BB areas).  This catches
    small-hole peaks that would otherwise all pass through SAM2 as valid
    exemplars of similar size.

    Returns filtered (points, expected_areas) keeping the larger-area group.
    """
    if len(points) <= 1 or not expected_areas:
        return points, expected_areas

    indices = sorted(range(len(expected_areas)), key=lambda i: expected_areas[i])

    max_gap_ratio = 0.0
    split_pos = -1
    for i in range(1, len(indices)):
        smaller = max(expected_areas[indices[i - 1]], 1e-6)
        ratio = expected_areas[indices[i]] / smaller
        if ratio > max_gap_ratio:
            max_gap_ratio = ratio
            split_pos = i

    if max_gap_ratio <= area_ratio:
        return points, expected_areas

    keep = indices[split_pos:]
    return (
        [points[i] for i in keep],
        [expected_areas[i] for i in keep],
    )


async def _request_aodc_predict_multi(contents: bytes, filename: str) -> dict:
    """Call AODC /predict_multi using a shared AsyncClient."""
    if _aodc_client is None:
        raise HTTPException(status_code=503, detail="AODC client is not initialized")

    try:
        resp = await _aodc_client.post(
            f"{AODC_SERVICE_URL}/predict_multi",
            files={"image": (filename, contents, "image/jpeg")},
        )
        resp.raise_for_status()
        return resp.json()
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


def _normalize_chw_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """In-place ImageNet normalization for CHW float tensor in [0,1]."""
    return tensor.sub_(_NORM_MEAN).div_(_NORM_STD)


async def _read_upload_image(image: UploadFile) -> Tuple[bytes, Image.Image, np.ndarray]:
    """Decode uploaded image once and return bytes, PIL(RGB), numpy(HWC uint8)."""
    try:
        contents = await image.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.asarray(pil, dtype=np.uint8)
        if not image_np.flags["C_CONTIGUOUS"]:
            image_np = np.ascontiguousarray(image_np)
        return contents, pil, image_np
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")


def _preprocess_image(image: np.ndarray) -> Tuple[torch.Tensor, float]:
    """Normalize, resize-and-pad to [1, 3, 1024, 1024] (zero-shot mode, no exemplar bboxes)."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
    tensor = _normalize_chw_tensor(tensor)
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
    tensor = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32).div_(255.0)
    tensor = _normalize_chw_tensor(tensor)
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
    expected_areas: Optional[List[float]] = None,
    need_masks: bool = True,
):
    """Core inference: image + points → boxes + count (two-pass with adaptive scaling).

    Pass 1: zero-shot preprocessing → backbone → SAM2 point→mask→bbox (get exemplars).
    Pass 2: re-preprocess with adaptive scaling using those bboxes → backbone → forward_detect.

    This matches demo_gradio.py / /predict_bbox quality because the detection pass
    uses the same adaptive scaling (~80 px objects) that the model was trained with.

    Args:
        expected_areas: Optional per-point expected object areas in original-image
            pixel² units (from AODC density map). When provided, SAM2 selects
            the mask whose area is closest to the expected size.
    """
    # --- Pass 1: SAM2 exemplar extraction at zero-shot scale ---
    img_tensor_zs, scale_zs = _preprocess_image(image)
    point_coords = torch.tensor(points, dtype=torch.float32, device=_device) * scale_zs
    point_labels = torch.tensor(labels, dtype=torch.int32, device=_device)

    with _inference_autocast():
        feats_zs = _pipeline.cnt.forward_backbone(img_tensor_zs)

    # Scale expected_areas from original-image space to zero-shot 1024px space
    ea_tensor = None
    if expected_areas is not None:
        ea_tensor = torch.tensor(expected_areas, dtype=torch.float32, device=_device)
        ea_tensor = ea_tensor * (scale_zs ** 2)

    with _inference_autocast():
        exemplar_masks, exemplar_ious, exemplar_bboxes = \
            _pipeline.cnt.sam_mask.predict_masks_from_points(
                backbone_feats=feats_zs,
                point_coords=point_coords,
                point_labels=point_labels,
                expected_areas=ea_tensor,
            )

    # Filter low-quality masks
    keep_mask = exemplar_ious >= _pipeline.iou_threshold
    if keep_mask.sum() == 0:
        keep_mask = exemplar_ious >= exemplar_ious.max() * 0.5
    exemplar_bboxes_px = exemplar_bboxes[keep_mask]  # pixel coords in the 1024-padded space

    if exemplar_bboxes_px.numel() == 0:
        return [], 0, exemplar_bboxes.cpu().tolist(), None

    # Convert SAM bboxes back to original image coordinates
    exemplar_bboxes_orig = (exemplar_bboxes_px / scale_zs).cpu().tolist()

    # Remove outlier exemplars (e.g. "hole" vs "whole object" with very different sizes)
    exemplar_bboxes_orig = _filter_exemplars_by_area(exemplar_bboxes_orig)

    # Remove BBs that engulf multiple AODC seed points (merged objects)
    exemplar_bboxes_orig = _filter_merged_exemplars(exemplar_bboxes_orig, points)

    # Remove BBs with anomalous aspect ratios (e.g. text regions among round objects)
    exemplar_bboxes_orig = _filter_exemplars_by_shape(exemplar_bboxes_orig)

    if not exemplar_bboxes_orig:
        return [], 0, [], None

    # Select the single best exemplar (median area) for GECO2
    if len(exemplar_bboxes_orig) > 1:
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in exemplar_bboxes_orig]
        order = sorted(range(len(areas)), key=lambda i: areas[i])
        median_idx = order[len(order) // 2]
        exemplar_bboxes_orig = [exemplar_bboxes_orig[median_idx]]

    # --- Pass 2: adaptive re-preprocessing + detection ---
    img_tensor, bboxes_scaled, scale = _preprocess_image_with_bboxes(image, exemplar_bboxes_orig)
    image_size = float(img_tensor.shape[-1])  # 1024.0

    with _inference_autocast():
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
        return [], 0, exemplar_bboxes_orig, None

    thr_inv = 1.0 / threshold
    sel = box_v > (box_v.max() / thr_inv)
    if sel.sum() == 0:
        return [], 0, exemplar_bboxes_orig, None

    keep = ops.nms(pred_boxes[sel], box_v[sel], 0.5)
    final_boxes = pred_boxes[sel][keep]
    final_boxes = torch.clamp(final_boxes, 0, 1)

    final_boxes_orig_t = final_boxes / scale * image_size
    masks_np = None
    if need_masks:
        # SAM2 masks on detected boxes (in 1024 space)
        boxes_1024 = final_boxes * image_size  # [0,1] → pixel in 1024-padded image
        masks_np = _generate_detection_masks(
            backbone_feats=feats,
            final_boxes_1024=boxes_1024,
            final_boxes_orig=final_boxes_orig_t,
            scale=scale,
            orig_h=image.shape[0],
            orig_w=image.shape[1],
        )

    final_boxes = final_boxes_orig_t.cpu().tolist()

    return final_boxes, len(final_boxes), exemplar_bboxes_orig, masks_np


@torch.no_grad()
def _run_inference_bbox(
    image: np.ndarray,
    bboxes: List[List[float]],
    threshold: float = 0.33,
    need_masks: bool = True,
):
    """Core inference with explicit exemplar bounding boxes (no SAM2 preprocessing).

    Matches demo_gradio.py: passes real bboxes to resize_and_pad for adaptive scaling,
    then feeds pixel-space bboxes directly to forward_detect (NOT normalized to [0,1]).
    """
    img_tensor, bboxes_scaled, scale = _preprocess_image_with_bboxes(image, bboxes)
    image_size = float(img_tensor.shape[-1])  # 1024.0

    with _inference_autocast():
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
        return [], 0, bboxes, None

    thr_inv = 1.0 / threshold
    sel = box_v > (box_v.max() / thr_inv)
    if sel.sum() == 0:
        return [], 0, bboxes, None

    keep = ops.nms(pred_boxes_raw[sel], box_v[sel], 0.5)
    final_boxes = pred_boxes_raw[sel][keep]
    final_boxes = torch.clamp(final_boxes, 0, 1)

    final_boxes_orig_t = final_boxes / scale * image_size
    masks_np = None
    if need_masks:
        # SAM2 masks on detected boxes (in 1024 space)
        boxes_1024 = final_boxes * image_size
        masks_np = _generate_detection_masks(
            backbone_feats=feats,
            final_boxes_1024=boxes_1024,
            final_boxes_orig=final_boxes_orig_t,
            scale=scale,
            orig_h=image.shape[0],
            orig_w=image.shape[1],
        )

    final_boxes = final_boxes_orig_t.cpu().tolist()

    return final_boxes, len(final_boxes), bboxes, masks_np


@torch.no_grad()
def _generate_detection_masks(
    backbone_feats: dict,
    final_boxes_1024: torch.Tensor,
    final_boxes_orig: torch.Tensor,
    scale: float,
    orig_h: int,
    orig_w: int,
) -> Optional[np.ndarray]:
    """Run SAM2 on detected box centres to produce per-object masks.

    Args:
        backbone_feats: Pass 2 backbone output (1024-space).
        final_boxes_1024: Detected boxes [N, 4] in 1024-padded pixel coords (x1, y1, x2, y2).
        final_boxes_orig: Detected boxes [N, 4] in original-image pixel coords.
        scale: Scale factor from resize_and_pad.
        orig_h, orig_w: Original image dimensions.

    Returns:
        masks_np: Boolean array [N, orig_h, orig_w] or None if no boxes.
    """
    if final_boxes_1024.numel() == 0:
        return None

    # Centre points of detected boxes
    cx = (final_boxes_1024[:, 0] + final_boxes_1024[:, 2]) / 2
    cy = (final_boxes_1024[:, 1] + final_boxes_1024[:, 3]) / 2
    point_coords = torch.stack([cx, cy], dim=1)  # [N, 2]
    point_labels = torch.ones(len(point_coords), dtype=torch.int32, device=point_coords.device)

    # Expected areas from box sizes (helps mask selection)
    box_areas = (final_boxes_1024[:, 2] - final_boxes_1024[:, 0]) * \
                (final_boxes_1024[:, 3] - final_boxes_1024[:, 1])

    with _inference_autocast():
        masks_1024, _, _ = _pipeline.cnt.sam_mask.predict_masks_from_points(
            backbone_feats=backbone_feats,
            point_coords=point_coords,
            point_labels=point_labels,
            expected_areas=box_areas,
        )  # [N, 1024, 1024] bool

    # Crop padded region and resize to original image size
    padded_h = int(round(orig_h * scale))
    padded_w = int(round(orig_w * scale))
    masks_crop = masks_1024[:, :padded_h, :padded_w]  # remove padding
    masks_resized = torch.nn.functional.interpolate(
        masks_crop.unsqueeze(1).float(),
        size=(orig_h, orig_w),
        mode="nearest",
    ).squeeze(1) > 0.5  # [N, orig_h, orig_w]

    masks_np = masks_resized.cpu().numpy()

    # Hard-clip each mask to its corresponding detected bounding box in original space.
    # This removes occasional SAM2 leakage outside the box sent to clients.
    clipped_masks = np.zeros_like(masks_np, dtype=bool)
    boxes_np = final_boxes_orig.detach().cpu().numpy()
    for i, (x1, y1, x2, y2) in enumerate(boxes_np):
        x1i = max(0, min(orig_w - 1, int(np.floor(x1))))
        y1i = max(0, min(orig_h - 1, int(np.floor(y1))))
        x2i = max(0, min(orig_w, int(np.ceil(x2))))
        y2i = max(0, min(orig_h, int(np.ceil(y2))))
        if x2i <= x1i or y2i <= y1i:
            continue
        clipped_masks[i, y1i:y2i, x1i:x2i] = masks_np[i, y1i:y2i, x1i:x2i]

    return clipped_masks


_INSTANCE_PALETTE: List[Tuple[int, int, int]] = [
    (255, 191, 71),   # amber
    (135, 206, 250),  # sky
    (153, 230, 179),  # mint
    (255, 179, 186),  # coral pink
    (176, 224, 230),  # powder cyan
    (255, 214, 165),  # apricot
    (199, 243, 205),  # soft lime
    (255, 205, 210),  # blush
    (173, 216, 230),  # airy blue
    (255, 228, 181),  # moccasin
]


def _reading_order_indices(boxes: List[List[float]]) -> List[int]:
    """Return indices sorted in reading order: top-to-bottom, left-to-right."""
    if not boxes:
        return []
    if len(boxes) == 1:
        return [0]

    heights = [max(1.0, b[3] - b[1]) for b in boxes]
    median_h = float(np.median(np.asarray(heights, dtype=np.float32)))
    row_band = max(20.0, 0.6 * median_h)

    def sort_key(i: int) -> Tuple[int, float, float]:
        x1, y1, x2, y2 = boxes[i]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        row = int(cy // row_band)
        return row, cx, cy

    return sorted(range(len(boxes)), key=sort_key)


def _draw_boxes(
    pil_image: Image.Image,
    pred_boxes: List[List[float]],
    exemplar_boxes: List[List[float]],
    count: int,
    masks: Optional[np.ndarray] = None,
) -> Image.Image:
    """Draw bounding boxes and optional masks on a copy of the image.

    pred_boxes     → green rectangles (detected objects)
    exemplar_boxes → red rectangles (exemplar / query objects)
    masks          → semi-transparent green overlay per detected object
    Count is rendered in the top-left corner.
    """
    img = pil_image.copy().convert("RGBA")
    ordered_indices = _reading_order_indices(pred_boxes)

    # Draw semi-transparent mask overlay
    if masks is not None and len(masks) > 0:
        overlay_np = np.zeros((img.size[1], img.size[0], 4), dtype=np.uint8)
        # Color masks in reading order, so colors/numbers align with displayed order.
        for rank, src_idx in enumerate(ordered_indices):
            if src_idx >= len(masks):
                continue
            mask = masks[src_idx]
            color = _INSTANCE_PALETTE[rank % len(_INSTANCE_PALETTE)]
            overlay_np[mask, 0] = color[0]
            overlay_np[mask, 1] = color[1]
            overlay_np[mask, 2] = color[2]
            overlay_np[mask, 3] = 130
        if np.any(overlay_np[..., 3]):
            overlay = Image.fromarray(overlay_np, mode="RGBA")
            img = Image.alpha_composite(img, overlay)

    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = None
    index_font = None
    for font_path in (
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            font = ImageFont.truetype(font_path, size=24)
            index_font = ImageFont.truetype(font_path, size=36)
            break
        except OSError:
            continue
    if font is None or index_font is None:
        font = ImageFont.load_default()
        index_font = font

    for i, box in enumerate(exemplar_boxes):
        x1, y1, x2, y2 = box
        # Exemplar BBs: lighter neutral style for distinction.
        draw.rectangle([x1, y1, x2, y2], outline=(160, 160, 160), width=3)
        tag = f"Ref{i + 1}"
        tx = max(0, int(x1))
        ty = max(0, int(y1) - 18)
        tbox = draw.textbbox((tx, ty), tag, font=font)
        draw.rectangle([tbox[0] - 3, tbox[1] - 2, tbox[2] + 3, tbox[3] + 2], fill=(235, 235, 235))
        draw.text((tx, ty), tag, fill=(70, 70, 70), font=font)

    for rank, src_idx in enumerate(ordered_indices):
        box = pred_boxes[src_idx]
        x1, y1, x2, y2 = box
        color = _INSTANCE_PALETTE[rank % len(_INSTANCE_PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
        # Detection index label: 1, 2, 3, ...
        tag = str(rank + 1)
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        tbox = draw.textbbox((0, 0), tag, font=index_font)
        tw = tbox[2] - tbox[0]
        th = tbox[3] - tbox[1]
        tx = max(0, min(img.width - tw, cx - tw // 2))
        ty = max(0, min(img.height - th, cy - th // 2))
        draw.text((tx, ty), tag, fill=(0, 0, 0), font=index_font)

    # Count label
    label = f"Count: {count}"
    margin = 6
    bbox = draw.textbbox((margin, margin), label, font=font)
    draw.rectangle([bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin],
                   fill=(0, 0, 0, 180))
    draw.text((margin, margin), label, fill=(255, 255, 255), font=font)

    return img


def _image_response(
    pil_image: Image.Image,
    response_format: str = DEFAULT_IMAGE_FORMAT,
    max_side: int = DEFAULT_IMAGE_MAX_SIDE,
    quality: int = DEFAULT_IMAGE_QUALITY,
) -> StreamingResponse:
    response_format = (response_format or DEFAULT_IMAGE_FORMAT).lower()
    if response_format not in _SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format: {response_format}")

    quality = max(1, min(100, int(quality)))
    max_side = max(0, int(max_side))

    img_out = pil_image
    if max_side > 0:
        w, h = img_out.size
        longer = max(w, h)
        if longer > max_side:
            scale = max_side / float(longer)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img_out = img_out.resize((new_w, new_h), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    if response_format == "png":
        img_out.save(buf, format="PNG", compress_level=PNG_COMPRESS_LEVEL, optimize=False)
        media_type = "image/png"
    elif response_format == "jpeg":
        img_out.convert("RGB").save(
            buf, format="JPEG", quality=quality, optimize=False, progressive=False
        )
        media_type = "image/jpeg"
    else:  # webp
        img_out.convert("RGB").save(
            buf, format="WEBP", quality=quality, method=4
        )
        media_type = "image/webp"
    buf.seek(0)
    return StreamingResponse(buf, media_type=media_type)




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
    pred_boxes, count, exemplar_boxes, _masks = _run_inference(
        image_np, points_list, labels_list, threshold=threshold, need_masks=False,
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
    with_masks: bool = Form(default=True, description="If true, overlays SAM2 masks on output image."),
    response_format: str = Form(default=DEFAULT_IMAGE_FORMAT, description="Response format: png|jpeg|webp."),
    max_side: int = Form(default=DEFAULT_IMAGE_MAX_SIDE, description="Resize output image so longest side <= max_side. 0 disables."),
    quality: int = Form(default=DEFAULT_IMAGE_QUALITY, description="JPEG/WEBP quality (1-100)."),
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

    _contents, pil, image_np = await _read_upload_image(image)

    pred_boxes, count, exemplar_boxes, masks = _run_inference(
        image_np, points_list, labels_list, threshold=threshold, need_masks=with_masks,
    )
    return _image_response(
        _draw_boxes(pil, pred_boxes, exemplar_boxes, count, masks=masks),
        response_format=response_format,
        max_side=max_side,
        quality=quality,
    )


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
    contents, _pil, image_np = await _read_upload_image(image)

    # --- Call AODC service to get multiple seed points ---
    aodc_result = await _request_aodc_predict_multi(contents, image.filename or "image.jpg")

    points = aodc_result.get("points", [])
    if not points:
        return JSONResponse({"count": 0, "pred_boxes": [], "exemplar_boxes": []})

    # Decode density map only when multiple seeds exist (single-point case doesn't need area filtering).
    expected_areas = None
    if len(points) > 1:
        expected_areas = _decode_aodc_areas(aodc_result, image_np.shape[0], image_np.shape[1])

    # Pre-filter: remove small-hole peaks before SAM2
    if expected_areas is not None:
        points, expected_areas = _filter_aodc_points_by_area(points, expected_areas)

    if not points:
        return JSONResponse({"count": 0, "pred_boxes": [], "exemplar_boxes": []})

    # --- Run local SAM2+GECO2 pipeline with the AODC seed points ---
    pred_boxes, count, exemplar_boxes, _masks = _run_inference(
        image_np,
        points=points,
        labels=[1] * len(points),
        threshold=threshold,
        expected_areas=expected_areas,
        need_masks=False,
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
    with_masks: bool = Form(default=True, description="If true, overlays SAM2 masks on output image."),
    response_format: str = Form(default=DEFAULT_IMAGE_FORMAT, description="Response format: png|jpeg|webp."),
    max_side: int = Form(default=DEFAULT_IMAGE_MAX_SIDE, description="Resize output image so longest side <= max_side. 0 disables."),
    quality: int = Form(default=DEFAULT_IMAGE_QUALITY, description="JPEG/WEBP quality (1-100)."),
):
    """Same as /predict_auto but returns a PNG image with bounding boxes drawn on it.

    - Green boxes: detected objects
    - Red boxes: exemplar (query) objects
    - Count label in top-left corner
    """
    contents, pil, image_np = await _read_upload_image(image)

    aodc_result = await _request_aodc_predict_multi(contents, image.filename or "image.jpg")

    points = aodc_result.get("points", [])
    if not points:
        return _image_response(_draw_boxes(pil, [], [], 0))

    # Decode density map only when multiple seeds exist (single-point case doesn't need area filtering).
    expected_areas = None
    if len(points) > 1:
        expected_areas = _decode_aodc_areas(aodc_result, image_np.shape[0], image_np.shape[1])

    # Pre-filter: remove small-hole peaks before SAM2
    if expected_areas is not None:
        points, expected_areas = _filter_aodc_points_by_area(points, expected_areas)

    if not points:
        return _image_response(_draw_boxes(pil, [], [], 0))

    pred_boxes, count, exemplar_boxes, masks = _run_inference(
        image_np, points=points, labels=[1] * len(points), threshold=threshold,
        expected_areas=expected_areas, need_masks=with_masks,
    )
    return _image_response(
        _draw_boxes(pil, pred_boxes, exemplar_boxes, count, masks=masks),
        response_format=response_format,
        max_side=max_side,
        quality=quality,
    )


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

    _contents, _pil, image_np = await _read_upload_image(image)

    pred_boxes, count, exemplar_boxes, _masks = _run_inference_bbox(
        image_np, bboxes_list, threshold=threshold, need_masks=False
    )
    return JSONResponse({"count": count, "pred_boxes": pred_boxes, "exemplar_boxes": exemplar_boxes})


@app.post("/predict_bbox/image", response_class=StreamingResponse)
async def predict_bbox_image(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    bboxes: str = Form(..., description='JSON array of exemplar bboxes [[x1,y1,x2,y2], ...] in original image pixel coordinates'),
    threshold: float = Form(default=0.33, description="Detection confidence threshold (0.05–0.95)"),
    with_masks: bool = Form(default=True, description="If true, overlays SAM2 masks on output image."),
    response_format: str = Form(default=DEFAULT_IMAGE_FORMAT, description="Response format: png|jpeg|webp."),
    max_side: int = Form(default=DEFAULT_IMAGE_MAX_SIDE, description="Resize output image so longest side <= max_side. 0 disables."),
    quality: int = Form(default=DEFAULT_IMAGE_QUALITY, description="JPEG/WEBP quality (1-100)."),
):
    """Same as /predict_bbox but returns a PNG image with bounding boxes drawn on it."""
    import json as _json

    try:
        bboxes_list: List[List[float]] = _json.loads(bboxes)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid bboxes JSON: {e}")
    if not bboxes_list or not all(len(b) == 4 for b in bboxes_list):
        raise HTTPException(status_code=400, detail="bboxes must be a non-empty array of [x1, y1, x2, y2]")

    _contents, pil, image_np = await _read_upload_image(image)

    pred_boxes, count, exemplar_boxes, masks = _run_inference_bbox(
        image_np, bboxes_list, threshold=threshold, need_masks=with_masks
    )
    return _image_response(
        _draw_boxes(pil, pred_boxes, exemplar_boxes, count, masks=masks),
        response_format=response_format,
        max_side=max_side,
        quality=quality,
    )


# ---------------------------------------------------------------------------
# SAM2 grid-based automatic exemplar generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_auto_exemplars_sam(
    image_np: np.ndarray,
    grid_size: int = 16,
    iou_thresh: float = 0.7,
    nms_thresh: float = 0.5,
    num_exemplars: int = 3,
) -> Optional[List[List[float]]]:
    """Generate exemplar BBs automatically via SAM2 grid-point mask generation.

    1. Preprocess image (zero-shot scale)
    2. Run backbone
    3. Create grid_size x grid_size uniform point prompts
    4. SAM2 predict_masks_from_points → masks, IoU scores, bboxes
    5. Filter by IoU → NMS
    6. Cluster surviving bboxes by area (log-scale histogram) → find mode
    7. Return top-N exemplars from the most common size group

    Returns exemplar bboxes in original image pixel coordinates, or None if no masks survived.
    """
    img_tensor, scale_zs = _preprocess_image(image_np)
    with _inference_autocast():
        feats = _pipeline.cnt.forward_backbone(img_tensor)

    # Uniform grid of foreground point prompts in 1024px padded space
    margin = 1024.0 / (grid_size + 1)
    grid = torch.linspace(margin, 1024.0 - margin, grid_size)
    ys, xs = torch.meshgrid(grid, grid, indexing="ij")
    point_coords = torch.stack([xs.flatten(), ys.flatten()], dim=1).to(_device)
    point_labels = torch.ones(len(point_coords), dtype=torch.int32, device=_device)

    with _inference_autocast():
        masks, ious, bboxes = _pipeline.cnt.sam_mask.predict_masks_from_points(
            backbone_feats=feats,
            point_coords=point_coords,
            point_labels=point_labels,
        )

    # Filter by IoU quality
    good = ious >= iou_thresh
    if good.sum() == 0:
        good = ious >= ious.max() * 0.5
    bboxes = bboxes[good]
    ious = ious[good]

    if bboxes.numel() == 0:
        return None

    # Remove near-zero-area boxes
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    valid = areas > 4.0  # at least 2x2 px
    bboxes, ious, areas = bboxes[valid], ious[valid], areas[valid]
    if bboxes.numel() == 0:
        return None

    # NMS to remove duplicates/overlaps
    keep = ops.nms(bboxes, ious, nms_thresh)
    bboxes, ious, areas = bboxes[keep], ious[keep], areas[keep]

    if len(bboxes) <= num_exemplars:
        return (bboxes / scale_zs).cpu().tolist()

    # Cluster by area: log-scale histogram → find the peak bin
    log_areas = torch.log(areas + 1).cpu().numpy()
    n_bins = max(5, min(20, len(log_areas) // 3))
    counts, bin_edges = np.histogram(log_areas, bins=n_bins)
    peak_bin = int(counts.argmax())
    bin_lo, bin_hi = bin_edges[peak_bin], bin_edges[peak_bin + 1]

    in_peak = (log_areas >= bin_lo) & (log_areas <= bin_hi)
    if in_peak.sum() == 0:
        in_peak = np.ones(len(log_areas), dtype=bool)

    peak_bboxes = bboxes[in_peak]
    peak_ious = ious[in_peak]

    # Take top exemplars by IoU from the most common size group
    top_k = min(num_exemplars, len(peak_bboxes))
    top_idx = peak_ious.argsort(descending=True)[:top_k]
    exemplar_bboxes = peak_bboxes[top_idx]

    return (exemplar_bboxes / scale_zs).cpu().tolist()


@torch.no_grad()
def _run_inference_auto_sam(
    image_np: np.ndarray,
    threshold: float = 0.33,
    grid_size: int = 16,
    num_exemplars: int = 3,
):
    """Automatic counting via SAM2 grid masks → exemplar BBs → GECO2 detection."""
    exemplar_bboxes_orig = _generate_auto_exemplars_sam(
        image_np, grid_size=grid_size, num_exemplars=num_exemplars,
    )
    if exemplar_bboxes_orig is None:
        return [], 0, [], None

    # Use the same adaptive-scaling detection path as /predict_bbox
    return _run_inference_bbox(image_np, exemplar_bboxes_orig, threshold=threshold)


@app.post("/predict_auto_sam")
async def predict_auto_sam(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    threshold: float = Form(default=0.33, description="Detection confidence threshold (0.05–0.95)"),
    grid_size: int = Form(default=16, description="Grid density per side (default 16 → 256 points)"),
    num_exemplars: int = Form(default=3, description="Number of exemplar BBs to select (1–10)"),
):
    """Fully automatic counting using SAM2 grid-based mask generation.

    Generates masks from a uniform grid of points using SAM2, clusters them
    by object size, selects representative exemplars, and feeds them to GECO2.

    No external service (AODC) needed — works entirely within the GECO2 container.

    **Request** (multipart/form-data):
    - `image`: image file
    - `threshold` *(optional)*: float, default 0.33
    - `grid_size` *(optional)*: int, default 16 (16×16=256 point prompts)
    - `num_exemplars` *(optional)*: int, default 3

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

    pred_boxes, count, exemplar_boxes, _masks = _run_inference_auto_sam(
        image_np, threshold=threshold, grid_size=grid_size, num_exemplars=num_exemplars,
    )
    return JSONResponse({"count": count, "pred_boxes": pred_boxes, "exemplar_boxes": exemplar_boxes})


@app.post("/predict_auto_sam/image", response_class=StreamingResponse)
async def predict_auto_sam_image(
    image: UploadFile = File(..., description="Input image (JPEG/PNG)"),
    threshold: float = Form(default=0.33, description="Detection confidence threshold (0.05–0.95)"),
    grid_size: int = Form(default=16, description="Grid density per side (default 16 → 256 points)"),
    num_exemplars: int = Form(default=3, description="Number of exemplar BBs to select (1–10)"),
):
    """Same as /predict_auto_sam but returns a PNG image with bounding boxes drawn on it."""
    try:
        contents = await image.read()
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(pil)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    pred_boxes, count, exemplar_boxes, masks = _run_inference_auto_sam(
        image_np, threshold=threshold, grid_size=grid_size, num_exemplars=num_exemplars,
    )
    return _image_response(_draw_boxes(pil, pred_boxes, exemplar_boxes, count, masks=masks))


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
