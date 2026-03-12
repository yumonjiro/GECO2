"""Inference script for point-prompt → SAM2 mask → GECO2 counting pipeline.

Usage:
    # Interactive mode via Gradio UI
    python inference_point.py

    # CLI mode with explicit points
    python inference_point.py --image path/to/image.jpg --points '[[100,200],[300,400]]' --labels '[1,1]'
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import DataParallel
from torchvision import transforms as T
import torchvision.ops as ops

from models.counter_infer import build_model
from models.point_to_count import PointToCountPipeline
from utils.arg_parser import get_argparser
from utils.data import resize_and_pad


def load_pipeline(checkpoint_path: str = "CNTQG_multitrain_ca44.pth", iou_threshold: float = 0.7):
    """Load the GECO2 model and wrap it in the PointToCountPipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_argparser().parse_args([])
    args.zero_shot = True
    cnt_model = build_model(args).to(device)

    # Load pretrained weights
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # DataParallel-saved checkpoints prefix keys with 'module.'
    state_dict = state["model"]
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
    cnt_model.load_state_dict(cleaned, strict=False)
    cnt_model.eval()

    pipeline = PointToCountPipeline(cnt_model, iou_threshold=iou_threshold).to(device)
    pipeline.eval()
    return pipeline, device


def preprocess_image(image: np.ndarray, device: torch.device):
    """Normalize and pad an HWC uint8 image to [1, 3, 1024, 1024] (zero-shot, no exemplar bboxes)."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
    # resize_and_pad needs dummy bboxes; just pass a single dummy box
    dummy_bbox = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
    padded_img, _, scale = resize_and_pad(tensor, dummy_bbox, size=1024.0, zero_shot=True)
    return padded_img.unsqueeze(0).to(device), scale


def preprocess_image_with_bboxes(image: np.ndarray, bboxes: list, device: torch.device):
    """Normalize, resize-and-pad with adaptive scaling (~80px objects)."""
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
    padded, bboxes_scaled, scale = resize_and_pad(tensor, bboxes_tensor, size=1024.0)
    return padded.unsqueeze(0).to(device), bboxes_scaled.unsqueeze(0).to(device), scale


def run_point_counting(
    pipeline: PointToCountPipeline,
    image: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    threshold: float = 0.33,
):
    """Run the full point-to-count pipeline on a single image (two-pass with adaptive scaling).

    Pass 1: zero-shot preprocessing → backbone → SAM2 point→mask→bbox (get exemplars).
    Pass 2: re-preprocess with adaptive scaling using those bboxes → backbone → detection.

    Args:
        pipeline: Loaded PointToCountPipeline.
        image: HWC uint8 numpy image.
        points: Nx2 array of (x, y) point coordinates in original image space.
        labels: N array of labels (1=foreground, 0=background).
        device: Torch device.
        threshold: Detection confidence threshold factor.

    Returns:
        pred_boxes: List of [x1, y1, x2, y2] in original image coordinates.
        count: Number of detected objects.
        exemplar_masks: Binary masks [N, H, W] of SAM2-generated exemplar regions.
        exemplar_bboxes: [N, 4] exemplar bounding boxes in original image pixel coordinates.
    """
    # --- Pass 1: SAM2 exemplar extraction at zero-shot scale ---
    img_tensor_zs, scale_zs = preprocess_image(image, device)
    point_coords = torch.tensor(points, dtype=torch.float32, device=device) * scale_zs
    point_labels = torch.tensor(labels, dtype=torch.int32, device=device)

    feats_zs = pipeline.cnt.forward_backbone(img_tensor_zs)
    exemplar_masks, exemplar_ious, exemplar_bboxes = \
        pipeline.cnt.sam_mask.predict_masks_from_points(
            backbone_feats=feats_zs,
            point_coords=point_coords,
            point_labels=point_labels,
        )

    # Filter low-quality masks
    keep_mask = exemplar_ious >= pipeline.iou_threshold
    if keep_mask.sum() == 0:
        keep_mask = exemplar_ious >= exemplar_ious.max() * 0.5
    bboxes_px = exemplar_bboxes[keep_mask]

    if bboxes_px.numel() == 0:
        return [], 0, exemplar_masks.cpu(), exemplar_bboxes.cpu()

    # Convert SAM bboxes back to original image coordinates
    bboxes_orig = (bboxes_px / scale_zs).cpu().tolist()

    # --- Pass 2: adaptive re-preprocessing + detection ---
    img_tensor, bboxes_scaled, scale = preprocess_image_with_bboxes(image, bboxes_orig, device)
    image_size = float(img_tensor.shape[-1])

    feats = pipeline.cnt.forward_backbone(img_tensor)
    det_results = pipeline.cnt.forward_detect(feats, bboxes_scaled, image_size=image_size)

    outputs = det_results[0]
    pred_boxes = outputs[0]["pred_boxes"]
    box_v = outputs[0]["box_v"]

    if pred_boxes.dim() == 3:
        pred_boxes = pred_boxes[0]
    if box_v.dim() == 2:
        box_v = box_v[0]

    thr_inv = 1.0 / threshold
    sel = box_v > (box_v.max() / thr_inv) if box_v.numel() > 0 else torch.zeros(0, dtype=torch.bool)

    if sel.sum() == 0:
        return [], 0, exemplar_masks.cpu(), torch.tensor(bboxes_orig)

    keep = ops.nms(pred_boxes[sel], box_v[sel], 0.5)
    pred_boxes = pred_boxes[sel][keep]
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    pred_boxes = (pred_boxes / scale * image_size).cpu().tolist()

    return pred_boxes, len(pred_boxes), exemplar_masks.cpu(), torch.tensor(bboxes_orig)


def visualize_result(
    image: np.ndarray,
    pred_boxes: list,
    exemplar_bboxes: torch.Tensor,
) -> Image.Image:
    """Draw detection and exemplar boxes on the image."""
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)

    # Draw detected boxes (orange)
    for box in pred_boxes:
        draw.rectangle(box, outline="orange", width=2)

    # Draw exemplar boxes (green, already in original image coordinates)
    for bbox in exemplar_bboxes:
        draw.rectangle(bbox.tolist(), outline="lime", width=3)

    # Counter badge
    w, h = pil_img.size
    sq = int(0.05 * w)
    x1, y1 = 10, h - sq - 10
    draw.rectangle([x1, y1, x1 + sq, y1 + sq], outline="black", fill="black")
    font = ImageFont.load_default()
    txt = str(len(pred_boxes))
    text_x = x1 + (sq - draw.textlength(txt, font=font)) / 2
    text_y = y1 + (sq - 10) / 2
    draw.text((text_x, text_y), txt, fill="white", font=font)

    return pil_img


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_gradio_app(pipeline, device):
    """Build a Gradio interface for interactive point-to-count inference."""
    import gradio as gr

    def process_click(image_np, point_type, threshold, evt: gr.SelectData):
        """Handle a click event on the image."""
        if image_np is None:
            return None, None, 0, [], []

        x, y = evt.index
        points_state = getattr(process_click, "_points", [])
        labels_state = getattr(process_click, "_labels", [])
        points_state.append([x, y])
        labels_state.append(1 if point_type == "Foreground" else 0)
        process_click._points = points_state
        process_click._labels = labels_state

        # Preview: show points on image
        preview = Image.fromarray(image_np)
        draw = ImageDraw.Draw(preview)
        for pt, lbl in zip(points_state, labels_state):
            color = "lime" if lbl == 1 else "red"
            r = 6
            draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=color, outline="white", width=2)
        return preview, image_np, 0, points_state, labels_state

    def run_counting(image_np, threshold, points_json, labels_json):
        """Run counting with accumulated points."""
        if image_np is None or not points_json:
            return None, 0

        points = np.array(points_json, dtype=np.float32)
        labels = np.array(labels_json, dtype=np.int32)

        pred_boxes, count, exemplar_masks, exemplar_bboxes = run_point_counting(
            pipeline, image_np, points, labels, device, threshold=threshold,
        )

        result = visualize_result(image_np, pred_boxes, exemplar_bboxes)
        return result, count

    def clear_points():
        """Reset accumulated points."""
        process_click._points = []
        process_click._labels = []
        return None, None, 0, [], []

    with gr.Blocks() as demo:
        gr.Markdown("""
# GECO2 — Point-to-Count Pipeline
**Click on example objects** in the image, then press **Count** to detect all similar objects.

### How to use:
1. Upload an image.
2. Click on one or more example objects (foreground points).
3. Press **Count** — SAM2 generates masks from your points, derives bounding boxes,
   and GECO2 counts all matching objects.
        """)

        image_state = gr.State(None)
        points_state = gr.State([])
        labels_state = gr.State([])

        with gr.Row():
            image_input = gr.Image(label="Upload image & click points", type="numpy")
            image_output = gr.Image(type="pil", label="Result")

        with gr.Row():
            point_type = gr.Radio(["Foreground", "Background"], value="Foreground", label="Point type")
            threshold = gr.Slider(0.05, 0.95, value=0.33, step=0.01, label="Threshold")
            count_output = gr.Number(label="Total Count", precision=0)

        with gr.Row():
            count_button = gr.Button("Count", variant="primary")
            clear_button = gr.Button("Clear Points")

        image_input.select(
            process_click,
            [image_input, point_type, threshold],
            [image_input, image_state, count_output, points_state, labels_state],
        )

        count_button.click(
            run_counting,
            [image_state, threshold, points_state, labels_state],
            [image_output, count_output],
        )

        clear_button.click(
            clear_points,
            [],
            [image_input, image_state, count_output, points_state, labels_state],
        )

    return demo


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Point-to-Count inference")
    parser.add_argument("--image", type=str, help="Path to input image (CLI mode)")
    parser.add_argument("--points", type=str, help="JSON list of [x,y] points (CLI mode)")
    parser.add_argument("--labels", type=str, help="JSON list of labels (1=fg, 0=bg)")
    parser.add_argument("--threshold", type=float, default=0.33)
    parser.add_argument("--iou_threshold", type=float, default=0.7)
    parser.add_argument("--checkpoint", type=str, default="CNTQG_multitrain_ca44.pth")
    parser.add_argument("--output", type=str, default=None, help="Output image path (CLI mode)")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio UI")
    args = parser.parse_args()

    pipeline, device = load_pipeline(args.checkpoint, iou_threshold=args.iou_threshold)

    if args.gradio or args.image is None:
        demo = build_gradio_app(pipeline, device)
        demo.queue(max_size=8)
        demo.launch(show_error=True, debug=True, share=True, max_threads=1)
    else:
        # CLI mode
        image = np.array(Image.open(args.image).convert("RGB"))
        points = np.array(json.loads(args.points), dtype=np.float32)
        labels = np.array(json.loads(args.labels), dtype=np.int32)

        pred_boxes, count, exemplar_masks, exemplar_bboxes = run_point_counting(
            pipeline, image, points, labels, device, threshold=args.threshold,
        )

        print(f"Detected: {count} objects")
        for i, box in enumerate(pred_boxes):
            print(f"  Box {i}: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

        if args.output:
            result = visualize_result(image, pred_boxes, exemplar_bboxes)
            result.save(args.output)
            print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
