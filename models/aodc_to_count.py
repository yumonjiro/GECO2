"""AutoCountPipeline: AODC → SAM2 → GECO2 end-to-end counting.

Chains AODCWrapper (density-map peak extraction) with PointToCountPipeline
(SAM2 exemplar masking + GECO2 detection) to produce fully automatic, zero-prompt
object counting from a single image.

Data flow:
    image_np
      ├─ AODCWrapper.run()  →  (x_orig, y_orig)
      │                         ↓  × geco2_scale
      │                   point_coords [1, 2]  (1024px space)
      └─ GECO2 preprocess  →  img_tensor [1,3,1024,1024] + scale
                               ↓
                         PointToCountPipeline.forward_with_exemplars()
                               ↓
                         (pred_boxes, count, exemplar_bboxes)
"""

from typing import List, Tuple

import numpy as np
import torch
import torchvision.ops as ops
from torchvision import transforms as T

from models.aodc_wrapper import AODCWrapper
from models.point_to_count import PointToCountPipeline
from utils.data import resize_and_pad

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def _preprocess_for_geco2(
    image_np: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Prepare image tensor for GECO2 (1024×1024 padded, ImageNet-normalised)."""
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    tensor = _normalize(tensor)
    dummy_bbox = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
    padded, _, scale = resize_and_pad(tensor, dummy_bbox, size=1024.0, zero_shot=True)
    return padded.unsqueeze(0).to(device), float(scale)


class AutoCountPipeline:
    """Zero-prompt counting pipeline: AODC seeds a single point → SAM2 + GECO2 counts.

    Args:
        aodc: Initialised AODCWrapper.
        ptc_pipeline: Initialised PointToCountPipeline (the existing GECO2 pipeline).
        device: torch.device (must match both models).
    """

    def __init__(
        self,
        aodc: AODCWrapper,
        ptc_pipeline: PointToCountPipeline,
        device: torch.device,
    ):
        self.aodc = aodc
        self.ptc = ptc_pipeline
        self.device = device

    def run(
        self,
        image_np: np.ndarray,
        threshold: float = 0.33,
    ) -> Tuple[List[List[float]], int, List[List[float]]]:
        """Run the full AODC→SAM2→GECO2 pipeline.

        Args:
            image_np: HWC uint8 RGB numpy array.
            threshold: NMS confidence threshold, same semantics as /predict endpoint.

        Returns:
            (pred_boxes, count, exemplar_boxes):
                pred_boxes      — [[x1,y1,x2,y2], ...] in original image pixels
                count           — number of detected objects
                exemplar_boxes  — [[x1,y1,x2,y2], ...] in 1024px padded space
        """
        # Step 1: AODC → top-1 density peak (original image coords)
        x_orig, y_orig = self.aodc.run(image_np)

        # Step 2: GECO2 preprocessing
        img_tensor, scale = _preprocess_for_geco2(image_np, self.device)

        # Step 3: Scale point to 1024px padded space
        point_coords = torch.tensor(
            [[x_orig * scale, y_orig * scale]], dtype=torch.float32, device=self.device
        )  # [1, 2]
        point_labels = torch.tensor([1], dtype=torch.int32, device=self.device)  # foreground

        # Step 4: SAM2 + GECO2 (backbone runs once)
        outputs, _, _, _, _masks, _ex_masks, _ex_ious, exemplar_bboxes = \
            self.ptc.forward_with_exemplars(img_tensor, point_coords, point_labels)

        # Step 5: Post-processing (mirrors _run_inference in api_server.py)
        pred_boxes_raw = outputs[0]["pred_boxes"]
        box_v = outputs[0]["box_v"]
        if pred_boxes_raw.dim() == 3:
            pred_boxes_raw = pred_boxes_raw[0]
        if box_v.dim() == 2:
            box_v = box_v[0]

        if box_v.numel() == 0:
            return [], 0, exemplar_bboxes.cpu().tolist()

        thr_inv = 1.0 / threshold
        sel = box_v > (box_v.max() / thr_inv)
        if sel.sum() == 0:
            return [], 0, exemplar_bboxes.cpu().tolist()

        keep = ops.nms(pred_boxes_raw[sel], box_v[sel], 0.5)
        final_boxes = pred_boxes_raw[sel][keep]
        final_boxes = torch.clamp(final_boxes, 0, 1)
        final_boxes = (final_boxes / scale * img_tensor.shape[-1]).cpu().tolist()

        return final_boxes, len(final_boxes), exemplar_bboxes.cpu().tolist()
