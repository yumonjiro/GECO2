"""Point-to-Count pipeline: SAM2 point-prompt segmentation → GECO2 counting.

Shares the backbone (Hiera + FPN) between SAM2 mask generation and GECO2 detection,
so image encoding runs only once.

Data flow:
    Image → Backbone → features (computed once)
      ├→ features + point_prompts → SAM2 MaskDecoder → masks → exemplar BBs
      └→ features + exemplar_BBs  → GECO2 detection  → count
"""

import torch
from torch import nn
from typing import Optional, Tuple, List


class PointToCountPipeline(nn.Module):
    """Wraps a GECO2 CNT model to accept point prompts instead of bounding boxes.

    Uses the CNT model's shared backbone and MaskProcessor to:
    1. Encode the image once via the shared backbone.
    2. Generate segmentation masks from user-specified point prompts using SAM2.
    3. Derive exemplar bounding boxes from the masks.
    4. Feed those bounding boxes into GECO2's detection pipeline.
    """

    def __init__(self, cnt_model: nn.Module, iou_threshold: float = 0.7):
        super().__init__()
        self.cnt = cnt_model
        self.iou_threshold = iou_threshold

    @torch.no_grad()
    def forward(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor, object]:
        """Run the full point-to-count pipeline.

        Args:
            image: Input image tensor [B, 3, H, W] (normalized, padded to 1024).
            point_coords: Point coordinates [N, 2] in pixel space (x, y).
            point_labels: Point labels [N], 1=foreground, 0=background.

        Returns:
            Same outputs as CNT.forward(): (outputs, ref_points, centerness, outputs_coord, masks)
        """
        # Step 1: Shared backbone encoding (runs once)
        feats = self.cnt.forward_backbone(image)

        # Step 2: SAM2 point-prompt → masks → bounding boxes
        masks, iou_scores, bboxes_px = self.cnt.sam_mask.predict_masks_from_points(
            backbone_feats=feats,
            point_coords=point_coords,
            point_labels=point_labels,
        )

        # Step 3: Filter low-quality masks by IoU score
        keep = iou_scores >= self.iou_threshold
        if keep.sum() == 0:
            # Fallback: keep at least the best mask per point
            keep = iou_scores >= iou_scores.max() * 0.5
        bboxes_px = bboxes_px[keep]
        masks = masks[keep]

        # Step 4: Normalize bboxes to [0, 1] range for GECO2
        image_size = float(image.shape[-1])
        bboxes_norm = bboxes_px / image_size  # [M, 4]
        bboxes_norm = bboxes_norm.unsqueeze(0).to(image.device)  # [1, M, 4]

        # Step 5: Run GECO2 detection with shared features
        return self.cnt.forward_detect(feats, bboxes_norm, image_size=image_size)

    @torch.no_grad()
    def predict_exemplar_masks(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate exemplar masks from points without running counting (useful for visualization).

        Returns:
            masks: Binary masks [N, H, W].
            iou_scores: IoU predictions [N].
            bboxes: Bounding boxes [N, 4] in pixel coordinates.
        """
        feats = self.cnt.forward_backbone(image)
        return self.cnt.sam_mask.predict_masks_from_points(
            backbone_feats=feats,
            point_coords=point_coords,
            point_labels=point_labels,
        )

    @torch.no_grad()
    def forward_with_exemplars(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
    ) -> Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor, object,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run backbone once, return both exemplar info and detection results.

        Returns:
            outputs, ref_points, centerness, outputs_coord, masks,
            exemplar_masks, exemplar_ious, exemplar_bboxes
        """
        # Backbone runs only once
        feats = self.cnt.forward_backbone(image)

        # SAM2 point-prompt → masks → exemplar bounding boxes
        exemplar_masks, exemplar_ious, exemplar_bboxes = \
            self.cnt.sam_mask.predict_masks_from_points(
                backbone_feats=feats,
                point_coords=point_coords,
                point_labels=point_labels,
            )

        # Filter low-quality masks
        keep = exemplar_ious >= self.iou_threshold
        if keep.sum() == 0:
            keep = exemplar_ious >= exemplar_ious.max() * 0.5
        bboxes_px = exemplar_bboxes[keep]

        # Normalize bboxes to [0, 1] for GECO2
        image_size = float(image.shape[-1])
        bboxes_norm = bboxes_px / image_size
        bboxes_norm = bboxes_norm.unsqueeze(0).to(image.device)

        # Detection with shared features (no second backbone call)
        det_results = self.cnt.forward_detect(feats, bboxes_norm, image_size=image_size)

        return (*det_results, exemplar_masks, exemplar_ious, exemplar_bboxes)
