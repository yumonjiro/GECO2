"""AODCWrapper: loads AODC model and extracts the top-1 density peak as a point.

The AODC model is used in reference-less mode (zero_shot=False, few_shot=False),
meaning it needs only the image as input and produces a density map via self-similarity.

Usage:
    wrapper = AODCWrapper("path/to/aodc.pth", device)
    x, y = wrapper.run(image_np)   # original-image pixel coordinates
"""

import sys
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T

# ---------------------------------------------------------------------------
# Make AODC importable from the GECO2/AODC/ subdirectory.
#
# IMPORTANT — naming convention:
#   AODC's models/ directory must be renamed to aodc_models/ after cloning,
#   to avoid shadowing GECO2's own 'models' package in sys.modules:
#
#     git clone https://github.com/WJWu20/AODC.git GECO2/AODC
#     mv GECO2/AODC/models GECO2/AODC/aodc_models
#
# AODC uses only relative imports internally, so the rename is safe.
# GECO2/AODC/ is added to sys.path solely for AODC's vendored 'clip/' package.
# ---------------------------------------------------------------------------
_AODC_ROOT = Path(__file__).parent.parent / "AODC"

_aodc_models_dir = _AODC_ROOT / "aodc_models"
if not _aodc_models_dir.exists():
    raise RuntimeError(
        f"AODC models directory not found at '{_aodc_models_dir}'.\n"
        "Please clone AODC and rename its models/ directory:\n"
        "  git clone https://github.com/WJWu20/AODC.git GECO2/AODC\n"
        "  mv GECO2/AODC/models GECO2/AODC/aodc_models"
    )

if str(_AODC_ROOT) not in sys.path:
    sys.path.append(str(_AODC_ROOT))  # append (not insert) to avoid shadowing GECO2 packages

from aodc_models import build_model as _aodc_build_model  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal config object (replaces yacs CfgNode for inference-only use)
# ---------------------------------------------------------------------------
class _AODCConfig:
    """Minimal stand-in for AODC's CfgNode — only the fields used by AODC(config)."""
    ZERO_SHOT: bool = False
    FEW_SHOT: bool = False
    FACTOR: float = 128.0
    NAME: str = "AODC"


# ---------------------------------------------------------------------------
# ImageNet normalisation (same stats AODC was trained with)
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


class AODCWrapper:
    """Wraps AODC for reference-less (zero_shot=False, few_shot=False) inference.

    Given a raw numpy image, runs AODC and returns the single pixel location
    (in original image coordinates) corresponding to the highest-density peak.

    Args:
        checkpoint_path: Path to the AODC checkpoint (.pth).
        device: torch.device to run inference on.
        input_size: (H, W) that the model was trained at. Default (384, 576) matches
                    AODC's gendata384x576.py preprocessing.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        input_size: Tuple[int, int] = (384, 576),
    ):
        self.device = device
        self.input_size = input_size  # (H, W)

        config = _AODCConfig()
        model, _ = _aodc_build_model(config)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Support checkpoints saved as {"model": state_dict} or raw state_dicts
        state_dict = state.get("model", state)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        self._model = model

        # Dummy boxes tensor (not used when zero_shot=False, few_shot=False)
        self._dummy_boxes = torch.zeros(1, 5, device=device)

    @torch.no_grad()
    def run(self, image_np: np.ndarray) -> Tuple[float, float]:
        """Return the top-1 density-map peak in original image pixel coordinates.

        Args:
            image_np: HWC uint8 numpy array (RGB).

        Returns:
            (x, y): float coordinates in original image pixel space.
        """
        orig_h, orig_w = image_np.shape[:2]
        aodc_h, aodc_w = self.input_size

        # Preprocess: HWC uint8 → CHW float32 → resize → normalize → batch
        img = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        img = F.interpolate(
            img.unsqueeze(0), size=(aodc_h, aodc_w), mode="bilinear", align_corners=False
        )  # [1, 3, aodc_h, aodc_w]
        img = _normalize(img.squeeze(0)).unsqueeze(0).to(self.device)  # [1, 3, aodc_h, aodc_w]

        # AODC forward (reference-less: boxes ignored, categories ignored)
        den_map = self._model(img, self._dummy_boxes, mode="test", categories=[])
        # den_map: [1, 1, H_den, W_den]
        den_map = F.relu(den_map.squeeze())  # [H_den, W_den]

        # Find peak: argmax over flattened density map
        idx = int(den_map.argmax())
        den_h, den_w = den_map.shape
        row = idx // den_w
        col = idx % den_w

        # Map from density-map space → original image space
        x = col * orig_w / den_w
        y = row * orig_h / den_h

        return float(x), float(y)
