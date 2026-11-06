"""Monocular metric depth from a single RGB frame.

Default model: Depth Anything V2, metric outdoor head, Small backbone (~25 M parameters).
It predicts depth in metres directly, but its scale is wrong for a low drone camera, so
the agent rescales it with :mod:`orchardscout.scale`. Any Hugging Face model with
``AutoModelForDepthEstimation`` can be swapped in via ``model_id``.

Speed on a GTX 1650 (4 GB), 640x480 frames, fp32: 136 ms at input 518, 71 ms at 392,
33 ms at 266. fp16 is 3-4x *slower* on this card (no tensor cores), so it is off by default.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class MonoDepth:
    """Wraps a metric monocular depth network: ``depth_m = MonoDepth()(rgb_uint8)``."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        fp16: bool = False,
        input_size: int = 518,
    ) -> None:
        import torch
        from transformers import AutoModelForDepthEstimation

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = fp16 and self.device.startswith("cuda")
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model = (model.half() if self.half else model).to(self.device).eval()
        self.model_id = model_id
        self.input_size = input_size

    def _input_shape(self, h: int, w: int) -> tuple[int, int]:
        # Short side -> input_size, both sides multiples of 14 (the ViT patch size).
        s = self.input_size / min(h, w)
        return max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14)

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Predict planar depth in metres for an (H, W, 3) uint8 RGB image."""
        torch = self.torch
        h, w = rgb.shape[:2]
        nh, nw = self._input_shape(h, w)
        x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        t = t.half() if self.half else t
        with torch.inference_mode():
            pred = self.model(pixel_values=t).predicted_depth  # (1, nh, nw)
            pred = torch.nn.functional.interpolate(
                pred[:, None].float(), size=(h, w), mode="bilinear", align_corners=False
            )[0, 0]
        return pred.cpu().numpy()
