"""Pixel masks computed from the camera image and the predicted depth (no ground truth)."""

from __future__ import annotations

import cv2
import numpy as np


def sky_mask(rgb: np.ndarray) -> np.ndarray:
    """True where the pixel is sky.

    A colour rule: sky is bright and bluish (blue above red and green), which grass,
    leaves, bark and soil never are. It is enough for this simulator. On real footage a
    segmentation network would replace it. The depth network gives sky a finite depth,
    so these pixels must be removed before mapping.
    """
    img = rgb.astype(np.int16)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    sky = (b > r + 15) & (b >= g) & (b > 150)
    sky = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return sky.astype(bool)


def depth_edge_mask(
    depth: np.ndarray, ratio: float = 1.15, size: int = 5, max_rel_grad: float = 0.05
) -> np.ndarray:
    """True near depth discontinuities (silhouettes of trunks and crowns).

    A network smooths a depth step into a ramp several pixels wide. Pixels on the ramp
    ("flying pixels") get a depth between foreground and background and would appear as
    ghost objects in the gap, so two tests are used: the max/min ratio in a window, and
    the depth gradient relative to depth (per pixel). Ground and trunk surfaces stay well
    below ``max_rel_grad`` within mapping range; ramps exceed it.
    """
    k = np.ones((size, size), np.uint8)
    d = depth.astype(np.float32)
    hi = cv2.dilate(d, k)
    lo = cv2.erode(d, k)
    edge = hi > ratio * np.maximum(lo, 1e-3)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    rel = np.sqrt(gx**2 + gy**2) / np.maximum(d, 1e-3)
    ramp = cv2.dilate((rel > max_rel_grad).astype(np.uint8), np.ones((3, 3), np.uint8))
    return edge | ramp.astype(bool)
