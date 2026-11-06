"""Per-frame perception front end: RGB -> metric depth, masks and ground plane.

This is the only place the agent turns pixels into geometry. It takes the raw depth from
the network (or, for diagnostics only, simulator depth), fixes its scale from the ground
plane, and marks pixels that should not be trusted (sky, depth edges).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .masks import depth_edge_mask, sky_mask
from .scale import GroundPlaneConfig, GroundPlaneScale


@dataclass
class Frame:
    rgb: np.ndarray
    gray: np.ndarray
    depth: np.ndarray  # metric (rescaled) planar depth, m
    sky: np.ndarray  # bool
    feature_mask: np.ndarray  # bool: pixels usable for VO features
    map_mask: np.ndarray  # bool: pixels usable for mapping
    up_cam: np.ndarray | None  # ground normal in the camera frame, if the fit was accepted
    scale: float  # factor applied to the raw depth


class Perception:
    """``ground_snap`` (off by default): pixels on the fitted ground plane get the plane's
    exact depth. It looked attractive, but it ties every ground point's error to the
    per-frame plane tilt, and in tests it made visual odometry's heading drift worse."""

    def __init__(
        self,
        K: np.ndarray,
        cam_pitch_deg: float,
        altitude: float,
        seed: int = 0,
        ground_snap: bool = False,
        snap_tol: float = 0.08,  # m, plus 3 % of range
        snap_max_range: float = 15.0,
    ):
        self.K = K
        self.altitude = altitude
        self.ground_snap = ground_snap
        self.snap_tol = snap_tol
        self.snap_max_range = snap_max_range
        self.ground = GroundPlaneScale(K, cam_pitch_deg, GroundPlaneConfig(altitude=altitude),
                                       seed=seed)
        self._rays: np.ndarray | None = None

    def _pixel_rays(self, h: int, w: int) -> np.ndarray:
        if self._rays is None or self._rays.shape[:2] != (h, w):
            v, u = np.mgrid[0:h, 0:w].astype(np.float64)
            self._rays = np.stack(
                [(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1],
                 np.ones_like(u)], axis=-1)
        return self._rays

    def process(self, rgb: np.ndarray, raw_depth: np.ndarray) -> Frame:
        scale = self.ground.update(raw_depth)
        if not np.isfinite(scale):
            scale = 1.0  # no ground seen yet; the caller should wait for a good fit
        depth = (raw_depth * scale).astype(np.float32)
        sky = sky_mask(rgb)
        fit = self.ground.last_fit
        up = fit.normal.copy() if fit is not None and fit.ok else None
        if self.ground_snap and up is not None:
            rays = self._pixel_rays(*depth.shape)
            down = -(rays @ up)  # > 0 for rays that hit the ground
            plane_z = np.where(down > 1e-3, self.altitude / np.maximum(down, 1e-3), np.inf)
            height = (rays * depth[..., None]) @ up + self.altitude  # height above the plane
            on_ground = (np.abs(height) < self.snap_tol + 0.03 * depth) & (
                plane_z < self.snap_max_range) & ~sky
            depth = np.where(on_ground, plane_z, depth).astype(np.float32)
        depth[sky] = np.inf
        valid = ~sky & ~depth_edge_mask(np.where(sky, 1e3, depth))
        feat = valid.copy()
        if up is not None:
            # "Rising ground": over open, featureless grass the network bends the ground
            # upwards beyond a few metres. Those points have the wrong depth, so visual
            # odometry must not use them (mapping may: it only frees space up to them).
            rays = self._pixel_rays(*depth.shape)
            height = (rays * np.where(np.isfinite(depth), depth, 0.0)[..., None]) @ up
            height = height + self.altitude
            rising = (height > 0.12) & (height < 0.6) & (depth > 2.5)
            feat &= ~rising
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return Frame(rgb, gray, depth, sky, feat, valid, up, scale)
