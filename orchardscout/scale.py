"""Metric scale for monocular depth from the ground plane and the known flight height.

A metric depth network trained on driving data gets the *shape* of the orchard right but
its absolute scale wrong (about 5x too far for a camera 1 m above the grass). Monocular
visual odometry for cars fixes scale with a known camera height above the road; we do the
same. The bottom of the image is mostly ground, so we fit a plane to the predicted 3-D
points there (RANSAC, scale-free threshold), read off the predicted camera height, and
rescale depth so it matches the nominal flight height that the flight controller holds.

Inputs: the predicted depth map, the intrinsics, the camera mount pitch and the nominal
flight height. No ground truth is used.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from . import geometry as geo


@dataclass
class GroundPlaneConfig:
    altitude: float = 1.0  # m, nominal camera height above ground
    # Bottom fraction of the image used for the fit. Narrow on purpose: the network
    # compresses depth with distance even on flat ground, so a wide band bends the fitted
    # plane and biases the scale (0.28 gave depth ~3-5 % too short at 0-4 m in flight,
    # and visual odometry ~4 % too short; 0.12 gives ~0-2 %).
    band: float = 0.12
    stride: int = 3  # pixel subsampling inside the band
    iters: int = 150  # RANSAC iterations
    inlier_rel: float = 0.03  # inlier threshold as a fraction of the plane distance
    min_inlier_frac: float = 0.5
    max_tilt_deg: float = 25.0  # allowed angle between the plane normal and expected 'up'
    smooth: int = 5  # median over the last k accepted per-frame scales


@dataclass
class PlaneFit:
    normal: np.ndarray  # unit normal in the camera frame, pointing up (towards the camera)
    distance: float  # camera height above the plane, in predicted (unscaled) units
    inlier_frac: float
    tilt_deg: float  # angle between normal and the expected up direction
    ok: bool


class GroundPlaneScale:
    """Per-frame scale factor so that ``scale * predicted_depth`` is metric."""

    def __init__(
        self,
        K: np.ndarray,
        cam_pitch_deg: float,
        config: GroundPlaneConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.K = K
        self.cfg = config or GroundPlaneConfig()
        p = np.deg2rad(cam_pitch_deg)
        # World 'up' in the OpenCV camera frame for a camera pitched down by p.
        self.up = np.array([0.0, -np.cos(p), -np.sin(p)])
        self.rng = np.random.default_rng(seed)
        self.history: deque[float] = deque(maxlen=self.cfg.smooth)
        self.last_fit: PlaneFit | None = None

    def fit_plane(self, depth: np.ndarray) -> PlaneFit:
        cfg = self.cfg
        h = depth.shape[0]
        masked = np.zeros_like(depth)
        r0 = int((1.0 - cfg.band) * h)
        masked[r0:] = depth[r0:]
        P, _ = geo.backproject(masked, self.K, stride=cfg.stride)
        fail = PlaneFit(self.up.copy(), float("nan"), 0.0, float("nan"), False)
        if len(P) < 30:
            return fail

        best, best_count = None, -1
        idx = self.rng.integers(0, len(P), size=(cfg.iters, 3))
        for i, j, k in idx:
            n = np.cross(P[j] - P[i], P[k] - P[i])
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            n /= norm
            d = -float(n @ P[i])
            if abs(d) < 1e-6:
                continue
            inl = np.abs(P @ n + d) < cfg.inlier_rel * abs(d)
            c = int(inl.sum())
            if c > best_count:
                best, best_count = inl, c
        if best is None:
            return fail

        Q = P[best]
        centroid = Q.mean(axis=0)
        _, _, vt = np.linalg.svd(Q - centroid, full_matrices=False)
        n = vt[2]
        d = -float(n @ centroid)
        if d < 0:
            n, d = -n, -d
        tilt = float(np.degrees(np.arccos(np.clip(n @ self.up, -1.0, 1.0))))
        frac = best_count / len(P)
        ok = frac >= cfg.min_inlier_frac and tilt <= cfg.max_tilt_deg and d > 0
        return PlaneFit(n, d, frac, tilt, ok)

    def update(self, depth: np.ndarray) -> float:
        """Fit the ground in this frame and return the smoothed scale factor.

        If the fit is rejected (e.g. a trunk fills the bottom of the image), the previous
        scale is kept. Returns NaN until the first good fit.
        """
        fit = self.fit_plane(depth)
        self.last_fit = fit
        if fit.ok:
            self.history.append(self.cfg.altitude / fit.distance)
        return float(np.median(self.history)) if self.history else float("nan")

