"""Evaluation metrics. These use simulator ground truth and are never fed to the agent."""

from __future__ import annotations

import numpy as np


def depth_errors(
    pred: np.ndarray, gt: np.ndarray, min_depth: float = 0.1, max_depth: float = 40.0
) -> dict[str, float]:
    """Standard monocular depth errors on pixels with min_depth < gt < max_depth.

    ``scale_gt_over_pred`` is the median ratio gt / pred: 1.0 means the metric scale is right.
    """
    m = (gt > min_depth) & (gt < max_depth) & np.isfinite(pred) & (pred > 0)
    p, g = pred[m].astype(np.float64), gt[m].astype(np.float64)
    if p.size == 0:
        return {"n": 0}
    ratio = np.maximum(p / g, g / p)
    return {
        "n": int(p.size),
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "scale_gt_over_pred": float(np.median(g / p)),
    }


def depth_errors_by_distance(
    pred: np.ndarray, gt: np.ndarray, edges: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
) -> list[dict]:
    """Depth errors in ground-truth distance bins, to show how error grows with range."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        e = depth_errors(pred, gt, min_depth=max(lo, 0.1), max_depth=hi)
        rows.append({"bin_m": [lo, hi], **e})
    return rows


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    """Least-squares similarity transform dst ~ s * R @ src + t for (N, 3) point sets."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / xs.var(0).sum()) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate_rmse(est_xyz: np.ndarray, gt_xyz: np.ndarray, align: str = "se3") -> dict[str, float]:
    """Absolute trajectory error after alignment ('none', 'se3' or 'sim3')."""
    if align == "none":
        aligned, s = est_xyz, 1.0
    else:
        s, R, t = umeyama(est_xyz, gt_xyz, with_scale=align == "sim3")
        aligned = (s * (R @ est_xyz.T)).T + t
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    length = float(np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)))
    return {
        "ate_rmse_m": float(np.sqrt(np.mean(err**2))),
        "ate_max_m": float(err.max()),
        "final_error_m": float(err[-1]),
        "path_length_m": length,
        "drift_pct": float(100.0 * err[-1] / length) if length > 0 else float("nan"),
        "scale": s,
    }
