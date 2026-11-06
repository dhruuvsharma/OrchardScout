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


def heading(T: np.ndarray) -> float:
    """Heading (rad) of a camera pose: its optical axis projected on the ground plane."""
    f = T[:3, 2]
    return float(np.arctan2(f[1], f[0]))


def start_alignment_2d(T_M_c0: np.ndarray, T_W_c0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with world_xy = R @ map_xy + t, from the first pose only (x, y, heading).

    The map frame is gravity-aligned, so this is the honest open-loop alignment: it uses
    no information from later in the run.
    """
    dyaw = heading(T_W_c0) - heading(T_M_c0)
    R = np.array([[np.cos(dyaw), -np.sin(dyaw)], [np.sin(dyaw), np.cos(dyaw)]])
    t = T_W_c0[:2, 3] - R @ T_M_c0[:2, 3]
    return R, t


def match_points(est: np.ndarray, gt: np.ndarray, radius: float = 0.5) -> dict[str, float]:
    """One-to-one nearest matching of estimated to true points within ``radius``."""
    from scipy.spatial import cKDTree

    if len(est) == 0 or len(gt) == 0:
        return {"precision": 0.0, "recall": 0.0, "matched": 0, "mean_error_m": float("nan")}
    d, j = cKDTree(gt).query(est, distance_upper_bound=radius)
    used, errs = set(), []
    for k in np.argsort(d):
        if np.isfinite(d[k]) and j[k] not in used:
            used.add(int(j[k]))
            errs.append(d[k])
    m = len(used)
    return {
        "precision": m / len(est),
        "recall": m / len(gt),
        "matched": m,
        "n_est": int(len(est)),
        "n_gt": int(len(gt)),
        "mean_error_m": float(np.mean(errs)) if errs else float("nan"),
    }


def icp_2d(src: np.ndarray, dst: np.ndarray, iters: int = 20, radius: float = 1.0):
    """Rigid 2-D alignment of point set ``src`` onto ``dst`` (nearest-neighbour ICP).

    Used only to evaluate map *shape* (how well trees are placed relative to each other)
    independently of where the drift happened. Returns (R, t) with dst ~ src @ R.T + t.
    """
    from scipy.spatial import cKDTree

    R, t = np.eye(2), np.zeros(2)
    tree = cKDTree(dst)
    for _ in range(iters):
        cur = src @ R.T + t
        d, j = tree.query(cur, distance_upper_bound=radius)
        ok = np.isfinite(d)
        if ok.sum() < 3:
            break
        a, b = cur[ok], dst[j[ok]]
        ma, mb = a.mean(0), b.mean(0)
        H = (a - ma).T @ (b - mb)
        ang = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        dR = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        R, t = dR @ R, dR @ t + mb - dR @ ma
    return R, t


def size_errors(est: dict, gt: dict) -> dict[str, float]:
    """Compare an orchard estimate with the layout's ground truth."""
    out = {}
    for k in ("length_m", "width_m", "row_spacing_m", "tree_spacing_m", "area_m2"):
        e, g = est.get(k), gt.get(k)
        if e is not None and g:
            out[f"{k}_est"] = e
            out[f"{k}_true"] = g
            out[f"{k}_err_pct"] = 100.0 * (e - g) / g
    out["n_rows_est"] = est.get("n_rows")
    out["n_rows_true"] = gt.get("n_rows")
    return out


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
