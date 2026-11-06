"""Estimate the orchard's layout and extent from mapped trunk positions.

1. Grid orientation: directions from each trunk to its nearest neighbours, folded modulo
   90 degrees, voted in a histogram. Trees in an orchard sit on a (jittered) grid.
2. Row direction: of the two grid axes, rows run along the one with the smaller spacing
   between neighbouring trees (in-row spacing < row spacing in a planted orchard).
3. Rows: trunk coordinates across the row direction form tight clusters, one per row.
4. Extent: length along the rows and width across them, from the outermost trees; plus
   row count, row spacing and tree spacing.

Works in any frame (the map frame is fine); every output is frame-independent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class OrchardEstimate:
    n_trees: int
    n_rows: int
    row_direction_deg: float  # in the input frame
    row_spacing_m: float
    tree_spacing_m: float
    length_m: float  # along the rows (map extent)
    width_m: float  # across the rows: (rows - 1) x locally measured row spacing if available
    area_m2: float
    corners: list  # 4 corners of the bounding rectangle, input frame
    width_map_m: float = float("nan")  # across-row extent read directly off the map
    spacing_source: str = "map"  # "local" when spacings come from single-frame measurements

    def to_dict(self) -> dict:
        return asdict(self)


def grid_orientation(xy: np.ndarray, k: int = 4, max_dist: float = 8.0) -> float:
    """Dominant grid angle in [0, 90) degrees from nearest-neighbour directions.

    Raises ValueError if no neighbour pairs fall within (0.3, max_dist) m.
    """
    if len(xy) < 2:
        raise ValueError("need at least two points")
    tree = cKDTree(xy)
    d, j = tree.query(xy, k=min(k + 1, len(xy)))
    vecs = []
    for col in range(1, d.shape[1]):
        ok = np.isfinite(d[:, col]) & (d[:, col] < max_dist) & (d[:, col] > 0.3)
        vecs.append(xy[j[ok, col]] - xy[ok])
    v = np.concatenate(vecs)
    if len(v) == 0:
        raise ValueError("no neighbour pairs in range")
    ang = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 90.0
    # circular mean on the 90-degree circle, weighted by inverse length
    w = 1.0 / np.linalg.norm(v, axis=1)
    phi = np.deg2rad(ang * 4.0)
    mean = np.degrees(np.arctan2(np.sum(w * np.sin(phi)), np.sum(w * np.cos(phi)))) / 4.0
    return float(mean % 90.0)


def line_orientation(
    xy: np.ndarray, tol: float = 0.25, min_inliers: int = 3, min_len: float = 2.5,
    max_lines: int = 3, seed: int = 0,
) -> tuple[float, int]:
    """Grid angle (degrees, modulo 90) from straight lines of trees, and the trees used.

    RANSAC on pairs of trees finds lines with >= ``min_inliers`` trees within ``tol``
    of the line; each line's direction is refined by PCA on its inliers. Unlike nearest-
    neighbour directions this ignores scattered or ghost detections, so it is precise
    enough (~1-2 deg) to correct heading from a handful of trees. Raises ValueError if no
    line is found.
    """
    pts = np.asarray(xy, dtype=np.float64)
    rng = np.random.default_rng(seed)
    angles, weights, used = [], [], 0
    for _ in range(max_lines):
        if len(pts) < min_inliers:
            break
        best = None
        n = len(pts)
        pairs = rng.integers(0, n, size=(min(200, n * n), 2))
        for i, j in pairs:
            d = pts[j] - pts[i]
            L = np.linalg.norm(d)
            if i == j or L < 1.0:
                continue
            nrm = np.array([-d[1], d[0]]) / L
            inl = np.abs((pts - pts[i]) @ nrm) < tol
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < min_inliers:
            break
        q = pts[best]
        c = q - q.mean(axis=0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        direction = vt[0]
        if np.ptp(c @ direction) < min_len:
            break
        angles.append(np.degrees(np.arctan2(direction[1], direction[0])) % 90.0)
        weights.append(best.sum())
        used += int(best.sum())
        pts = pts[~best]
    if not angles:
        raise ValueError("no line of trees")
    phi = np.deg2rad(np.array(angles) * 4.0)
    w = np.array(weights, dtype=float)
    mean = np.degrees(np.arctan2(np.sum(w * np.sin(phi)), np.sum(w * np.cos(phi)))) / 4.0
    return float(mean % 90.0), used


def find_lines(
    v: np.ndarray, sigma: float = 0.2, min_sep: float = 1.2, assign: float = 0.6,
    min_trees: int = 3,
) -> list[np.ndarray]:
    """Group points into parallel lines: peaks of a smoothed density of their v coordinate.

    Points further than ``assign`` from every peak (ghost detections between rows) are
    left out. Returns index arrays, one per line, ordered by v.
    """
    grid = np.arange(v.min() - 1.0, v.max() + 1.0, 0.05)
    dens = np.exp(-0.5 * ((grid[:, None] - v[None, :]) / sigma) ** 2).sum(axis=1)
    is_peak = (dens[1:-1] > dens[:-2]) & (dens[1:-1] >= dens[2:])
    peaks = grid[1:-1][is_peak]
    heights = dens[1:-1][is_peak]
    if len(peaks) == 0:
        return []
    keep = heights >= 0.25 * np.percentile(heights, 80)  # drop weak peaks (sparse ghosts)
    peaks, heights = peaks[keep], heights[keep]
    chosen: list[float] = []  # non-maximum suppression, strongest first
    for k in np.argsort(-heights):
        if all(abs(peaks[k] - c) >= min_sep for c in chosen):
            chosen.append(peaks[k])
    lines = []
    for c in sorted(chosen):
        idx = np.nonzero(np.abs(v - c) <= assign)[0]
        if len(idx) >= min_trees:
            lines.append(idx)
    return lines


def _runs(u: np.ndarray, max_gap: float, min_trees: int = 3) -> list[tuple[float, float]]:
    """Extents of runs of trees along a line, split at gaps larger than ``max_gap``."""
    s = np.sort(u)
    cuts = np.nonzero(np.diff(s) > max_gap)[0] + 1
    return [(g[0], g[-1]) for g in np.split(s, cuts) if len(g) >= min_trees]


def _in_line_spacing(u: np.ndarray, lines: list[np.ndarray]) -> float:
    diffs = [np.diff(np.sort(u[g])) for g in lines]
    d = np.concatenate(diffs) if diffs else np.array([])
    d = d[d > 0.5]
    return float(np.median(d)) if len(d) else np.inf


def estimate_orchard(
    xy: np.ndarray,
    local_row_spacing: list[float] | None = None,
    local_tree_spacing: list[float] | None = None,
    min_local: int = 20,
) -> OrchardEstimate | None:
    """Layout estimate from trunk positions (N, 2), or None if there are too few trees.

    ``local_*_spacing``: spacings measured inside single frames (see ``Mapper``). They do
    not depend on odometry drift, so when there are enough of them the width is computed
    as (rows - 1) x row spacing instead of being read off the map, where sideways drift
    from headland turns would distort it.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 6:
        return None
    try:
        theta = np.deg2rad(grid_orientation(xy))
    except ValueError:
        return None
    # With a locally measured row spacing, two peaks closer than 0.6 x that spacing are the
    # same row mapped twice (sideways drift), not two rows.
    local_ok = local_row_spacing is not None and len(local_row_spacing) >= min_local
    min_sep = max(1.2, 0.6 * float(np.median(local_row_spacing))) if local_ok else 1.2
    assign = max(0.6, 0.45 * min_sep)  # trees this close to a row line belong to it
    best = None
    for alt in (theta, theta + np.pi / 2):  # try both grid axes as the row direction
        R = np.array([[np.cos(alt), np.sin(alt)], [-np.sin(alt), np.cos(alt)]])
        uv = xy @ R.T  # u along the candidate row direction, v across
        lines = find_lines(uv[:, 1], min_sep=min_sep, assign=assign)
        spacing = _in_line_spacing(uv[:, 0], lines)
        if lines and (best is None or spacing < best[0]):
            best = (spacing, alt, uv, lines)
    if best is None:
        return None
    spacing, alt, uv, lines = best
    centres = np.array([np.median(uv[g, 1]) for g in lines])
    row_spacing = float(np.median(np.diff(centres))) if len(centres) > 1 else float("nan")
    # along-row extent from runs of >= 3 trees (isolated ghosts cannot stretch it)
    runs = [r for g in lines for r in _runs(uv[g, 0], 3.5 * spacing)]
    if not runs:
        return None
    u0, u1 = min(r[0] for r in runs), max(r[1] for r in runs)
    v0, v1 = centres.min(), centres.max()
    width_map = float(v1 - v0)
    width, source = width_map, "map"
    if local_row_spacing is not None and len(local_row_spacing) >= min_local:
        row_spacing = float(np.median(local_row_spacing))
        width, source = (len(lines) - 1) * row_spacing, "local"
        vc = 0.5 * (v0 + v1)
        v0, v1 = vc - 0.5 * width, vc + 0.5 * width
    if local_tree_spacing is not None and len(local_tree_spacing) >= min_local:
        spacing = float(np.median(local_tree_spacing))
    Rinv = np.array([[np.cos(alt), -np.sin(alt)], [np.sin(alt), np.cos(alt)]])
    corners = (np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]) @ Rinv.T).tolist()
    return OrchardEstimate(
        n_trees=int(len(xy)),
        n_rows=int(len(lines)),
        row_direction_deg=float(np.degrees(alt) % 180.0),
        row_spacing_m=row_spacing,
        tree_spacing_m=float(spacing),
        length_m=float(u1 - u0),
        width_m=float(width),
        area_m2=float((u1 - u0) * width),
        corners=corners,
        width_map_m=width_map,
        spacing_source=source,
    )
