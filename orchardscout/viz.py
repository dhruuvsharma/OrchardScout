"""Visualisation helpers: depth colouring and layout plots."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def colorize_depth(depth: np.ndarray, max_depth: float = 30.0) -> np.ndarray:
    """Depth (m) to an RGB uint8 image; near = bright, far / sky = dark."""
    d = np.nan_to_num(depth.astype(np.float32), nan=max_depth, posinf=max_depth)
    t = 1.0 - np.clip(d / max_depth, 0.0, 1.0)
    img = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_trajectories(layout, p_gt, p_est, dist, err, path: str | Path, title: str = "") -> None:
    """Top-down ground-truth vs estimated trajectory over the orchard, plus error vs distance."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.6, 1.0]}
    )
    ax.scatter(layout.trunks[:, 0], layout.trunks[:, 1], s=6, c="#5b3a1e", label="trunks (GT)")
    ax.plot(p_gt[:, 0], p_gt[:, 1], "-", c="#2b7a3d", lw=2, label="ground truth")
    ax.plot(p_est[:, 0], p_est[:, 1], "-", c="#d1495b", lw=1.5, label="visual odometry")
    ax.plot(*p_gt[0, :2], "o", c="k", ms=6, label="start")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    ax2.plot(dist, err, c="#d1495b")
    ax2.set_xlabel("distance flown (m)")
    ax2.set_ylabel("horizontal position error (m)")
    ax2.grid(alpha=0.3)
    ax2.set_title("open-loop error (aligned at the start pose only)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_map(grid, trunks_w, gt_trunks, traj_gt, traj_est, corners_w, path, title="",
             transform=None) -> None:
    """Occupancy grid (in world coordinates via ``transform``), trunks, paths, estimate box.

    ``transform``: (R, t) taking map-frame xy to world xy. The grid is drawn in the map
    frame rotated into the world, so it is resampled with nearest neighbour.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R, t = transform
    allxy = np.vstack([gt_trunks[:, :2], traj_gt[:, :2]])
    lo, hi = allxy.min(0) - 8.0, allxy.max(0) + 8.0
    # resample the grid into a world-aligned raster
    res = grid.cfg.resolution
    xs = np.arange(lo[0], hi[0], res)
    ys = np.arange(lo[1], hi[1], res)
    X, Y = np.meshgrid(xs, ys)
    W = np.stack([X.ravel(), Y.ravel()], 1)
    Mxy = (W - t) @ R  # inverse rotation: map = R^T (world - t)
    ij = grid.to_cell(Mxy)
    ok = grid.inside(ij)
    st = np.zeros(len(W), np.uint8)
    state = grid.state()
    st[ok] = state[ij[ok, 0], ij[ok, 1]]
    img = np.full((len(W), 3), 0.80)
    img[st == grid.FREE] = 1.0
    img[st == grid.OCCUPIED] = 0.2
    img = img.reshape(len(ys), len(xs), 3)

    fig, ax = plt.subplots(figsize=(10, 10 * (hi[1] - lo[1]) / (hi[0] - lo[0]) + 0.8))
    ax.imshow(img, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]], interpolation="nearest")
    ax.scatter(gt_trunks[:, 0], gt_trunks[:, 1], s=30, facecolors="none", edgecolors="#8a5a00",
               lw=1.0, label="true trunks")
    if len(trunks_w):
        ax.scatter(trunks_w[:, 0], trunks_w[:, 1], s=9, c="#d1495b", label="mapped trunks")
    ax.plot(traj_gt[:, 0], traj_gt[:, 1], c="#2b7a3d", lw=1.5, label="true path")
    ax.plot(traj_est[:, 0], traj_est[:, 1], c="#3366cc", lw=1.0, ls="--", label="VO path")
    if corners_w is not None:
        c = np.vstack([corners_w, corners_w[:1]])
        ax.plot(c[:, 0], c[:, 1], c="#d1495b", lw=1.5, ls=":", label="estimated extent")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_size_convergence(hist: list[dict], gt: dict, path) -> None:
    """Estimated length / width / rows against distance flown, with the true values."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not hist:
        return
    d = [h["distance_m"] for h in hist]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, key, label in zip(axs, ("length_m", "width_m", "n_rows"),
                              ("length along rows (m)", "width across rows (m)", "rows"),
                              strict=True):
        ax.plot(d, [h.get(key, np.nan) for h in hist], c="#d1495b", label="estimate")
        ax.axhline(gt[key], c="#2b7a3d", ls="--", label="true")
        ax.set_xlabel("distance flown (m)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_map_panel(grid, trunks, traj_xy, T_M_cam, plan=None, frontiers=None, est=None,
                     size=(720, 800), text_lines=()) -> np.ndarray:
    """Top-down live map (agent's own map frame) as an RGB image, for the video."""
    H, W = size
    img = np.full((H, W, 3), 45, np.uint8)
    # frame the trees and the flown path (plus a margin), not all the free space seen
    pts = [np.atleast_2d(T_M_cam[:2, 3])]
    if len(traj_xy):
        pts.append(np.asarray(traj_xy))
    if len(trunks):
        pts.append(trunks)
    allp = np.vstack(pts)
    lo = allp.min(axis=0) - 6.0
    hi = allp.max(axis=0) + 6.0
    span = max(hi[0] - lo[0], hi[1] - lo[1], 20.0)
    mid = 0.5 * (lo + hi)
    lo, hi = mid - 0.5 * span, mid + 0.5 * span
    scale = min((W - 20) / (hi[0] - lo[0]), (H - 70) / (hi[1] - lo[1]))
    cx, cy = 0.5 * (lo + hi)
    ox, oy = W / 2 - cx * scale, 40 + (H - 60) / 2 + cy * scale

    def px(xy):
        xy = np.atleast_2d(xy)
        return np.column_stack([ox + xy[:, 0] * scale, oy - xy[:, 1] * scale]).astype(np.int32)

    v, u = np.mgrid[0:H, 0:W]
    wx = (u + 0.5 - ox) / scale
    wy = (oy - (v + 0.5)) / scale
    ij = grid.to_cell(np.stack([wx.ravel(), wy.ravel()], 1))
    ok = grid.inside(ij)
    st = np.zeros(H * W, np.uint8)
    state = grid.state()
    st[ok] = state[ij[ok, 0], ij[ok, 1]]
    st = st.reshape(H, W)
    img[st == grid.FREE] = (228, 228, 218)
    img[st == grid.OCCUPIED] = (70, 70, 70)
    if frontiers is not None and len(frontiers):
        for p in px(frontiers):
            cv2.circle(img, tuple(p), 1, (0, 170, 200), -1)
    if len(traj_xy) > 1:
        cv2.polylines(img, [px(traj_xy)], False, (51, 102, 204), 2, cv2.LINE_AA)
    if plan is not None:
        cv2.polylines(img, [px(plan.path)], False, (40, 160, 70), 2, cv2.LINE_AA)
        cv2.circle(img, tuple(px(plan.goal)[0]), 6, (40, 160, 70), 2, cv2.LINE_AA)
    r = max(2, int(0.2 * scale))
    for p in px(trunks) if len(trunks) else []:
        cv2.circle(img, tuple(p), r, (200, 50, 60), -1, cv2.LINE_AA)
    if est is not None:
        c = px(np.array(est.corners))
        cv2.polylines(img, [c], True, (180, 60, 200), 2, cv2.LINE_AA)
    p = px(T_M_cam[:2, 3])[0]
    f = T_M_cam[:2, 2] / (np.linalg.norm(T_M_cam[:2, 2]) + 1e-9)
    tip = (p + 18 * np.array([f[0], -f[1]])).astype(np.int32)
    cv2.arrowedLine(img, tuple(p), tuple(tip), (255, 140, 0), 3, cv2.LINE_AA, tipLength=0.4)
    cv2.rectangle(img, (0, 0), (W, 34 + 22 * max(0, len(text_lines) - 1)), (25, 25, 25), -1)
    for k, line in enumerate(text_lines):
        cv2.putText(img, line, (10, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240),
                    1, cv2.LINE_AA)
    return img


def compose_frame(rgb: np.ndarray, depth: np.ndarray, panel: np.ndarray,
                  caption: str = "") -> np.ndarray:
    """1280x720 video frame: onboard RGB and predicted depth (left), live map (right)."""
    left = np.concatenate([
        cv2.resize(rgb, (480, 360), interpolation=cv2.INTER_AREA),
        cv2.resize(colorize_depth(depth, 12.0), (480, 360), interpolation=cv2.INTER_AREA),
    ], axis=0)
    for k, label in enumerate(("onboard camera (only input)", "monocular depth (metric)")):
        cv2.rectangle(left, (0, 360 * k), (300, 360 * k + 26), (25, 25, 25), -1)
        cv2.putText(left, label, (8, 360 * k + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 240, 240), 1, cv2.LINE_AA)
    frame = np.concatenate([left, panel], axis=1)
    if caption:
        cv2.putText(frame, caption, (490, 712), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200),
                    1, cv2.LINE_AA)
    return frame


def plot_coverage(hist: list[dict], path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not hist:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot([h["distance_m"] for h in hist], [100 * h["coverage"] for h in hist], c="#3366cc")
    ax.set_xlabel("distance flown (m)")
    ax.set_ylabel("orchard area mapped (%)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_layout(layout, path: str | Path, starts=None) -> None:
    """Top-down plot of trunks and crowns (ground truth), with optional start poses."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    xmin, ymin, xmax, ymax = layout.bbox()
    span = max(xmax - xmin, ymax - ymin) + 12.0
    fig, ax = plt.subplots(figsize=(9, 9 * (ymax - ymin + 12.0) / span + 1.0))
    for _, x, y, _, rx, ry, _ in layout.canopies:
        ax.add_patch(Ellipse((x, y), 2 * rx, 2 * ry, color="#7fb069", alpha=0.25, lw=0))
    for x, y, r, _ in layout.trunks:
        ax.add_patch(Circle((x, y), max(r, 0.08), color="#5b3a1e"))
    for x, y, yaw in starts or []:
        ax.plot(x, y, "o", color="#d1495b", ms=6)
        ax.arrow(x, y, 2.0 * np.cos(yaw), 2.0 * np.sin(yaw), color="#d1495b", width=0.12)
    gt = layout.ground_truth()
    ax.set_title(
        f"seed {layout.seed}: {gt['n_rows']} rows, {gt['n_trees']} trees, "
        f"{gt['length_m']:.1f} x {gt['width_m']:.1f} m"
    )
    ax.set_xlim(xmin - 6, xmax + 6)
    ax.set_ylim(ymin - 6, ymax + 6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
