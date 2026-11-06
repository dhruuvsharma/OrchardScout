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
