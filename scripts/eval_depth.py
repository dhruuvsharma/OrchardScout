"""Measure monocular depth quality on simulated orchard frames against ground truth.

Renders frames from random drone poses (some while accelerating, so the body is tilted),
runs the depth network, rescales it with the ground-plane method the agent uses, and
reports AbsRel, RMSE, delta<1.25 and the median scale ratio, raw and rescaled, overall and
per distance bin. Saves a figure: RGB | ground truth | rescaled prediction | relative error.

    python scripts/eval_depth.py --size medium --seed 1 --frames 16 --out runs/depth_eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchardscout.depth import DEFAULT_MODEL, MonoDepth  # noqa: E402
from orchardscout.metrics import depth_errors, depth_errors_by_distance  # noqa: E402
from orchardscout.orchard import generate_layout, sample_start_pose  # noqa: E402
from orchardscout.scale import GroundPlaneConfig, GroundPlaneScale  # noqa: E402
from orchardscout.sim import OrchardSim  # noqa: E402
from orchardscout.viz import colorize_depth  # noqa: E402


def error_image(pred: np.ndarray, gt: np.ndarray, max_depth: float) -> np.ndarray:
    """Relative error, 0 (blue) to >=50 % (red); pixels beyond max_depth are black."""
    rel = np.abs(pred - gt) / np.maximum(gt, 1e-3)
    img = cv2.applyColorMap((np.clip(rel / 0.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img[gt >= max_depth] = 0
    return img


def summarise(P: np.ndarray, G: np.ndarray, max_depth: float) -> dict:
    return {
        "overall": depth_errors(P, G, max_depth=max_depth),
        "by_distance": depth_errors_by_distance(P, G),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--size", default="medium", choices=["small", "medium", "large", "any"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--max-depth", type=float, default=30.0, help="ignore GT beyond this (sky)")
    ap.add_argument("--out", default="runs/depth_eval")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layout = generate_layout(args.seed, args.size)
    sim = OrchardSim(layout, seed=args.seed)
    net = MonoDepth(args.model, input_size=args.input_size)
    net(sim.rgb())  # warm-up
    gps = GroundPlaneScale(sim.K, sim.cam_cfg.pitch_deg,
                           GroundPlaneConfig(altitude=sim.drone_cfg.altitude, smooth=1))

    rng = np.random.default_rng(args.seed)
    raw, scaled, gts, rows, times, scales = [], [], [], [], [], []
    for i in range(args.frames):
        sim.reset(*sample_start_pose(layout, rng, "alley" if i % 3 else "edge"))
        for _ in range(int(rng.integers(0, 12))):  # fly a little so the body tilts
            sim.step((1.5, 0.0), float(rng.uniform(-0.5, 0.5)))
        rgb, gt = sim.rgb(), sim.depth_gt()
        t = time.perf_counter()
        pred = net(rgb)
        times.append(time.perf_counter() - t)
        s = gps.update(pred)
        if not np.isfinite(s):
            print(f"frame {i:2d}: ground plane not found, skipped")
            continue
        e0 = depth_errors(pred, gt, max_depth=args.max_depth)
        e1 = depth_errors(s * pred, gt, max_depth=args.max_depth)
        scales.append({"estimated": s, "true_median": e0["scale_gt_over_pred"]})
        print(f"frame {i:2d}: scale est {s:.3f} vs true {e0['scale_gt_over_pred']:.3f} | "
              f"abs_rel raw {e0['abs_rel']:.2f} -> rescaled {e1['abs_rel']:.3f}")
        raw.append(pred)
        scaled.append(s * pred)
        gts.append(gt)
        if len(rows) < 6:
            rows.append(np.concatenate([
                rgb,
                colorize_depth(gt, 30.0),
                colorize_depth(s * pred, 30.0),
                error_image(s * pred, gt, args.max_depth),
            ], axis=1))

    G = np.stack(gts)
    summary = {
        "model": args.model,
        "input_size": args.input_size,
        "frames": len(gts),
        "max_depth_m": args.max_depth,
        "orchard": layout.ground_truth(),
        "ms_per_frame": 1e3 * float(np.median(times)),
        "scales": scales,
        "raw": summarise(np.stack(raw), G, args.max_depth),
        "rescaled": summarise(np.stack(scaled), G, args.max_depth),
    }
    print(f"\n{'bin (m)':>10} {'raw AbsRel':>11} {'rescaled AbsRel':>16} {'gt/pred after':>14}")
    bins = zip(summary["raw"]["by_distance"], summary["rescaled"]["by_distance"], strict=True)
    for a, b in bins:
        lo, hi = a["bin_m"]
        print(f"{lo:>4.0f}-{hi:<5.0f} {a['abs_rel']:>11.3f} {b['abs_rel']:>16.3f} "
              f"{b['scale_gt_over_pred']:>14.3f}")
    o0, o1 = summary["raw"]["overall"], summary["rescaled"]["overall"]
    print(f"{'all':>10} {o0['abs_rel']:>11.3f} {o1['abs_rel']:>16.3f} "
          f"{o1['scale_gt_over_pred']:>14.3f}   ({summary['ms_per_frame']:.0f} ms/frame)")
    (out / "depth_eval.json").write_text(json.dumps(summary, indent=1))
    grid = np.concatenate(rows, axis=0)
    grid = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "depth_eval.jpg"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    sim.close()
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()
