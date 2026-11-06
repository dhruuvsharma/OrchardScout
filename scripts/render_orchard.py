"""Generate an orchard and save preview images: onboard RGB, ground-truth depth, chase view,
orthographic top-down view and a plot of the layout.

    python scripts/render_orchard.py --size medium --seed 3 --views 4 --out runs/preview
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

from orchardscout.orchard import generate_layout, sample_start_pose  # noqa: E402
from orchardscout.sim import OrchardSim  # noqa: E402
from orchardscout.viz import colorize_depth, plot_layout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--size", default="medium", choices=["small", "medium", "large", "any"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", type=int, default=3, help="number of random drone poses")
    ap.add_argument("--out", default="runs/preview")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    layout = generate_layout(args.seed, args.size)
    t1 = time.perf_counter()
    sim = OrchardSim(layout, seed=args.seed)
    t2 = time.perf_counter()
    gt = layout.ground_truth()
    print(json.dumps(gt, indent=1))
    print(f"layout {1e3 * (t1 - t0):.0f} ms, scene build {1e3 * (t2 - t1):.0f} ms, "
          f"{sim.model.ngeom} geoms")
    layout.save_json(out / "layout.json")

    rng = np.random.default_rng(args.seed)
    starts = []
    for i in range(args.views):
        x, y, yaw = sample_start_pose(layout, rng, "alley" if i % 2 == 0 else "edge")
        sim.reset(x, y, yaw)
        starts.append((x, y, yaw))
        t = time.perf_counter()
        rgb = sim.rgb()
        t_rgb = time.perf_counter() - t
        t = time.perf_counter()
        depth = sim.depth_gt()
        t_depth = time.perf_counter() - t
        chase = sim.render("chase")
        print(f"view {i}: x={x:.1f} y={y:.1f} yaw={np.rad2deg(yaw):.0f} deg | "
              f"rgb {1e3 * t_rgb:.0f} ms, depth {1e3 * t_depth:.0f} ms | "
              f"depth range {depth.min():.2f}..{np.percentile(depth, 95):.1f} m")
        row = np.concatenate([rgb, colorize_depth(depth, max_depth=30.0), chase], axis=1)
        cv2.imwrite(str(out / f"view{i}.jpg"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    top = sim.render("overview", size=(960, 1280), fog=False)
    cv2.imwrite(str(out / "overview.jpg"), cv2.cvtColor(top, cv2.COLOR_RGB2BGR))
    plot_layout(layout, out / "layout.png", starts=starts)
    sim.close()
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()
