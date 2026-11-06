"""Mapping and size estimation on a scripted flight (before autonomous exploration).

The drone flies a lawnmower route along the alleys (steered with simulator ground truth:
this is a test harness). The agent sees only the camera. It runs depth, odometry and
mapping, and estimates the orchard's size. Outputs: map figure, size-over-time figure,
JSON metrics.

    python scripts/eval_mapping.py --size small --seed 0 --depth net --out runs/mapping
    python scripts/eval_mapping.py ... --pose gt     # mapping with true poses (diagnostic)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchardscout import geometry as geo  # noqa: E402
from orchardscout.agent import Agent, AgentConfig  # noqa: E402
from orchardscout.control import FollowerConfig, PathFollower, lawnmower_route  # noqa: E402
from orchardscout.metrics import (  # noqa: E402
    icp_2d,
    match_points,
    size_errors,
    start_alignment_2d,
)
from orchardscout.orchard import generate_layout  # noqa: E402
from orchardscout.sim import OrchardSim  # noqa: E402
from orchardscout.viz import plot_map, plot_size_convergence  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--size", default="small", choices=["small", "medium", "large", "any"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alleys", type=int, default=0, help="0 = every alley")
    ap.add_argument("--depth", default="net", choices=["net", "gt"])
    ap.add_argument("--pose", default="vo", choices=["vo", "gt"])
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--no-correct", action="store_true", help="disable landmark pose correction")
    ap.add_argument("--vo-max-depth", type=float, default=None)
    ap.add_argument("--depth-cache", default=None, help="cache directory for network depth")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="runs/mapping")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.size}_s{args.seed}_{args.depth}_{args.pose}{args.tag}"

    layout = generate_layout(args.seed, args.size)
    gt = layout.ground_truth()
    sim = OrchardSim(layout, seed=args.seed)
    n_alleys = args.alleys or layout.n_rows - 1
    route = lawnmower_route(layout, start_alley=0, n_alleys=n_alleys)
    sim.reset(route[0][0], route[0][1], np.arctan2(*(route[8] - route[0])[::-1]) + 0.15)
    # same gentle speed and turn rate as the explorer, so the test is representative
    follower = PathFollower(FollowerConfig(cruise_speed=1.0, max_yaw_rate=0.6))
    follower.set_path(route)

    if args.depth == "net":
        from orchardscout.depth import CachedDepth, MonoDepth

        depth_fn = MonoDepth(input_size=args.input_size)
        if args.depth_cache:
            key = f"{args.size}_s{args.seed}_a{n_alleys}_i{args.input_size}_lawnmower_v2"
            depth_fn = CachedDepth(depth_fn, Path(args.depth_cache) / key)
    else:
        depth_fn = lambda rgb: sim.depth_gt()  # noqa: E731  diagnostic only

    cfg = AgentConfig()
    cfg.trunk.correct = not args.no_correct
    if args.vo_max_depth:
        cfg.vo.max_depth = args.vo_max_depth
    agent = Agent(sim.K, sim.cam_cfg.pitch_deg, sim.drone_cfg.altitude, depth_fn, cfg,
                  seed=args.seed)
    T_M_W = None
    traj_gt, traj_est, hist = [], [], []
    distance = 0.0
    for step in range(args.max_steps):
        T_W = sim.T_world_cam()
        override = None
        if args.pose == "gt" and T_M_W is not None:
            override = T_M_W @ T_W
        res = agent.step(sim.rgb(), override)
        if res.T_M_cam is not None:
            if T_M_W is None:
                T_M_W = res.T_M_cam @ geo.invert_T(T_W)
            if traj_gt:
                distance += float(np.linalg.norm(T_W[:2, 3] - traj_gt[-1][:2, 3]))
            traj_gt.append(T_W)
            traj_est.append(agent.T_M_cam.copy())
        if step % 25 == 0:
            est = agent.estimate()
            if est is not None:
                hist.append({"step": step, "distance_m": distance, **est.to_dict()})
        if step % 200 == 0:
            e = hist[-1] if hist else {}
            n_trees = len(agent.mapper.trunks.confirmed())
            print(f"step {step:5d} dist {distance:6.1f} m | trunks {n_trees:4d}"
                  f" | est rows {e.get('n_rows')} length {e.get('length_m', float('nan')):.1f}"
                  f" width {e.get('width_m', float('nan')):.1f}", flush=True)
        x, y, _ = sim.state.pos
        v, w = follower.command(x, y, sim.state.yaw)  # ground-truth steering (harness)
        if follower.done:
            break
        sim.step(v, w)

    traj_gt, traj_est = np.stack(traj_gt), np.stack(traj_est)
    R, t = start_alignment_2d(traj_est[0], traj_gt[0])
    to_world = lambda xy: xy @ R.T + t  # noqa: E731
    p_est = to_world(traj_est[:, :2, 3])
    trunks_w = to_world(agent.mapper.trunks.confirmed())
    err = np.linalg.norm(p_est - traj_gt[:, :2, 3], axis=1)

    # trees near the flown path are the ones the agent could have seen
    from scipy.spatial import cKDTree

    near = cKDTree(traj_gt[:, :2, 3]).query(layout.trunks[:, :2])[0] < 6.0
    est = agent.estimate()
    icp = None
    if len(trunks_w) >= 3:  # map shape quality, independent of where drift happened
        Ri, ti = icp_2d(trunks_w, layout.trunks[:, :2])
        icp = match_points(trunks_w @ Ri.T + ti, layout.trunks[:, :2], 0.5)
    result = {
        "orchard": gt,
        "depth_source": args.depth,
        "pose_source": args.pose,
        "frames": len(traj_gt),
        "distance_m": distance,
        "vo_final_xy_error_m": float(err[-1]),
        "vo_drift_pct": float(100 * err[-1] / max(distance, 1e-9)),
        "trunks_all": match_points(trunks_w, layout.trunks[:, :2], 0.5),
        "trunks_near_path": match_points(trunks_w, layout.trunks[near, :2], 0.5),
        "trunks_icp_aligned": icp,
        "estimate": est.to_dict() if est else None,
        "size": size_errors(est.to_dict(), gt) if est else None,
        "collisions": len(sim.collisions),
        **agent.timing_summary(),
    }
    brief = {k: v for k, v in result.items() if k not in ("orchard", "estimate")}
    print(json.dumps(brief, indent=1))
    (out / f"map_{tag}.json").write_text(json.dumps({**result, "history": hist}, indent=1))
    corners_w = to_world(np.array(est.corners)) if est else None
    title = (f"{args.depth} depth, {args.pose} pose: {distance:.0f} m flown | "
             + (f"est {est.n_rows} rows, {est.length_m:.1f} x {est.width_m:.1f} m | "
                f"true {gt['n_rows']} rows, {gt['length_m']:.1f} x {gt['width_m']:.1f} m"
                if est else "no estimate"))
    plot_map(agent.mapper.grid, trunks_w, layout.trunks, traj_gt[:, :2, 3], p_est, corners_w,
             out / f"map_{tag}.png", title=title, transform=(R, t))
    plot_size_convergence(hist, gt, out / f"size_{tag}.png")
    sim.close()
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()
