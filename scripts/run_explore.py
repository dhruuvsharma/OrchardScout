"""Autonomous exploration of a randomly generated orchard of unknown size.

The agent sees only the onboard camera (plus the nominal flight height). It estimates
depth, tracks itself with visual odometry, maps trunks and free space, explores frontiers
near trees until none are left, and reports the orchard's size. The simulator's ground
truth is used only to score the run afterwards.

    python scripts/run_explore.py --size small --seed 0 --start alley --video --out runs/explore
    python scripts/run_explore.py ... --depth gt   # perfect depth (diagnostic upper bound)
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

from orchardscout import geometry as geo  # noqa: E402
from orchardscout.agent import Agent, AgentConfig  # noqa: E402
from orchardscout.explore import Explorer  # noqa: E402
from orchardscout.metrics import icp_2d, match_points, size_errors, start_alignment_2d  # noqa: E402
from orchardscout.orchard import generate_layout, sample_start_pose  # noqa: E402
from orchardscout.sim import OrchardSim  # noqa: E402
from orchardscout.viz import (  # noqa: E402
    compose_frame,
    plot_coverage,
    plot_map,
    plot_size_convergence,
    render_map_panel,
)


def coverage(grid, layout, R, t) -> float:
    """Fraction of the orchard (trunk bounding box + 1 m) that the map has observed."""
    xmin, ymin, xmax, ymax = layout.bbox()
    xs, ys = np.meshgrid(np.arange(xmin - 1, xmax + 1, 0.5), np.arange(ymin - 1, ymax + 1, 0.5))
    W = np.stack([xs.ravel(), ys.ravel()], 1)
    ij = grid.to_cell((W - t) @ R)  # world -> map frame
    ok = grid.inside(ij)
    return float(np.mean(grid.seen[ij[ok, 0], ij[ok, 1]])) if ok.any() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--size", default="small", choices=["small", "medium", "large", "any"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="alley", choices=["alley", "edge"])
    ap.add_argument("--depth", default="net", choices=["net", "gt"])
    ap.add_argument("--pose", default="vo", choices=["vo", "gt"])
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--video-every", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-correct", action="store_true", help="no landmark/compass/lattice")
    ap.add_argument("--no-compass", action="store_true")
    ap.add_argument("--no-lattice", action="store_true")
    ap.add_argument("--no-landmark", action="store_true", help="no landmark re-sighting")
    ap.add_argument("--out", default="runs/explore")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.size}_s{args.seed}_{args.start}_{args.depth}_{args.pose}{args.tag}"

    layout = generate_layout(args.seed, args.size)
    gt = layout.ground_truth()
    sim = OrchardSim(layout, seed=args.seed)
    rng = np.random.default_rng(args.seed + 100)
    sim.reset(*sample_start_pose(layout, rng, args.start))

    if args.depth == "net":
        from orchardscout.depth import MonoDepth

        depth_fn = MonoDepth(input_size=args.input_size)
    else:
        depth_fn = lambda rgb: sim.depth_gt()  # noqa: E731  diagnostic only
    cfg = AgentConfig()
    cfg.trunk.correct = not (args.no_correct or args.no_landmark)
    if args.no_correct or args.no_compass:
        cfg.trunk.compass_gain = 0.0
    if args.no_correct or args.no_lattice:
        cfg.trunk.lattice_gain = 0.0
    agent = Agent(sim.K, sim.cam_cfg.pitch_deg, sim.drone_cfg.altitude, depth_fn, cfg,
                  seed=args.seed)
    explorer = Explorer()
    explorer.debug = args.debug

    writer = None
    if args.video:
        writer = cv2.VideoWriter(str(out / f"explore_{tag}_raw.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))

    T_M_W = None
    traj_gt, traj_est, hist = [], [], []
    distance, R, t = 0.0, None, None
    t_start = time.perf_counter()
    status = "max_steps"
    for step in range(args.max_steps):
        T_W = sim.T_world_cam()
        override = T_M_W @ T_W if (args.pose == "gt" and T_M_W is not None) else None
        rgb = sim.rgb()
        res = agent.step(rgb, override)
        if res.T_M_cam is None:  # no ground plane yet: turn slowly until one is visible
            sim.step((0.0, 0.0), 0.3)
            continue
        if T_M_W is None:
            T_M_W = res.T_M_cam @ geo.invert_T(T_W)
            R, t = start_alignment_2d(res.T_M_cam, T_W)
        if traj_gt:
            distance += float(np.linalg.norm(T_W[:2, 3] - traj_gt[-1][:2, 3]))
        traj_gt.append(T_W)
        traj_est.append(res.T_M_cam)

        trunks = agent.mapper.trunks.confirmed()
        v, w = explorer.command(res.T_M_cam, agent.mapper.grid,
                                agent.mapper.trunks.provisional(explorer.cfg.relevance_min_obs),
                                agent.mapper.trunks.provisional())

        est = None
        if step % 25 == 0 or explorer.done:
            est = agent.estimate()
            hist.append({"step": step, "distance_m": distance,
                         "coverage": coverage(agent.mapper.grid, layout, R, t),
                         "n_trunks": int(len(trunks)), **(est.to_dict() if est else {})})
        if args.debug and step % 50 == 0 and R is not None:
            pe = res.T_M_cam[:2, 3] @ R.T + t
            fe, fg = res.T_M_cam[:2, 2] @ R.T, T_W[:2, 2]
            yerr = np.degrees(np.arctan2(fe[0] * fg[1] - fe[1] * fg[0], fe @ fg))
            m = agent.mapper
            print(f"  dbg {step}: xy err {np.linalg.norm(pe - T_W[:2, 3]):.2f} m, yaw err "
                  f"{-yerr:+.1f} deg, compass n={len(m.compass_log)} last "
                  f"{(m.compass_log or [0])[-1]:+.1f}, lattice n={len(m.lattice_log)} last "
                  f"{(m.lattice_log or [0])[-1]:+.2f}, mode {explorer.mode}", flush=True)
        if step % 200 == 0:
            h = hist[-1]
            print(f"step {step:5d} {explorer.mode:6s} dist {distance:6.1f} m cov "
                  f"{100 * h['coverage']:5.1f}% trunks {len(trunks):4d} rows {h.get('n_rows')} "
                  f"L {h.get('length_m', float('nan')):.1f} W {h.get('width_m', float('nan')):.1f} "
                  f"coll {len(sim.collisions)} | {time.perf_counter() - t_start:.0f}s", flush=True)
        if writer is not None and step % args.video_every == 0:
            e = est or agent.estimate()
            lines = [f"distance {distance:5.1f} m   trees mapped {len(trunks)}   "
                     f"mode {explorer.mode}"]
            if e is not None:
                lines.append(f"estimate: {e.n_rows} rows, {e.length_m:.1f} m x {e.width_m:.1f} m, "
                             f"row spacing {e.row_spacing_m:.2f} m")
            path_xy = np.array([T[:2, 3] for T in traj_est])
            panel = render_map_panel(agent.mapper.grid, trunks, path_xy, res.T_M_cam,
                                     explorer.plan, explorer.last_frontiers, e, text_lines=lines)
            frame = compose_frame(rgb, res.frame.depth, panel,
                                  caption="map: agent's own estimate (no ground truth)")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if explorer.done:
            status = explorer.status
            break
        hit = sim.step(v, w)
        if hit.collided and args.debug:
            p_m = res.T_M_cam[:2, 3]
            dmin = (np.min(np.linalg.norm(trunks - p_m, axis=1)) if len(trunks) else np.inf)
            print(f"  collision step {step} with {hit.collision_with} at gt {sim.state.pos[:2]}"
                  f" est {p_m} mode {explorer.mode} cmd v={v} w={w:.2f} goal "
                  f"{None if explorer.plan is None else explorer.plan.goal} nearest mapped trunk"
                  f" {dmin:.2f} m", flush=True)

    wall = time.perf_counter() - t_start
    if writer is not None:
        writer.release()
    traj_gt, traj_est = np.stack(traj_gt), np.stack(traj_est)
    to_world = lambda xy: xy @ R.T + t  # noqa: E731
    p_est = to_world(traj_est[:, :2, 3])
    err = np.linalg.norm(p_est - traj_gt[:, :2, 3], axis=1)
    trunks_w = to_world(agent.mapper.trunks.confirmed())
    est = agent.estimate()
    icp = None
    if len(trunks_w) >= 3:
        Ri, ti = icp_2d(trunks_w, layout.trunks[:, :2])
        icp = match_points(trunks_w @ Ri.T + ti, layout.trunks[:, :2], 0.5)
    result = {
        "orchard": gt,
        "start": args.start,
        "depth_source": args.depth,
        "pose_source": args.pose,
        "status": status,
        "steps": len(traj_gt),
        "sim_time_s": round(sim.t, 1),
        "wall_time_s": round(wall, 1),
        "distance_m": distance,
        "coverage": hist[-1]["coverage"] if hist else 0.0,
        "collisions": len(sim.collisions),
        "vo_final_xy_error_m": float(err[-1]),
        "vo_drift_pct": float(100 * err[-1] / max(distance, 1e-9)),
        "trunks_start_aligned": match_points(trunks_w, layout.trunks[:, :2], 0.5),
        "trunks_icp_aligned": icp,
        "estimate": est.to_dict() if est else None,
        "size": size_errors(est.to_dict(), gt) if est else None,
        "plans": explorer.n_plans,
        "safety_interventions": explorer.n_safety,
        **agent.timing_summary(),
    }
    brief = {k: v for k, v in result.items() if k not in ("orchard", "estimate")}
    print(json.dumps(brief, indent=1))
    (out / f"explore_{tag}.json").write_text(json.dumps({**result, "history": hist}, indent=1))
    corners_w = to_world(np.array(est.corners)) if est else None
    title = (f"{args.depth} depth, {args.pose} pose: {distance:.0f} m flown, {status} | "
             + (f"est {est.n_rows} rows, {est.length_m:.1f} x {est.width_m:.1f} m | "
                f"true {gt['n_rows']} rows, {gt['length_m']:.1f} x {gt['width_m']:.1f} m"
                if est else "no estimate"))
    plot_map(agent.mapper.grid, trunks_w, layout.trunks, traj_gt[:, :2, 3], p_est, corners_w,
             out / f"explore_{tag}_map.png", title=title, transform=(R, t))
    plot_size_convergence(hist, gt, out / f"explore_{tag}_size.png")
    plot_coverage(hist, out / f"explore_{tag}_coverage.png")
    sim.close()
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()
