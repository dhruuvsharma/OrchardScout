"""Evaluate visual odometry on a scripted flight through the orchard.

The drone follows a lawnmower route along several alleys. The route and the steering use
simulator ground truth (this is a test harness for odometry, not the exploring agent).
Visual odometry runs only on the camera images and estimates the trajectory, which is then
compared with the true one.

    python scripts/eval_vo.py --size medium --seed 2 --alleys 3 --depth net --out runs/vo
    python scripts/eval_vo.py ... --depth gt     # upper bound with perfect depth (diagnostic)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchardscout import geometry as geo  # noqa: E402
from orchardscout.control import PathFollower, lawnmower_route  # noqa: E402
from orchardscout.metrics import ate_rmse  # noqa: E402
from orchardscout.orchard import generate_layout  # noqa: E402
from orchardscout.perception import Perception  # noqa: E402
from orchardscout.sim import OrchardSim  # noqa: E402
from orchardscout.viz import plot_trajectories  # noqa: E402
from orchardscout.vo import VisualOdometry, VOConfig  # noqa: E402


def yaw_of(T: np.ndarray) -> float:
    f = T[:3, 2]  # camera optical axis in the world frame
    return float(np.arctan2(f[1], f[0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--size", default="medium", choices=["small", "medium", "large", "any"])
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--alleys", type=int, default=3)
    ap.add_argument("--depth", default="net",
                    choices=["net", "gt", "gt_compress", "gt_noise", "gt_smooth"],
                    help="gt_* corrupt simulator depth in one controlled way (diagnostics)")
    ap.add_argument("--input-size", type=int, default=518)
    ap.add_argument("--ground-alpha", type=float, default=0.3,
                    help="0 disables the ground constraint")
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--snap", action="store_true", help="enable ground-plane depth snapping")
    ap.add_argument("--vo-max-depth", type=float, default=12.0)
    ap.add_argument("--depth-cache", default=None,
                    help="directory to cache network depth per frame (the scripted flight is "
                         "deterministic, so variants can be compared quickly)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="runs/vo")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layout = generate_layout(args.seed, args.size)
    sim = OrchardSim(layout, seed=args.seed)
    first = max(0, layout.n_rows // 2 - 2)
    route = lawnmower_route(layout, start_alley=first, n_alleys=args.alleys)
    start_yaw = np.arctan2(*(route[8] - route[0])[::-1])
    sim.reset(route[0][0], route[0][1], start_yaw + np.deg2rad(10))
    follower = PathFollower()
    follower.set_path(route)

    net = None
    if args.depth == "net":
        from orchardscout.depth import MonoDepth

        net = MonoDepth(input_size=args.input_size)

    cache_dir = None
    if args.depth_cache and net is not None:
        key = f"{args.size}_s{args.seed}_a{args.alleys}_i{args.input_size}"
        cache_dir = Path(args.depth_cache) / key
        cache_dir.mkdir(parents=True, exist_ok=True)
    frame_no = [0]

    noise_rng = np.random.default_rng(1234)

    def corrupt(d: np.ndarray) -> np.ndarray:
        if args.depth == "gt_compress":  # like the network: squashed beyond 6 m
            return np.where(d < 6.0, d, 6.0 + (d - 6.0) / 1.6)
        if args.depth == "gt_noise":  # 10 % independent per-pixel noise
            return d * np.exp(noise_rng.normal(0.0, 0.1, d.shape))
        if args.depth == "gt_smooth":  # 10 % smooth, low-frequency noise
            import cv2

            low = noise_rng.normal(0.0, 0.1, (6, 8))
            return d * np.exp(cv2.resize(low, d.shape[::-1], interpolation=cv2.INTER_CUBIC))
        return d

    def raw_depth(rgb):
        if net is None:
            return corrupt(sim.depth_gt())
        if cache_dir is not None:
            f = cache_dir / f"{frame_no[0]:05d}.npy"
            frame_no[0] += 1
            if f.exists():
                return np.load(f).astype(np.float32)
            d = net(rgb)
            np.save(f, d.astype(np.float16))
            return d.astype(np.float16).astype(np.float32)
        return net(rgb)

    per = Perception(sim.K, sim.cam_cfg.pitch_deg, sim.drone_cfg.altitude, seed=args.seed,
                     ground_snap=args.snap)
    vo_cfg = VOConfig(ground_alpha=args.ground_alpha, max_depth=args.vo_max_depth)
    vo = VisualOdometry(sim.K, vo_cfg)

    # wait for a ground-plane fit, then initialise
    frame = per.process(sim.rgb(), raw_depth(sim.rgb()))
    assert frame.up_cam is not None, "no ground plane in the first frame"
    T_est = [vo.start(frame.gray, frame.depth, frame.feature_mask, frame.up_cam, per.altitude)]
    T_gt = [sim.T_world_cam()]
    stats, timing = [], {"depth": [], "vo": []}

    for step in range(args.max_steps):
        x, y, _ = sim.state.pos
        v, w = follower.command(x, y, sim.state.yaw)  # ground-truth steering (test harness)
        if follower.done:
            break
        sim.step(v, w)
        rgb = sim.rgb()
        t0 = time.perf_counter()
        d = raw_depth(rgb)
        t1 = time.perf_counter()
        frame = per.process(rgb, d)
        res = vo.track(frame.gray, frame.depth, frame.feature_mask, frame.up_cam, per.altitude)
        t2 = time.perf_counter()
        timing["depth"].append(t1 - t0)
        timing["vo"].append(t2 - t1)
        T_est.append(res.T_M_cam)
        T_gt.append(sim.T_world_cam())
        stats.append((res.n_tracked, res.n_inliers, res.new_keyframe, res.lost, frame.scale))
        if step % 100 == 0:
            print(f"step {step:5d}: tracked {res.n_tracked:4d} inliers {res.n_inliers:4d} "
                  f"kf {vo.n_keyframes:4d} lost {vo.n_lost:3d} scale {frame.scale:.3f}", flush=True)

    T_est, T_gt = np.stack(T_est), np.stack(T_gt)
    # Open-loop alignment: the map frame is gravity-aligned, so only the first pose's
    # x, y and heading are used to place it in the world (no information from later frames).
    yaw_est = np.array([yaw_of(T) for T in T_est])
    yaw_gt = np.array([yaw_of(T) for T in T_gt])
    dyaw = yaw_gt[0] - yaw_est[0]
    R2 = geo.rot_z(dyaw)[:2, :2]
    p_est = T_est[:, :3, 3].copy()
    p_gt = T_gt[:, :3, 3]
    p_est[:, :2] = (p_est[:, :2] - p_est[0, :2]) @ R2.T + p_gt[0, :2]
    err = np.linalg.norm(p_est[:, :2] - p_gt[:, :2], axis=1)
    seg = np.linalg.norm(np.diff(p_gt[:, :2], axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    yaw_err = np.degrees(geo.wrap_angle(yaw_est + dyaw - yaw_gt))

    result = {
        "orchard": layout.ground_truth(),
        "depth_source": args.depth,
        "ground_alpha": args.ground_alpha,
        "frames": len(T_gt),
        "path_length_m": float(dist[-1]),
        "open_loop": {
            "final_xy_error_m": float(err[-1]),
            "final_drift_pct": float(100 * err[-1] / dist[-1]),
            "max_xy_error_m": float(err.max()),
            "final_yaw_error_deg": float(yaw_err[-1]),
            "max_abs_yaw_error_deg": float(np.abs(yaw_err).max()),
            "height_rmse_m": float(np.sqrt(np.mean((p_est[:, 2] - p_gt[:, 2]) ** 2))),
        },
        "ate_se3": ate_rmse(p_est, p_gt, "se3"),
        "ate_sim3": ate_rmse(p_est, p_gt, "sim3"),
        "keyframes": vo.n_keyframes,
        "lost_frames": vo.n_lost,
        "median_inliers": float(np.median([s[1] for s in stats])),
        "ms_depth": 1e3 * float(np.median(timing["depth"])),
        "ms_vo": 1e3 * float(np.median(timing["vo"])),
        "collisions": len(sim.collisions),
    }
    print(json.dumps(result, indent=1))
    tag = f"{args.size}_s{args.seed}_{args.depth}{args.tag}"
    (out / f"vo_{tag}.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(out / f"vo_{tag}_traj.npz", T_est_map=T_est, p_est_aligned=p_est,
                        T_gt=T_gt, dist=dist)
    plot_trajectories(layout, p_gt, p_est, dist, err, out / f"vo_{tag}.png",
                      title=f"VO ({args.depth} depth), {dist[-1]:.0f} m flown, "
                            f"final drift {result['open_loop']['final_drift_pct']:.1f} %")
    sim.close()
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()
