"""Visual odometry with simulator depth (perfect depth): drift must be tiny."""

import numpy as np

from orchardscout import geometry as geo
from orchardscout.control import PathFollower, densify
from orchardscout.metrics import heading, start_alignment_2d
from orchardscout.orchard import generate_layout
from orchardscout.perception import Perception
from orchardscout.sim import OrchardSim
from orchardscout.vo import VisualOdometry, map_frame_from_up, rotation_between


def test_rotation_helpers():
    a = np.array([0.1, -0.9, 0.2])
    a /= np.linalg.norm(a)
    R = rotation_between(a, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(R @ a, [0, 0, 1], atol=1e-12)
    M = map_frame_from_up(np.array([0.0, -np.cos(0.1), -np.sin(0.1)]))
    np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(M @ [0.0, -np.cos(0.1), -np.sin(0.1)], [0, 0, 1], atol=1e-12)


def test_vo_with_perfect_depth_drifts_little():
    lay = generate_layout(6, "small")
    sim = OrchardSim(lay, seed=6)
    try:
        xmin, ymin, xmax, ymax = lay.bbox()
        y = 0.5 * (lay.row_y[1] + lay.row_y[2])
        route = densify(np.array([[xmin, y], [xmax, y], [xmax + 2.5, y + lay.row_spacing]]), 0.25)
        sim.reset(route[0][0], route[0][1], 0.1)
        fol = PathFollower()
        fol.set_path(route)
        per = Perception(sim.K, sim.cam_cfg.pitch_deg, sim.drone_cfg.altitude)
        vo = VisualOdometry(sim.K)
        f = per.process(sim.rgb(), sim.depth_gt())
        est = [vo.start(f.gray, f.depth, f.feature_mask, f.up_cam, 1.0)]
        gt = [sim.T_world_cam()]
        for _ in range(600):
            v, w = fol.command(sim.state.pos[0], sim.state.pos[1], sim.state.yaw)
            if fol.done:
                break
            sim.step(v, w)
            f = per.process(sim.rgb(), sim.depth_gt())
            est.append(vo.track(f.gray, f.depth, f.feature_mask, f.up_cam, 1.0).T_M_cam)
            gt.append(sim.T_world_cam())
        R, t = start_alignment_2d(est[0], gt[0])
        p = np.array([T[:2, 3] for T in est]) @ R.T + t
        g = np.array([T[:2, 3] for T in gt])
        length = np.sum(np.linalg.norm(np.diff(g, axis=0), axis=1))
        assert length > 15.0
        assert np.linalg.norm(p[-1] - g[-1]) < 0.01 * length + 0.05
        dyaw = heading(est[-1]) + (heading(gt[0]) - heading(est[0])) - heading(gt[-1])
        assert abs(np.degrees(geo.wrap_angle(dyaw))) < 1.0
        assert vo.n_lost == 0
    finally:
        sim.close()
