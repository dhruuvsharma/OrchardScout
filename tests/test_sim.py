"""Simulator checks. These render with OpenGL, so they need a display-capable GPU or
software OpenGL (set MUJOCO_GL=osmesa / egl on headless Linux)."""

import numpy as np
import pytest

from orchardscout import geometry as geo
from orchardscout.orchard import generate_layout
from orchardscout.sim import OrchardSim


@pytest.fixture(scope="module")
def sim():
    s = OrchardSim(generate_layout(2, "small"), seed=0)
    yield s
    s.close()


def _end_tree(layout):
    """The tree with the smallest x in row 0: nothing stands between it and the headland."""
    in_row = np.where(layout.row_index == 0)[0]
    return layout.trunks[in_row[np.argmin(layout.trunks[in_row, 0])]]


def test_gt_ground_points_lie_on_the_ground(sim):
    sim.reset(*sim.layout.bbox()[:2], yaw=0.3)  # a corner of the orchard
    depth = sim.depth_gt()
    h = depth.shape[0]
    bottom = np.zeros_like(depth)
    bottom[int(0.9 * h):] = depth[int(0.9 * h):]  # bottom rows see the ground near the drone
    P, _ = geo.backproject(bottom, sim.K, stride=8)
    Pw = geo.transform_points(sim.T_world_cam(), P)
    assert np.abs(Pw[:, 2]).max() < 0.02  # ground (or the 1 cm soil strip)


def test_depth_and_pose_agree_on_a_trunk(sim):
    x, y, r, _ = _end_tree(sim.layout)
    sim.reset(x - 3.0, y, yaw=0.0)
    depth = sim.depth_gt()
    T = sim.T_world_cam()
    # The optical axis hits the trunk: planar depth equals the range along the axis.
    pitch = np.deg2rad(sim.cam_cfg.pitch_deg)
    cam_x = T[0, 3]
    expected = (x - r - cam_x) / np.cos(pitch)
    h, w = depth.shape
    assert abs(depth[(h - 1) // 2, (w - 1) // 2] - expected) < 0.03
    # Projecting that 3-D point with the GT pose lands on the image centre.
    hit = T[:3, 3] + expected * T[:3, 2]
    uv = geo.project(geo.transform_points(geo.invert_T(T), hit[None]), sim.K)[0]
    assert np.allclose(uv, [(w - 1) / 2, (h - 1) / 2], atol=0.5)


def test_flying_into_a_trunk_is_a_collision(sim):
    x, y, r, _ = _end_tree(sim.layout)
    sim.reset(x - 2.0, y, yaw=0.0)
    hits = [sim.step((1.0, 0.0), 0.0) for _ in range(40)]
    assert any(h.collided for h in hits)
    assert sim.state.pos[0] < x - r - sim.drone_cfg.radius + 1e-6
    assert sim.collisions


def test_rgb_frame(sim):
    sim.reset()
    rgb = sim.rgb()
    assert rgb.shape == (sim.cam_cfg.height, sim.cam_cfg.width, 3) and rgb.dtype == np.uint8
    assert 20 < rgb.mean() < 235
