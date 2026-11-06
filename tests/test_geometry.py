import mujoco
import numpy as np

from orchardscout import geometry as geo


def test_quaternion_matches_mujoco():
    rng = np.random.default_rng(0)
    for _ in range(50):
        roll, pitch, yaw = rng.uniform(-np.pi, np.pi, 3) * np.array([0.5, 0.5, 1.0])
        R = geo.euler_zyx(roll, pitch, yaw)
        q = geo.quat_wxyz_from_matrix(R)
        back = np.zeros(9)
        mujoco.mju_quat2Mat(back, q)
        np.testing.assert_allclose(back.reshape(3, 3), R, atol=1e-9)


def test_backproject_then_project_is_identity():
    K = geo.intrinsics(64, 48, 75.0)
    depth = np.random.default_rng(1).uniform(0.5, 20.0, (48, 64))
    P, uv = geo.backproject(depth, K)
    np.testing.assert_allclose(geo.project(P, K), uv, atol=1e-9)
    np.testing.assert_allclose(P[:, 2], depth.ravel())


def test_intrinsics_field_of_view():
    K = geo.intrinsics(640, 480, 75.0)
    half_v = np.degrees(np.arctan((480 / 2) / K[1, 1]))
    assert abs(2 * half_v - 75.0) < 1e-9


def test_invert_transform():
    T = geo.make_T(geo.euler_zyx(0.1, -0.2, 0.3), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(T @ geo.invert_T(T), np.eye(4), atol=1e-12)
