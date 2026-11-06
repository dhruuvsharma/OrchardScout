"""Camera models, rigid transforms and back-projection.

Conventions used throughout the project:

* World frame: x east, y north, z up (metres). Orchard rows run along world x.
* Camera frame: OpenCV convention, x right, y down, z forward (optical axis).
  MuJoCo cameras look along their -z axis with +y up; ``MJ_TO_CV`` converts.
* ``T_a_b`` is a 4x4 homogeneous transform that maps points from frame b to frame a.
"""

from __future__ import annotations

import numpy as np

MJ_TO_CV = np.diag([1.0, -1.0, -1.0])


def intrinsics(width: int, height: int, fovy_deg: float) -> np.ndarray:
    """Pinhole intrinsics of a MuJoCo camera (square pixels, centred principal point).

    MuJoCo's ``fovy`` is the vertical field of view. Pixel centres are at integer
    coordinates (OpenCV convention), so the principal point is ((w-1)/2, (h-1)/2).
    """
    fy = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return np.array(
        [[fy, 0.0, (width - 1) / 2.0], [0.0, fy, (height - 1) / 2.0], [0.0, 0.0, 1.0]]
    )


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-world rotation R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def quat_wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix (Shepperd's method)."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N, 3) array of points."""
    return P @ T[:3, :3].T + T[:3, 3]


def pixel_grid(height: int, width: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates (u, v) of a strided grid, each shaped (h', w')."""
    v, u = np.mgrid[0:height:stride, 0:width:stride]
    return u.astype(np.float64), v.astype(np.float64)


def backproject(
    depth: np.ndarray,
    K: np.ndarray,
    stride: int = 1,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift a planar (z-buffer) depth map to 3-D points in the OpenCV camera frame.

    Returns ``(points, uv)``: points (N, 3) and their pixel coordinates (N, 2), keeping
    only finite, positive depths (and depths below ``max_depth`` if given).
    """
    h, w = depth.shape
    u, v = pixel_grid(h, w, stride)
    z = depth[::stride, ::stride].astype(np.float64)
    keep = np.isfinite(z) & (z > 0.0)
    if max_depth is not None:
        keep &= z < max_depth
    z, u, v = z[keep], u[keep], v[keep]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1), np.stack([u, v], axis=1)


def project(P_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project (N, 3) camera-frame points to (N, 2) pixel coordinates."""
    z = P_cam[:, 2:3]
    return (P_cam[:, :2] / z) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def wrap_angle(a: float | np.ndarray) -> float | np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi
