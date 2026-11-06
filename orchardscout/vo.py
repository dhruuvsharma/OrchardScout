"""Visual odometry from one RGB camera plus monocular metric depth (RGB-D style).

* Features: Shi-Tomasi corners, spread over a grid. Pixels on the sky, near depth edges
  or with unreliable depth are excluded.
* Tracking: pyramidal Lucas-Kanade optical flow from frame to frame, with a forward-backward
  check. Local tracking is safer than descriptor matching in an orchard, where every tree
  looks like its neighbour.
* Pose: PnP + RANSAC between the keyframe's 3-D points (from predicted depth) and their
  tracked pixels in the current frame, refined with Levenberg-Marquardt. A new keyframe is
  made when too few points survive or the camera has moved far enough.
* Ground constraint: the ground plane fitted for depth scale (``orchardscout.scale``) also
  gives the gravity direction and the camera height in every frame. These are blended into
  the pose, so roll, pitch and height do not drift; only x, y and heading are integrated.

The map frame M is gravity-aligned: origin on the ground under the first camera position,
z up, x along the first camera heading.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import geometry as geo


@dataclass
class VOConfig:
    max_features: int = 600
    grid: tuple[int, int] = (4, 6)  # rows, cols for evenly spread detection
    quality: float = 0.005
    min_distance: int = 9
    min_depth: float = 0.4  # m
    max_depth: float = 12.0  # m; monocular depth is unreliable beyond ~10 m
    klt_win: int = 21
    klt_levels: int = 4
    fb_max_px: float = 1.0
    ransac_px: float = 2.0
    min_inliers: int = 25
    max_step_translation: float = 0.5  # m per frame; larger jumps are rejected as failures
    max_step_rotation_deg: float = 20.0
    kf_min_tracked: int = 150
    kf_max_translation: float = 1.0  # m
    kf_max_rotation_deg: float = 15.0
    ground_alpha: float = 0.3  # 0 disables the ground constraint


@dataclass
class VOStep:
    T_M_cam: np.ndarray
    n_tracked: int
    n_inliers: int
    new_keyframe: bool
    lost: bool


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3)  # a is (anti)parallel to b; the anti case never happens here
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def scaled_rotation(R: np.ndarray, alpha: float) -> np.ndarray:
    """R raised to the power alpha (same axis, alpha times the angle)."""
    rvec, _ = cv2.Rodrigues(R)
    return cv2.Rodrigues(rvec * alpha)[0]


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Nearest rotation matrix (removes accumulated floating-point drift)."""
    U, _, Vt = np.linalg.svd(R)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def map_frame_from_up(up_cam: np.ndarray) -> np.ndarray:
    """R_M_cam for a gravity-aligned map frame whose x axis is the camera heading."""
    z = up_cam / np.linalg.norm(up_cam)
    f = np.array([0.0, 0.0, 1.0])
    x = f - (f @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1).T  # rows: map axes in camera coordinates


class VisualOdometry:
    def __init__(self, K: np.ndarray, config: VOConfig | None = None) -> None:
        self.K = K
        self.cfg = config or VOConfig()
        c = self.cfg
        self._lk = dict(
            winSize=(c.klt_win, c.klt_win),
            maxLevel=c.klt_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.T_M_cam: np.ndarray | None = None
        self.n_keyframes = 0
        self.n_lost = 0

    # ---------------------------------------------------------------- helpers
    def _detect(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        c = self.cfg
        h, w = gray.shape
        rows, cols = c.grid
        per_cell = max(1, c.max_features // (rows * cols))
        pts = []
        for i in range(rows):
            for j in range(cols):
                y0, y1 = i * h // rows, (i + 1) * h // rows
                x0, x1 = j * w // cols, (j + 1) * w // cols
                cell_mask = np.zeros_like(mask)
                cell_mask[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
                p = cv2.goodFeaturesToTrack(
                    gray, per_cell, c.quality, c.min_distance, mask=cell_mask, blockSize=7
                )
                if p is not None:
                    pts.append(p.reshape(-1, 2))
        return np.concatenate(pts).astype(np.float32) if pts else np.zeros((0, 2), np.float32)

    def _new_keyframe(self, gray: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> None:
        c = self.cfg
        valid = mask & (depth > c.min_depth) & (depth < c.max_depth)
        pts = self._detect(gray, valid.astype(np.uint8) * 255)
        u = np.clip(np.round(pts[:, 0]).astype(int), 0, gray.shape[1] - 1)
        v = np.clip(np.round(pts[:, 1]).astype(int), 0, gray.shape[0] - 1)
        z = depth[v, u].astype(np.float64)
        X = np.stack(
            [(pts[:, 0] - self.K[0, 2]) * z / self.K[0, 0],
             (pts[:, 1] - self.K[1, 2]) * z / self.K[1, 1], z], axis=1)
        self.kf_points = X  # (N, 3) in the keyframe camera frame
        self.tracks = pts  # current pixel positions of those points
        self.T_M_kf = self.T_M_cam.copy()
        self.T_cur_kf = np.eye(4)
        self.n_keyframes += 1

    def _predict_pixels(self, T_cur_kf: np.ndarray) -> np.ndarray:
        """Where the keyframe points should appear under a predicted pose."""
        P = geo.transform_points(T_cur_kf, self.kf_points)
        uv = self.tracks.copy()
        front = P[:, 2] > 0.1
        uv[front] = geo.project(P[front], self.K).astype(np.float32)
        return uv

    def _track(self, gray: np.ndarray, guess: np.ndarray | None = None) -> np.ndarray:
        """Advance self.tracks into ``gray``; returns a boolean mask of survivors.

        ``guess``: predicted pixel positions (from the motion model) to start the search.
        """
        if len(self.tracks) == 0:
            return np.zeros(0, bool)
        p0 = self.tracks.reshape(-1, 1, 2)
        if guess is not None:
            p1 = guess.reshape(-1, 1, 2).astype(np.float32).copy()
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, p0, p1, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self._lk
            )
        else:
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self._lk)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, p1, p0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self._lk
        )
        p1 = p1.reshape(-1, 2)
        fb = np.linalg.norm(p0.reshape(-1, 2) - p0b.reshape(-1, 2), axis=1)
        h, w = gray.shape
        ok = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.cfg.fb_max_px)
        ok &= (p1[:, 0] >= 0) & (p1[:, 0] < w) & (p1[:, 1] >= 0) & (p1[:, 1] < h)
        self.tracks = p1
        return ok

    def _apply_ground(self, up_cam: np.ndarray, cam_height: float) -> None:
        """Pull roll/pitch and height towards the ground-plane estimate (complementary filter).

        The correction is applied to the keyframe anchor too, so it persists.
        """
        a = self.cfg.ground_alpha
        T_old = self.T_M_cam
        R = T_old[:3, :3]
        R_fix = scaled_rotation(rotation_between(R @ up_cam, np.array([0.0, 0.0, 1.0])), a)
        T_new = T_old.copy()
        T_new[:3, :3] = orthonormalize(R_fix @ R)
        T_new[2, 3] = (1 - a) * T_old[2, 3] + a * cam_height
        # Move the keyframe with the camera so the relative pose T_cur_kf is unchanged.
        self.T_M_kf = T_new @ self.T_cur_kf
        self.T_M_cam = T_new

    def apply_correction(self, C: np.ndarray) -> None:
        """Apply an external correction (map frame, ``T <- C @ T``), e.g. from landmarks.

        The keyframe anchor moves with the camera so tracking continues seamlessly.
        """
        T_new = C @ self.T_M_cam
        T_new[:3, :3] = orthonormalize(T_new[:3, :3])
        self.T_M_kf = T_new @ self.T_cur_kf
        self.T_M_cam = T_new

    # ---------------------------------------------------------------- public
    def start(self, gray: np.ndarray, depth: np.ndarray, mask: np.ndarray,
              up_cam: np.ndarray, cam_height: float) -> np.ndarray:
        """Initialise on the first frame. Needs a ground-plane normal (camera frame)."""
        self.T_M_cam = geo.make_T(map_frame_from_up(up_cam), np.array([0.0, 0.0, cam_height]))
        self._velocity = np.eye(4)  # last frame-to-frame motion, for the motion model
        self._new_keyframe(gray, depth, mask)
        self.prev_gray = gray
        return self.T_M_cam.copy()

    def track(
        self,
        gray: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
        up_cam: np.ndarray | None = None,
        cam_height: float | None = None,
    ) -> VOStep:
        """Estimate the pose of a new frame.

        ``mask``: pixels allowed for new features (e.g. not sky, not depth edges).
        ``up_cam``: ground-plane normal in this camera frame, if the fit succeeded.
        """
        c = self.cfg
        T_prev = self.T_M_cam.copy()
        T_pred = T_prev @ self._velocity  # constant-velocity prediction
        T_cur_kf_pred = geo.invert_T(T_pred) @ self.T_M_kf
        ok = self._track(gray, self._predict_pixels(T_cur_kf_pred))
        n_tracked = int(ok.sum())
        n_inl, lost = 0, True

        if n_tracked >= c.min_inliers:
            obj = self.kf_points[ok].astype(np.float64)
            img = self.tracks[ok].astype(np.float64)
            rvec0, _ = cv2.Rodrigues(T_cur_kf_pred[:3, :3])
            tvec0 = T_cur_kf_pred[:3, 3].reshape(3, 1).copy()
            success, rvec, tvec, inl = cv2.solvePnPRansac(
                obj, img, self.K, None, rvec0, tvec0, useExtrinsicGuess=True,
                iterationsCount=100, reprojectionError=c.ransac_px, confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if success and inl is not None and len(inl) >= c.min_inliers:
                inl = inl.ravel()
                rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
                T_cur_kf = geo.make_T(cv2.Rodrigues(rvec)[0], tvec.ravel())
                T_new = self.T_M_kf @ geo.invert_T(T_cur_kf)
                T_new[:3, :3] = orthonormalize(T_new[:3, :3])
                step = geo.invert_T(T_prev) @ T_new
                jump = np.linalg.norm(step[:3, 3]) > c.max_step_translation or np.degrees(
                    np.linalg.norm(cv2.Rodrigues(step[:3, :3])[0])) > c.max_step_rotation_deg
                if not jump:
                    lost = False
                    n_inl = len(inl)
                    self.T_cur_kf = T_cur_kf
                    self.T_M_cam = T_new
                    keep = np.zeros(len(ok), bool)  # keep only inlier tracks
                    keep[np.where(ok)[0][inl]] = True
                    ok = keep

        if lost:  # keep the constant-velocity guess, then restart from a fresh keyframe
            self.n_lost += 1
            self.T_M_cam = T_pred
        else:
            self._velocity = geo.invert_T(T_prev) @ self.T_M_cam

        if c.ground_alpha > 0 and up_cam is not None and cam_height is not None:
            self._apply_ground(up_cam, cam_height)

        self.kf_points, self.tracks = self.kf_points[ok], self.tracks[ok]
        T_kf_cur = geo.invert_T(self.T_M_kf) @ self.T_M_cam
        moved = np.linalg.norm(T_kf_cur[:3, 3])
        turned = np.degrees(np.linalg.norm(cv2.Rodrigues(T_kf_cur[:3, :3])[0]))
        new_kf = (
            lost
            or n_inl < c.kf_min_tracked
            or moved > c.kf_max_translation
            or turned > c.kf_max_rotation_deg
        )
        if new_kf:
            self._new_keyframe(gray, depth, mask)
        self.prev_gray = gray
        return VOStep(self.T_M_cam.copy(), n_tracked, n_inl, new_kf, lost)
