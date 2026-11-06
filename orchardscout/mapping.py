"""Mapping in the gravity-aligned map frame M (z up, ground at z = 0).

Two maps are built from each frame's metric depth and the visual-odometry pose:

* ``OccupancyGrid``: a 2-D log-odds grid at flight height, used for exploration
  (frontiers) and path planning. Every visible point frees the horizontal line of sight up
  to it, because the only obstacles at flight height are trunks, which span from the ground
  to the canopy and would block the ray. Points at trunk height mark their cell occupied.
* ``TrunkMap``: per-frame trunk detections (clusters of points at trunk height) fused into
  landmarks. These give the orchard size estimate and can correct odometry drift.

Depth beyond ``max_range`` is not used: monocular depth saturates past ~8-10 m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from . import geometry as geo


@dataclass
class MapConfig:
    resolution: float = 0.2  # m per cell
    size_m: float = 340.0  # square map centred on the start; larger than any orchard here
    max_range: float = 8.0  # m (planar depth) for free space
    hit_range: float = 5.0  # m; obstacles and trunks only this close (depth is accurate)
    pixel_stride: int = 8
    trunk_stride: int = 2
    trunk_voxel: float = 0.05  # m, points are binned before clustering
    ground_max_z: float = 0.15
    obstacle_min_z: float = 0.25
    obstacle_max_z: float = 1.45  # below the canopy base
    hit_margin: float = 0.3  # m kept unknown in front of an obstacle hit
    l_free: float = -0.4
    l_occ: float = 0.85
    l_min: float = -4.0
    l_max: float = 4.0
    occ_thresh: float = 1.0
    free_thresh: float = -0.8


class OccupancyGrid:
    UNKNOWN, FREE, OCCUPIED = 0, 1, 2

    def __init__(self, config: MapConfig | None = None) -> None:
        self.cfg = config or MapConfig()
        n = int(round(self.cfg.size_m / self.cfg.resolution))
        self.n = n
        self.logodds = np.zeros((n, n), np.float32)  # [ix, iy]
        self.seen = np.zeros((n, n), bool)
        self.origin = -0.5 * self.cfg.size_m  # world coordinate of cell edge 0

    # ---- coordinates
    def to_cell(self, xy: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(xy) - self.origin) / self.cfg.resolution).astype(np.int64)

    def to_world(self, ij: np.ndarray) -> np.ndarray:
        return (np.asarray(ij) + 0.5) * self.cfg.resolution + self.origin

    def inside(self, ij: np.ndarray) -> np.ndarray:
        return (ij[..., 0] >= 0) & (ij[..., 0] < self.n) & (ij[..., 1] >= 0) & (ij[..., 1] < self.n)

    def state(self) -> np.ndarray:
        s = np.full(self.logodds.shape, self.UNKNOWN, np.uint8)
        s[self.seen & (self.logodds <= self.cfg.free_thresh)] = self.FREE
        s[self.logodds >= self.cfg.occ_thresh] = self.OCCUPIED
        return s

    # ---- update
    def _flat(self, ij: np.ndarray) -> np.ndarray:
        ij = ij[self.inside(ij)]
        return np.unique(ij[:, 0] * self.n + ij[:, 1])

    def disk_cells(self, centres: np.ndarray, radius: float) -> np.ndarray:
        """Grid cells (K, 2) within ``radius`` of any of the given centres."""
        if len(centres) == 0:
            return np.zeros((0, 2), np.int64)
        r = int(np.ceil(radius / self.cfg.resolution))
        di, dj = np.mgrid[-r:r + 1, -r:r + 1]
        offs = np.stack([di.ravel(), dj.ravel()], 1)
        offs = offs[np.hypot(offs[:, 0], offs[:, 1]) * self.cfg.resolution <= radius + 1e-9]
        base = self.to_cell(centres)
        return (base[:, None, :] + offs[None, :, :]).reshape(-1, 2)

    def integrate(
        self,
        cam_xy: np.ndarray,
        ends_xy: np.ndarray,
        hit: np.ndarray,
        stop_short: np.ndarray | None = None,
        occ_xy: np.ndarray | None = None,
        occ_radius: float = 0.25,
    ) -> None:
        """Free the segments cam -> end and mark obstacles occupied.

        Obstacles are the ends flagged ``hit`` plus disks of ``occ_radius`` around
        ``occ_xy`` (detected trunks). Segments ending on something at obstacle height
        (``stop_short``, default ``hit``) stop ``hit_margin`` short, so the free ray does
        not erase what it ends on.
        """
        c = self.cfg
        if stop_short is None:
            stop_short = hit
        d = ends_xy - cam_xy
        length = np.linalg.norm(d, axis=1)
        reach = np.where(stop_short, np.maximum(length - c.hit_margin, 0.0), length)
        step = 0.5 * c.resolution
        n = np.ceil(reach / step).astype(np.int64)
        keep = n > 0
        if keep.any():
            idx = np.repeat(np.nonzero(keep)[0], n[keep])
            # sample positions t in (0, reach] along each ray
            starts = np.cumsum(n[keep]) - n[keep]
            k = np.arange(len(idx)) - np.repeat(starts, n[keep]) + 1
            t = (k * step) / np.maximum(length[idx], 1e-9)
            pts = cam_xy + d[idx] * t[:, None]
            free = self._flat(self.to_cell(pts))
        else:
            free = np.zeros(0, np.int64)
        occ_cells = [self.to_cell(ends_xy[hit])] if hit.any() else []
        if occ_xy is not None and len(occ_xy):
            occ_cells.append(self.disk_cells(occ_xy, occ_radius))
        occ = self._flat(np.vstack(occ_cells)) if occ_cells else np.zeros(0, np.int64)
        free = np.setdiff1d(free, occ, assume_unique=True)
        L, S = self.logodds.reshape(-1), self.seen.reshape(-1)
        L[free] = np.clip(L[free] + c.l_free, c.l_min, c.l_max)
        L[occ] = np.clip(L[occ] + c.l_occ, c.l_min, c.l_max)
        S[free] = True
        S[occ] = True


@dataclass
class TrunkConfig:
    cluster_eps: float = 0.3  # m
    min_points: int = 12
    max_extent: float = 0.9  # m; wider clusters are not single trunks
    min_height_span: float = 0.5  # m; a trunk is a vertical column of points
    gate: float = 0.6  # m, association distance to an existing landmark (along rows)
    # Across the rows (once their direction is known) the gate is wider: rows are >= 3 m
    # apart, so a 1.2 m gate cannot jump to the wrong row, and the sideways error built up
    # during a headland turn can be pulled back when the shared row is seen again.
    gate_across: float = 1.2
    min_obs: int = 5  # observations before a landmark counts as a tree
    occupied_radius: float = 0.25  # m drawn as occupied around each confirmed trunk
    # pose correction from landmarks
    correct: bool = True
    correct_gain: float = 0.15  # fraction of the fitted translation applied per frame
    # Rotation from a few nearby trunks is noisy (0.2 m noise over a 3 m spread is ~4 deg)
    # and, applied every frame, random-walks the heading; it is only used when the
    # matched trunks are many and spread out, and with a much smaller gain.
    correct_rot_gain: float = 0.0
    correct_rot_min_matches: int = 6
    correct_rot_min_spread: float = 4.0  # m
    correct_min_matches: int = 3
    correct_max_rot_deg: float = 6.0
    correct_max_shift: float = 1.3  # m (sideways error after a headland turn can be ~1 m)
    # When several matches agree on the same offset, it is a re-sighting of mapped trees
    # after drift: snap most of the way back at once instead of creeping.
    snap_gain: float = 0.5
    snap_min_matches: int = 4
    snap_max_spread: float = 0.25  # m, std of the per-match offsets
    # Row compass: rows are straight and parallel, so the grid orientation of recently
    # seen trees must match the orientation frozen from the first mapped trees.
    compass_gain: float = 0.1  # fraction of the heading error removed per frame
    compass_min_landmarks: int = 25  # confirmed trees needed to freeze the orientation
    compass_window: int = 15  # frames of recent detections used for the local estimate
    compass_min_trees: int = 4
    # Line-based local estimates are precise, so a moderate window suffices; a wide window
    # (35 deg) with noisy neighbour-direction estimates let wrong corrections through.
    compass_max_err_deg: float = 15.0
    # Row lattice: rows lie at a fixed spacing (measured inside single frames), so every
    # row line seen in a frame must fall on that lattice. The residual (modulo the
    # spacing) corrects the sideways position, as the compass corrects the heading.
    lattice_gain: float = 0.15
    lattice_min_samples: int = 20  # local row-spacing measurements before freezing
    lattice_max_spread: float = 0.25  # residuals of lines in one frame must agree (x spacing)


class TrunkMap:
    """Trunk landmarks fused from per-frame detections (weighted by 1 / range^2)."""

    def __init__(self, config: TrunkConfig | None = None) -> None:
        self.cfg = config or TrunkConfig()
        self.xy = np.zeros((0, 2))
        self.weight = np.zeros(0)
        self.count = np.zeros(0, np.int64)
        self._A: np.ndarray | None = None  # anisotropic matching metric, once rows are known

    def set_row_frame(self, theta_deg: float, along_axis: int) -> None:
        """Use a gate of ``gate`` along the rows and ``gate_across`` across them."""
        a = np.deg2rad(theta_deg)
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        s = np.ones(2)
        s[1 - along_axis] = self.cfg.gate / self.cfg.gate_across
        self._A = np.diag(s) @ R

    def _query(self, pts: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._A is None:
            return cKDTree(targets).query(pts, distance_upper_bound=self.cfg.gate)
        A = self._A.T
        return cKDTree(targets @ A).query(pts @ A, distance_upper_bound=self.cfg.gate)

    def confirmed(self) -> np.ndarray:
        return self.xy[self.count >= self.cfg.min_obs]

    def provisional(self, min_obs: int = 2) -> np.ndarray:
        """Trunks seen at least ``min_obs`` times: used as obstacles before confirmation."""
        return self.xy[self.count >= min_obs]

    def detect(self, pts: np.ndarray, voxel: float = 0.05,
               cam_xy: np.ndarray | None = None) -> np.ndarray:
        """Cluster trunk-height points (N, 3) into trunk centres (M, 2).

        Points are binned into ``voxel`` cells in x-y first (with counts and height range),
        which keeps clustering cheap when a nearby trunk contributes thousands of points.

        The camera only sees the front half of a trunk, whose points average ``2 r / pi``
        in front of the axis. With ``cam_xy`` given, each centre is pushed back along the
        viewing ray by that amount, with r estimated from the cluster's visible width.
        """
        c = self.cfg
        if len(pts) < c.min_points:
            return np.zeros((0, 2))
        keys = np.floor(pts[:, :2] / voxel).astype(np.int64)
        cells, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        zmin = np.full(len(cells), np.inf)
        zmax = np.full(len(cells), -np.inf)
        np.minimum.at(zmin, inv, pts[:, 2])
        np.maximum.at(zmax, inv, pts[:, 2])
        centres_xy = (cells + 0.5) * voxel
        pairs = cKDTree(centres_xy).query_pairs(c.cluster_eps, output_type="ndarray")
        labels = _connected_components(len(centres_xy), pairs)
        out = []
        for lab in np.unique(labels):
            m = labels == lab
            if counts[m].sum() < c.min_points:
                continue
            p, w = centres_xy[m], counts[m]
            if np.ptp(p, axis=0).max() > c.max_extent:
                continue
            if zmax[m].max() - zmin[m].min() < c.min_height_span:
                continue
            centre = (p * w[:, None]).sum(axis=0) / w.sum()
            if cam_xy is not None:
                ray = centre - cam_xy
                ray /= max(np.linalg.norm(ray), 1e-6)
                side = np.array([-ray[1], ray[0]])
                width = np.ptp(p @ side) + voxel  # visible width ~ trunk diameter
                r = float(np.clip(0.5 * width, 0.04, 0.3))
                centre = centre + ray * (2.0 * r / np.pi)
            out.append(centre)
        return np.array(out).reshape(-1, 2)

    def fit_correction(
        self, dets: np.ndarray, cam_xy: np.ndarray
    ) -> tuple[float, np.ndarray] | None:
        """Gain-scaled 2-D correction (angle, camera shift) moving detections onto landmarks.

        Returns None if there are too few confident matches or the fit is implausible.
        """
        c = self.cfg
        lm = self.confirmed()
        if len(dets) < c.correct_min_matches or len(lm) < c.correct_min_matches:
            return None
        d, j = self._query(dets, lm)
        ok = np.isfinite(d)
        if ok.sum() < c.correct_min_matches or len(np.unique(j[ok])) < ok.sum():
            return None
        src, dst = dets[ok], lm[j[ok]]
        w = 1.0 / (0.5 + np.sum((src - cam_xy) ** 2, axis=1))
        ms, md = (w[:, None] * src).sum(0) / w.sum(), (w[:, None] * dst).sum(0) / w.sum()
        H = ((src - ms) * w[:, None]).T @ (dst - md)
        ang = float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
        offset = md - ms
        if (abs(np.degrees(ang)) > c.correct_max_rot_deg
                or np.linalg.norm(offset) > c.correct_max_shift):
            return None
        spread = np.ptp(src, axis=0).max()
        rot_ok = len(src) >= c.correct_rot_min_matches and spread >= c.correct_rot_min_spread
        consistent = (len(src) >= c.snap_min_matches
                      and np.std(dst - src, axis=0).max() <= c.snap_max_spread)
        gain = c.snap_gain if consistent else c.correct_gain
        if rot_ok and c.correct_rot_gain > 0:
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            shift = R @ cam_xy + md - R @ ms - cam_xy
            return c.correct_rot_gain * ang, gain * shift
        return 0.0, gain * offset  # translation only: weighted mean offset of the matches

    @staticmethod
    def _row_axis(xy_rot: np.ndarray) -> int:
        """Which axis (0 or 1) of grid-aligned points the rows run along: the one with the
        smaller median nearest-neighbour gap between points sharing a line."""
        if len(xy_rot) < 4:
            return 0
        d, j = cKDTree(xy_rot).query(xy_rot, k=2)
        v = np.abs(xy_rot[j[:, 1]] - xy_rot)
        return 0 if np.sum(v[:, 0] > v[:, 1]) >= np.sum(v[:, 1] > v[:, 0]) else 1

    def add(self, dets: np.ndarray, cam_xy: np.ndarray) -> None:
        if len(dets) == 0:
            return
        w = 1.0 / (0.5 + np.sum((dets - cam_xy) ** 2, axis=1))
        if len(self.xy):
            dist, j = self._query(dets, self.xy)
        else:
            dist, j = np.full(len(dets), np.inf), np.zeros(len(dets), np.int64)
        new = []
        for k in range(len(dets)):
            if np.isfinite(dist[k]):
                i = j[k]
                tot = self.weight[i] + w[k]
                self.xy[i] = (self.xy[i] * self.weight[i] + dets[k] * w[k]) / tot
                self.weight[i] = tot
                self.count[i] += 1
            else:
                new.append(k)
        if new:
            self.xy = np.vstack([self.xy, dets[new]])
            self.weight = np.concatenate([self.weight, w[new]])
            self.count = np.concatenate([self.count, np.ones(len(new), np.int64)])


def _connected_components(n: int, pairs: np.ndarray) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if len(pairs) == 0:
        return np.arange(n)
    a = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(a, directed=False)[1]


def merge_points(xy: np.ndarray, radius: float) -> np.ndarray:
    """Merge points closer than ``radius`` (connected components) into their means."""
    if len(xy) == 0:
        return xy
    pairs = cKDTree(xy).query_pairs(radius, output_type="ndarray")
    labels = _connected_components(len(xy), pairs)
    return np.array([xy[labels == lab].mean(axis=0) for lab in np.unique(labels)])


def rotate_about(xy: np.ndarray, centre: np.ndarray, ang: float) -> np.ndarray:
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return (xy - centre) @ R.T + centre


class Mapper:
    """Feeds frames into the occupancy grid and the trunk map, and corrects drift."""

    def __init__(self, K: np.ndarray, config: MapConfig | None = None,
                 trunk_config: TrunkConfig | None = None) -> None:
        from collections import deque

        self.K = K
        self.cfg = config or MapConfig()
        self.grid = OccupancyGrid(self.cfg)
        self.trunks = TrunkMap(trunk_config)
        self.last_detections = np.zeros((0, 2))
        self.recent: deque[np.ndarray] = deque(maxlen=self.trunks.cfg.compass_window)
        self.row_theta: float | None = None  # frozen grid orientation, degrees in [0, 90)
        self.compass_log: list[float] = []
        # spacings measured inside single frames (immune to odometry drift)
        self.local_row_spacing: list[float] = []
        self.local_tree_spacing: list[float] = []
        self._row_axis: int | None = None
        self.frames = 0
        self.lattice_phase: float | None = None  # cross-row offset of the row lattice (m)
        self.lattice_log: list[float] = []

    def _row_rotation(self) -> np.ndarray:
        a = np.deg2rad(self.row_theta)
        return np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])

    def _measure_local_spacing(self, dets: np.ndarray) -> list[float]:
        """Row and tree spacing from the trunks seen in one frame (within ~5 m).

        Positions within a single frame do not depend on accumulated drift, so these
        measurements stay accurate however far the drone has flown. Returns the cross-row
        coordinates of the row lines seen in this frame.
        """
        if self.row_theta is None or len(dets) < 3:
            return []
        a = np.deg2rad(self.row_theta)
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        uv = dets @ R.T
        u, v = uv[:, self._row_axis], uv[:, 1 - self._row_axis]
        order = np.argsort(v)
        groups = np.split(order, np.nonzero(np.diff(v[order]) > 1.0)[0] + 1)
        centres = []
        for g in groups:
            if len(g) >= 2:
                centres.append(float(np.median(v[g])))
                du = np.diff(np.sort(u[g]))
                self.local_tree_spacing.extend(du[(du > 1.0) & (du < 5.0)].tolist())
        dc = np.diff(sorted(centres))
        self.local_row_spacing.extend(dc[(dc > 2.0) & (dc < 8.0)].tolist())
        return centres

    def _lattice(self, centres: list[float]) -> float:
        """Sideways correction (m, along the cross-row axis) from the row lattice."""
        tc = self.trunks.cfg
        if tc.lattice_gain <= 0 or self.row_theta is None:
            return 0.0
        if len(self.local_row_spacing) < tc.lattice_min_samples:
            return 0.0
        sp = float(np.median(self.local_row_spacing))
        if self.lattice_phase is None:  # freeze from the trees mapped so far
            lm = self.trunks.confirmed()
            if len(lm) < tc.compass_min_landmarks:
                return 0.0
            v = (lm @ self._row_rotation().T)[:, 1 - self._row_axis]
            ang = 2 * np.pi * v / sp
            self.lattice_phase = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
                                       * sp / (2 * np.pi))
            return 0.0
        if not centres:
            return 0.0
        res = ((np.asarray(centres) - self.lattice_phase + sp / 2) % sp) - sp / 2
        if len(res) > 1 and np.ptp(res) > tc.lattice_max_spread * sp:
            return 0.0
        r = float(np.mean(res))
        self.lattice_log.append(r)
        return -tc.lattice_gain * r

    def _compass(self, cam_xy: np.ndarray) -> float:
        """Heading correction (rad) from the row compass; 0 if not applicable."""
        from .size_estimation import line_orientation

        tc = self.trunks.cfg
        if tc.compass_gain <= 0:
            return 0.0
        if self.row_theta is None:
            return 0.0
        pts = merge_points(np.vstack(list(self.recent)), 0.4) if self.recent else np.zeros((0, 2))
        if len(pts) < tc.compass_min_trees:
            return 0.0
        try:
            # straight lines of >= 3 trees (rows, or row ends at a headland); scattered or
            # ghost detections are ignored, so the estimate is precise to ~1-2 deg
            local, _ = line_orientation(pts, seed=self.frames)
        except ValueError:
            return 0.0
        err = (local - self.row_theta + 45.0) % 90.0 - 45.0
        if abs(err) > tc.compass_max_err_deg:
            return 0.0
        self.compass_log.append(err)
        return -np.deg2rad(tc.compass_gain * err)

    def update(self, depth: np.ndarray, valid: np.ndarray, T_M_cam: np.ndarray,
               correct: bool = True) -> np.ndarray:
        """Integrate one frame. Returns the 4x4 pose correction C (map frame) that was
        applied from trunk landmarks; the corrected pose is ``C @ T_M_cam``."""
        c = self.cfg
        d = np.where(valid, depth, 0.0)
        T = T_M_cam.copy()
        cam_xy = T[:2, 3].copy()

        # trunks: denser pixels at trunk height, close enough for accurate depth
        Pt, _ = geo.backproject(d, self.K, stride=c.trunk_stride, max_depth=c.hit_range)
        Ptm = geo.transform_points(T, Pt)
        band = (Ptm[:, 2] > c.obstacle_min_z) & (Ptm[:, 2] < c.obstacle_max_z)
        dets = self.trunks.detect(Ptm[band], c.trunk_voxel, cam_xy)

        C = np.eye(4)
        if correct and self.trunks.cfg.correct:
            fit = self.trunks.fit_correction(dets, cam_xy)
            if fit is not None:
                ang, shift = fit  # already scaled by the gains
                R2 = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
                t2 = cam_xy + shift - R2 @ cam_xy  # rotate about the camera, then shift
                C[:2, :2], C[:2, 3] = R2, t2
                T = C @ T
                dets = dets @ R2.T + t2
                cam_xy = T[:2, 3].copy()
        self.frames += 1
        if self.row_theta is None:
            lm = self.trunks.confirmed()
            if len(lm) >= self.trunks.cfg.compass_min_landmarks:
                from .size_estimation import grid_orientation

                self.row_theta = grid_orientation(lm)  # frozen from the first mapped trees
                a = np.deg2rad(self.row_theta)
                R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
                self._row_axis = TrunkMap._row_axis(lm @ R.T)
                self.trunks.set_row_frame(self.row_theta, self._row_axis)
        if correct:
            self.recent.append(dets.copy())
            ang = self._compass(cam_xy)
            if ang != 0.0:
                R2 = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
                Cc = np.eye(4)
                Cc[:2, :2], Cc[:2, 3] = R2, cam_xy - R2 @ cam_xy  # rotate about the camera
                C, T = Cc @ C, Cc @ T
                dets = rotate_about(dets, cam_xy, ang)
                for k in range(len(self.recent)):
                    self.recent[k] = rotate_about(self.recent[k], cam_xy, ang)
        centres = self._measure_local_spacing(dets)
        if correct:
            dv = self._lattice(centres)
            if dv != 0.0:
                shift = dv * self._row_rotation()[1 - self._row_axis]  # cross-row unit vector
                Cl = np.eye(4)
                Cl[:2, 3] = shift
                C, T = Cl @ C, Cl @ T
                dets = dets + shift
                cam_xy = T[:2, 3].copy()
        self.last_detections = dets
        self.trunks.add(dets, cam_xy)

        # Occupancy: free space along every ray (to max_range). Obstacles come only from
        # trunk detections: raw points at trunk height are not trusted as obstacles,
        # because the network bends featureless open ground upwards ("rising" grass).
        # Those points still stop their ray short, which is safe: predicted depth there is
        # too close, so the real surface lies beyond.
        P, _ = geo.backproject(np.minimum(d, 1e3), self.K, stride=c.pixel_stride)
        far = P[:, 2] > c.max_range  # beyond range: free up to max_range only
        P[far] *= (c.max_range / P[far, 2])[:, None]
        Pm = geo.transform_points(T, P)
        z = Pm[:, 2]
        in_band = (~far) & (z > c.obstacle_min_z)
        no_hit = np.zeros(len(Pm), bool)
        self.grid.integrate(cam_xy, Pm[:, :2], no_hit, stop_short=in_band, occ_xy=dets,
                            occ_radius=self.trunks.cfg.occupied_radius)
        return C
