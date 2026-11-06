"""Frontier exploration of an orchard whose extent is unknown.

The explorer runs on the agent's own estimates only (visual-odometry pose, occupancy grid,
trunk landmarks). It never sees the simulator's ground truth.

* Planning grid (0.4 m): obstacles are occupied cells and mapped trunks, inflated by the
  drone radius plus a safety margin. Unknown cells are passable at a higher cost (the
  camera looks ahead, so they are checked before the drone gets there; plans are redone
  often). Shortest paths come from Dijkstra on an 8-connected grid graph.
* Frontiers: known-free cells next to unknown cells, grouped into clusters.
* Unknown extent: a frontier is *relevant* only if it is within ``relevance_radius`` of a
  mapped trunk (or, before any tree is known, near the start). Open field beyond the
  outermost rows therefore stops attracting the drone, and exploration ends by itself once
  every relevant frontier is gone: the orchard and its edges have been seen.
* Behaviour: a slow 360-degree look-around at the start; then fly to the nearest relevant
  frontier, turn to face the unknown area there, replan; stuck goals are blacklisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .control import FollowerConfig, PathFollower
from .geometry import wrap_angle
from .mapping import OccupancyGrid


@dataclass
class ExploreConfig:
    plan_res: float = 0.4  # m per planning cell (2 x the map resolution)
    inflate: float = 0.5  # m: drone radius 0.18 + safety margin
    trunk_radius: float = 0.25  # m drawn around each mapped trunk before inflation
    unknown_cost: float = 2.5
    # Open field is the worst place for monocular depth and odometry (featureless grass,
    # no rows for drift correction), so cells far from every mapped trunk cost more and
    # paths stay in the alleys and headlands.
    far_from_trees: float = 3.0  # m
    far_cost: float = 2.5  # multiplier
    relevance_min_obs: int = 4  # sightings before a tree makes nearby frontiers relevant
    final_spins: int = 3  # look around before declaring the orchard explored (at most)
    min_frontier_cells: int = 3
    relevance_radius: float = 5.0  # m from a mapped trunk
    # ...or this many row spacings, if larger: always look at least one row beyond the
    # outermost mapped row (with 5.4 m rows a fixed 5 m radius missed an outer row)
    relevance_rows: float = 1.5
    start_radius: float = 10.0  # m around the start, before any tree is mapped
    replan_every: int = 15  # steps
    switch_ratio: float = 0.7  # switch goals only if the new one is this much closer
    spin_rate: float = 0.6  # rad/s for look-arounds
    face_tolerance_deg: float = 12.0
    stuck_steps: int = 100  # > the longest turn in place (pi / 0.6 rad/s = 52 steps)
    stuck_dist: float = 0.3  # m moved within stuck_steps
    blacklist_radius: float = 1.5  # m around a goal that could not be reached
    window_margin: float = 12.0  # m around the seen area
    safety_radius: float = 0.45  # m: closer than this to a trunk centre -> back away
    # When nothing is reachable (often a ghost trunk blocking an alley), replan with a
    # thinner margin around confirmed trees only before declaring the rest unreachable.
    fallback_inflate: float = 0.3
    lookahead_check: float = 0.9  # m of commanded motion checked for trunks each step
    path_check: float = 4.0  # m of the remaining path re-checked against trunks each step
    follower: FollowerConfig = field(
        default_factory=lambda: FollowerConfig(cruise_speed=1.0, max_yaw_rate=0.6)
    )


@dataclass
class Plan:
    goal: np.ndarray
    path: np.ndarray
    cost: float
    face: float | None  # heading to face on arrival (towards the unknown)


class Explorer:
    SPIN, FOLLOW, FACE, DONE = "spin", "follow", "face", "done"

    def __init__(self, config: ExploreConfig | None = None) -> None:
        self.cfg = config or ExploreConfig()
        self.follower = PathFollower(self.cfg.follower)
        self.mode = self.SPIN
        self.spun = 0.0
        self.step_i = 0
        self.plan: Plan | None = None
        self.start_xy: np.ndarray | None = None
        self.blacklist: list[np.ndarray] = []
        self._hist: list[np.ndarray] = []
        self.n_plans = 0
        self.last_frontiers = np.zeros((0, 2))
        self.debug = False
        self.n_relevant = 0
        self.status = "running"
        self.n_safety = 0
        self.n_fallback = 0
        self.n_final_spins = 0
        self.row_spacing: float | None = None  # measured by the mapper, set by the caller
        self._force_replan = False

    # ------------------------------------------------------------ planning
    def _planning_grid(self, grid: OccupancyGrid, trunks: np.ndarray,
                       inflate: float | None = None, use_grid_occ: bool = True):
        c = self.cfg
        inflate = c.inflate if inflate is None else inflate
        st = grid.state()
        seen = np.nonzero(grid.seen.any(axis=1))[0], np.nonzero(grid.seen.any(axis=0))[0]
        f = int(round(c.plan_res / grid.cfg.resolution))
        m = int(c.window_margin / grid.cfg.resolution)
        i0, i1 = max(seen[0].min() - m, 0), min(seen[0].max() + m, grid.n)
        j0, j1 = max(seen[1].min() - m, 0), min(seen[1].max() + m, grid.n)
        i1 = i0 + (i1 - i0) // f * f
        j1 = j0 + (j1 - j0) // f * f
        sub = st[i0:i1, j0:j1]
        occ = (sub == grid.OCCUPIED) if use_grid_occ else np.zeros(sub.shape, bool)
        if len(trunks):
            cells = grid.disk_cells(trunks, c.trunk_radius) - [i0, j0]
            ok = ((cells[:, 0] >= 0) & (cells[:, 0] < occ.shape[0])
                  & (cells[:, 1] >= 0) & (cells[:, 1] < occ.shape[1]))
            occ[cells[ok, 0], cells[ok, 1]] = True
        r = int(np.ceil(inflate / grid.cfg.resolution))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        occ = ndimage.binary_dilation(occ, structure=(xx**2 + yy**2) <= r * r)
        H, W = occ.shape[0] // f, occ.shape[1] // f
        blk = lambda a: a.reshape(H, f, W, f)  # noqa: E731
        obstacle = blk(occ).any(axis=(1, 3))
        free = (blk(sub == grid.FREE).mean(axis=(1, 3)) >= 0.5) & ~obstacle
        unknown = ~obstacle & ~free
        origin = grid.to_world(np.array([i0, j0])) - 0.5 * grid.cfg.resolution
        return obstacle, free, unknown, origin

    def _graph(self, obstacle: np.ndarray, free: np.ndarray,
               far: np.ndarray | None = None) -> csr_matrix:
        H, W = obstacle.shape
        cost = np.where(free, 1.0, self.cfg.unknown_cost)
        if far is not None:
            cost = cost * np.where(far, self.cfg.far_cost, 1.0)
        idx = np.arange(H * W).reshape(H, W)
        rows, cols, wts = [], [], []
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            i0, i1 = max(0, -di), H - max(0, di)
            j0, j1 = max(0, -dj), W - max(0, dj)
            a = idx[i0:i1, j0:j1].ravel()
            b = idx[i0 + di:i1 + di, j0 + dj:j1 + dj].ravel()
            ok = ~obstacle.ravel()[a] & ~obstacle.ravel()[b]
            w = np.hypot(di, dj) * 0.5 * (cost.ravel()[a] + cost.ravel()[b])
            rows.append(a[ok])
            cols.append(b[ok])
            wts.append(w[ok])
        r, c_, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
        n = H * W
        return csr_matrix((np.concatenate([w, w]), (np.concatenate([r, c_]),
                                                     np.concatenate([c_, r]))), shape=(n, n))

    def _make_plan(self, xy: np.ndarray, grid: OccupancyGrid, trunks: np.ndarray,
                   obstacles: np.ndarray, fallback: bool = False) -> Plan | None:
        c = self.cfg
        if fallback:
            obstacle, free, unknown, origin = self._planning_grid(
                grid, trunks, inflate=c.fallback_inflate, use_grid_occ=False)
        else:
            obstacle, free, unknown, origin = self._planning_grid(grid, obstacles)
        H, W = obstacle.shape
        res = c.plan_res
        to_cell = lambda p: np.floor((p - origin) / res).astype(int)  # noqa: E731
        to_xy = lambda ij: origin + (np.asarray(ij) + 0.5) * res  # noqa: E731

        # frontier cells: free with an unknown 4-neighbour
        nb = np.zeros_like(unknown)
        nb[1:, :] |= unknown[:-1, :]
        nb[:-1, :] |= unknown[1:, :]
        nb[:, 1:] |= unknown[:, :-1]
        nb[:, :-1] |= unknown[:, 1:]
        frontier = free & nb
        labels, n_lab = ndimage.label(frontier, structure=np.ones((3, 3)))
        if n_lab == 0:
            self.last_frontiers = np.zeros((0, 2))
            return None

        start = to_cell(xy)
        start = np.clip(start, 0, [H - 1, W - 1])
        if obstacle[start[0], start[1]]:  # inside an inflated obstacle: use nearest open cell
            d, (ii, jj) = ndimage.distance_transform_edt(obstacle, return_indices=True)
            start = np.array([ii[start[0], start[1]], jj[start[0], start[1]]])
        far = None
        if len(obstacles):
            ctr = to_xy(np.argwhere(np.ones((H, W), bool)))
            far = (cKDTree(obstacles).query(ctr)[0] > c.far_from_trees).reshape(H, W)
        G = self._graph(obstacle, free, far)
        dist, pred = dijkstra(G, indices=start[0] * W + start[1], return_predecessors=True)
        dist = dist.reshape(H, W)

        tree = cKDTree(trunks) if len(trunks) >= 3 else None
        best = None
        fr_pts = []
        for lab in range(1, n_lab + 1):
            cells = np.argwhere(labels == lab)
            if len(cells) < c.min_frontier_cells:
                continue
            pts = to_xy(cells)
            if tree is not None:
                radius = c.relevance_radius
                if self.row_spacing is not None and np.isfinite(self.row_spacing):
                    radius = max(radius, c.relevance_rows * self.row_spacing)
                near = tree.query(pts)[0] <= radius
            else:
                near = np.linalg.norm(pts - self.start_xy, axis=1) <= c.start_radius
            if not near.any():
                continue
            cells, pts = cells[near], pts[near]
            fr_pts.append(pts)
            dd = dist[cells[:, 0], cells[:, 1]]
            k = int(np.argmin(dd))
            if not np.isfinite(dd[k]):
                continue
            if any(np.linalg.norm(pts[k] - b) < c.blacklist_radius for b in self.blacklist):
                continue
            if best is None or dd[k] < best[0]:
                best = (dd[k], cells[k], pts)
        self.last_frontiers = np.vstack(fr_pts) if fr_pts else np.zeros((0, 2))
        self.n_relevant = sum(len(p) for p in fr_pts)
        if best is None:
            return None
        cost, goal_cell, cluster_pts = best
        # path by walking predecessors back to the start
        node, path = goal_cell[0] * W + goal_cell[1], []
        while node >= 0 and len(path) < 100000:
            path.append(divmod(node, W))
            node = pred[node]
        path = to_xy(np.array(path[::-1]))
        if np.linalg.norm(path[0] - xy) > 0.3:  # start was inside an inflated obstacle
            path = np.vstack([xy, path])
        # on arrival, face the unknown cells around the goal
        gi, gj = goal_cell
        win = unknown[max(gi - 8, 0):gi + 9, max(gj - 8, 0):gj + 9]
        face = None
        if win.any():
            uc = np.argwhere(win).mean(axis=0) + [max(gi - 8, 0), max(gj - 8, 0)]
            d = to_xy(uc) - to_xy(goal_cell)
            face = float(np.arctan2(d[1], d[0]))
        self.n_plans += 1
        if self.debug:
            goal_xy = np.round(to_xy(goal_cell), 1)
            print(f"  plan {self.n_plans}: from {np.round(xy, 1)} goal {goal_xy} "
                  f"cost {cost:.1f} clusters {n_lab} relevant {len(fr_pts)}", flush=True)
        return Plan(goal=to_xy(goal_cell), path=path, cost=float(cost), face=face)

    # ------------------------------------------------------------ control
    def command(self, T_M_cam: np.ndarray, grid: OccupancyGrid, trunks: np.ndarray,
                obstacles: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Velocity (forward, left) and yaw rate from the current pose estimate and maps.

        ``trunks``: confirmed trees (decide which frontiers matter). ``obstacles``: every
        trunk seen at least twice (for safety); defaults to ``trunks``.
        """
        if obstacles is None:
            obstacles = trunks
        v, w = self._command(T_M_cam, grid, trunks, obstacles)
        return self._safety(T_M_cam, obstacles, v, w)

    def _safety(self, T_M_cam: np.ndarray, obstacles: np.ndarray, v: np.ndarray,
                w: float) -> tuple[np.ndarray, float]:
        """Back away from a trunk that is too close; never move forward into one."""
        c = self.cfg
        if len(obstacles) == 0:
            return v, w
        xy = T_M_cam[:2, 3]
        f = T_M_cam[:2, 2] / (np.linalg.norm(T_M_cam[:2, 2]) + 1e-9)  # heading on the ground
        d = np.linalg.norm(obstacles - xy, axis=1)
        k = int(np.argmin(d))
        if d[k] < c.safety_radius:
            away = (xy - obstacles[k]) / max(d[k], 1e-6)
            body = np.array([f @ away, -f[1] * away[0] + f[0] * away[1]])  # world -> body
            self.n_safety += 1
            self._force_replan = True
            return 0.3 * body, 0.0
        if v[0] > 0:
            ahead = xy + f * np.linspace(0.0, c.lookahead_check, 5)[:, None]
            dd = np.linalg.norm(ahead[:, None, :] - obstacles[None, :, :], axis=2).min()
            if dd < c.safety_radius:
                self.n_safety += 1
                self._force_replan = True
                return np.zeros(2), w
        return v, w

    def _path_blocked(self, xy: np.ndarray, obstacles: np.ndarray) -> bool:
        c = self.cfg
        if len(obstacles) == 0 or self.follower.done:
            return False
        p = self.follower.path
        seg = np.linalg.norm(np.diff(np.vstack([xy, p]), axis=0), axis=1)
        p = p[np.cumsum(seg) <= c.path_check]
        if len(p) == 0:
            return False
        dmin = cKDTree(obstacles).query(p)[0].min()
        return dmin < c.inflate + c.trunk_radius - 0.1

    def _command(self, T_M_cam: np.ndarray, grid: OccupancyGrid, trunks: np.ndarray,
                 obstacles: np.ndarray) -> tuple[np.ndarray, float]:
        c = self.cfg
        self.step_i += 1
        xy = T_M_cam[:2, 3].copy()
        f = T_M_cam[:3, 2]
        yaw = float(np.arctan2(f[1], f[0]))
        if self.start_xy is None:
            self.start_xy = xy.copy()

        if self.mode == self.SPIN:
            self.spun += c.spin_rate * 0.1
            if self.spun < 2 * np.pi:
                return np.zeros(2), c.spin_rate
            self.mode = self.FOLLOW
            self.plan = None
            self._hist.clear()

        if self.mode == self.FACE:
            if self.plan is None or self.plan.face is None:
                self.mode = self.FOLLOW
            else:
                err = float(wrap_angle(self.plan.face - yaw))
                if abs(err) > np.deg2rad(c.face_tolerance_deg):
                    return np.zeros(2), float(np.clip(1.5 * err, -c.spin_rate, c.spin_rate))
                self.mode = self.FOLLOW
                self.plan = None
                self._hist.clear()

        if self.mode == self.FOLLOW:
            self._hist.append(xy)
            blocked = self._path_blocked(xy, obstacles)
            need = (self.plan is None or self.follower.done or blocked or self._force_replan
                    or self.step_i % c.replan_every == 0)
            if need:
                self._force_replan = False
                new = self._make_plan(xy, grid, trunks, obstacles)
                if new is None and self.n_relevant > 0:
                    new = self._make_plan(xy, grid, trunks, obstacles, fallback=True)
                    if new is None and self.blacklist:
                        self.blacklist.clear()
                        new = self._make_plan(xy, grid, trunks, obstacles, fallback=True)
                    if new is not None:
                        self.n_fallback += 1
                if new is None and (self.plan is None or self.follower.done or blocked):
                    if self.n_final_spins < c.final_spins:
                        # nothing left to explore: look all around once more first, since
                        # trees at the edge of earlier views may not be confirmed yet
                        self.n_final_spins += 1
                        self.mode, self.spun, self.plan = self.SPIN, 0.0, None
                        return np.zeros(2), c.spin_rate
                    self.mode = self.DONE
                    # explored = nothing relevant left; otherwise what is left is unreachable
                    self.status = "explored" if self.n_relevant == 0 else "unreachable"
                    return np.zeros(2), 0.0
                if new is not None:
                    remaining = self._remaining_cost(xy)
                    # the old goal is obsolete once no relevant frontier is left near it
                    obsolete = self.plan is not None and (
                        len(self.last_frontiers) == 0
                        or np.min(np.linalg.norm(self.last_frontiers - self.plan.goal, axis=1))
                        > 1.0)
                    if (self.plan is None or self.follower.done or blocked or obsolete
                            or new.cost < c.switch_ratio * remaining
                            or np.linalg.norm(new.goal - self.plan.goal) < 1.0):
                        self.plan = new
                        self.follower.set_path(new.path)
            if self._stuck():
                if self.plan is not None:
                    self.blacklist.append(self.plan.goal.copy())
                self.plan = None
                self._hist.clear()
                return np.zeros(2), 0.0
            v, w = self.follower.command(xy[0], xy[1], yaw)
            if self.follower.done:
                self.mode = self.FACE
                return np.zeros(2), 0.0
            return v, w
        return np.zeros(2), 0.0

    def _remaining_cost(self, xy: np.ndarray) -> float:
        if self.plan is None or self.follower.done:
            return np.inf
        p = self.follower.path
        return float(np.linalg.norm(p[0] - xy) + np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))

    def _stuck(self) -> bool:
        c = self.cfg
        if len(self._hist) < c.stuck_steps:
            return False
        moved = np.linalg.norm(self._hist[-1] - self._hist[-c.stuck_steps])
        self._hist = self._hist[-c.stuck_steps:]
        return moved < c.stuck_dist

    @property
    def done(self) -> bool:
        return self.mode == self.DONE
