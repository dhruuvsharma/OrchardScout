"""Path following for a forward-looking multirotor.

The camera faces forward, so the drone turns to face where it is going before it speeds
up: forward speed falls off with heading error, and above ``turn_in_place_deg`` it only
rotates. Steering is pure pursuit on a look-ahead point along the path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import wrap_angle


@dataclass
class FollowerConfig:
    cruise_speed: float = 1.2  # m/s
    lookahead: float = 1.2  # m
    yaw_gain: float = 1.8  # rad/s per rad of heading error
    max_yaw_rate: float = 1.0  # rad/s
    turn_in_place_deg: float = 35.0
    goal_tolerance: float = 0.4  # m
    slow_radius: float = 1.5  # m, slow down approaching the goal


class PathFollower:
    def __init__(self, config: FollowerConfig | None = None) -> None:
        self.cfg = config or FollowerConfig()
        self.path = np.zeros((0, 2))
        self._i = 0

    def set_path(self, path: np.ndarray) -> None:
        self.path = np.asarray(path, dtype=np.float64).reshape(-1, 2)
        self._i = 0

    @property
    def done(self) -> bool:
        return len(self.path) == 0

    def command(self, x: float, y: float, yaw: float) -> tuple[np.ndarray, float]:
        """Body-frame velocity (forward, left) and yaw rate for the current pose."""
        c = self.cfg
        if self.done:
            return np.zeros(2), 0.0
        p = np.array([x, y])
        # advance the progress index to the closest point ahead of us
        d = np.linalg.norm(self.path[self._i:] - p, axis=1)
        self._i += int(np.argmin(d))
        goal = self.path[-1]
        dist_goal = float(np.linalg.norm(goal - p))
        if self._i == len(self.path) - 1 and dist_goal < c.goal_tolerance:
            self.path = np.zeros((0, 2))
            return np.zeros(2), 0.0
        # look-ahead point
        target = goal
        acc = 0.0
        for j in range(self._i, len(self.path) - 1):
            acc += float(np.linalg.norm(self.path[j + 1] - self.path[j]))
            if acc >= c.lookahead:
                target = self.path[j + 1]
                break
        dx, dy = target - p
        err = float(wrap_angle(np.arctan2(dy, dx) - yaw))
        yaw_rate = float(np.clip(c.yaw_gain * err, -c.max_yaw_rate, c.max_yaw_rate))
        if abs(err) > np.deg2rad(c.turn_in_place_deg):
            return np.zeros(2), yaw_rate
        speed = c.cruise_speed * np.cos(err) ** 2
        speed = min(speed, c.cruise_speed * max(dist_goal / c.slow_radius, 0.25))
        return np.array([speed, 0.0]), yaw_rate


def lawnmower_route(layout, start_alley: int, n_alleys: int, overshoot: float = 2.5) -> np.ndarray:
    """Waypoints along consecutive alleys, turning in the headland past the row ends.

    Uses the ground-truth layout, so it is a test harness for odometry and mapping, not
    part of the agent.
    """
    xmin, _, xmax, _ = layout.bbox()
    ys = 0.5 * (layout.row_y[:-1] + layout.row_y[1:])
    order = list(range(start_alley, min(start_alley + n_alleys, len(ys))))
    pts = []
    for k, i in enumerate(order):
        a, b = xmin - overshoot, xmax + overshoot
        xs = (a, b) if k % 2 == 0 else (b, a)
        pts.append((xs[0], ys[i]))
        pts.append((xs[1], ys[i]))
    return densify(np.array(pts), 0.25)


def densify(pts: np.ndarray, step: float) -> np.ndarray:
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    return np.array(out)
