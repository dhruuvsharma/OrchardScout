"""The agent: everything that would run on the drone.

Input per step: one RGB frame. Prior: the nominal flight height (for depth scale).
Output: the camera pose in the gravity-aligned map frame, the occupancy grid, the trunk
landmark map and the current orchard size estimate.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .mapping import MapConfig, Mapper, TrunkConfig
from .perception import Frame, Perception
from .size_estimation import OrchardEstimate, estimate_orchard
from .vo import VisualOdometry, VOConfig, VOStep


@dataclass
class AgentConfig:
    vo: VOConfig = field(default_factory=VOConfig)
    map: MapConfig = field(default_factory=MapConfig)
    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    ground_snap: bool = False


@dataclass
class AgentStep:
    T_M_cam: np.ndarray | None  # None until the first ground-plane fit
    frame: Frame
    vo: VOStep | None


class Agent:
    def __init__(
        self,
        K: np.ndarray,
        cam_pitch_deg: float,
        altitude: float,
        depth_fn: Callable[[np.ndarray], np.ndarray],
        config: AgentConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.cfg = config or AgentConfig()
        self.depth_fn = depth_fn
        self.altitude = altitude
        self.perception = Perception(K, cam_pitch_deg, altitude, seed=seed,
                                     ground_snap=self.cfg.ground_snap)
        self.vo = VisualOdometry(K, self.cfg.vo)
        self.mapper = Mapper(K, self.cfg.map, self.cfg.trunk)
        self.T_M_cam: np.ndarray | None = None
        self.timings: dict[str, list[float]] = defaultdict(list)

    def step(self, rgb: np.ndarray, T_override: np.ndarray | None = None) -> AgentStep:
        """Process one frame. ``T_override`` replaces the VO pose (diagnostics only)."""
        t0 = time.perf_counter()
        raw = self.depth_fn(rgb)
        t1 = time.perf_counter()
        frame = self.perception.process(rgb, raw)
        res = None
        if self.T_M_cam is None:
            if frame.up_cam is None:
                return AgentStep(None, frame, None)  # wait for a visible ground plane
            self.T_M_cam = self.vo.start(frame.gray, frame.depth, frame.feature_mask,
                                         frame.up_cam, self.altitude)
        else:
            res = self.vo.track(frame.gray, frame.depth, frame.feature_mask, frame.up_cam,
                                self.altitude)
            self.T_M_cam = res.T_M_cam
        t2 = time.perf_counter()
        pose = T_override if T_override is not None else self.T_M_cam
        C = self.mapper.update(frame.depth, frame.map_mask, pose,
                               correct=T_override is None)
        if T_override is None and not np.allclose(C, np.eye(4)):
            self.vo.apply_correction(C)  # landmark-based drift correction
            self.T_M_cam = self.vo.T_M_cam.copy()
            pose = self.T_M_cam
        t3 = time.perf_counter()
        self.timings["depth"].append(t1 - t0)
        self.timings["vo"].append(t2 - t1)
        self.timings["map"].append(t3 - t2)
        return AgentStep(pose.copy(), frame, res)

    def estimate(self) -> OrchardEstimate | None:
        m = self.mapper
        return estimate_orchard(m.trunks.confirmed(), m.local_row_spacing, m.local_tree_spacing)

    def timing_summary(self) -> dict[str, float]:
        return {f"ms_{k}": 1e3 * float(np.median(v)) for k, v in self.timings.items() if v}
