"""MuJoCo simulator: a small multirotor with one forward-looking RGB camera in an orchard.

Flight model (deliberately simple, stated in the README): a velocity-commanded,
acceleration-limited body with yaw, held at a fixed altitude below the canopy by its
flight controller. Body roll and pitch follow the commanded acceleration (small-angle
multirotor approximation) plus a little noise, so the camera does not move unrealistically
smoothly. Collisions are checked exactly against the trunk cylinders and crown ellipsoids.

The agent may only use :meth:`OrchardSim.rgb`. Everything else (ground-truth depth,
ground-truth pose, the layout) exists for evaluation and visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from . import geometry as geo
from .orchard import OrchardLayout, sample_start_pose, world_elements

GRAVITY = 9.81


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fovy_deg: float = 75.0  # vertical field of view (horizontal is ~90 deg at 4:3)
    pitch_deg: float = 5.0  # downward tilt of the optical axis
    forward_offset: float = 0.10  # m in front of the body centre


@dataclass
class DroneConfig:
    altitude: float = 1.0  # m above ground, below the canopy base
    radius: float = 0.18  # m, collision sphere
    max_speed: float = 1.5  # m/s, horizontal
    max_accel: float = 2.0  # m/s^2
    max_yaw_rate_deg: float = 60.0
    dt: float = 0.1  # s per step = camera period (10 Hz)
    tilt_from_accel: bool = True
    attitude_noise_deg: float = 0.3


@dataclass
class RenderConfig:
    shadows: bool = True
    fog: bool = True
    fog_start: float = 60.0  # m
    fog_end: float = 220.0  # m
    haze: float = 0.3
    sun_dir: tuple[float, float, float] = (-0.45, -0.30, -1.0)
    sun_diffuse: float = 0.70
    ambient: float = 0.35
    shadow_size: int = 4096
    max_geom: int = 40000


@dataclass
class DroneState:
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    yaw_rate: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0


@dataclass
class StepResult:
    t: float
    collided: bool
    collision_with: str | None = None


def _f(v: float) -> str:
    return f"{v:.6g}"


def build_scene_xml(
    layout: OrchardLayout,
    cam: CameraConfig,
    drone: DroneConfig,
    render: RenderConfig,
) -> tuple[str, dict[str, bytes]]:
    """Full MJCF for the orchard plus the drone, and the texture assets it references."""
    asset_xml, world_xml, assets = world_elements(layout)
    xmin, ymin, xmax, ymax = layout.bbox()
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    # MuJoCo expresses near/far/fog distances relative to the model extent.
    extent = 0.5 * max(xmax - xmin, ymax - ymin) + 20.0
    sun = np.asarray(render.sun_dir, dtype=float)
    sun /= np.linalg.norm(sun)
    p = np.deg2rad(cam.pitch_deg)
    # Orthographic top-down view: vertical extent in metres, 4:3 image.
    ov = max(ymax - ymin + 16.0, 0.75 * (xmax - xmin + 16.0))
    a, s = render.ambient, render.sun_diffuse
    rotors = "\n".join(
        f'      <geom type="cylinder" size="0.07 0.004" pos="{dx} {dy} 0.03" '
        f'rgba="0.85 0.85 0.85 0.6" group="3"/>'
        for dx, dy in ((0.12, 0.12), (0.12, -0.12), (-0.12, 0.12), (-0.12, -0.12))
    )
    xml = f"""<mujoco model="orchard">
  <compiler angle="radian"/>
  <option timestep="{_f(drone.dt)}"/>
  <statistic center="{_f(cx)} {_f(cy)} 0" extent="{_f(extent)}"/>
  <visual>
    <global offwidth="1920" offheight="1440"/>
    <quality shadowsize="{render.shadow_size}" offsamples="4" numslices="24" numstacks="12"/>
    <map znear="{_f(0.03 / extent)}" zfar="{_f(500.0 / extent)}"
         fogstart="{_f(render.fog_start / extent)}" fogend="{_f(render.fog_end / extent)}"
         haze="{_f(render.haze)}"/>
    <headlight ambient="{a} {a} {a}" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
    <rgba fog="0.80 0.86 0.93 1" haze="0.86 0.90 0.96 1"/>
  </visual>
  <default>
    <geom contype="0" conaffinity="0"/>
  </default>
{asset_xml}
  <worldbody>
    <light name="sun" type="directional" castshadow="{str(render.shadows).lower()}"
           pos="{_f(cx)} {_f(cy)} 80" dir="{_f(sun[0])} {_f(sun[1])} {_f(sun[2])}"
           diffuse="{s} {s} {0.95 * s}" specular="0.05 0.05 0.05"/>
{world_xml}
    <camera name="overview" projection="orthographic" fovy="{_f(ov)}"
            pos="{_f(cx)} {_f(cy)} 150" xyaxes="1 0 0 0 1 0"/>
    <body name="drone" mocap="true" pos="{_f(cx)} {_f(cy)} {_f(drone.altitude)}">
      <geom name="drone_frame" type="box" size="0.10 0.10 0.025" rgba="0.12 0.12 0.12 1"
            group="3"/>
      <geom name="drone_nose" type="box" size="0.04 0.03 0.02" pos="0.11 0 0"
            rgba="0.9 0.3 0.1 1" group="3"/>
{rotors}
      <camera name="onboard" pos="{_f(cam.forward_offset)} 0 0"
              xyaxes="0 -1 0 {_f(np.sin(p))} 0 {_f(np.cos(p))}" fovy="{_f(cam.fovy_deg)}"/>
      <camera name="chase" pos="-2.2 0 0.4" xyaxes="0 -1 0 0.21 0 0.98" fovy="75"/>
    </body>
  </worldbody>
</mujoco>
"""
    return xml, assets


class OrchardSim:
    """Kinematic multirotor + onboard camera in a MuJoCo orchard scene."""

    def __init__(
        self,
        layout: OrchardLayout,
        camera: CameraConfig | None = None,
        drone: DroneConfig | None = None,
        render: RenderConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.layout = layout
        self.cam_cfg = camera or CameraConfig()
        self.drone_cfg = drone or DroneConfig()
        self.render_cfg = render or RenderConfig()
        if self.drone_cfg.altitude + self.drone_cfg.radius >= layout.min_canopy_z():
            raise ValueError("flight altitude must be below the canopy")

        xml, assets = build_scene_xml(layout, self.cam_cfg, self.drone_cfg, self.render_cfg)
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        self.K = geo.intrinsics(self.cam_cfg.width, self.cam_cfg.height, self.cam_cfg.fovy_deg)
        self._cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "onboard")
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

        self._trunk_tree = cKDTree(layout.trunks[:, :2])
        self._crown_tree = cKDTree(layout.canopies[:, 1:3])
        self._max_trunk_r = float(layout.trunks[:, 2].max())
        self._max_crown_r = float(layout.canopies[:, 4:6].max())

        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(
        self,
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
        mode: str = "alley",
    ) -> None:
        if x is None or y is None:
            x, y, sampled_yaw = sample_start_pose(self.layout, self.rng, mode)
            yaw = sampled_yaw if yaw is None else yaw
        pos = np.array([x, y, self.drone_cfg.altitude])
        self.state = DroneState(pos=pos, yaw=float(yaw or 0.0))
        self.t = 0.0
        self.distance = 0.0
        self.collisions: list[tuple[float, str]] = []
        self._apply_state()

    def _apply_state(self) -> None:
        s = self.state
        R = geo.euler_zyx(s.roll, s.pitch, s.yaw)
        self.data.mocap_pos[0] = s.pos
        self.data.mocap_quat[0] = geo.quat_wxyz_from_matrix(R)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def step(self, v_body: np.ndarray | tuple, yaw_rate: float) -> StepResult:
        """Advance one camera period.

        ``v_body`` = (forward, left) or (forward, left, up) velocity command in m/s in the
        body frame; ``yaw_rate`` in rad/s (positive = turn left). Altitude is held.
        """
        cfg, s = self.drone_cfg, self.state
        dt = cfg.dt
        vb = np.zeros(3)
        vb[: len(v_body)] = v_body
        c, si = np.cos(s.yaw), np.sin(s.yaw)
        v_des = np.array([c * vb[0] - si * vb[1], si * vb[0] + c * vb[1], 0.0])
        speed = np.linalg.norm(v_des[:2])
        if speed > cfg.max_speed:
            v_des[:2] *= cfg.max_speed / speed
        dv = v_des - s.vel
        max_dv = cfg.max_accel * dt
        n = np.linalg.norm(dv)
        if n > max_dv:
            dv *= max_dv / n
        acc = dv / dt
        new_vel = s.vel + dv
        new_pos = s.pos + new_vel * dt

        max_wz = np.deg2rad(cfg.max_yaw_rate_deg)
        s.yaw_rate = float(np.clip(yaw_rate, -max_wz, max_wz))
        s.yaw = float(geo.wrap_angle(s.yaw + s.yaw_rate * dt))

        if cfg.tilt_from_accel:
            a_fwd = c * acc[0] + si * acc[1]
            a_left = -si * acc[0] + c * acc[1]
            noise = np.deg2rad(cfg.attitude_noise_deg)
            s.pitch = float(np.arctan2(a_fwd, GRAVITY) + self.rng.normal(0.0, noise))
            s.roll = float(-np.arctan2(a_left, GRAVITY) + self.rng.normal(0.0, noise))

        hit = self.collision_at(new_pos)
        if hit is None:
            self.distance += float(np.linalg.norm(new_pos - s.pos))
            s.pos, s.vel = new_pos, new_vel
        else:  # stop at the obstacle instead of passing through it
            s.vel = np.zeros(3)
            self.collisions.append((self.t + dt, hit))
        self.t += dt
        self._apply_state()
        return StepResult(t=self.t, collided=hit is not None, collision_with=hit)

    def collision_at(self, p: np.ndarray) -> str | None:
        """Name of the obstacle a drone centred at ``p`` would touch, or None."""
        r = self.drone_cfg.radius
        if p[2] < r:
            return "ground"
        for k in self._trunk_tree.query_ball_point(p[:2], r + self._max_trunk_r):
            x, y, tr, top = self.layout.trunks[k]
            if np.hypot(p[0] - x, p[1] - y) < r + tr and p[2] - r < top:
                return f"trunk{k}"
        for k in self._crown_tree.query_ball_point(p[:2], r + self._max_crown_r):
            _, x, y, z, rx, ry, rz = self.layout.canopies[k]
            q = ((p[0] - x) / (rx + r)) ** 2 + ((p[1] - y) / (ry + r)) ** 2
            if q + ((p[2] - z) / (rz + r)) ** 2 < 1.0:
                return f"crown{k}"
        return None

    # ------------------------------------------------------------- rendering
    def _renderer(self, height: int, width: int) -> mujoco.Renderer:
        key = (height, width)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(
                self.model, height, width, max_geom=self.render_cfg.max_geom
            )
        return self._renderers[key]

    def render(
        self,
        camera: str = "onboard",
        depth: bool = False,
        size: tuple[int, int] | None = None,
        show_drone: bool | None = None,
        fog: bool | None = None,
    ) -> np.ndarray:
        """Render a camera. ``size`` is (height, width); defaults to the onboard camera size."""
        h, w = size or (self.cam_cfg.height, self.cam_cfg.width)
        r = self._renderer(h, w)
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = int(camera != "onboard") if show_drone is None else int(show_drone)
        r.update_scene(self.data, camera=camera, scene_option=opt)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = self.render_cfg.fog if fog is None else fog
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self.render_cfg.shadows
        if not depth:
            return r.render()
        r.enable_depth_rendering()
        try:
            return r.render().astype(np.float32)
        finally:
            r.disable_depth_rendering()

    def rgb(self) -> np.ndarray:
        """The onboard camera image: the agent's only sensor input."""
        return self.render("onboard")

    def depth_gt(self) -> np.ndarray:
        """Ground-truth planar depth (m) of the onboard camera. Evaluation only."""
        return self.render("onboard", depth=True)

    def T_world_cam(self) -> np.ndarray:
        """Ground-truth onboard camera pose (OpenCV camera frame). Evaluation only."""
        R = self.data.cam_xmat[self._cam_id].reshape(3, 3) @ geo.MJ_TO_CV
        return geo.make_T(R, self.data.cam_xpos[self._cam_id].copy())

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
