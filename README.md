# OrchardScout

**Camera-only drone exploration of an orchard whose size is unknown.**
A small multirotor flies under the canopy of a procedurally generated orchard with one RGB
camera. From the camera alone it estimates depth (monocular metric depth), tracks its own
motion (visual odometry), builds a map, explores until it has found the orchard's edges on
every side, and reports how big the orchard is: rows, row length, row spacing and overall
extent.

> Status: **work in progress** (proof of concept, November 2026). The simulator, the
> orchard generator and the monocular depth stage with metric-scale recovery are working
> and tested. Visual odometry, mapping, exploration and size estimation are next; see the
> [roadmap](#roadmap).

![Onboard camera, ground-truth depth and chase view in two generated orchards](docs/images/scene_views.jpg)
*Left to right: onboard RGB (the only thing the agent sees), simulator depth (evaluation
only), chase camera. Top: inside an alley. Bottom: at the orchard edge.*

## The task

| Requirement | How it is handled |
|---|---|
| Simulated orchard, different every run | Procedural generator: rows, row length, spacings, tree sizes, missing trees and position jitter are random per episode |
| Camera only at run time | One forward RGB camera (640x480, ~90 deg horizontal FOV). No LiDAR, depth sensor, GPS or ground-truth pose |
| Monocular depth | Depth Anything V2 (metric, outdoor, Small), rescaled online from the ground plane (below) |
| Visual odometry | ORB features + PnP on predicted depth (in progress) |
| Exploration + mapping | 2-D occupancy grid + frontier exploration (planned) |
| Unknown extent | The agent is never told the orchard size; it detects trunks, fits rows and estimates the extent while flying (planned) |

Simulator ground truth (depth, pose, layout) is used **only for evaluation**, never as an
input to the agent.

**One stated prior:** the drone flies at a fixed, known height (1.0 m) held by its flight
controller, as a real multirotor does with a barometer or downward range sensor. The agent
uses that number, and nothing else, to fix the metric scale of monocular depth, the same
way monocular visual odometry on cars uses the known camera height above the road.

## Pipeline

```
RGB frame ─► Depth Anything V2 (metric) ─► ground-plane rescale ─► metric depth ─┐
     │                                                                            ├─► occupancy grid
     └─► ORB features ─► PnP with metric depth ─► camera pose (VO) ───────────────┘
occupancy grid ─► frontier planner ─► waypoint ─► velocity controller ─► drone
occupancy grid ─► trunk detection ─► row fit ─► orchard size estimate (updated while flying)
simulator ground truth ─► evaluation only
```

## Simulator

The simulator is [MuJoCo](https://mujoco.org). The original plan was Isaac Lab on Isaac
Sim, but Isaac Sim needs an RTX GPU, and the development machine has a GTX 1650 (4 GB).
The agent only ever receives RGB frames, so the perception and exploration code does not
depend on the simulator and can be moved to Isaac Lab or PX4 on an RTX machine later.

* **Orchard** (`orchardscout/orchard.py`): rows along one axis, trunks as cylinders,
  crowns as three overlapping ellipsoids, a bare-soil strip under each row and open grass
  around the orchard. All textures are generated procedurally from noise
  (`orchardscout/textures.py`), so there are no downloaded assets.
* **Drone** (`orchardscout/sim.py`): a velocity-commanded, acceleration-limited body with
  yaw, at a fixed altitude below the canopy. Roll and pitch follow the commanded
  acceleration (small-angle multirotor model) plus noise. Collisions are checked exactly
  against trunk cylinders and crown ellipsoids.
* **Rendering**: offscreen OpenGL, with sun shadows, a sky gradient and distance fog.
  About 17 ms for a 640x480 RGB frame on a laptop.

| Parameter (drawn per episode) | Range |
|---|---|
| Rows | 4–20 (presets: small 4–6, medium 8–12, large 14–20) |
| Row length | 20–120 m (small 20–35, medium 40–70, large 80–120) |
| Row spacing | 3–6 m |
| Tree spacing in a row | 1.5–4 m |
| Trunk radius | 7–15 cm |
| Canopy base height | 1.7–2.3 m (flight height 1.0 m) |
| Missing trees | 5 % |
| Position jitter | 12 cm (std) |
| Start pose | random alley, or 2–5 m outside a random edge; heading uniform |

![Top-down view of a generated medium orchard](docs/images/overview.jpg)

## First result: monocular depth in the orchard

Depth Anything V2 (metric outdoor) recovers the scene's *structure* well, but its
*scale* is far off: it puts everything about 5x too far away, because it was trained on
car-height driving images. Rescaling every frame from the ground plane (RANSAC plane fit
on the bottom of the image, scaled to the known 1.0 m flight height) fixes this. Per frame,
the estimated scale (0.199–0.215) is within 4.6 % of the true scale on average (worst
frame 8.8 %).

16 frames from a medium orchard, some taken while the drone accelerates and tilts
(`python scripts/eval_depth.py --size medium --seed 1 --frames 16`):

| Distance (m) | AbsRel, raw network | AbsRel, after ground-plane rescale | GT / prediction after |
|---|---|---|---|
| 0–2 | 4.10 | **0.076** | 0.95 |
| 2–5 | 4.03 | **0.096** | 0.95 |
| 5–10 | 3.27 | **0.153** | 1.11 |
| 10–20 | 2.29 | 0.317 | 1.45 |
| 20–40 | 1.19 | 0.543 | 2.24 |
| all (< 30 m) | 3.75 | **0.122** | 0.97 |

Depth is reliable to about 10 m, so the map uses only that range. Beyond it, distant trees
are placed too close. The network also gives the sky a finite depth, so sky pixels must be
masked before mapping. Speed: 135 ms per frame (fp32, input 518 px) on the GTX 1650;
fp16 is 3–4x slower on this card, and input 392 px takes 71 ms.

![RGB, ground-truth depth, rescaled prediction and relative error](docs/images/depth_eval.jpg)
*RGB | ground-truth depth | prediction after rescaling | relative error (blue 0 %, red
≥ 50 %; sky in black is excluded).*

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or the CPU build
pip install -r requirements.txt

python scripts/check_env.py                                   # versions, GPU, render test
python scripts/render_orchard.py --size medium --seed 3       # preview images -> runs/preview
python scripts/eval_depth.py --size medium --seed 1           # depth vs ground truth -> runs/depth_eval
pytest                                                        # 19 tests
```

On a headless Linux machine, set `MUJOCO_GL=egl` (NVIDIA) or `MUJOCO_GL=osmesa`.

## Repository layout

```
orchardscout/
  orchard.py    procedural orchard layout (ground truth) + MuJoCo geometry
  textures.py   procedural tileable textures (grass, soil, bark, leaves)
  sim.py        MuJoCo scene, drone flight model, cameras, collisions
  geometry.py   camera intrinsics, transforms, back-projection
  depth.py      monocular metric depth network wrapper
  scale.py      metric scale from the ground plane and the known flight height
  metrics.py    depth errors, trajectory error (ATE with SE(3)/Sim(3) alignment)
  viz.py        depth colouring, layout plots
scripts/        check_env, render_orchard, eval_depth
tests/          layout, geometry, simulator and scale tests
docs/images/    figures used in this README
```

## Roadmap

- [x] Procedural orchard with randomised, unknown size; ground truth for evaluation
- [x] MuJoCo drone simulator with onboard RGB camera, collisions and ground-truth depth/pose
- [x] Monocular metric depth + online ground-plane scale recovery, evaluated against ground truth
- [ ] Visual odometry: ORB + PnP on metric depth, keyframes; ATE and drift vs ground truth
- [ ] Sky mask and range-limited depth-to-occupancy mapping (2-D grid, plus a 3-D point cloud for figures)
- [ ] Frontier exploration with an A* planner, limited to frontiers near detected trees, with a stop rule
- [ ] Trunk detection, row fitting and orchard size estimation over time
- [ ] Results on small / medium / large orchards: size error, coverage, ATE, trunk precision/recall, collisions, FPS
- [ ] Video and short write-up
- [ ] Extras: robustness to lighting and fog, learned exploration policy, loop closure

## Limitations (so far)

* The flight model is kinematic with tilt from acceleration, not a full rotor-dynamics
  model.
* The scene is rendered with OpenGL rasterisation, so it looks synthetic. The depth
  network still transfers, but numbers on real orchard footage would differ.
* Metric scale relies on flat ground near the drone and a known flight height.

## License

MIT
