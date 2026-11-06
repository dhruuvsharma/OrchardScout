"""Procedural orchard: a randomised layout (the ground truth) and the MuJoCo geometry for it.

Each episode draws a new orchard: number of rows, row length, row spacing, tree spacing,
trunk and canopy sizes, missing trees and position jitter are all random. The agent is
never told any of these; the layout is used only to build the scene and to score the
agent afterwards.

Trees are modelled for under-canopy flight (nut / olive style): a bare trunk up to a
canopy base of ~1.6-2.3 m, then a crown made of a few overlapping ellipsoids. A strip of
bare soil runs under each row, as in a managed orchard.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import textures

SIZE_PRESETS: dict[str, dict[str, tuple]] = {
    "small": {"n_rows": (4, 6), "row_length": (20.0, 35.0)},
    "medium": {"n_rows": (8, 12), "row_length": (40.0, 70.0)},
    "large": {"n_rows": (14, 20), "row_length": (80.0, 120.0)},
    "any": {"n_rows": (4, 20), "row_length": (20.0, 120.0)},
}


@dataclass
class OrchardConfig:
    """Ranges that orchard parameters are drawn from (uniformly, per episode)."""

    n_rows: tuple[int, int] = (4, 20)
    row_length: tuple[float, float] = (20.0, 120.0)  # m, planted length of each row
    row_spacing: tuple[float, float] = (3.0, 6.0)  # m between row centre lines
    tree_spacing: tuple[float, float] = (1.5, 4.0)  # m between trees in a row
    trunk_radius: tuple[float, float] = (0.07, 0.15)  # m
    canopy_base: tuple[float, float] = (1.7, 2.3)  # m, height of the lowest leaves
    missing_prob: float = 0.05  # chance that a planting position is empty
    jitter: float = 0.12  # m, std of tree position noise
    field_margin: float = 300.0  # m of open field around the orchard

    @classmethod
    def preset(cls, size: str = "any", **overrides) -> OrchardConfig:
        if size not in SIZE_PRESETS:
            raise ValueError(f"unknown size {size!r}; choose from {sorted(SIZE_PRESETS)}")
        return cls(**{**SIZE_PRESETS[size], **overrides})


@dataclass
class OrchardLayout:
    """Ground-truth description of one generated orchard."""

    seed: int
    n_rows: int
    row_length: float
    row_spacing: float
    tree_spacing: float
    canopy_base: float
    field_margin: float
    row_y: np.ndarray  # (n_rows,) nominal y of each row centre line
    trunks: np.ndarray  # (N, 4): x, y, radius, top z
    canopies: np.ndarray  # (M, 7): tree index, cx, cy, cz, rx, ry, rz
    row_index: np.ndarray  # (N,) row of each tree
    config: dict = field(default_factory=dict)

    @property
    def n_trees(self) -> int:
        return len(self.trunks)

    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box of the trunk centres: (xmin, ymin, xmax, ymax)."""
        xy = self.trunks[:, :2]
        return (*xy.min(axis=0).tolist(), *xy.max(axis=0).tolist())

    def min_canopy_z(self) -> float:
        """Lowest point of any canopy ellipsoid (the ceiling for under-canopy flight)."""
        return float((self.canopies[:, 3] - self.canopies[:, 6]).min())

    def clearance(self, x: float, y: float) -> float:
        """Horizontal distance from (x, y) to the nearest trunk surface."""
        d = np.hypot(self.trunks[:, 0] - x, self.trunks[:, 1] - y) - self.trunks[:, 2]
        return float(d.min())

    def ground_truth(self) -> dict:
        """The numbers the agent has to recover, for evaluation only."""
        xmin, ymin, xmax, ymax = self.bbox()
        rows_present = np.unique(self.row_index)
        return {
            "n_rows": int(len(rows_present)),
            "n_trees": self.n_trees,
            "row_spacing_m": round(self.row_spacing, 3),
            "tree_spacing_m": round(self.tree_spacing, 3),
            "length_m": round(xmax - xmin, 3),  # along the rows
            "width_m": round(ymax - ymin, 3),  # across the rows
            "area_m2": round((xmax - xmin) * (ymax - ymin), 2),
            "bbox_xyxy": [round(v, 3) for v in (xmin, ymin, xmax, ymax)],
            "canopy_base_m": round(self.canopy_base, 3),
            "min_canopy_z_m": round(self.min_canopy_z(), 3),
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        d["ground_truth"] = self.ground_truth()
        return d

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load_json(cls, path: str | Path) -> OrchardLayout:
        d = json.loads(Path(path).read_text())
        d.pop("ground_truth", None)
        for k in ("row_y", "trunks", "canopies", "row_index"):
            d[k] = np.asarray(d[k])
        return cls(**d)


def generate_layout(
    seed: int, size: str = "any", config: OrchardConfig | None = None
) -> OrchardLayout:
    """Draw a random orchard. The same ``seed`` and config always give the same orchard."""
    cfg = config or OrchardConfig.preset(size)
    rng = np.random.default_rng(seed)

    n_rows = int(rng.integers(cfg.n_rows[0], cfg.n_rows[1] + 1))
    row_length = float(rng.uniform(*cfg.row_length))
    row_spacing = float(rng.uniform(*cfg.row_spacing))
    tree_spacing = float(rng.uniform(*cfg.tree_spacing))
    trunk_r_mean = float(rng.uniform(*cfg.trunk_radius))
    base_mean = float(rng.uniform(*cfg.canopy_base))
    crown_r_mean = float(np.clip(rng.uniform(0.32, 0.42) * row_spacing, 0.9, 2.2))

    n_per_row = int(np.floor(row_length / tree_spacing)) + 1
    row_y = (np.arange(n_rows) - (n_rows - 1) / 2.0) * row_spacing
    x0 = -(n_per_row - 1) * tree_spacing / 2.0

    trunks, canopies, row_index = [], [], []
    for i, y_row in enumerate(row_y):
        for j in range(n_per_row):
            if rng.random() < cfg.missing_prob:
                continue
            x = x0 + j * tree_spacing + rng.normal(0.0, cfg.jitter)
            y = y_row + rng.normal(0.0, cfg.jitter)
            radius = trunk_r_mean * rng.uniform(0.85, 1.15)
            base = max(base_mean + rng.uniform(-0.1, 0.1), cfg.canopy_base[0] - 0.1)
            rx = crown_r_mean * rng.uniform(0.85, 1.15)
            ry = rx * rng.uniform(0.9, 1.1)
            rz = rx * rng.uniform(0.7, 0.9)
            k = len(trunks)
            canopies.append([k, x, y, base + rz, rx, ry, rz])
            for _ in range(2):  # two smaller lobes make the crown look less like a ball
                a = rng.uniform(0.0, 2.0 * np.pi)
                d = rng.uniform(0.3, 0.6) * rx
                r2 = rx * rng.uniform(0.5, 0.75)
                rz2 = 0.8 * r2
                cz = base + rz2 + rng.uniform(0.2, 0.7) * rz
                canopies.append([k, x + d * np.cos(a), y + d * np.sin(a), cz, r2, r2, rz2])
            trunks.append([x, y, radius, base + rz])
            row_index.append(i)

    return OrchardLayout(
        seed=seed,
        n_rows=n_rows,
        row_length=row_length,
        row_spacing=row_spacing,
        tree_spacing=tree_spacing,
        canopy_base=base_mean,
        field_margin=cfg.field_margin,
        row_y=row_y,
        trunks=np.asarray(trunks, dtype=np.float64),
        canopies=np.asarray(canopies, dtype=np.float64),
        row_index=np.asarray(row_index, dtype=np.int64),
        config=asdict(cfg),
    )


def sample_start_pose(
    layout: OrchardLayout,
    rng: np.random.Generator,
    mode: str = "alley",
    clearance: float = 0.8,
) -> tuple[float, float, float]:
    """Random collision-free start (x, y, yaw).

    ``alley``: in the middle of a random alley between two rows.
    ``edge``: 2-5 m outside a random side of the orchard.
    The heading is uniform, so the agent cannot assume the rows are aligned with it.
    """
    xmin, ymin, xmax, ymax = layout.bbox()
    for _ in range(1000):
        if mode == "alley":
            i = int(rng.integers(0, layout.n_rows - 1))
            y = 0.5 * (layout.row_y[i] + layout.row_y[i + 1])
            x = rng.uniform(xmin + 1.0, xmax - 1.0)
        elif mode == "edge":
            off = rng.uniform(2.0, 5.0)
            side = int(rng.integers(4))
            if side == 0:
                x, y = xmin - off, rng.uniform(ymin, ymax)
            elif side == 1:
                x, y = xmax + off, rng.uniform(ymin, ymax)
            elif side == 2:
                x, y = rng.uniform(xmin, xmax), ymin - off
            else:
                x, y = rng.uniform(xmin, xmax), ymax + off
        else:
            raise ValueError(f"unknown start mode {mode!r}")
        if layout.clearance(x, y) >= clearance:
            return float(x), float(y), float(rng.uniform(-np.pi, np.pi))
    raise RuntimeError("could not find a collision-free start pose")


def _f(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def world_elements(layout: OrchardLayout) -> tuple[str, str, dict[str, bytes]]:
    """MJCF ``<asset>`` block, worldbody geoms and texture files for an orchard."""
    assets = textures.make_textures(layout.seed)
    asset_xml = """
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.40 0.58 0.84"
             rgb2="0.86 0.90 0.96" width="512" height="3072"/>
    <texture name="grass" type="2d" file="grass.png"/>
    <texture name="soil" type="2d" file="soil.png"/>
    <texture name="bark" type="cube" file="bark.png"/>
    <texture name="leaves" type="cube" file="leaves.png"/>
    <material name="grass" texture="grass" texrepeat="0.25 0.25" texuniform="true"
              specular="0" shininess="0"/>
    <material name="soil" texture="soil" texrepeat="0.5 0.5" texuniform="true"
              specular="0" shininess="0"/>
    <material name="bark" texture="bark" texrepeat="2 2" texuniform="true"
              specular="0.05" shininess="0.1"/>
    <material name="leaves" texture="leaves" texrepeat="1 1" texuniform="true"
              specular="0.1" shininess="0.2"/>
  </asset>"""

    rng = np.random.default_rng(layout.seed + 7919)
    xmin, ymin, xmax, ymax = layout.bbox()
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    gx = 0.5 * (xmax - xmin) + layout.field_margin
    gy = 0.5 * (ymax - ymin) + layout.field_margin

    lines = [
        f'    <geom name="ground" type="plane" size="{_f(gx)} {_f(gy)} 10" '
        f'pos="{_f(cx)} {_f(cy)} 0" material="grass"/>'
    ]
    strip_len = 0.5 * (xmax - xmin) + 1.0
    for i, y in enumerate(layout.row_y):
        half_w = rng.uniform(0.4, 0.7)
        lines.append(
            f'    <geom name="strip{i}" type="plane" size="{_f(strip_len)} {_f(half_w)} 2" '
            f'pos="{_f(cx)} {_f(y)} 0.01" material="soil"/>'
        )
    for k, (x, y, r, top) in enumerate(layout.trunks):
        t = rng.uniform(0.8, 1.0)
        lines.append(
            f'    <geom name="trunk{k}" type="cylinder" size="{_f(r)} {_f(top / 2)}" '
            f'pos="{_f(x)} {_f(y)} {_f(top / 2)}" material="bark" '
            f'rgba="{_f(t)} {_f(t * rng.uniform(0.95, 1.0))} {_f(t * rng.uniform(0.9, 1.0))} 1"/>'
        )
    for k, (_, x, y, z, rx, ry, rz) in enumerate(layout.canopies):
        tint = (rng.uniform(0.8, 1.0), rng.uniform(0.85, 1.0), rng.uniform(0.75, 0.95))
        lines.append(
            f'    <geom name="crown{k}" type="ellipsoid" size="{_f(rx)} {_f(ry)} {_f(rz)}" '
            f'pos="{_f(x)} {_f(y)} {_f(z)}" material="leaves" '
            f'rgba="{_f(tint[0])} {_f(tint[1])} {_f(tint[2])} 1"/>'
        )
    return asset_xml, "\n".join(lines), assets
