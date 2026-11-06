"""Check that the machine can run OrchardScout: Python packages, GPU, OpenGL rendering.

    python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    print(f"python {sys.version.split()[0]} on {platform.platform()}")
    for name in ("numpy", "scipy", "cv2", "mujoco", "matplotlib", "torch", "transformers"):
        try:
            mod = importlib.import_module(name)
            print(f"  {name:<13} {getattr(mod, '__version__', '?')}")
        except ImportError as e:
            print(f"  {name:<13} MISSING ({e})")

    import torch

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB, compute {p.major}.{p.minor}")
        if p.major < 7 or "RTX" not in p.name:
            print("  note: not an RTX card, so Isaac Sim will not run here; MuJoCo is fine")
    else:
        print("GPU: none visible to PyTorch (depth will run on the CPU, slowly)")

    free = shutil.disk_usage(Path.cwd()).free / 2**30
    print(f"free disk here: {free:.1f} GB" + ("  (low: runs write videos)" if free < 2 else ""))

    from orchardscout.orchard import generate_layout
    from orchardscout.sim import OrchardSim

    sim = OrchardSim(generate_layout(0, "small"))
    t = time.perf_counter()
    for _ in range(10):
        sim.rgb()
    print(f"MuJoCo offscreen render OK: {100 * (time.perf_counter() - t):.1f} ms per 640x480 frame")
    sim.close()


if __name__ == "__main__":
    main()
