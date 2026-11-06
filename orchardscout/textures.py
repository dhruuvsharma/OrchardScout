"""Procedural, tileable textures generated with numpy.

The scene needs no downloaded assets. Every texture is built from spectrally filtered
noise, so it is reproducible from a seed, tiles without seams and is safe to publish.
The textures also matter for perception: visual odometry needs surface detail to track,
and the depth network behaves better on natural-looking surfaces than on flat colours.
"""

from __future__ import annotations

import cv2
import numpy as np


def tileable_noise(
    size: int,
    rng: np.random.Generator,
    beta: float,
    stretch: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Zero-mean, unit-std noise with a 1/f**beta power spectrum.

    The noise is made by filtering white noise in the Fourier domain, so it is periodic
    and tiles seamlessly. ``stretch=(sy, sx)`` greater than 1 along an axis smooths the
    noise along that axis; ``(8, 1)`` gives vertical streaks, like bark.
    """
    white = rng.standard_normal((size, size))
    fy = np.fft.fftfreq(size)[:, None] * stretch[0]
    fx = np.fft.fftfreq(size)[None, :] * stretch[1]
    f = np.hypot(fx, fy)
    f[0, 0] = np.inf  # removes the DC term
    noise = np.fft.ifft2(np.fft.fft2(white) * f ** (-beta / 2.0)).real
    noise -= noise.mean()
    return noise / (noise.std() + 1e-9)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _palette(t: np.ndarray, colors: list[tuple[float, float, float]]) -> np.ndarray:
    """Map values in [0, 1] through a piecewise-linear colour ramp (RGB in [0, 1])."""
    ramp = np.asarray(colors, dtype=np.float64)
    pos = np.clip(t, 0.0, 1.0) * (len(ramp) - 1)
    i0 = np.minimum(pos.astype(int), len(ramp) - 2)
    w = (pos - i0)[..., None]
    return ramp[i0] * (1.0 - w) + ramp[i0 + 1] * w


def _modulate(rgb: np.ndarray, noise: np.ndarray, amount: float) -> np.ndarray:
    return rgb * (1.0 + amount * np.tanh(noise))[..., None]


def grass(rng: np.random.Generator, size: int = 512) -> np.ndarray:
    base = 0.75 * tileable_noise(size, rng, 2.4) + 0.45 * tileable_noise(size, rng, 1.2)
    rgb = _palette(
        _sigmoid(1.3 * base),
        [(0.15, 0.25, 0.08), (0.25, 0.40, 0.13), (0.37, 0.50, 0.19), (0.52, 0.52, 0.29)],
    )
    return _modulate(rgb, tileable_noise(size, rng, 0.4, stretch=(2.5, 1.0)), 0.22)


def soil(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    base = 0.8 * tileable_noise(size, rng, 2.2) + 0.4 * tileable_noise(size, rng, 1.0)
    rgb = _palette(
        _sigmoid(1.2 * base), [(0.22, 0.16, 0.11), (0.34, 0.26, 0.18), (0.47, 0.38, 0.27)]
    )
    pebbles = tileable_noise(size, rng, 0.2)
    rgb = _modulate(rgb, pebbles, 0.18)
    rgb[pebbles > 2.6] = (0.62, 0.58, 0.52)
    return rgb


def bark(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    grooves = tileable_noise(size, rng, 1.6, stretch=(8.0, 1.0))
    rgb = _palette(
        _sigmoid(1.6 * grooves), [(0.14, 0.11, 0.08), (0.29, 0.22, 0.16), (0.43, 0.36, 0.28)]
    )
    return _modulate(rgb, tileable_noise(size, rng, 0.6, stretch=(3.0, 1.0)), 0.25)


def leaves(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    base = tileable_noise(size, rng, 1.6)
    rgb = _palette(
        _sigmoid(1.5 * base),
        [(0.06, 0.15, 0.04), (0.13, 0.30, 0.08), (0.26, 0.46, 0.14), (0.42, 0.58, 0.22)],
    )
    clusters = tileable_noise(size, rng, 0.9)
    rgb *= (0.45 + 0.55 * _sigmoid(4.0 * (clusters + 0.9)))[..., None]  # dark gaps
    return _modulate(rgb, tileable_noise(size, rng, 0.3), 0.15)


def to_png(rgb: np.ndarray) -> bytes:
    """Encode an RGB float image in [0, 1] as PNG bytes."""
    img = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def make_textures(seed: int) -> dict[str, bytes]:
    """All scene textures as ``{filename: png_bytes}``, ready to pass to MuJoCo as assets."""
    rng = np.random.default_rng(seed)
    return {
        "grass.png": to_png(grass(rng)),
        "soil.png": to_png(soil(rng)),
        "bark.png": to_png(bark(rng)),
        "leaves.png": to_png(leaves(rng)),
    }
