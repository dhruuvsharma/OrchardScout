"""Assemble the demo video from raw exploration recordings (run_explore.py --video).

Adds a title card, a caption per segment and an end card with results, speeds each
segment up, and encodes H.264 with ffmpeg.

    python scripts/make_video.py \
        --segment runs/explore/explore_small_..._raw.mp4 4 "Small orchard" \
        --segment runs/explore/explore_medium_..._raw.mp4 8 "Medium orchard" \
        --results "small: 6/6 rows ..." --out docs/media/orchardscout_demo.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1280, 720, 30


def card(lines: list[tuple[str, float, tuple[int, int, int]]], seconds: float) -> list[np.ndarray]:
    img = np.full((H, W, 3), (28, 30, 28), np.uint8)
    y = 200
    for text, scale, color in lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.putText(img, text, ((W - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2,
                    cv2.LINE_AA)
        y += int(th * 2.2) + 6
    return [img] * int(seconds * FPS)


def segment(path: Path, speed: float, caption: str) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    step = speed * src_fps / FPS  # source frames per output frame
    frames, k, t = [], 0, 0.0
    ok, f = cap.read()
    while ok:
        while t <= k:
            g = cv2.resize(f, (W, H))
            cv2.rectangle(g, (480, H - 58), (W, H - 30), (25, 25, 25), -1)
            cv2.putText(g, caption, (490, H - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 120), 1, cv2.LINE_AA)
            frames.append(g)
            t += step
        ok, f = cap.read()
        k += 1
    cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--segment", nargs=3, action="append", metavar=("MP4", "SPEED", "CAPTION"),
                    required=True, help="SPEED: playback speed relative to the raw recording")
    ap.add_argument("--results", nargs="*", default=[], help="lines for the end card")
    ap.add_argument("--out", default="docs/media/orchardscout_demo.mp4")
    args = ap.parse_args()

    white, grey, gold = (240, 240, 240), (170, 170, 170), (120, 220, 255)
    frames = card([
        ("OrchardScout", 1.6, white),
        ("camera-only exploration of an orchard of unknown size", 0.9, white),
        ("input: one RGB camera (+ known 1.0 m flight height)", 0.7, grey),
        ("monocular metric depth  -  visual odometry  -  trunk mapping", 0.7, grey),
        ("drift correction  -  frontier exploration  -  size estimation", 0.7, grey),
        ("simulated orchards (MuJoCo), new random layout every run", 0.7, gold),
    ], 4.0)
    for mp4, speed, caption in args.segment:
        frames += card([(caption, 1.1, white)], 1.5)
        frames += segment(Path(mp4), float(speed), caption)
    if args.results:
        frames += card([("Results", 1.2, white)] + [(r, 0.65, grey) for r in args.results], 6.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in frames:
            wr.write(f)
        wr.release()
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264",
                            "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(out)], check=True)
        else:
            shutil.copy(raw, out)
    print(f"{len(frames) / FPS:.0f} s video -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
