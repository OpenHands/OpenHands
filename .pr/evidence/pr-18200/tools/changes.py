"""List the video times where a clip's picture changes (and roughly where)."""

import subprocess
import sys

import numpy as np


def frames(path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    w, h = int(probe[0]), int(probe[1])
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w), w, h


def changes(path, fps=10, threshold=1.5):
    f, w, h = frames(path)
    out = []
    for i in range(1, len(f)):
        d = np.abs(f[i].astype(np.int16) - f[i - 1].astype(np.int16))
        if d.mean() > 0.01 or d.max() > 40:
            ys, xs = np.nonzero(d > 24)
            region = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(xs) else None
            out.append({"t": i / fps, "mean": float(d.mean()), "region": region})
    return out


if __name__ == "__main__":
    for c in changes(sys.argv[1]):
        print(c)
