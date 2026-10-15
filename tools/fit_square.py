"""Crop transparent renders to their alpha bounding box and fit them into an NxN transparent canvas.

    ./.venv/Scripts/python.exe tools/fit_square.py <in_dir> <out_dir> [--size 512] [--margin 0.04]

Input: PNGs from tools/render_bdsp_vs.mjs (<game>_<id>_f<frame>.png). Output: <id>.png, bottom-aligned, scaled to fill.
"""
import argparse
import glob
import os
import re

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--margin", type=float, default=0.04)
a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)
for f in sorted(glob.glob(os.path.join(a.src, "*.png"))):
    im = Image.open(f).convert("RGBA")
    bb = im.getbbox()
    if not bb:
        continue
    im = im.crop(bb)
    inner = round(a.size * (1 - 2 * a.margin))
    s = inner / max(im.size)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    out = Image.new("RGBA", (a.size, a.size), (0, 0, 0, 0))
    out.paste(im, ((a.size - im.width) // 2, a.size - im.height - round(a.size * a.margin)), im)
    name = re.sub(r"^[a-z]+_", "", os.path.basename(f))
    name = re.sub(r"_f[0-9.]+\.png$", ".png", name)
    out.save(os.path.join(a.dst, name))
