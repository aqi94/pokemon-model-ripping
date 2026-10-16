"""Grid of renders: one row per lighting preset, one column per model, labelled, on a neutral grey.

    ./.venv/Scripts/python.exe tools/lighting_sheet.py out/sv_lights out/sv_lighting_sheet.png [--suffix _full] [--width 260]

Expects <dir>/<preset>/<game>_<id><suffix>.png as written by tools/render_models.mjs (one folder per preset).
"""

from __future__ import annotations

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("out")
ap.add_argument("--suffix", default="_full")
ap.add_argument("--width", type=int, default=260)
ap.add_argument("--bg", default="#404040")
args = ap.parse_args()

presets = sorted(d for d in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, d)))
presets.sort(key=lambda d: (d != "studio", d))
models: list[str] = []
for pr in presets:
    for f in sorted(os.listdir(os.path.join(args.dir, pr))):
        if f.endswith(args.suffix + ".png"):
            m = f[: -len(args.suffix) - 4]
            if m not in models:
                models.append(m)

W = args.width
cell_h = None
tiles: dict[tuple[str, str], Image.Image] = {}
for pr in presets:
    for m in models:
        p = os.path.join(args.dir, pr, m + args.suffix + ".png")
        if os.path.isfile(p):
            im = Image.open(p).convert("RGBA")
            s = W / im.width
            im = im.resize((W, round(im.height * s)), Image.LANCZOS)
            tiles[(pr, m)] = im
            cell_h = max(cell_h or 0, im.height)
cell_h = cell_h or W
label_w, head_h = 190, 24
try:
    font = ImageFont.truetype("arial.ttf", 14)
except OSError:
    font = ImageFont.load_default()
sheet = Image.new("RGB", (label_w + W * len(models), head_h + cell_h * len(presets)), args.bg)
draw = ImageDraw.Draw(sheet)
for j, m in enumerate(models):
    draw.text((label_w + j * W + 4, 4), m, fill="white", font=font)
for i, pr in enumerate(presets):
    y = head_h + i * cell_h
    draw.text((4, y + 4), pr, fill="white", font=font)
    for j, m in enumerate(models):
        im = tiles.get((pr, m))
        if im is None:
            continue
        sheet.alpha_composite(im, (label_w + j * W, y)) if sheet.mode == "RGBA" else sheet.paste(im, (label_w + j * W, y), im)
sheet.save(args.out)
print("wrote", args.out, sheet.size)
