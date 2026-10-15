"""Labelled contact sheet of render PNGs, with a transparent background like every PNG export.

    ./.venv/Scripts/python.exe tools/contact_sheet.py <dir> <suffix> <out.png> [--cols 6]

Takes <dir>/za_<id>_<suffix>.png (tools/render_models.mjs output); labels come from out/index.json.
"""
import argparse
import glob
import json
import os

from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("suffix")
ap.add_argument("out")
ap.add_argument("--cols", type=int, default=6)
a = ap.parse_args()
names = {m["id"]: m["displayName"] for m in json.load(open("out/index.json"))["models"]}
files = sorted(glob.glob(os.path.join(a.dir, f"za_*_{a.suffix}.png")))
cw, ch, lh = 270, 360, 26
rows = (len(files) + a.cols - 1) // a.cols
sheet = Image.new("RGBA", (cw * a.cols, (ch + lh) * rows), (0, 0, 0, 0))
d = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("arial.ttf", 16)
except OSError:
    font = ImageFont.load_default()
for k, f in enumerate(files):
    mid = os.path.basename(f)[3:-len(a.suffix) - 5]
    x, y = (k % a.cols) * cw, (k // a.cols) * (ch + lh)
    sheet.alpha_composite(Image.open(f).convert("RGBA").resize((cw, ch)), (x, y + lh))
    # light text with a dark outline reads on any background the sheet is placed on
    d.text((x + 6, y + 4), f"{mid}  {names.get(mid, '')}", fill="#f0f0f0", font=font, stroke_width=2, stroke_fill="#000000")
sheet.save(a.out)
print(a.out, sheet.size, len(files), "renders")
