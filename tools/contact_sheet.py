"""Labelled contact sheet of render PNGs, with a transparent background like every PNG export.

    ./.venv/Scripts/python.exe tools/contact_sheet.py <dir> <suffix> <out.png> [--cols 6] [--cell 270x360]
        [--pad 0] [--gap 0] [--bg transparent|#rrggbb] [--fit contain|stretch] [--no-label]

Takes <dir>/<game>_<id>_<suffix>.png (tools/render_models.mjs output); labels come from out/index.json.
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
ap.add_argument("--game", default="za")
ap.add_argument("--cell", default="270x360", help="image size per cell WxH")
ap.add_argument("--pad", type=int, default=0, help="padding inside each cell around the image")
ap.add_argument("--gap", type=int, default=0, help="space between cells and around the sheet")
ap.add_argument("--bg", default="transparent")
ap.add_argument("--fit", choices=["contain", "stretch"], default="contain", help="contain keeps the aspect, bottom-aligned")
ap.add_argument("--no-label", action="store_true")
ap.add_argument("--per-page", type=int, default=0, help="split into <out>_01.png, _02.png ... of this many cells")
a = ap.parse_args()
names = {(m["game"], m["id"]): m["displayName"] for m in json.load(open("out/index.json"))["models"]}
all_files = sorted(glob.glob(os.path.join(a.dir, f"{a.game}_*_{a.suffix}.png")))
cw, ch = (int(v) for v in a.cell.split("x"))


def build(files, out):
    lh = 0 if a.no_label else 26
    rows = (len(files) + a.cols - 1) // a.cols
    stepx, stepy = cw + a.gap, ch + lh + a.gap
    bg = (0, 0, 0, 0) if a.bg == "transparent" else a.bg
    sheet = Image.new("RGBA", (stepx * a.cols + a.gap, stepy * rows + a.gap), bg)
    d = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for k, f in enumerate(files):
        mid = os.path.basename(f)[len(a.game) + 1:-len(a.suffix) - 5]
        x, y = a.gap + (k % a.cols) * stepx, a.gap + (k // a.cols) * stepy
        im = Image.open(f).convert("RGBA")
        iw, ih = max(1, cw - 2 * a.pad), max(1, ch - 2 * a.pad)
        if a.fit == "stretch":
            im = im.resize((iw, ih), Image.LANCZOS)
            ox, oy = a.pad, a.pad
        else:
            s = min(iw / im.width, ih / im.height)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
            ox, oy = a.pad + (iw - im.width) // 2, a.pad + ih - im.height
        sheet.alpha_composite(im, (x + ox, y + lh + oy))
        if a.no_label:
            continue
        # light text with a dark outline reads on any background the sheet is placed on
        d.text((x + 6, y + 4), f"{mid}  {names.get((a.game, mid), '')}", fill="#f0f0f0", font=font, stroke_width=2, stroke_fill="#000000")
    sheet.save(out)
    print(out, sheet.size, len(files), "renders")


n = a.per_page or len(all_files) or 1
pages = [all_files[i:i + n] for i in range(0, len(all_files), n)] or [[]]
for i, page in enumerate(pages):
    build(page, a.out if len(pages) == 1 else f"{os.path.splitext(a.out)[0]}_{i + 1:02d}.png")
