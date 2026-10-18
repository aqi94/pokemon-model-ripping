"""Post-process the BDSP battle idle renders: Doubles-exclusive trainers ship as two models (<id>_00 / <id>_01) and become one image.

    ./.venv/Scripts/python.exe tools/combine_doubles.py [--dir out/bdsp_battle_idle] [--gap 144]

Both renders already share one scale and ground line, so the pair is only shifted horizontally: _00 on the left, _01 on the right, a
gap between the two silhouettes like tr1020 (kid + parent in one model: ~145 px, group centred), the gap shrunk if the pair would not
fit in the 1024 frame (tr2003's wide poses). The single renders move to <dir>/singles/ (rerunning reads them from there), the combined
image is <id>_00_01.png.
"""
import argparse
import os
import re
import shutil
import time

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join("out", "bdsp_battle_idle"))
ap.add_argument("--gap", type=int, default=144)
ap.add_argument("--margin", type=int, default=24)
args = ap.parse_args()
SWAP = {"tr1008"}  # _01 on the left: the Twins read backwards the other way round
single = os.path.join(args.dir, "singles")
os.makedirs(single, exist_ok=True)


def src(name):
    for d in (args.dir, single):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


stems = sorted({m.group(1) for f in os.listdir(args.dir) + os.listdir(single) if (m := re.match(r"^(tr\d{4})_01\.png$", f))})
for stem in stems:
    pa, pb = src(f"{stem}_00.png"), src(f"{stem}_01.png")
    if not (pa and pb):
        continue
    ims = [Image.open(p).convert("RGBA") for p in ((pb, pa) if stem in SWAP else (pa, pb))]
    box = [np.where((np.asarray(im)[..., 3] > 8).any(0))[0] for im in ims]
    x0, x1 = [int(b[0]) for b in box], [int(b[-1]) + 1 for b in box]
    w = [x1[i] - x0[i] for i in range(2)]
    size = ims[0].size[0]
    gap = max(0, min(args.gap, size - 2 * args.margin - sum(w)))
    left = (size - (sum(w) + gap)) // 2
    pos = [left, left + w[0] + gap]
    out = Image.new("RGBA", ims[0].size, (0, 0, 0, 0))
    for i in range(2):
        out.alpha_composite(ims[i], (pos[i] - x0[i], 0))
    dst = os.path.join(args.dir, f"{stem}_00_01.png")
    for attempt in range(5):  # a viewer/preview holding the PNG open makes Windows refuse the write
        try:
            out.save(dst)
            break
        except OSError:
            time.sleep(1)
    else:
        print(stem, "WRITE FAILED (file locked? close any preview of", dst + ")")
        continue
    for p in (pa, pb):
        if os.path.dirname(p) == args.dir:
            os.replace(p, os.path.join(single, os.path.basename(p)))
    print(stem, "widths", w, "gap", gap)
