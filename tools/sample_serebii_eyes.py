"""Sample the player's eye-contact colours from Serebii's Legends Z-A customisation thumbnails.

The rip has no eye-colour table (the player's eye materials ship one fixed brown and the game recolours them from
parameter data that is not in the zips), but https://www.serebii.net/legendsz-a/looks.shtml shows all 32 contacts on
the same face render. The irises are the only pixels that differ between the thumbnails, so the iris mask is the
per-pixel std across them; per thumbnail the iris colour is the median of the 20-50% luminance pixels (pupil
and glint trimmed) and the lighter inner iris the median of the 50-85% ones. Renders are ~0.92x the palette value (checked on
the lip thumbnails against the true `p0_face_pallete01` cells), hence GAIN. Thumbnails 27-32 are patterned (void,
starry, lovestruck, fired up, cyber, Poke Ball), not a colour: skipped.

    ./.venv/Scripts/python.exe -m tools.sample_serebii_eyes [DIR]   # downloads cc1..32.jpg into DIR, writes player_eye_samples.json
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.request

import numpy as np
from PIL import Image

URL = "https://www.serebii.net/legendsz-a/custom/look/th/colorcontacts/{}.jpg"
OUT = os.path.join(os.path.dirname(__file__), "player_eye_samples.json")
GAIN = 1.08
CHROMA = 1.5  # the median blends the iris with lashes, sclera and the eyelid shadow: pull it back out from grey
# Thumbnail n -> colour word (read off the thumbnails; Serebii's alt text order does not match the file numbers).
WORDS = ["black", "slate", "brown", "light blue", "pitch black", "dark red", "navy", "olive grey", "violet", "poppy red",
         "marine", "grey", "rose", "sapphire", "lime", "lavender", "amber", "turquoise", "silver", "pale pink", "blue",
         "emerald", "pale lavender", "gold", "aqua", "khaki"]


def main() -> None:
    d = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    os.makedirs(d, exist_ok=True)
    ims = []
    for i in range(1, 33):
        p = os.path.join(d, f"cc{i}.jpg")
        if not os.path.exists(p):
            req = urllib.request.Request(URL.format(i), headers={"User-Agent": "Mozilla/5.0"})
            with open(p, "wb") as f:
                f.write(urllib.request.urlopen(req).read())
        ims.append(np.asarray(Image.open(p).convert("RGB")).astype(np.float32))
    cc = np.stack(ims)
    mask = cc.std(0).sum(-1) > 40
    mask[:100] = False
    mask[140:] = False
    out = []
    for i, word in enumerate(WORDS):
        px = cc[i][mask]
        o = np.argsort(px.mean(1))
        n = len(o)
        def pick(a: float, b: float) -> np.ndarray:
            c = np.median(px[o[int(n * a):int(n * b)]], 0) * GAIN
            lum = float(c @ np.asarray([0.299, 0.587, 0.114]))
            return np.clip(lum + (c - lum) * CHROMA, 0, 255)

        iris, inner = pick(0.2, 0.5), pick(0.5, 0.85)
        out.append({"label": f"Contact {i + 1:02d} \u00b7 {word}", "iris": [int(round(v)) for v in iris], "inner": [int(round(v)) for v in inner]})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"{len(out)} contact colours -> {OUT}")


if __name__ == "__main__":
    main()
