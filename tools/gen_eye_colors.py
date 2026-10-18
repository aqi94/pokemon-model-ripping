"""Write the player-customisation palettes the viewer offers as "custom" colours on NPC parts that ship none
(pipeline/importers/za/{eye,eyebrow,lip}_colors.json).

The player's colour choices are the one place the game has a master list (https://www.serebii.net/legendsz-a/looks.shtml):
- eyebrows / eyelashes / hair: the 32 zones of `p0_hairstyle_pallete01_90_alb` (ZA-TrainersSharedTex);
- lips: zones 5-13 of `p0_face_pallete01_90_alb` (beige, pink, coral, orange, brown, rouge, blue, green, black; "No color"
  is the lip layer off); zones 1-4 are the skin tones, 14-16 other layers;
- eye contacts: NOT in the rip (the player's eye materials ship a fixed brown, the game recolours them from parameter data),
  so they come from `tools/player_eye_samples.json`, sampled from the Serebii thumbnails by `tools/sample_serebii_eyes.py`.

    ./.venv/Scripts/python.exe -m tools.gen_eye_colors [--data DIR]
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os

from pipeline.catalog.za import ZaCatalog
from pipeline.importers.bntx import parse_bntx
from pipeline.importers.za.importer import table_shadow_rgb

ZA = os.path.join(os.path.dirname(__file__), "..", "pipeline", "importers", "za")
SAMPLES = os.path.join(os.path.dirname(__file__), "player_eye_samples.json")
LIP_NAMES = ["Beige", "Pink", "Coral", "Orange", "Brown", "Rouge", "Blue", "Green", "Black"]  # face palette zones 5..13
HUE_WORDS = [(15, "red"), (40, "orange"), (65, "yellow"), (95, "lime"), (165, "green"), (200, "teal"), (255, "blue"), (290, "purple"), (335, "pink"), (361, "red")]


def colour_word(rgb) -> str:
    h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
    if v < 0.2:
        return "black"
    if s < 0.12:
        return "white" if v > 0.85 else "light grey" if v > 0.62 else "grey" if v > 0.35 else "charcoal"
    word = next(w for hi, w in HUE_WORDS if h * 360 < hi)
    if word in ("red", "orange", "yellow") and v < 0.7 and s < 0.55:
        word = "brown"
    return ("pale " if s < 0.35 and v > 0.7 else "dark " if v < 0.45 else "") + word


def palette(cat: ZaCatalog, name: str) -> list[tuple[list[int], list[int]]]:
    """(lit, shadow) of every zone of a shared player palette."""
    for zp in cat.zips(cat.TRAINER_SHARED_TEX_ZIPS):
        p = f"ik_chara/share/pallete/{name}/{name}.bntx"
        if p in zp.namelist():
            t = parse_bntx(zp.read(p))[0].rgba
            th, tw = t.shape[:2]
            zones = tw // 2
            out = []
            for z in range(zones):
                x = min((2 * z + 1) * tw // (2 * zones), tw - 1)
                lit = [int(v) for v in t[th // 4, x, :3]]
                out.append((lit, table_shadow_rgb(lit, [int(v) for v in t[3 * th // 4, x, :3]])))
            return out
    raise SystemExit(f"{name} not found")


def dump(name: str, data: list) -> None:
    path = os.path.normpath(os.path.join(ZA, name))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print(f"{len(data)} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("POKEMODELS_DATA") or os.path.expanduser("~/Documents/MEGA downloads/PokéModels"))
    cat = ZaCatalog(ap.parse_args().data)

    brows = [{"label": f"Player {i + 1:02d} \u00b7 {colour_word(lit)}", "lit": [lit], "shadow": [shd]}
             for i, (lit, shd) in enumerate(palette(cat, "p0_hairstyle_pallete01_90_alb"))]
    dump("eyebrow_colors.json", brows)

    face = palette(cat, "p0_face_pallete01_90_alb")
    dump("lip_colors.json", [{"label": n, "lit": [face[4 + i][0]], "shadow": [face[4 + i][1]]} for i, n in enumerate(LIP_NAMES)])

    with open(SAMPLES, encoding="utf-8") as f:
        eyes = json.load(f)
    # Shadow side: the tables' shadow cells sit at about 0.78x their lit cell.
    dim = lambda c: [round(v * 0.78) for v in c]
    dump("eye_colors.json", [{"label": e["label"], "lit": [e["iris"], e["inner"]], "shadow": [dim(e["iris"]), dim(e["inner"])]} for e in eyes])


if __name__ == "__main__":
    main()
