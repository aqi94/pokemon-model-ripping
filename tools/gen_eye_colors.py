"""Write the player-customisation palettes the viewer offers as "custom" colours on NPC parts that ship none
(pipeline/importers/za/{eye,eyebrow,lip}_colors.json).

The player's colour choices are the one place the game has a master list (https://www.serebii.net/legendsz-a/looks.shtml):
- eyebrows / eyelashes / hair: the 32 zones of `p0_hairstyle_pallete01_90_alb` (ZA-TrainersSharedTex);
- lips: zones 5-13 of `p0_face_pallete01_90_alb` (beige, pink, coral, orange, brown, rouge, blue, green, black; "No color"
  is the lip layer off); zones 1-4 are the skin tones, 14-16 other layers;
- eye contacts: the `color_eye` variation group (ids 300..325) in the player faces' `.trmmt` keys the eye materials'
  `BaseColorLayer1-3` per contact (exact; the 6 pattern contacts on Serebii are not in it). The older Serebii-thumbnail
  samples (`tools/player_eye_samples.json`, `tools/sample_serebii_eyes.py`) are superseded and kept for reference.

    ./.venv/Scripts/python.exe -m tools.gen_eye_colors [--data DIR]
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os

import numpy as np

from pipeline.catalog.za import ZaCatalog
from pipeline.importers.bntx import parse_bntx
from pipeline.importers.za import trinity as T
from pipeline.importers.za.importer import linear_to_srgb, table_shadow_rgb

ZA = os.path.join(os.path.dirname(__file__), "..", "pipeline", "importers", "za")
LIP_NAMES = ["Beige", "Pink", "Coral", "Orange", "Brown", "Rouge", "Blue", "Green", "Black"]  # face palette zones 5..13
NEAR_RGB = 8  # RGB distance under which two hair colours count as the same swatch
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


def table_zones(rgba) -> list[tuple[list[int], list[int]]]:
    """(lit, shadow) of every zone of a colour-table texture (2 px per zone, top half lit, bottom half shadow)."""
    th, tw = rgba.shape[:2]
    zones = max(1, tw // 2)
    out = []
    for z in range(zones):
        x = min((2 * z + 1) * tw // (2 * zones), tw - 1)
        lit = [int(v) for v in rgba[th // 4, x, :3]]
        out.append((lit, table_shadow_rgb(lit, [int(v) for v in rgba[3 * th // 4, x, :3]])))
    return out


def npc_hair_colors(cat: ZaCatalog, known: list[list[int]]) -> list[tuple[list[int], list[int]]]:
    """Distinct colours of every NPC hair part's colour table (`*_hairstyle_90_alb`), minus those within NEAR_RGB of `known`
    or of each other. The game applies one hair colour to hair, brows and facial hair, so the NPC tables hold shades the
    player's 32 do not (the Tourist's plum browns)."""
    kept: list[tuple[list[int], list[int]]] = []
    seen = [list(c) for c in known]
    for zp in cat.zips(cat.CC_MODEL_ZIPS):
        for n in sorted(zp.namelist()):
            if "/model_cc_vr/hrs/" not in n or not n.endswith("_hairstyle_90_alb.bntx"):
                continue
            for lit, shd in table_zones(parse_bntx(zp.read(n))[0].rgba):
                if any(sum((a - b) ** 2 for a, b in zip(lit, k)) < NEAR_RGB ** 2 for k in seen):
                    continue
                seen.append(lit)
                kept.append((lit, shd))
    # Browsable order: neutrals dark to light, then by hue.
    def key(c):
        h, sat, v = colorsys.rgb_to_hsv(*(x / 255 for x in c[0]))
        return (0, v, 0) if sat < 0.12 else (1, h, v)

    return sorted(kept, key=key)


def player_eye_contacts(cat: ZaCatalog) -> list[tuple[list[int], list[int]]]:
    """(iris, inner iris) sRGB of the player's eye contacts: the `color_eye` variation group (ids 300..325) of a player
    face `.trmmt` keys `BaseColorLayer2/3` of the eye materials per contact (layer 1 = pupil)."""
    stem = "p0_face0000_00"
    for zp in cat.zips(cat.PC_MODEL_ZIPS):
        path = next((n for n in zp.namelist() if n.endswith(f"/{stem}.trmmt")), None)
        if path is None:
            continue
        base = next(s for s in T.parse_trmmt(zp.read(path)) if s.name == "normal")
        group = next(g for g in base.groups if g.name == "color_eye")
        tracks = {(t.material, t.param): t for t in group.tracks}
        out = []
        for k in range(group.first, group.last + 1):
            pair = []
            for layer in (2, 3):
                v = T.track_value(tracks[("l_eye", f"BaseColorLayer{layer}")], k)
                pair.append([int(round(float(c) * 255)) for c in linear_to_srgb(np.array(v[:3]))])
            out.append((pair[0], pair[1]))
        return out
    raise SystemExit(f"{stem}.trmmt not found")


def npc_eye_pairs(cat: ZaCatalog, known: list[list[list[int]]]) -> list[tuple[list[list[int]], list[list[int]]]]:
    """(lit, shadow) iris + inner-iris pairs of every NPC face's eye `ColorTableMap` that the player contacts lack
    (Veteran F's violet #7a73d7/#bfbbf5, which Soignée's eyes match). The game recolours Common faces per NPC (data not in
    the rip), and the NPC tables are the only other place it keeps eye colours.

    Only tables laid out as (pupil, iris, inner iris) triples are read: layers 2 and 3 on consecutive default zones with
    the pupil (layer 1) on the zone before, so zone `i1 - 1 + 3k` starts triple k. Reading stops at the first triple whose
    pupil is not near-black (the tables continue with other layers) or whose iris and inner iris are both near-black
    (padding). Triples whose iris is near-black or whose inner iris is darker than the iris are skipped (ring/pupil layouts). Pairs within NEAR_RGB of a known pair (or of each other) are dropped. Two-zone (pupil, iris) tables and
    faces whose layers do not follow this layout are skipped: their pairing is ambiguous."""
    kept: list[tuple[list[list[int]], list[list[int]]]] = []
    seen = [[list(c) for c in k] for k in known]
    near = lambda a, b: all(sum((x - y) ** 2 for x, y in zip(p, q)) < NEAR_RGB ** 2 for p, q in zip(a, b))
    done: set[tuple] = set()
    for zp in cat.zips(cat.CC_MODEL_ZIPS):
        names = set(zp.namelist())
        for n in sorted(names):
            if "/model_cc_vr/face/" not in n or not n.endswith(".trmtr"):
                continue
            folder = n.rsplit("/", 1)[0]
            for tm in T.parse_trmtr(zp.read(n)):
                tab = next((t.file for t in tm.textures if t.slot == "ColorTableMap"), None)
                if tm.name.split("__")[-1] not in ("l_eye", "r_eye") or tab is None:
                    continue
                i1, i2, i3 = (tm.ints.get(f"BaseColorIndex{i}", 1) for i in (1, 2, 3))
                on = lambda i: tm.floats.get(f"LayerMaskScale{i}", 1.0) > 0
                path = f"{folder}/{tab}"
                if i3 != i2 + 1 or i1 != i2 - 1 or not (on(2) and on(3)) or path not in names or (path, i1) in done:
                    continue
                done.add((path, i1))
                zones = table_zones(parse_bntx(zp.read(path))[0].rgba)
                for k in range((len(zones) - (i1 - 1)) // 3):
                    p, iris, inner = zones[i1 - 1 + 3 * k: i1 + 2 + 3 * k]
                    if max(p[0]) > 100 or (max(iris[0]) < 40 and max(inner[0]) < 40):
                        break
                    if max(iris[0]) < 90 or sum(inner[0]) < 0.9 * sum(iris[0]):  # not an iris + lighter inner iris (rings, pupils)
                        continue
                    pair = ([iris[0], inner[0]], [iris[1], inner[1]])
                    if any(near(pair[0], s) for s in seen):
                        continue
                    seen.append(pair[0])
                    kept.append(pair)

    def key(c):
        h, sat, v = colorsys.rgb_to_hsv(*(x / 255 for x in c[0][0]))
        return (0, v, 0) if sat < 0.12 else (1, h, v)

    return sorted(kept, key=key)


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
    players = [b["lit"][0] for b in brows]
    brows += [{"label": f"NPC {i + 1:02d} · {colour_word(lit)}", "lit": [lit], "shadow": [shd]}
              for i, (lit, shd) in enumerate(npc_hair_colors(cat, players))]
    dump("eyebrow_colors.json", brows)

    face = palette(cat, "p0_face_pallete01_90_alb")
    dump("lip_colors.json", [{"label": n, "lit": [face[4 + i][0]], "shadow": [face[4 + i][1]]} for i, n in enumerate(LIP_NAMES)])

    # Shadow side: the tables' shadow cells sit at about 0.78x their lit cell.
    dim = lambda c: [round(v * 0.78) for v in c]
    contacts = player_eye_contacts(cat)
    eyes = [{"label": f"Contact {i + 1:02d} · {colour_word(iris)}", "lit": [iris, inner], "shadow": [dim(iris), dim(inner)]}
            for i, (iris, inner) in enumerate(contacts)]
    eyes += [{"label": f"NPC {i + 1:02d} · {colour_word(lit[0])}", "lit": lit, "shadow": shd}
             for i, (lit, shd) in enumerate(npc_eye_pairs(cat, [list(c) for c in contacts]))]
    dump("eye_colors.json", eyes)


if __name__ == "__main__":
    main()
