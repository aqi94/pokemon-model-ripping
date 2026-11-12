"""Pair Bulbapedia holowear art (samples/unite_holowear) with converted UNITE skins and build review sheets.

    ./.venv/Scripts/python.exe tools/holowear_pairs.py ids   > ids.txt     # models to render (lobby + battle per skin)
    node tools/render_models.mjs --out out/unite_holowear_audit/renders --size 512x512 --views back unite/<id> ...
    ./.venv/Scripts/python.exe tools/holowear_pairs.py pair                # -> out/unite_holowear_audit/pairs.json
    ./.venv/Scripts/python.exe tools/holowear_pairs.py sheets              # -> out/unite_holowear_audit/sheets/*.png

A skin = one `extras.holowear` key (dex4 + skin2 [+ form]) with its lobby and/or battle model. Skin 00 (default look)
and forms >= 50 (Megas) are not holowear and only appear as context rows. The proposed pairing is a Hungarian
assignment on alpha-masked CIELAB histograms (art vs the skin's front+back renders), with skins whose scene theme
word appears in the holowear name pinned. It is a proposal for review, not ground truth.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont

ART = "samples/unite_holowear"
AUD = "out/unite_holowear_audit"
REN = f"{AUD}/renders"
RARITY = {"rgb(113, 198, 113)": "green", "rgb(119, 141, 230)": "blue", "rgb(184, 132, 221)": "purple",
          "rgb(140, 230, 255)": "crystal", "rgb(255, 172, 89)": "orange"}


def norm(s: str) -> str:
    s = re.sub(r"^(alolan|galarian|hisuian|paldean)\s+", "", s.lower())
    return re.sub(r"[^a-z0-9]", "", s)


def skins() -> dict[str, list[dict]]:
    """normalised species name -> list of skins {key, skin, lobby, battle, theme, label}."""
    by: dict[str, dict[str, dict]] = {}
    for m in glob.glob("out/unite/*/manifest.json"):
        j = json.load(open(m, encoding="utf-8"))
        e = j.get("extras") or {}
        if j.get("kind") != "pokemon" or not e.get("holowear"):
            continue
        name = j["displayName"].split(" · ")[0].replace(" (lobby)", "").strip()
        sp = by.setdefault(norm(name), {})
        s = sp.setdefault(e["holowear"], {"key": e["holowear"], "skin": e.get("skin"), "dex": e.get("dex"),
                                          "lobby": None, "battle": None, "theme": None, "label": None})
        s["lobby" if e.get("battle") is False else "battle"] = j["id"]
        lab = j["displayName"].split(" · ")[1].replace(" (lobby)", "") if " · " in j["displayName"] else None
        if e.get("theme"):
            s["theme"] = e["theme"]
        if lab and not lab.startswith("skin ") and not s["label"]:
            s["label"] = lab
    return {k: sorted(v.values(), key=lambda s: s["key"]) for k, v in by.items()}


def is_holowear(s: dict) -> bool:
    try:
        n = int(re.match(r"\d+", s["skin"] or "0").group())
    except AttributeError:
        return True
    return 0 < n < 50


def art() -> list[dict]:
    a = json.load(open(f"{ART}/sources.json", encoding="utf-8"))
    for i, e in enumerate(a):
        e["i"] = i
        e["rarity"] = RARITY.get(e.get("bg"), e.get("bg"))
    return a


_LAB = ImageCms.buildTransformFromOpenProfiles(ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB"), "RGB", "LAB")


def hist(paths: list[str]) -> np.ndarray | None:
    hs = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        im = Image.open(p).convert("RGBA")
        im.thumbnail((256, 256))
        a = np.asarray(im)[..., 3] > 128
        lab = np.asarray(ImageCms.applyTransform(im.convert("RGB"), _LAB)).astype(np.float32)
        px = lab[a]
        if len(px) < 50:
            continue
        L = px[:, 0] / 255 * 100
        aa = px[:, 1].astype(np.int16).astype(np.float32)
        bb = px[:, 2].astype(np.int16).astype(np.float32)
        aa = np.where(aa > 127, aa - 256, aa)
        bb = np.where(bb > 127, bb - 256, bb)
        h, _ = np.histogramdd(np.stack([L, aa, bb], 1), bins=(4, 10, 10), range=((0, 100), (-80, 80), (-80, 80)))
        hs.append(h.ravel() / h.sum())
    return np.mean(hs, 0) if hs else None


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Min-cost assignment for a square matrix (Kuhn-Munkres, O(n^3))."""
    n = cost.shape[0]
    u, v, p, way = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1, int), np.zeros(n + 1, int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(p[j] - 1, j - 1) for j in range(1, n + 1) if p[j]]


def cmd_ids(_a) -> None:
    sk = skins()
    names = {norm(e["species"]) for e in art()}
    for sp in sorted(names):
        for s in sk.get(sp, []):
            for i in (s["lobby"], s["battle"]):
                if i:
                    print(f"unite/{i}")
    missing = sorted(n for n in names if n not in sk)
    if missing:
        print("species without models:", missing, file=sys.stderr)


def cmd_pair(_a) -> None:
    sk = skins()
    A = art()
    out = {"species": {}}
    for spn in sorted({norm(e["species"]) for e in A}):
        arts = [e for e in A if norm(e["species"]) == spn]
        cands = [s for s in sk.get(spn, []) if is_holowear(s)]
        ah = [hist([f"{ART}/{e['local']}"]) for e in arts]
        ch = []
        for s in cands:
            mid = s["lobby"] or s["battle"]
            ch.append(hist([f"{REN}/unite_{mid}_full.png", f"{REN}/unite_{mid}_back.png"]))
        D = np.full((len(arts), len(cands)), 1.0)
        for i, h in enumerate(ah):
            for j, c in enumerate(ch):
                if h is not None and c is not None:
                    D[i, j] = 1 - float(np.sqrt(h * c).sum())  # 1 - Bhattacharyya coefficient
        C = D.copy()
        pins = {}
        for j, s in enumerate(cands):
            word = norm(s["theme"] or s["label"] or "")
            if len(word) >= 4:
                hits = [i for i, e in enumerate(arts) if word in norm(e["name"]) or norm(e["name"].replace(" Style", "")) in word]
                if len(hits) == 1:
                    pins[hits[0]] = j
        for i, j in pins.items():
            C[i, :] += 10
            C[:, j] += 10
            C[i, j] = -10
        n = max(len(arts), len(cands))
        UNMATCHED = 0.75  # an art/skin pair worse than this is left unpaired
        sq = np.full((n, n), UNMATCHED)
        sq[:len(arts), :len(cands)] = np.minimum(C, UNMATCHED + 0.001) if len(cands) else 0
        assign = dict(hungarian(sq)) if n else {}
        rows = []
        used = set()
        for i, e in enumerate(arts):
            j = assign.get(i)
            ok = j is not None and j < len(cands) and sq[i, j] <= UNMATCHED
            alt = [cands[k]["key"] for k in np.argsort(D[i])[:3]] if len(cands) else []
            rows.append({"art": e["file"], "local": e["local"], "name": e["name"], "released": e["released"],
                         "rarity": e["rarity"], "skin": cands[j]["key"] if ok else None,
                         "lobby": cands[j]["lobby"] if ok else None, "battle": cands[j]["battle"] if ok else None,
                         "dist": round(float(D[i, j]), 3) if ok else None, "pinned": pins.get(i) == j if ok else False,
                         "alternatives": alt})
            if ok:
                used.add(j)
        context = [s for s in sk.get(spn, []) if not is_holowear(s)]
        out["species"][spn] = {"label": arts[0]["species"], "pairs": rows,
                               "unmatchedSkins": [cands[j] for j in range(len(cands)) if j not in used],
                               "contextSkins": context}
    json.dump(out, open(f"{AUD}/pairs.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    n_pairs = sum(1 for s in out["species"].values() for r in s["pairs"] if r["skin"])
    n_art = sum(len(s["pairs"]) for s in out["species"].values())
    n_un = sum(len(s["unmatchedSkins"]) for s in out["species"].values())
    print(f"{n_pairs}/{n_art} art paired, {n_un} holowear skins without art pair -> {AUD}/pairs.json")


def cell(path: str | None, size: int) -> Image.Image:
    c = Image.new("RGBA", (size, size), (235, 235, 235, 255))
    if path and os.path.isfile(path):
        im = Image.open(path).convert("RGBA")
        bb = im.getbbox()
        if bb:
            im = im.crop(bb)
        im.thumbnail((size - 8, size - 8))
        c.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
    return c


def cmd_sheets(a) -> None:
    P = json.load(open(f"{AUD}/pairs.json", encoding="utf-8"))
    os.makedirs(f"{AUD}/sheets", exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    S, LH, per = a.cell, 40, a.rows
    heads = ["Bulbapedia art", "lobby front", "lobby back", "battle front", "battle back"]
    index = []
    for spn, sp in P["species"].items():
        rows = []
        for r in sp["pairs"]:
            lab = f"{r['name']} ({r['rarity']}, {r['released']})"
            mid = f"-> skin {r['skin']}  dist {r['dist']}{'  [pinned]' if r['pinned'] else ''}" if r["skin"] else "-> no proposed skin"
            rows.append((f"{ART}/{r['local']}", r["lobby"], r["battle"], lab + "   " + mid + f"   lobby {r['lobby']}  battle {r['battle']}"))
        for s in sp["unmatchedSkins"]:
            rows.append((None, s["lobby"], s["battle"], f"UNPAIRED skin {s['key']}   lobby {s['lobby']}  battle {s['battle']}"))
        for s in sp["contextSkins"]:
            rows.append((None, s["lobby"], s["battle"], f"default/form skin {s['key']} (no holowear)   lobby {s['lobby']}  battle {s['battle']}"))
        for pg in range(0, len(rows), per):
            chunk = rows[pg:pg + per]
            W = S * 5
            sheet = Image.new("RGBA", (W, 24 + len(chunk) * (S + LH)), (255, 255, 255, 255))
            d = ImageDraw.Draw(sheet)
            for k, h in enumerate(heads):
                d.text((k * S + 6, 4), h, fill=(0, 0, 0), font=font)
            for r, (artp, lob, bat, text) in enumerate(chunk):
                y = 24 + r * (S + LH)
                d.text((6, y + 4), text, fill=(0, 0, 0), font=font)
                imgs = [artp,
                        f"{REN}/unite_{lob}_full.png" if lob else None, f"{REN}/unite_{lob}_back.png" if lob else None,
                        f"{REN}/unite_{bat}_full.png" if bat else None, f"{REN}/unite_{bat}_back.png" if bat else None]
                for k, p in enumerate(imgs):
                    sheet.alpha_composite(cell(p, S), (k * S, y + LH - 18))
                    d.rectangle([k * S, y + LH - 18, (k + 1) * S - 1, y + LH - 18 + S - 1], outline=(180, 180, 180))
            fn = f"{AUD}/sheets/{spn}_{pg // per + 1:02d}.png"
            sheet.convert("RGB").save(fn)
            index.append({"species": sp["label"], "key": spn, "file": fn, "rows": len(chunk)})
    json.dump(index, open(f"{AUD}/sheets/index.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(len(index), "sheets")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ids")
    sub.add_parser("pair")
    p = sub.add_parser("sheets")
    p.add_argument("--cell", type=int, default=300)
    p.add_argument("--rows", type=int, default=5)
    a = ap.parse_args()
    {"ids": cmd_ids, "pair": cmd_pair, "sheets": cmd_sheets}[a.cmd](a)


if __name__ == "__main__":
    main()
