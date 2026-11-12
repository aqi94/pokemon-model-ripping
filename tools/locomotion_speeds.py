"""Z-A locomotion speeds: metres per second of every Pokémon's walk/run clips, per movement category.

Movement speed is authored as root motion: the `origin` bone moves at a constant rate over a looping
clip, and the animator parameters (`move_playspeed` default 1) are the same for every species. Clip
index sets are the `.tracs` movement categories (index // 10000):
0 ground, 1 water, 2 sky, 3 other01, 4 other02 (species-specific, e.g. Pikachu/Spinarak wall climbing).

    python -m tools.locomotion_speeds [--out out/za_locomotion_speeds]

Writes <out>.csv (one row per clip), <out>.json (same rows) and <out>_wide.csv (one row per model).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import sys

import numpy as np

from pipeline.catalog.za import ZaCatalog
from pipeline.importers.za import trinity

CATEGORIES = {0: "ground", 1: "water", 2: "sky", 3: "other01", 4: "other02"}
CLIPS = ("walk01_loop", "run01_loop", "backwalk01_loop")
SHORT = {"walk01_loop": "walk", "run01_loop": "run", "backwalk01_loop": "backwalk"}
FIELDS = [
    "id", "dex", "name", "form", "category", "clip", "clip_index", "frames", "fps", "duration_s",
    "distance_m", "speed_m_s", "direction", "has_root_motion", "model_height_m",
]


def glb_height(path: str) -> float | None:
    """Rest-pose height of a converted model (LOD0 POSITION accessor bounds), None if not converted."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        f.read(12)
        n, _ = struct.unpack("<II", f.read(8))
        gltf = json.loads(f.read(n))
    lo, hi = [], []
    for mesh in gltf.get("meshes", []):
        if "_lod" in mesh.get("name", ""):
            continue
        for prim in mesh["primitives"]:
            acc = gltf["accessors"][prim["attributes"]["POSITION"]]
            lo.append(acc["min"][1])
            hi.append(acc["max"][1])
    return round(max(hi) - min(lo), 3) if hi else None


def display_name(ref, out_dir: str) -> str:
    """Manifest name when converted (carries relabelled internal ids), else the catalog's."""
    man = os.path.join(out_dir, ref.id, "manifest.json")
    if os.path.exists(man):
        try:
            return json.load(open(man, encoding="utf-8")).get("displayName") or ref.display_name
        except (OSError, ValueError):
            pass
    return ref.display_name


def direction(d: np.ndarray, dist: float) -> str:
    if dist < 1e-4:
        return "none"
    axis = int(np.argmax(np.abs(d)))
    if axis == 2:
        return "forward" if d[2] > 0 else "backward"
    if axis == 1:
        return "vertical"
    return "sideways"


def clip_row(ref, name: str, height, af) -> dict:
    t = trinity.parse_tranm(af.data)
    fps = t.frame_rate
    duration = (t.frame_count - 1) / fps if t.frame_count > 1 else 0.0
    origin = next((tr for tr in t.tracks if tr.bone == "origin" and tr.translation is not None), None)
    d = np.zeros(3)
    if origin is not None and len(origin.translation.values) > 1:
        d = (origin.translation.values[-1] - origin.translation.values[0]).astype(float)
    dist = float(np.linalg.norm(d))
    return {
        "id": ref.id,
        "dex": ref.extras.get("dex"),
        "name": name,
        "form": ref.form,
        "category": CATEGORIES.get(af.index // 10000, f"set{af.index // 10000}"),
        "clip": af.name,
        "clip_index": af.index,
        "frames": t.frame_count,
        "fps": fps,
        "duration_s": round(duration, 4),
        "distance_m": round(dist, 4),
        "speed_m_s": round(dist / duration, 3) if duration else 0.0,
        "direction": direction(d, dist),
        "has_root_motion": dist > 1e-4,
        "model_height_m": height,
    }


def collect(cat: ZaCatalog, out_dir: str) -> list[dict]:
    rows = []
    refs = cat.pokemon()
    for i, ref in enumerate(refs):
        try:
            anims = cat.animations(ref)
        except Exception as e:  # missing/corrupt anim folder: report and go on
            print(f"  {ref.id}: {e}", file=sys.stderr)
            continue
        seen = set()
        picked = []
        for af in sorted(anims, key=lambda a: a.index):
            key = (af.index // 10000, af.name)
            if af.name in CLIPS and af.data and key not in seen:
                seen.add(key)
                picked.append(af)
        if not picked:
            continue
        name = display_name(ref, out_dir)
        height = glb_height(os.path.join(out_dir, ref.id, "model.glb"))
        for af in picked:
            try:
                rows.append(clip_row(ref, name, height, af))
            except Exception as e:
                print(f"  {ref.id} {af.index} {af.name}: {e}", file=sys.stderr)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(refs)} models", file=sys.stderr)
    return rows


def write(rows: list[dict], base: str) -> None:
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    cols = [f"{c}_{SHORT[n]}" for c in CATEGORIES.values() for n in CLIPS]
    wide: dict[str, dict] = {}
    for r in rows:
        m = wide.setdefault(r["id"], {"id": r["id"], "dex": r["dex"], "name": r["name"], "model_height_m": r["model_height_m"]})
        m[f"{r['category']}_{SHORT[r['clip']]}"] = r["speed_m_s"]
    used = [c for c in cols if any(c in m for m in wide.values())]
    with open(base + "_wide.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "dex", "name", "model_height_m"] + used)
        w.writeheader()
        w.writerows(wide.values())


def summary(rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["category"], r["clip"]), []).append(r)
    order = {c: i for i, c in enumerate(CATEGORIES.values())}
    for (cat, clip), rs in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1])):
        sp = np.array([r["speed_m_s"] for r in rs])
        ranked = sorted(rs, key=lambda r: -r["speed_m_s"])
        top = ", ".join(f"{r['name']} {r['speed_m_s']}" for r in ranked[:5])
        print(f"{cat:8s} {clip:16s} n={len(rs):3d} median {np.median(sp):5.2f}  max {sp.max():5.2f}  top: {top}")
        for k, r in enumerate(ranked):
            if r["id"] == "pm0544_00_00":
                print(f"{'':26s}Whirlipede {r['speed_m_s']} m/s, rank {k + 1}/{len(rs)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/za_locomotion_speeds", help="output path without extension")
    ap.add_argument("--models", default="out/za", help="converted models (for names/heights)")
    args = ap.parse_args(argv)
    rows = collect(ZaCatalog(), args.models)
    write(rows, args.out)
    summary(rows)
    print(f"{len(rows)} rows -> {args.out}.csv / .json / _wide.csv")


if __name__ == "__main__":
    main()
