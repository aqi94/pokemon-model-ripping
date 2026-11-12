"""Write pipeline/catalog/sv_motion_sets.json: which body motion sets each SV generic trainer plays.

SV's generic trainer classes (`chara/model_tr/tr0330_karate` ...) only have face/cloth layers in `motion_tr/<folder>/`.
Their body clips live in `motion_cc_base/`: 14 shared personality sets (`cc0001_kid` .. `cc0014_fat_f`, alternatives
with the same clip indexes) plus per-trainer folders (`motion_cc_base/<folder>/`: a few class clips such as
`04500_karate01_loop`, or a whole own set for the Hiker/Chef/Waitress/Nurse...). Which ones a trainer uses is authored
in its setup file `romfs/chara/sav_tr/<id>/<id>.ccdata` (full SV dump `SV-Everything.7z`), which names the `.tracn`
animator files. The dump is base-game only, so DLC generics get a borrowed Adult set (`BORROWED`, flagged `guess`).

The same setup files name each NPC's body type (`bsm/chara_create_value_list_<t>.trbsm` + weights); those go to
pipeline/catalog/sv_body_types.json (decoded tables + per-trainer weights, read by `pipeline/importers/za/trbsm.py`).

    python -m tools.gen_sv_motion_sets [--sav-dir <extracted romfs/chara>] [--archive SV-Everything.7z]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile

ARCHIVE = os.path.expanduser("~/Documents/MEGA downloads/SV Every File/SV-Everything.7z")
OUT = os.path.join(os.path.dirname(__file__), "..", "pipeline", "catalog", "sv_motion_sets.json")
BODY_OUT = os.path.join(os.path.dirname(__file__), "..", "pipeline", "catalog", "sv_body_types.json")

# Full sets come first when choosing the set that keeps bare clip names (the others are tagged " [pos_f]").
FULL_SETS = ("cc0004_adl_m", "cc0005_adl_f", "cc0001_kid")

# DLC generics (no motion data in the dump) on the shared cc_base rig: copy a base-game counterpart's plan, else a
# generic Adult set by the NPC's sex (from the class name). Unique DLC characters (tr1000-1140, 1020, 1720, 1900,
# 1920) and the Kitakami children (tr1320/1330, Kid proportions differ) stay static.
BORROWED = {
    "tr1300_00": "cc0004_adl_m",  # Scary M
    "tr1310_00": "cc0005_adl_f",  # Scary F
    "tr1340_00": "cc0005_adl_f",  # Kitakami youth F
    "tr1350_00": "cc0004_adl_m",  # Grandfather
    "tr1360_00": "cc0005_adl_f",  # Grandmother
    "tr1370_00": "tr0510_00",  # Kitakami Pokémon Center nurse = base-game nurse
    "tr1380_00": "tr0520_00",  # Kitakami shop clerk = base-game clerk
    "tr1390_00": "cc0004_adl_m",  # Janitor
    "tr1400_00": "cc0005_adl_f",  # Blueberry researcher
    "tr1500_00": "cc0004_adl_m",  # Villager M
    "tr1500_20": "cc0004_adl_m",
    "tr1510_00": "cc0005_adl_f",  # Villager F
    "tr1510_20": "cc0005_adl_f",
    "tr1800_00": "cc0004_adl_m",  # Blueberry teacher M
    "tr1810_00": "cc0005_adl_f",  # Blueberry teacher F
    "tr1820_00": "cc0004_adl_m",  # Sweets
    "tr2350_00": "cc0004_adl_m",  # Grandfather (P)
    "tr2360_00": "cc0005_adl_f",  # Grandmother (P)
    "tr2500_00": "cc0004_adl_m",  # Poison M
    "tr2500_20": "cc0004_adl_m",
    "tr2510_00": "cc0005_adl_f",  # Poison F
    "tr2510_20": "cc0005_adl_f",
}

TRACN_RE = re.compile(rb"(?:\.\./)*motion_cc_base/([A-Za-z0-9_]+)/([A-Za-z0-9_]+)\.tracn")


def plan_from_ccdata(data: bytes) -> dict:
    """{"own": folder | "", "sets": [cc####_x...], "clips": ["<folder>/<stem>"...]} from one .ccdata."""
    own, sets, clips = "", [], []
    for folder, stem in TRACN_RE.findall(data):
        folder, stem = folder.decode(), stem.decode()
        if re.fullmatch(r"cc\d{4}_[a-z_]+", folder):
            if stem == folder:
                sets.append(folder)  # the whole set's animator (cc0012_msl.tracn)
            else:
                clips.append(f"{folder}/{stem}")  # one clip of a set (cc0004_adl_m_04107_squat01_loop)
        else:
            own = folder  # per-trainer folder: the whole folder is loaded
    sets = sorted(set(sets), key=lambda s: (s not in FULL_SETS, FULL_SETS.index(s) if s in FULL_SETS else 0, s))
    return {"own": own, "sets": sets, "clips": sorted(set(clips))}


def merge(a: dict, b: dict) -> dict:
    sets = list(dict.fromkeys(a["sets"] + b["sets"]))
    sets.sort(key=lambda s: (s not in FULL_SETS, FULL_SETS.index(s) if s in FULL_SETS else 0, s))
    return {"own": a["own"] or b["own"], "sets": sets, "clips": sorted(set(a["clips"]) | set(b["clips"]))}


def read_sav(sav_dir: str) -> dict[str, bytes]:
    out = {}
    for root, _, files in os.walk(os.path.join(sav_dir, "sav_tr")):
        for f in files:
            if f.endswith(".ccdata"):
                with open(os.path.join(root, f), "rb") as fh:
                    out[f[: -len(".ccdata")]] = fh.read()
    return out


def read_bsm(sav_dir: str) -> dict[str, bytes]:
    out = {}
    for root, _, files in os.walk(os.path.join(sav_dir, "bsm")):
        for f in files:
            if f.endswith(".trbsm"):
                with open(os.path.join(root, f), "rb") as fh:
                    out[f[: -len(".trbsm")].replace("chara_create_value_list_", "")] = fh.read()
    return out


def build_bodies(sav: dict[str, bytes], bsm: dict[str, bytes]) -> dict:
    """{"tables": {type: decoded trbsm}, "trainers": {model id: {"table", "sliders", "scalers"}}}."""
    from pipeline.importers.za.trbsm import parse_ccdata_body, parse_trbsm

    trainers: dict[str, dict] = {}
    for sid in sorted(sav):
        mid = sid if sid.endswith("_00") or sid[:6] + "_00" not in sav else sid[:6] + "_00"
        body = parse_ccdata_body(sav[sid])
        if body and mid not in trainers:  # setup variants (_20) share the model's body
            trainers[mid] = body
    used = {b["table"] for b in trainers.values()}
    return {"tables": {t: parse_trbsm(bsm[t]) for t in sorted(bsm) if t in used}, "trainers": trainers}


def extract_sav(archive: str, dest: str) -> str:
    import py7zr

    with py7zr.SevenZipFile(archive, "r") as z:
        targets = [n for n in z.getnames() if n.startswith(("romfs/chara/sav_tr/", "romfs/chara/bsm/"))]
    with py7zr.SevenZipFile(archive, "r") as z:
        z.extract(path=dest, targets=targets)
    return os.path.join(dest, "romfs", "chara")


def build(sav: dict[str, bytes]) -> dict:
    plans: dict[str, dict] = {}
    for sid in sorted(sav):
        # ccdata variants (tr0340_20 = the "working" pose setup) belong to model tr0340_00
        mid = sid if sid.endswith("_00") or sid[:6] + "_00" not in sav else sid[:6] + "_00"
        p = plan_from_ccdata(sav[sid])
        plans[mid] = merge(plans[mid], p) if mid in plans else p
    for mid, src in BORROWED.items():
        base = plans[src] if src in plans else {"own": "", "sets": [src], "clips": []}
        plans[mid] = {**base, "guess": True, "borrowedFrom": src}
    return plans


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sav-dir", help="already extracted romfs/chara folder (holds sav_tr/)")
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if args.sav_dir:
        sav, bsm = read_sav(args.sav_dir), read_bsm(args.sav_dir)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            d = extract_sav(args.archive, tmp)
            sav, bsm = read_sav(d), read_bsm(d)
    plans = build(sav)
    bodies = build_bodies(sav, bsm)
    with open(BODY_OUT, "w", encoding="utf-8") as f:
        json.dump(bodies, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{len(bodies['trainers'])} body types -> {os.path.normpath(BODY_OUT)}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{len(plans)} trainers -> {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
