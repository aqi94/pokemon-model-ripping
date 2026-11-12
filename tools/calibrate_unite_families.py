"""Calibrate / audit pipeline.importers.unite.families.infer_family on the lobby scene materials that have JSONs, and
count the families of the FBX-only materials (battle + standalone lobby). Needs the UNITE rip. Usage:
python -m tools.calibrate_unite_families [data.pkl]  (data.pkl = cache of fbx.material_blocks per FBX; built if absent)"""
from __future__ import annotations

import collections
import json
import os
import pickle
import re
import sys
from multiprocessing import Pool

from pipeline.importers.unite import fbx as F
from pipeline.importers.unite.families import CLOTH_GROUP, family_of, infer_family

D = os.path.expanduser("~/Documents/MEGA downloads/Pokémon UNITE Assets/3D Assets")
P = os.path.expanduser("~/Documents/pokemon_raw_data/unite/Pokemon/3D Models")
CHAR = re.compile(r"^mat_(lob|bat|lobby)", re.I)


def _job(a):
    kind, path = a
    try:
        blocks, links = F.material_blocks(path), F.material_textures(path)
    except Exception as e:  # noqa: BLE001
        return kind, path, None, repr(e)
    shaders = {}
    md = os.path.join(os.path.dirname(path), "Materials")
    for fn in os.listdir(md) if os.path.isdir(md) else []:
        if fn.endswith(".json"):
            j = json.load(open(os.path.join(md, fn), encoding="utf-8"))
            shaders[os.path.splitext(fn)[0].lower()] = ((j.get("m_Shader") or {}).get("Name")) or ""
    return kind, path, {"blocks": blocks, "links": links, "shaders": shaders}, None


def collect(cache: str) -> list:
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    jobs = [("scene", os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(D, "Scenes")) for f in fs if f.lower().endswith(".fbx")]
    for k, rel in (("battle", "Battle Models"), ("lobby", "Lobby Models")):
        jobs += [(k, os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(P, rel)) for f in fs if f.lower().endswith(".fbx")]
    with Pool(8) as p:
        res = p.map(_job, jobs, chunksize=4)
    pickle.dump(res, open(cache, "wb"))
    return res


def signature(m: str, b) -> tuple:
    h = lambda rx: any(re.match(rx, x, re.I) for x in b)  # noqa: E731
    return ("eye" if "eye" in m.lower() else "mouth" if re.search("mouth|mouse", m, re.I) else "oth",
            h("t_matcap"), h("t_mipmap03"), h("t_m_"), h("t_noise"), h("t_ramp"), h("t_nm_"), h("t_fx_changeskin"), h("t_vc_"),
            h("t_mroe_"), "body" in m.lower(), bool(re.search(r"\d{6}[a-z]?(_|$)", m)))


def loo(rows: dict) -> None:
    """Leave-one-out accuracy of the data-driven alternative: the majority truth among the OTHER materials with the same
    feature signature (falling back to the rules when the signature is unseen), per true token. Bounds how much the
    hand rules could overfit: rules whose in-sample accuracy is not above this are not memorising."""
    by = collections.defaultdict(list)
    for k, t in rows.items():
        by[signature(k[0], k[2])].append((k, t))
    hit, tot = collections.Counter(), collections.Counter()
    for sig, items in by.items():
        for k, t in items:
            rest = collections.Counter(tt for kk, tt in items if kk != k)
            pred = rest.most_common(1)[0][0] if rest else infer_family(k[0], list(k[2]), "lobby")
            tot[t] += 1
            hit[t] += pred == t
    print("LOO (signature majority):", f"{sum(hit.values())}/{sum(tot.values())}")
    for t, n in tot.most_common():
        print(f"  {t:14s} {hit[t]}/{n}")


def main() -> None:
    res = collect(sys.argv[1] if len(sys.argv) > 1 else "unite_blocks.pkl")
    rows = {}
    for kind, path, d, err in res:
        if d and kind == "scene":
            for m, sh in d["shaders"].items():
                if CHAR.match(m):
                    b = next((x for k, x in d["blocks"].items() if k.lower() == m), None)
                    if b is not None:
                        rows[(m, sh, tuple(b))] = family_of(sh)
    print(len(rows), "distinct character materials with JSON + block")
    conf = collections.defaultdict(collections.Counter)
    for (m, sh, b), truth in rows.items():
        conf[truth][infer_family(m, list(b), "lobby")] += 1
    CHARF = {"basepbr", "basepbreye", "basepbrmouth", "cloth", "lengcy", "matcapMask", "fur", "player", "bubble"}
    tot = ok = 0
    for t, c in sorted(conf.items(), key=lambda x: -sum(x[1].values())):
        n = sum(c.values())
        tot += n
        ok += c[t]
        print(f"{t:14s} n={n:4d} recall={c[t] / n:.2f}  predicted: {dict(c.most_common())}")
    print(f"overall {ok}/{tot} = {ok / tot:.3f}")
    ct = [(t, p) for t, c in conf.items() if t in CHARF for p, n in c.items() for _ in range(n)]
    print(f"character families only: {sum(t == p for t, p in ct)}/{len(ct)} = {sum(t == p for t, p in ct) / len(ct):.3f}")
    ct2 = [(t if t not in CLOTH_GROUP else "clothgroup", p if p not in CLOTH_GROUP else "clothgroup") for t, p in ct]
    print(f"  with cloth/lengcy/matcapMask merged: {sum(t == p for t, p in ct2)}/{len(ct2)} = {sum(t == p for t, p in ct2) / len(ct2):.3f}")
    loo(rows)
    fam = collections.Counter()
    for kind, path, d, err in res:
        if d and kind in ("battle", "lobby"):
            for m, b in (d["blocks"] or {m: [] for m in d["links"]}).items():
                fam[(kind, infer_family(m, b, kind))] += 1
    for k, v in sorted(fam.items()):
        print(k, v)


if __name__ == "__main__":
    main()
