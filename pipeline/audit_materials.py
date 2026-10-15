"""Which material parameters do the game files use that the importer never reads?

    python -m pipeline.audit_materials [--kind trainer|pokemon|object] [--all-values]

Walks every .trmtr of the chosen models, collects shader flags, colours, floats,
ints and texture slots, and checks each name against the importer source. A name
that appears in the data but nowhere in the importer is an effect we drop
silently; the report lists how many materials/models use it with a non-neutral
value, with examples, so mis-colourings like Grisham's double skin tint can be
found by data instead of by eye.
"""
from __future__ import annotations

import argparse
import collections
import inspect

from pipeline.catalog.za import ZaCatalog
from pipeline.importers.za import importer as imp
from pipeline.importers.za import trinity as T

# Values that mean "feature off / no effect", per parameter kind.
def _neutral(kind: str, value) -> bool:
    if kind == "flag":
        return value in ("False", "0", "")
    if kind == "color":
        return all(abs(x - 1.0) < 1e-3 for x in value[:3]) or all(abs(x) < 1e-3 for x in value[:4])
    if kind in ("float", "int"):
        return value == 0
    return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="trainer", choices=["trainer", "pokemon", "object"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--all-values", action="store_true", help="also count neutral values")
    a = ap.parse_args(argv)
    cat = ZaCatalog(a.data) if a.data else ZaCatalog()
    refs = {"trainer": cat.trainers, "pokemon": cat.pokemon, "object": cat.objects}[a.kind]()
    src_text = inspect.getsource(imp)

    uses: dict[tuple[str, str], dict] = collections.defaultdict(lambda: {"mats": 0, "models": set(), "examples": []})
    n_mats = 0
    for ref in refs:
        src = cat.model_source(ref)
        try:
            mdl = T.parse_trmdl(src.read(f"{ref.id}.trmdl"))
        except Exception:
            continue
        files = list(mdl.materials or [f"{ref.id}.trmtr"])
        if src.exists(f"{ref.id}.trmmt"):
            for s in T.parse_trmmt(src.read(f"{ref.id}.trmmt")):
                files += s.material_files or []
        for fn in dict.fromkeys(files):
            if not src.exists(fn):
                continue
            for tm in T.parse_trmtr(src.read(fn)):
                n_mats += 1
                items = [("flag", k, v) for k, v in tm.shader_flags.items()]
                items += [("color", k, v) for k, v in tm.colors.items()]
                items += [("float", k, v) for k, v in tm.floats.items()]
                items += [("int", k, v) for k, v in tm.ints.items()]
                items += [("texture", t.slot, t.file.split("/")[-1]) for t in tm.textures]
                for kind, name, value in items:
                    if not a.all_values and _neutral(kind, value):
                        continue
                    u = uses[(kind, name)]
                    u["mats"] += 1
                    u["models"].add(ref.id)
                    if len(u["examples"]) < 3:
                        v = tuple(round(x, 3) for x in value) if isinstance(value, tuple) else value
                        u["examples"].append(f"{ref.id}/{tm.name}={v}")

    def read(name: str) -> bool:
        # Names are used as literals ("EnableHighlight") or built with an index (f"BaseColorLayer{i}").
        if f'"{name}"' in src_text or f"'{name}'" in src_text:
            return True
        stem = name.rstrip("0123456789")
        return stem != name and (f'"{stem}{{' in src_text or f'f"{stem}{{' in src_text)

    print(f"{len(refs)} {a.kind} models, {n_mats} materials\n")
    print("NOT READ by the importer (non-neutral values):")
    for (kind, name), u in sorted(uses.items(), key=lambda kv: -len(kv[1]["models"])):
        if read(name):
            continue
        print(f"  {kind:7} {name:34} {u['mats']:4} mats {len(u['models']):3} models  e.g. {'; '.join(u['examples'])}")


if __name__ == "__main__":
    main()
