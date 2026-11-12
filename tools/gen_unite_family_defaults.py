"""Write pipeline/importers/unite/family_defaults.json: per UNITE shader, the Properties block defaults from the
decompiled `Assets/Shaders/Compiled/*.txt` (Marcy4000/Unite clone). Run: python -m tools.gen_unite_family_defaults"""
from __future__ import annotations

import json
import os
import re

COMPILED = os.path.join(os.environ.get("POKEMODELS_UNITE_MARCY_ROOT") or os.path.join(
    os.path.expanduser("~"), "Documents", "pokemon_raw_data", "unite", "marcy"), "Assets", "Shaders", "Compiled")
OUT = os.path.join(os.path.dirname(__file__), "..", "pipeline", "importers", "unite", "family_defaults.json")
PROP = re.compile(r'^(?:\[[^\]]*\]\s*)*(_\w+)\s*\("[^"]*",\s*(\w+)(?:\s*\([^)]*\))?\)\s*=\s*(.+?)\s*$')


def parse_properties(lines: list[str]) -> dict:
    out, on = {}, False
    for l in lines:
        s = l.strip()
        if s.startswith("Properties"):
            on = True
            continue
        if not on:
            continue
        if s.startswith("SubShader") or s == "}":
            break
        m = PROP.match(s)
        if not m:
            continue
        name, typ, val = m.groups()
        if typ in ("Float", "Range", "Int"):
            try:
                out[name] = float(val)
            except ValueError:
                pass
        elif typ in ("Color", "Vector"):
            nums = re.findall(r"-?[\d.]+(?:e-?\d+)?", val)
            if len(nums) == 4:
                out[name] = [float(x) for x in nums]
    return out


def main() -> None:
    fam = json.load(open(os.path.join(os.path.dirname(__file__), "..", "pipeline", "importers", "unite", "shader_files.json")))
    res = {}
    for shader, fn in fam.items():
        p = os.path.join(COMPILED, fn) if fn else ""
        if fn and os.path.exists(p):
            res[shader] = parse_properties(open(p, encoding="utf-8", errors="replace").read().split("\n"))
    json.dump(res, open(OUT, "w"), separators=(",", ":"), sort_keys=True)
    print(len(res), "shaders,", sum(len(v) for v in res.values()), "defaults")


if __name__ == "__main__":
    main()
