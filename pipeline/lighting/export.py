"""`pipeline lights export`: every rig as JSON (+ probe cube PNGs) under out/sv/lights/."""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from .catalog import SvLightCatalog


def rig_json(cat: SvLightCatalog, name: str) -> dict:
    ref = cat.find(name)
    rig = cat.load(name)
    return {
        "name": name, "group": ref.group, "res0": rig.res0,
        "lights": [asdict(l) for l in rig.lights],
    }


def export_rigs(cat: SvLightCatalog, names: list[str], out_dir: str, probes: bool = True) -> int:
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for name in names:
        data = rig_json(cat, name)
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        n += 1
        if probes:
            ref = cat.find(name)
            for tex in ("probemain_diffuse.bntx", "probemain_specular.bntx"):
                p = ref.texture(tex)
                if p:
                    try:
                        from PIL import Image

                        from .probes import cube_to_cross, read_cube
                        with open(p, "rb") as f:
                            cube = read_cube(f.read())
                        Image.fromarray(cube_to_cross(cube)).save(os.path.join(out_dir, f"{name}_{tex[:-5]}.png"))
                    except Exception as e:  # BC6H support is best effort
                        print(f"  probe {name}/{tex}: {e}")
    return n
