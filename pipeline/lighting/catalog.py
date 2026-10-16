"""Where the extracted SV light rigs live and how they are grouped.

`SV-Everything.7z` -> `romfs/env/**`, `romfs/field_graphic/{terrain_env,indoor_shieldmap}/**` and
`romfs/world/scene/parts/env/field_env/**` extracted once into `~/Documents/pokemon_raw_data/sv/env`
(env `POKEMODELS_SV_ENV`). Rig names are the file stems (`master`, `weather_rain`, `fel_interior`, `ter_01_03`...).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..importers.za.trlgt import LightRig, parse_trlgt

SV_ENV_DIR = os.environ.get("POKEMODELS_SV_ENV") or os.path.join(os.path.expanduser("~"), "Documents", "pokemon_raw_data", "sv", "env")

_GROUPS = [  # (group, path prefix under romfs/)
    ("master", "env/master/"),
    ("weather", "env/offset/"),
    ("area", "env/fel/"),
    ("cutscene", "env/spl/"),
    ("terrain", "field_graphic/terrain_env/"),
    ("shieldmap", "field_graphic/indoor_shieldmap/"),
]
_FILTERS = {"blur", "cool", "crossFilter", "fisheyeLens", "kawaii", "monochrome", "retro", "scratch", "sepia", "toy", "vivid"}


@dataclass(frozen=True)
class RigRef:
    name: str
    group: str
    path: str

    @property
    def folder(self) -> str:
        return os.path.dirname(self.path)

    def texture(self, file_name: str) -> str | None:
        p = os.path.join(self.folder, file_name)
        return p if os.path.isfile(p) else None


class SvLightCatalog:
    def __init__(self, root: str = SV_ENV_DIR):
        self.root = root
        self.romfs = os.path.join(root, "romfs")

    def available(self) -> bool:
        return os.path.isdir(self.romfs)

    def rigs(self) -> list[RigRef]:
        out: list[RigRef] = []
        for group, prefix in _GROUPS:
            base = os.path.join(self.romfs, *prefix.strip("/").split("/"))
            if not os.path.isdir(base):
                continue
            for dirpath, _dirs, files in os.walk(base):
                for f in sorted(files):
                    if f.endswith(".trlgt"):
                        stem = f[:-6]
                        g = "filter" if group == "cutscene" and stem in _FILTERS else group
                        out.append(RigRef(stem, g, os.path.join(dirpath, f)))
        return out

    def find(self, name: str) -> RigRef:
        for r in self.rigs():
            if r.name == name:
                return r
        raise KeyError(f"no rig named {name!r} under {self.romfs}")

    def load(self, name: str) -> LightRig:
        with open(self.find(name).path, "rb") as f:
            return parse_trlgt(f.read())

    def master(self) -> LightRig:
        return self.load("master")


def merge(base: LightRig, overlay: LightRig) -> LightRig:
    """Area/weather rigs are partial: every light type they list overrides the same type in the base
    parameter by parameter (verified: `fel_interior`'s `Global` only carries Gamma/Contrast/CbCr/Tonemap/Bloom
    keys, so the rest of the post-process stays as in `master`)."""
    from copy import deepcopy

    out = deepcopy(base)
    for ol in overlay.lights:
        bl = out.get(ol.type)
        if bl is None:
            out.lights.append(deepcopy(ol))
            continue
        bl.name = ol.name or bl.name
        if ol.rotation is not None:
            bl.scale, bl.rotation, bl.translation = ol.scale, ol.rotation, ol.translation
        bl.floats.update(ol.floats)
        bl.colors.update(ol.colors)
        bl.textures.update(ol.textures)
    return out
