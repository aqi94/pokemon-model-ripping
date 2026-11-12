"""Catalog of Pokémon HOME models (Unity 2022.3 AssetBundles, ripped to `pokemon_home_unity`).

* `pokemons/pmNNNN_FF_VV.unity3d` (+ `_rare.unity3d` = shiny): the prefab (hierarchy, SkinnedMeshRenderers, Materials, the
  controller MonoBehaviour listing the clips). Meshes, textures and AnimationClips sit in `pokemons/dependencies/<hash>.unity3d`,
  named in the prefab's AssetBundle `m_Dependencies`; the PPtrs inside the prefab address them by CAB name (`externals`).
* `objects/ob####_VV.unity3d`: props, same layout.
* `OLD Prefabs/`: older prefab bundles, not catalogued (the first sample fails LZ4 decompression).
"""

from __future__ import annotations

import os
import re

from ..importers.home.importer import dependency_paths, import_bundle
from .za import ModelRef, _load_names

DEFAULT_DATA_DIR = os.environ.get("POKEMODELS_HOME_DATA") or os.path.join(os.path.expanduser("~"), "Documents", "pokemon_home_unity")

POKE_RE = re.compile(r"^(pm\d{4})_(\d{2})_(\d{2})\.unity3d$")
OBJECT_RE = re.compile(r"^(ob\d{4})_(\d{2})\.unity3d$")

# Internal numbers that are not National Dex numbers
SPECIAL_NAMES = {0: "Egg"}


class HomeCatalog:
    GAME = "home"

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self.names = _load_names()

    def _files(self, sub: str) -> list[str]:
        d = os.path.join(self.data_dir, sub)
        return sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))) if os.path.isdir(d) else []

    def pokemon(self) -> list[ModelRef]:
        found = [m for f in self._files("pokemons") if (m := POKE_RE.match(f))]
        forms: dict[str, set[str]] = {}
        for m in found:
            forms.setdefault(m.group(1), set()).add(f"{m.group(2)}_{m.group(3)}")
        refs = []
        for m in found:
            sp, ff, vv = m.groups()
            mid = f"{sp}_{ff}_{vv}"
            dex = int(sp[2:])
            base = SPECIAL_NAMES.get(dex) or self.names.get(str(dex), sp)
            # Same labelling as the other games: forms are unlabelled until a form table exists.
            display = base if (ff, vv) == ("00", "00") else f"{base} (form {ff}_{vv})"
            shiny = os.path.join(self.data_dir, "pokemons", f"{mid}_rare.unity3d")
            refs.append(
                ModelRef(
                    game=self.GAME,
                    kind="pokemon",
                    id=mid,
                    folder=f"pokemons/{mid}.unity3d",
                    display_name=display,
                    species=sp,
                    form=f"{ff}_{vv}",
                    extras={"dex": dex, "rare": f"pokemons/{mid}_rare.unity3d" if os.path.isfile(shiny) else None},
                )
            )
        return refs

    def objects(self) -> list[ModelRef]:
        refs = []
        for f in self._files("objects"):
            m = OBJECT_RE.match(f)
            if m:
                mid = f[: -len(".unity3d")]
                refs.append(ModelRef(game=self.GAME, kind="object", id=mid, folder=f"objects/{f}", display_name=mid, species=m.group(1), form=m.group(2)))
        return refs

    def trainers(self) -> list[ModelRef]:
        return []

    def all(self) -> list[ModelRef]:
        return self.pokemon() + self.objects()

    def find(self, model_id: str) -> ModelRef | None:
        return next((r for r in self.all() if r.id == model_id), None)

    def import_ref(self, ref: ModelRef, with_anim: bool = True):
        path = os.path.join(self.data_dir, ref.folder)
        rare = ref.extras.get("rare")
        rare_path = os.path.join(self.data_dir, rare) if rare else None
        return import_bundle(
            path,
            ref.id,
            ref.display_name,
            kind=ref.kind,
            game=ref.game,
            with_anim=with_anim,
            deps=dependency_paths(path, self.data_dir),
            rare_path=rare_path,
            rare_deps=dependency_paths(rare_path, self.data_dir) if rare_path else None,
        )
