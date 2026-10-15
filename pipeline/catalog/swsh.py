"""Catalog of Pokemon Sword/Shield models inside the ripped SwSh-* zips (.gfbmdl / .gfbanm / .bntx).

Pokemon: `SwSh-PokeGen*.zip` hold `bin/pokemon/pmDDDD_FF/mdl/` (model, `_rare` shiny model, shaders, textures) and
`SwSh-PokeAnim.zip` holds `bin/pokemon/pmDDDD_FF/anm/` (one `.gfbanm` per clip, plus `.gfbanmcfg` state machines).
Trainers/objects live in 7z archives that must be extracted once (see `TRAINER_DIR`); they are not catalogued yet.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from functools import lru_cache

import numpy as np

from ..importers.bntx import parse_bntx
from ..importers.swsh.dds import parse_dds
from ..importers.swsh.importer import CLIP_NAME_RE, clip_from_gfbanm, import_gfbmdl
from .za import DEFAULT_DATA_DIR, ModelRef  # noqa: F401  (DEFAULT_DATA_DIR re-exported for the CLI)

POKE_MODEL_ZIPS = ["SwSh-PokeGen1.zip", "SwSh-PokeGen2-3.zip", "SwSh-PokeGen4-5.zip", "SwSh-PokeGen6-7.zip", "SwSh-PokeGen8.zip"]
POKE_ANIM_ZIPS = ["SwSh-PokeAnim.zip"]
POKE_RE = re.compile(r"^bin/pokemon/(pm\d{4})_(\d{2})/mdl/(pm\d{4}_\d{2})\.gfbmdl$")
ANM_RE = re.compile(r"^bin/pokemon/(pm\d{4}_\d{2})/anm/(.+)\.gfbanm$")

# The trainer/object 7z archives are extracted once (py7zr) into one folder per archive under EXTRACTED_DIR:
# trainers, trainers_dlc1, trainers_dlc2, misc, misc_dlc1, misc_dlc2 (later archives override earlier ones).
EXTRACTED_DIR = os.environ.get("POKEMODELS_SWSH_DATA") or os.path.join(os.path.expanduser("~"), "Documents", "pokemon_raw_data", "swsh")
TRAINER_DIRS = ["trainers", "trainers_dlc1", "trainers_dlc2"]
OBJECT_DIRS = ["misc", "misc_dlc1", "misc_dlc2"]
# <root>/chara/data/(tr|tr_m)[/mdl]/<trNNNN_VV_name>/<same>.gfbmdl and <root>/chara/data/ob[/mdl]/<obNNNN_VV_name>/...
TRAINER_RE = re.compile(r"^chara/data/(?:tr|tr_m)/(?:mdl/)?(tr\d{4}_\d{2})_([^/]+)/\1_\2\.gfbmdl$")
OBJECT_RE = re.compile(r"^chara/data/ob/(?:mdl/)?(ob\d{4}_\d{2})_([^/]+)/\1_\2\.gfbmdl$")


@lru_cache(maxsize=1)
def _load_names() -> dict[str, str]:
    path = os.path.join(os.path.dirname(__file__), "pokemon_names.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


class SwshCatalog:
    GAME = "swsh"

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, raw_dir: str = EXTRACTED_DIR):
        self.data_dir = data_dir
        self.raw_dir = raw_dir
        self._zips: dict[str, zipfile.ZipFile | None] = {}
        self.names = _load_names()
        self._members: dict[str, tuple[zipfile.ZipFile, str]] | None = None
        self._anims: dict[str, list[tuple[zipfile.ZipFile, str, str]]] | None = None

    # -- zips ---------------------------------------------------------------
    def zip(self, name: str) -> zipfile.ZipFile | None:
        if name not in self._zips:
            path = os.path.join(self.data_dir, name)
            self._zips[name] = zipfile.ZipFile(path) if os.path.isfile(path) else None
        return self._zips[name]

    def _model_members(self) -> dict[str, tuple[zipfile.ZipFile, str]]:
        """zip member name -> (zip, name) for every file under bin/pokemon/*/mdl/."""
        if self._members is None:
            self._members = {}
            for zn in POKE_MODEL_ZIPS:
                z = self.zip(zn)
                if z is None:
                    continue
                for n in z.namelist():
                    if "/mdl/" in n and not n.endswith("/"):
                        self._members[n] = (z, n)
        return self._members

    def _anim_index(self) -> dict[str, list[tuple[zipfile.ZipFile, str, str]]]:
        if self._anims is None:
            self._anims = {}
            for zn in POKE_ANIM_ZIPS:
                z = self.zip(zn)
                if z is None:
                    continue
                for n in z.namelist():
                    m = ANM_RE.match(n)
                    if m:
                        self._anims.setdefault(m.group(1), []).append((z, n, m.group(2)))
        return self._anims

    # -- discovery ----------------------------------------------------------
    def pokemon(self) -> list[ModelRef]:
        refs: dict[str, ModelRef] = {}
        for n in self._model_members():
            m = POKE_RE.match(n)
            if not m or m.group(3) in refs:
                continue
            species, form, mid = m.group(1), m.group(2), m.group(3)
            dex = int(species[2:])
            base = self.names.get(str(dex), species)
            display = base if form == "00" else f"{base} (form {form})"
            refs[mid] = ModelRef(
                game=self.GAME,
                kind="pokemon",
                id=mid,
                folder=f"bin/pokemon/{mid}/",
                display_name=display,
                species=species,
                form=form,
                anim_folder=f"bin/pokemon/{mid}/anm/",
                extras={"dex": dex},
            )
        return [refs[k] for k in sorted(refs)]

    def _scan_extracted(self, dirs: list[str], rx: re.Pattern, kind: str) -> list[ModelRef]:
        refs: dict[str, ModelRef] = {}
        for d in dirs:
            root = os.path.join(self.raw_dir, d)
            if not os.path.isdir(root):
                continue
            for dp, _, files in os.walk(root):
                for fn in files:
                    if not fn.endswith(".gfbmdl"):
                        continue
                    rel = os.path.relpath(os.path.join(dp, fn), root).replace("\\", "/")
                    m = rx.match(rel)
                    if not m:
                        continue
                    mid, label = m.group(1), m.group(2)
                    refs[mid] = ModelRef(
                        game=self.GAME,
                        kind=kind,
                        id=mid,
                        folder=os.path.join(root, os.path.dirname(rel)),
                        display_name=re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label).replace("_", " ").title(),
                        species=mid[:6],
                        extras={"folder": f"{mid}_{label}", "model": fn},
                    )
        return [refs[k] for k in sorted(refs)]

    def trainers(self) -> list[ModelRef]:
        """NPC trainers (`tr####_VV_name`); static rigged models, no animation data was ripped for them."""
        return self._scan_extracted(TRAINER_DIRS, TRAINER_RE, "trainer")

    def objects(self) -> list[ModelRef]:
        """Props under chara/data/ob (Poke Balls, taxi, bicycle...)."""
        return self._scan_extracted(OBJECT_DIRS, OBJECT_RE, "object")

    def all(self) -> list[ModelRef]:
        return self.pokemon() + self.trainers() + self.objects()

    def find(self, model_id: str) -> ModelRef | None:
        return next((r for r in self.all() if r.id == model_id), None)

    # -- import ---------------------------------------------------------------
    def _import_extracted(self, ref: ModelRef):
        def load_tex(stem: str) -> np.ndarray | None:
            path = os.path.join(ref.folder, stem + ".dds")
            if not os.path.isfile(path):
                return None
            with open(path, "rb") as f:
                return parse_dds(f.read())[0]

        with open(os.path.join(ref.folder, ref.extras["model"]), "rb") as f:
            data = f.read()
        return import_gfbmdl(data, None, load_tex, ref.id, ref.display_name, kind=ref.kind, game=ref.game)

    def import_ref(self, ref: ModelRef, with_anim: bool = True):
        if ref.kind != "pokemon":
            return self._import_extracted(ref)
        members = self._model_members()
        mdl_dir = f"{ref.folder}mdl/"

        def read(name: str) -> bytes | None:
            hit = members.get(mdl_dir + name)
            return hit[0].read(hit[1]) if hit else None

        def load_tex(stem: str) -> np.ndarray | None:
            data = read(stem + ".bntx")
            return parse_bntx(data)[0].rgba if data else None

        data = read(f"{ref.id}.gfbmdl")
        if data is None:
            raise FileNotFoundError(f"{ref.id}.gfbmdl")
        model = import_gfbmdl(data, read(f"{ref.id}_rare.gfbmdl"), load_tex, ref.id, ref.display_name, kind=ref.kind, game=ref.game)
        if with_anim:
            model.clips = self.clips(ref, model)
            model.default_clip = next((c.name for c in model.clips if re.search(r"wait", c.name)), model.clips[0].name if model.clips else None)
        return model

    def clips(self, ref: ModelRef, model) -> list:
        out = []
        for z, member, stem in sorted(self._anim_index().get(ref.id, []), key=lambda t: t[2]):
            m = CLIP_NAME_RE.match(member.rsplit("/", 1)[-1])
            name = m.group("name") if m and m.group("name") else None
            if name is None:
                continue  # `<id>.gfbanm` alone is the bind/rest pose animation, not a clip
            try:
                clip = clip_from_gfbanm(z.read(member), name, model.skeleton, index=len(out))
            except Exception:  # noqa: BLE001 - one broken clip must not lose the model
                continue
            if clip is not None:
                out.append(clip)
        # the default idle clip first: viewers/Blender pose the model with the first clip
        out.sort(key=lambda c: (not re.search(r"^ba10_waitA", c.name), not re.search(r"wait", c.name), c.name))
        for i, c in enumerate(out):
            c.index = i
        return out
