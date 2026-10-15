"""Catalog of Pokémon Legends: Z-A models inside the ripped zip archives."""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache

from ..importers.za.composite import CcPart, import_cc_model, part_label
from ..importers.za.importer import AnimationFile, ZipSource, clip_name_from_file, import_model

DEFAULT_DATA_DIR = os.environ.get("POKEMODELS_DATA") or os.path.join(os.path.expanduser("~"), "Documents", "MEGA downloads", "PokéModels")

POKE_MODEL_ZIPS = ["ZA-Poke.zip", "ZA-PokeDLC.zip"]
POKE_ANIM_ZIPS = ["ZA-PokeAnim.zip", "ZA-PokeAnimDLC.zip"]
TRAINER_MODEL_ZIPS = ["ZA-TrainersNPC.zip", "ZA-TrainersNPCDLC.zip"]
TRAINER_ANIM_ZIPS = ["ZA-TrainersAnim.zip", "ZA-TrainersAnimDLC.zip"]
TRAINER_SHARED_TEX_ZIPS = ["ZA-TrainersSharedTex.zip", "ZA-TrainersSharedTexDLC.zip"]

POKE_RE = re.compile(r"^ik_pokemon/data/(pm\d{4})/(pm\d{4}_\d{2}_\d{2})/\2\.trmdl$")
TRAINER_UQ_RE = re.compile(r"^ik_chara/model_uq/(tr\d{4}_\d{2}_[^/]+)/(tr\d{4}_\d{2})\.trmdl$")
BALL_RE = re.compile(r"^ik_chara/model_ob/(ob02\d{2}_\d{2}_([a-z]+ball))/(ob02\d{2}_\d{2})\.trmdl$")
CC_BASE_RE = re.compile(r"^ik_chara/model_cc_base/(\w+)/\1_base\.trmdl$")
CC_PART_RE = re.compile(
    r"^ik_chara/model_cc_vr/(?P<grp>drs|face|hrs)/(?P<base>[a-z]{2})_(?P=grp)(?P<num>\d{4})_(?P<var>\d{2})_(?P<name>[^/]+)"
    r"/(?P=base)_(?P=grp)(?P=num)_(?P=var)\.trmdl$"
)
CC_BASE_NAMES = {"bu": "Kid", "dm": "Adult M", "em": "Stocky M", "fm": "Fit M", "gm": "Tall M", "rv": "Rival", "df": "Adult F", "gf": "Tall F", "ff": "Fit F"}
# The female bases are not in the rip but share the male rig (identical bind positions and clip bone
# values: df = dm, gf = gm, ff = fm).
CC_SKELETON_DONORS = {"df": "dm", "gf": "gm", "ff": "fm"}
BALL_NAMES = {"monsterball": "Poké Ball", "masterball": "Master Ball", "ultraball": "Ultra Ball", "greatball": "Great Ball"}


@dataclass
class ModelRef:
    game: str
    kind: str  # pokemon | trainer
    id: str  # model id used for filenames (pm0001_00_00 / tr0003_00)
    folder: str  # zip-internal folder prefix
    display_name: str
    species: str = ""  # pm0001 / tr0003
    form: str = ""
    anim_folder: str = ""
    extras: dict = field(default_factory=dict)


class ZaCatalog:
    """Z-A layout; `SvCatalog` overrides the class attributes below."""

    GAME = "za"
    POKE_MODEL_ZIPS = POKE_MODEL_ZIPS
    POKE_ANIM_ZIPS = POKE_ANIM_ZIPS
    TRAINER_MODEL_ZIPS = TRAINER_MODEL_ZIPS
    TRAINER_ANIM_ZIPS = TRAINER_ANIM_ZIPS
    TRAINER_SHARED_TEX_ZIPS = TRAINER_SHARED_TEX_ZIPS
    POKE_RE = POKE_RE
    TRAINER_RE = TRAINER_UQ_RE
    OBJECT_RE = BALL_RE
    POKE_ROOT = "ik_pokemon/data"
    TRAINER_ANIM_ROOT = "ik_chara/motion_uq"  # "" = trainers have no clips
    CC_MODEL_ZIPS = ["ZA-TrainersNPCCommon.zip"]  # [] = no modular generic NPCs
    CC_ANIM_ROOT = "ik_chara/motion_cc_base"

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self._zips: dict[str, zipfile.ZipFile] = {}
        self.names = _load_names()

    # -- zips -----------------------------------------------------------
    def zip(self, name: str) -> zipfile.ZipFile | None:
        if name not in self._zips:
            path = os.path.join(self.data_dir, name)
            self._zips[name] = zipfile.ZipFile(path) if os.path.isfile(path) else None
        return self._zips[name]

    def zips(self, names: list[str]) -> list[zipfile.ZipFile]:
        return [z for z in (self.zip(n) for n in names) if z is not None]

    # -- discovery -------------------------------------------------------
    def pokemon(self) -> list[ModelRef]:
        refs: dict[str, ModelRef] = {}
        for z in self.zips(self.POKE_MODEL_ZIPS):
            for n in z.namelist():
                m = self.POKE_RE.match(n)
                if not m:
                    continue
                species, mid = m.group(1), m.group(2)
                dex = int(species[2:])
                form = mid[7:]  # "00_00"
                base = self.names.get(str(dex), species)
                display = base if form == "00_00" else f"{base} ({form})"
                refs[mid] = ModelRef(
                    game=self.GAME,
                    kind="pokemon",
                    id=mid,
                    folder=f"{self.POKE_ROOT}/{species}/{mid}/",
                    display_name=display,
                    species=species,
                    form=form,
                    anim_folder=f"{self.POKE_ROOT}/{species}/{mid}/",
                    extras={"dex": dex},
                )
        return [refs[k] for k in sorted(refs)]

    def trainers(self) -> list[ModelRef]:
        refs: dict[str, ModelRef] = {}
        for z in self.zips(self.TRAINER_MODEL_ZIPS):
            for n in z.namelist():
                m = self.TRAINER_RE.match(n)
                if not m:
                    continue
                folder_name, mid = m.group(1), m.group(2)
                label = re.sub(r"^tr\d{4}_(\d{2}_)?", "", folder_name).replace("_", " ").title() or folder_name
                refs[mid] = ModelRef(
                    game=self.GAME,
                    kind="trainer",
                    id=mid,
                    folder=n[: n.rindex("/") + 1],
                    display_name=label,
                    species=mid[:6],
                    anim_folder=f"{self.TRAINER_ANIM_ROOT}/{folder_name}/" if self.TRAINER_ANIM_ROOT else "",
                    extras={"folder": folder_name},
                )
        return [refs[k] for k in sorted(refs)] + self.cc_models()

    # -- modular generic NPCs (model_cc_*) ----------------------------------------
    def _cc_parts_by_base(self) -> dict[str, list[CcPart]]:
        found: dict[str, dict[str, CcPart]] = {}
        for z in self.zips(self.CC_MODEL_ZIPS):
            for n in z.namelist():
                m = CC_PART_RE.match(n)
                if not m:
                    continue
                base, kind, num, var, name = m["base"], m["grp"], m["num"], m["var"], m["name"]
                pid = f"{base}_{kind}{num}_{var}"
                found.setdefault(base, {})[pid] = CcPart(group=kind, id=pid, label=part_label(name, int(var)), folder=n[: n.rindex("/") + 1])
        out: dict[str, list[CcPart]] = {}
        for base, parts in found.items():
            lst = sorted(parts.values(), key=lambda p: p.id)
            for g in {p.group for p in lst}:
                next(p for p in lst if p.group == g).default = True
            out[base] = lst
        return out

    def cc_models(self) -> list[ModelRef]:
        """One composite per base body that has a skeleton and parts."""
        if not self.CC_MODEL_ZIPS:
            return []
        parts = self._cc_parts_by_base()
        refs = []
        anim_zips = self.zips(self.TRAINER_ANIM_ZIPS)
        for z in self.zips(self.CC_MODEL_ZIPS):
            for n in z.namelist():
                m = CC_BASE_RE.match(n)
                if not m:
                    continue
                skel = m.group(1)
                # Female parts (df/gf/ff) have no base skeleton in the rip: they bind to the
                # same 74-joint rig as the matching male base (CC_SKELETON_DONORS).
                for b in [p for p in parts if CC_SKELETON_DONORS.get(p, p) == skel]:
                    anim_b = b if any(a.startswith(f"{self.CC_ANIM_ROOT}/{b}/") for az in anim_zips for a in az.namelist()) else skel
                    refs.append(
                        ModelRef(
                            game=self.GAME,
                            kind="trainer",
                            id=f"cc_{b}",
                            folder=n[: n.rindex("/") + 1],
                            display_name=f"NPC {CC_BASE_NAMES.get(b, b.upper())}",
                            species=f"cc_{b}",
                            anim_folder=f"{self.CC_ANIM_ROOT}/{anim_b}/",
                            extras={"folder": b, "cc_base": b, "cc_skeleton": skel},
                        )
                    )
        return sorted(refs, key=lambda r: r.id)

    def objects(self) -> list[ModelRef]:
        """Held items: the Poké Ball models under model_ob."""
        refs: dict[str, ModelRef] = {}
        for z in self.zips(self.TRAINER_MODEL_ZIPS):
            for n in z.namelist():
                m = self.OBJECT_RE.match(n)
                if not m:
                    continue
                folder_name, ball, mid = m.group(1), m.group(2), m.group(3)
                label = BALL_NAMES.get(ball) or (ball[: -len("ball")].title() + " Ball" if ball.endswith("ball") else ball.replace("_", " ").title())
                refs[mid] = ModelRef(
                    game=self.GAME,
                    kind="object",
                    id=mid,
                    folder=n[: n.rindex("/") + 1],
                    display_name=label,
                    species=mid[:6],
                    extras={"folder": folder_name, "item": ball},
                )
        return [refs[k] for k in sorted(refs)]

    def all(self) -> list[ModelRef]:
        return self.pokemon() + self.trainers() + self.objects()

    def find(self, model_id: str) -> ModelRef | None:
        for r in self.all():
            if r.id == model_id:
                return r
        return None

    def import_ref(self, ref: ModelRef, with_anim: bool = True):
        anims = self.animations(ref) if with_anim else []
        if ref.extras.get("cc_base"):
            b = ref.extras["cc_base"]
            zips = self.CC_MODEL_ZIPS + self.TRAINER_SHARED_TEX_ZIPS
            return import_cc_model(
                ZipSource(self.zips(zips), ref.folder),
                f"{ref.extras['cc_skeleton']}_base",
                self._cc_parts_by_base()[b],
                lambda p: ZipSource(self.zips(zips), p.folder),
                ref.id,
                ref.display_name,
                animations=anims,
                game=ref.game,
            )
        return import_model(self.model_source(ref), ref.id, ref.display_name, kind=ref.kind, animations=anims, game=ref.game)

    # -- sources -----------------------------------------------------------
    def model_source(self, ref: ModelRef) -> ZipSource:
        zips = self.POKE_MODEL_ZIPS if ref.kind == "pokemon" else self.TRAINER_MODEL_ZIPS + self.TRAINER_SHARED_TEX_ZIPS
        return ZipSource(self.zips(zips), ref.folder)

    def animations(self, ref: ModelRef) -> list[AnimationFile]:
        if not ref.anim_folder:
            return []
        zips = self.POKE_ANIM_ZIPS if ref.kind == "pokemon" else self.TRAINER_ANIM_ZIPS
        src = ZipSource(self.zips(zips), ref.anim_folder)
        if ref.extras.get("cc_base"):
            return self._cc_animations(src)
        out = []
        for n in src.list():
            base = os.path.basename(n)
            if "/" not in n and base.endswith(".tracm") and ("visoff" in base or "vison" in base) and not src.exists(n[:-6] + ".tranm"):
                parsed = clip_name_from_file(base[:-6] + ".tranm")
                if parsed:
                    out.append(AnimationFile(name=parsed[0], index=parsed[1], data=b"", config=src.read(n)))
                continue
            if "/" in n or not base.endswith(".tranm"):
                continue
            parsed = clip_name_from_file(base)
            if parsed:
                name, idx = parsed
            else:
                name, idx = os.path.splitext(base)[0], 99999
            cfg = n[: -len(".tranm")] + ".tracm"
            out.append(AnimationFile(name=name, index=idx, data=src.read(n), config=src.read(cfg) if src.exists(cfg) else None))
        return out


    @staticmethod
    def _cc_animations(src: ZipSource) -> list[AnimationFile]:
        """Base bodies animate from one folder per motion set (`cc0003_adl_m`, `cc0005_cla_m`...).

        The first folder keeps bare clip names, the others are suffixed ` [adl_m]` and grouped.
        """
        folders = sorted({n.split("/")[0] for n in src.list() if "/" in n})
        out = []
        for k, folder in enumerate(folders):
            tag = folder.split("_", 1)[1] if "_" in folder else folder
            for n in src.list():
                if not n.startswith(folder + "/") or not n.endswith(".tranm"):
                    continue
                parsed = clip_name_from_file(n)
                if not parsed:
                    continue
                name, idx = parsed
                cfg = n[: -len(".tranm")] + ".tracm"
                out.append(
                    AnimationFile(
                        name=name,
                        index=idx + k * 100000,
                        data=src.read(n),
                        config=src.read(cfg) if src.exists(cfg) else None,
                        group=tag if k else "",
                    )
                )
        return out


@lru_cache(maxsize=1)
def _load_names() -> dict[str, str]:
    path = os.path.join(os.path.dirname(__file__), "pokemon_names.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}
