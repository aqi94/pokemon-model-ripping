"""Catalog of Pokémon Legends: Z-A models inside the ripped zip archives."""

from __future__ import annotations

import json
import os
import re
import zipfile
import dataclasses
from dataclasses import dataclass, field
from functools import lru_cache

from ..importers.za.composite import CcPart, apply_appearance, import_cc_model, part_label
from ..importers.za.form_layers import apply_form_layer, is_layer_candidate, select_form_layers
from ..importers.za.trvat import attach_vat_clips
from .za_demo import demo_animations
from ..importers.za.importer import AnimationFile, ZipSource, clip_name_from_file, import_model

DEFAULT_DATA_DIR = os.environ.get("POKEMODELS_DATA") or os.path.join(os.path.expanduser("~"), "Documents", "MEGA downloads", "PokéModels")

POKE_MODEL_ZIPS = ["ZA-Poke.zip", "ZA-PokeDLC.zip"]
POKE_ANIM_ZIPS = ["ZA-PokeAnim.zip", "ZA-PokeAnimDLC.zip"]
TRAINER_MODEL_ZIPS = ["ZA-TrainersNPC.zip", "ZA-TrainersNPCDLC.zip"]
TRAINER_ANIM_ZIPS = ["ZA-TrainersAnim.zip", "ZA-TrainersAnimDLC.zip"]
TRAINER_SHARED_TEX_ZIPS = ["ZA-TrainersSharedTex.zip", "ZA-TrainersSharedTexDLC.zip"]

POKE_RE = re.compile(r"^ik_pokemon/data/(pm\d{4})/(pm\d{4}_\d{2}_\d{2})/\2\.trmdl$")
TRAINER_UQ_RE = re.compile(r"^ik_chara/model_uq/(tr\d{4}_\d{2}_[^/]+)/(tr\d{4}_\d{2})\.trmdl$")
# Every model_ob prop: the 27 Poké Balls (ob02NN), key items, café food and DLC props. ob9999 is a joint locator.
BALL_RE = re.compile(
    r"^ik_chara/model_ob/((?:ob(?!9999)\d{4}_\d{2}|po\d{4}_\d{2}_\d{2})(?:_([a-z]+))?)/(ob\d{4}_\d{2}|po\d{4}_\d{2}_\d{2})\.trmdl$"
)
# Standalone NPCs built on a cc base (rival tr0001/tr0002, Co members tr0030/tr0031): ZA-TrainersNPCCommon, clips in motion_cc_ir.
CC_IR_RE = re.compile(r"^ik_chara/model_cc_ir/(tr\d{4}_\d{2}_[^/]+)/(tr\d{4}_\d{2})\.trmdl$")
CC_BASE_RE = re.compile(r"^ik_chara/model_cc_base/(\w+)/\1_base\.trmdl$")
CC_PART_RE = re.compile(
    r"^ik_chara/model_cc_vr/(?P<grp>drs|face|hrs)/(?P<base>[a-z]{2})_(?P=grp)(?P<num>\d{4})_(?P<var>\d{2})_(?P<name>[^/]+)"
    r"/(?P=base)_(?P=grp)(?P=num)_(?P=var)\.trmdl$"
)
CC_BASE_NAMES = {"bu": "Kid", "dm": "Adult M", "em": "Stocky M", "fm": "Fit M", "gm": "Tall M", "rv": "Rival", "df": "Adult F", "gf": "Tall F", "ff": "Fit F"}
# The female bases are not in the rip but share the male rig (identical bind positions and clip bone
# values: df = dm, gf = gm, ff = fm).
CC_SKELETON_DONORS = {"df": "dm", "gf": "gm", "ff": "fm"}
# cc_ir NPCs whose motion_cc_ir folder only holds face layers (Co members, the Rust Syndicate grunts): their body clips
# are the matching adult base's sets (verified: skeleton = dm_base, df shares that rig).
CC_IR_BODY_DONORS = {"tr0030_00": "dm", "tr0031_00": "df"}
# Colour variants that share the base prop's rig and use its clips (ob0009_01 is the red flower, ob0009_00 the blue one).
OBJECT_CLIP_DONORS = {"ob0009_01": "ob0009_00"}
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
    # Z-A ids above Genesect are internal species numbers, not national dex (no table in the rip): keep the raw id.
    NAME_MAX_ID = 649
    TRAINER_ANIM_ROOT = "ik_chara/motion_uq"  # "" = trainers have no clips
    CC_MODEL_ZIPS = ["ZA-TrainersNPCCommon.zip"]  # [] = no modular generic NPCs
    CC_ANIM_ROOT = "ik_chara/motion_cc_base"
    CC_IR_ANIM_ROOT = "ik_chara/motion_cc_ir"  # "" = none
    OBJECT_ANIM_ROOT = "ik_chara/motion_ob"  # "" = props have no clips
    OBJECT_DEMO_ROOT = "ik_chara/demo_ob"

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
                known = dex <= self.NAME_MAX_ID
                base = self.names.get(str(dex), species) if known else species
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
                    extras={"dex": dex if known else None},
                )
        return self._split_form_layers([refs[k] for k in sorted(refs)])

    def _split_form_layers(self, refs: list[ModelRef]) -> list[ModelRef]:
        """One model per form layer (importers/za/form_layers.py; Mega Tatsugiri): `<id>_loopNN`."""
        cands: dict[str, dict[str, str]] = {}  # anim folder -> clip source -> zip member stem
        for z in self.zips(self.POKE_ANIM_ZIPS):
            for n in z.namelist():
                if not n.endswith("_loop.tranm"):
                    continue
                folder, base = n[: n.rindex("/") + 1], n[n.rindex("/") + 1 :]
                parsed = clip_name_from_file(base)
                if parsed and is_layer_candidate(*parsed):
                    cands.setdefault(folder, {})[f"{parsed[1]:05d}_{parsed[0]}"] = base[: -len(".tranm")]
        out: list[ModelRef] = []
        for ref in refs:
            found = cands.get(ref.anim_folder, {})
            layers: list[str] = []
            if len(found) >= 2:
                src = ZipSource(self.zips(self.POKE_ANIM_ZIPS), ref.anim_folder)
                data = {k: (src.read(v + ".tranm"), src.read(v + ".tracm") if src.exists(v + ".tracm") else None) for k, v in found.items()}
                layers = select_form_layers(data)
            if not layers:
                out.append(ref)
                continue
            for layer in layers:
                tag = layer.split("_", 1)[1].removesuffix("_loop")  # loop01
                out.append(dataclasses.replace(
                    ref,
                    id=f"{ref.id}_{tag}",
                    display_name=f"{ref.display_name} {tag}",
                    extras={**ref.extras, "stem": ref.id, "form_layer": layer, "form_layers": layers},
                ))
        return out

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
        if self.CC_IR_ANIM_ROOT:
            for z in self.zips(self.CC_MODEL_ZIPS):
                for n in z.namelist():
                    m = CC_IR_RE.match(n)
                    if not m or m.group(2) in refs:
                        continue
                    folder_name, mid = m.group(1), m.group(2)
                    label = re.sub(r"^tr\d{4}_(\d{2}_)?", "", folder_name).replace("_", " ").title()
                    refs[mid] = ModelRef(
                        game=self.GAME,
                        kind="trainer",
                        id=mid,
                        folder=n[: n.rindex("/") + 1],
                        display_name=label,
                        species=mid[:6],
                        anim_folder=f"{self.CC_IR_ANIM_ROOT}/{folder_name}/",
                        extras={"folder": folder_name, "cc_ir": True, **({"cc_donor": CC_IR_BODY_DONORS[mid]} if mid in CC_IR_BODY_DONORS else {})},
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
        """Held items and props: every model under model_ob (Poké Balls, key items, café food)."""
        refs: dict[str, ModelRef] = {}
        for z in self.zips(self.TRAINER_MODEL_ZIPS):
            for n in z.namelist():
                m = self.OBJECT_RE.match(n)
                if not m:
                    continue
                folder_name, ball, mid = m.group(1), m.group(2), m.group(3)
                ball = ball or folder_name
                label = BALL_NAMES.get(ball) or (ball[: -len("ball")].title() + " Ball" if ball.endswith("ball") else ball.replace("_", " ").title())
                refs[mid] = ModelRef(
                    game=self.GAME,
                    kind="object",
                    id=mid,
                    folder=n[: n.rindex("/") + 1],
                    display_name=label,
                    species=mid[:6],
                    anim_folder=f"{self.OBJECT_ANIM_ROOT}/{folder_name}/" if self.OBJECT_ANIM_ROOT else "",
                    extras={"folder": folder_name, "item": ball},
                )
        for mid, donor in OBJECT_CLIP_DONORS.items():
            if mid in refs and donor in refs and self.OBJECT_ANIM_ROOT:
                refs[mid].anim_folder = refs[donor].anim_folder
                refs[mid].extras["clip_id"] = donor
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
        model = import_model(self.model_source(ref), ref.extras.get("stem", ref.id), ref.display_name, kind=ref.kind, animations=anims, game=ref.game, skin_tones=bool(ref.extras.get("cc_ir")))
        if ref.extras.get("cc_ir"):
            apply_appearance(model)
        if ref.extras.get("form_layer"):
            apply_form_layer(model, ref.extras["form_layer"], ref.extras["form_layers"])
            model.id = ref.id
        if ref.kind == "trainer" and ref.anim_folder:
            # Crowd NPCs (tr05xx) animate through baked vertex textures (.trvat), see importers/za/trvat.py.
            if attach_vat_clips(model, ZipSource(self.zips(self.TRAINER_ANIM_ZIPS), ref.anim_folder)):
                model.clips.sort(key=lambda c: (c.index, c.name))
        return model

    # -- sources -----------------------------------------------------------
    def model_source(self, ref: ModelRef) -> ZipSource:
        zips = self.POKE_MODEL_ZIPS if ref.kind == "pokemon" else self.TRAINER_MODEL_ZIPS + self.TRAINER_SHARED_TEX_ZIPS
        if ref.extras.get("cc_ir"):
            zips = self.CC_MODEL_ZIPS + self.TRAINER_SHARED_TEX_ZIPS
        return ZipSource(self.zips(zips), ref.folder)

    def animations(self, ref: ModelRef) -> list[AnimationFile]:
        if not ref.anim_folder:
            return []
        zips = self.POKE_ANIM_ZIPS if ref.kind == "pokemon" else self.TRAINER_ANIM_ZIPS
        src = ZipSource(self.zips(zips), ref.anim_folder)
        if ref.extras.get("cc_base"):
            return self._cc_animations(src) + self._demo_animations(ref)
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
            if ref.kind == "object" and not base.startswith(ref.extras.get("clip_id", ref.id)):
                continue
            parsed = clip_name_from_file(base)
            if parsed:
                name, idx = parsed
            else:
                name, idx = os.path.splitext(base)[0], 99999
            cfg = n[: -len(".tranm")] + ".tracm"
            out.append(AnimationFile(name=name, index=idx, data=src.read(n), config=src.read(cfg) if src.exists(cfg) else None))
        if ref.extras.get("cc_ir"):
            self._attach_rival_body(out, zips)
        if ref.extras.get("cc_donor"):
            donor = ZipSource(self.zips(zips), f"{self.CC_ANIM_ROOT}/{ref.extras['cc_donor']}/")
            out += self._cc_animations(donor, only_sets=True)
        return out + self._demo_animations(ref)

    # The rival's body motion is shared on the player's rig: motion_cc_base/rv/<rival_f>/ has the waist/spine/limb tracks,
    # motion_cc_ir/<rival>/ only the face/hair/dummy layer for the same clip index.
    RIVAL_BODY_FOLDER = "ik_chara/motion_cc_base/rv/tr0001_00_rival_f/"

    def _attach_rival_body(self, files: list[AnimationFile], zips: list[str]) -> None:
        body = ZipSource(self.zips(zips), self.RIVAL_BODY_FOLDER)
        by_index = {}
        for n in body.list():
            if "/" not in n and n.endswith(".tranm"):
                parsed = clip_name_from_file(os.path.basename(n))
                if parsed:
                    by_index[parsed[1]] = n
        for af in files:
            if af.index in by_index:
                af.extra = body.read(by_index[af.index])

    # Cutscene clips (`ik_chara/demo_*`): ordinary bone .tranm, see catalog/za_demo.py.
    DEMO_ROOT = "ik_chara"

    def _demo_animations(self, ref: ModelRef) -> list[AnimationFile]:
        zips = self.zips(self.TRAINER_ANIM_ZIPS)
        if ref.kind == "object":
            # One prop folder serves several models (ob0005_00_mf holds ob0005_00..04): keep this model's clips.
            if not self.OBJECT_DEMO_ROOT:
                return []
            files = demo_animations(ZipSource(zips, f"{self.OBJECT_DEMO_ROOT}/{ref.extras.get('folder', '')}"))
            return [f for f in files if f"_{ref.id}_" in f"_{f.name}_"]
        if not self.DEMO_ROOT or ref.kind != "trainer":
            return []
        out: list[AnimationFile] = []
        if ref.extras.get("cc_base"):
            skel = ref.extras["cc_skeleton"]
            for sub in dict.fromkeys([ref.extras["cc_base"], skel]):
                out += demo_animations(ZipSource(zips, f"{self.DEMO_ROOT}/demo_cc_base/{sub}"))
            if ref.extras["cc_base"] == "rv":
                out += demo_animations(ZipSource(zips, f"{self.DEMO_ROOT}/demo_cc_ir"))
        else:
            out += demo_animations(ZipSource(zips, f"{self.DEMO_ROOT}/demo_uq/{ref.extras.get('folder', '')}"))
        return out

    @staticmethod
    def _cc_animations(src: ZipSource, only_sets: bool = False) -> list[AnimationFile]:
        """Base bodies animate from one folder per motion set (`cc0003_adl_m`, `cc0005_cla_m`...).

        The first folder keeps bare clip names, the others are suffixed ` [adl_m]` and grouped.
        """
        folders = sorted({n.split("/")[0] for n in src.list() if "/" in n})
        if only_sets:  # skip the per-NPC folders (tr0037_00_driver_m...), keep the shared cc#### sets
            folders = [f for f in folders if re.match(r"cc\d{4}_", f)]
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
