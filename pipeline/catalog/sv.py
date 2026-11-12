"""Catalog of Pokémon Scarlet/Violet models inside the ripped SV-* zips (same Trinity formats as Z-A)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from functools import lru_cache

import numpy as np

from ..importers.za.importer import AnimationFile, ZipSource, clip_name_from_file
from .sv_species import DLC_GEN9_FIRST_FOLDER, DLC_GEN9_FOLDER_OFFSET, national9
from .za import DEFAULT_DATA_DIR, ModelRef, ZaCatalog  # noqa: F401  (DEFAULT_DATA_DIR re-exported for the CLI)

MOTION_CC_ROOT = "chara/motion_cc_base"
# Body sets key the eye bones for the generic base face (sometimes one eye only: the Hiker's defaultwait01_loop moves the
# left pupil but not the right). In game the body layer's mask (<set>.tracl) leaves them out and the face layer
# (motion_tr/*_head00) owns them, so applying them gave crooked eyes.
BODY_SKIP_BONES = re.compile(r"^(eyescale|(left|right)_(eyeball_segoff|pupil_\d+|eyehilight_\d+))$")


@lru_cache(maxsize=1)
def motion_sets() -> dict[str, dict]:
    """Body motion plan per generic trainer (tools/gen_sv_motion_sets.py, from the game's sav_tr/*.ccdata)."""
    path = os.path.join(os.path.dirname(__file__), "sv_motion_sets.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def body_types() -> dict:
    """Body type per generic trainer + the decoded `.trbsm` tables (tools/gen_sv_motion_sets.py; see importers/za/trbsm.py)."""
    path = os.path.join(os.path.dirname(__file__), "sv_body_types.json")
    if not os.path.isfile(path):
        return {"tables": {}, "trainers": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def model_dex() -> dict[str, int]:
    """Model folder -> national dex from the game's poke_resource_table (tools/gen_sv_model_dex.py)."""
    path = os.path.join(os.path.dirname(__file__), "sv_model_dex.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class SvCatalog(ZaCatalog):
    GAME = "sv"
    # Folders equal dex numbers only up to pm0703; from pm0704 they are internal ids (pm0720 = Chespin, pm1013 =
    # Fuecoco). `sv_model_dex.json` (the game's own catalog) maps the base game; DLC Gen 9 folders follow a fixed
    # offset (sv_species.py); DLC Gen 6-8 folders absent from the table stay unnamed.
    NAME_MAX_ID = 703

    def species_dex(self, species: str) -> tuple[int, bool]:
        dex = model_dex().get(species)
        if dex is not None:
            return dex, True
        folder = int(species[2:])
        if DLC_GEN9_FIRST_FOLDER <= folder < 9000:
            return national9(folder - DLC_GEN9_FOLDER_OFFSET), True
        return super().species_dex(species)
    POKE_MODEL_ZIPS = ["SV-Poke.zip", "SV-PokeDLC1.zip", "SV-PokeDLC2.zip"]
    POKE_ANIM_ZIPS = ["SV-PokeAnim.zip", "SV-PokeAnimDLC1.zip", "SV-PokeAnimDLC2.zip"]
    TRAINER_MODEL_ZIPS = [
        "SV-TrainersNPC.zip",
        "SV-TrainersNPCCommon.zip",
        "SV-TrainersDLC1.zip",
        "SV-TrainersDLC2.zip",
        "SV-TrainersPlayer.zip",
        "SV-Misc.zip",
    ]
    TRAINER_ANIM_ZIPS = ["SV-TrainersAnim.zip"]  # built from the "SV Every File" 7z: chara/motion_* only
    TRAINER_SHARED_TEX_ZIPS = ["SV-TrainersSharedTex.zip"]
    POKE_RE = re.compile(r"^pokemon/data/(pm\d{4})/(pm\d{4}_\d{2}_\d{2})/\2\.trmdl$")
    TRAINER_RE = re.compile(r"^chara/model_(?:uq|tr)/(tr\d{4}_[^/]+)/(tr\d{4}_\d{2})\.trmdl$")
    OBJECT_RE = re.compile(r"^chara/model_ob/(ob\d{4}_([^/]+))/(ob\d{4}_\d{2})\.trmdl$")
    POKE_ROOT = "pokemon/data"
    TRAINER_ANIM_ROOT = ""
    OBJECT_ANIM_ROOT = ""
    OBJECT_DEMO_ROOT = ""
    CC_MODEL_ZIPS: list[str] = []  # SV's modular NPCs (model_vr) are not catalogued
    PC_MODEL_ZIPS: list[str] = []  # SV's player parts (model_pc, motion_pc) are not catalogued

    def _anim_folders(self) -> set[str]:
        if not hasattr(self, "_anim_folder_cache"):
            found: set[str] = set()
            for z in self.zips(self.TRAINER_ANIM_ZIPS):
                for n in z.namelist():
                    parts = n.split("/")
                    if len(parts) > 3 and parts[1].startswith("motion_"):
                        found.add("/".join(parts[:3]) + "/")
            self._anim_folder_cache = found
        return self._anim_folder_cache

    def _with_motion(self, refs):
        # model_uq/model_tr/model_ob/<folder> pairs with motion_uq/motion_tr/motion_ob/<folder>
        have = self._anim_folders()
        for r in refs:
            cand = r.folder.replace("/model_", "/motion_", 1).rsplit("/", 2)[0] + f"/{r.extras['folder']}/"
            if cand in have:
                r.anim_folder = cand
            plan = motion_sets().get(r.id) if r.kind == "trainer" else None
            if plan:
                r.extras["sv_motion"] = plan
            bt = body_types()
            w = bt["trainers"].get(r.id) if r.kind == "trainer" else None
            if w and w["table"] in bt["tables"]:
                r.extras["sv_body"] = {"table": bt["tables"][w["table"]], "type": w["table"], "weights": w}
        self._variant_donors(refs)
        return refs

    @staticmethod
    def _class_name(ref: ModelRef) -> str:
        return ref.extras.get("folder", "").split("_", 1)[-1]

    def _variant_donors(self, refs) -> None:
        """Alternate versions of a character without motion data borrow the original's clips.

        The DLC re-models (Team Star in school uniform `tr1100_sfire` <- `tr0280_fire`, `tr1140_seviil` <- `tr0320_eviil`;
        `tr1900_primitiveo` <- `tr0000_primitive`) are named after the original's folder plus a letter, and the
        dump has no motion for them. Candidate = a model_uq trainer with motion whose class name is
        this one's plus one letter; the rig check happens when the clips are loaded (`_borrowed_animations`)."""
        donors = {self._class_name(r): r for r in refs if r.kind == "trainer" and r.anim_folder and "/model_uq/" in r.folder}
        for r in refs:
            if r.kind != "trainer" or r.anim_folder or r.extras.get("sv_motion") or "/model_uq/" not in r.folder:
                continue
            own = self._class_name(r)
            # Exactly one letter added in front or behind (`s` = school uniform, `o` = ...): the Blueberry `bb*` staff
            # (`bbfire`, `bbprincipal`) are different people who only share a role word.
            hits = sorted((c for c in donors if len(c) >= 2 and len(own) == len(c) + 1 and (own[1:] == c or own[:-1] == c)),
                          key=lambda c: donors[c].id)
            if hits:
                r.extras["clip_donor"] = donors[hits[0]].id
                r.extras["clip_donor_folder"] = donors[hits[0]].anim_folder

    def trainers(self):
        return self._with_motion(super().trainers())

    def objects(self):
        return self._with_motion(super().objects())

    def animations(self, ref: ModelRef) -> list[AnimationFile]:
        if ref.extras.get("clip_donor") and not ref.anim_folder:
            return self._borrowed_animations(ref)
        out = super().animations(ref)
        plan = ref.extras.get("sv_motion")
        if plan:
            out = merge_layers(out, self._body_animations(plan))
        return out

    BORROW_MIN_SHARED = 0.5  # borrower must have at least half of the donor's joints (Team Star alternates: 0.6-0.8)
    BORROW_BIND_TOL = 0.002  # metres; shared joints whose bind offset differs more keep rotations only

    def _borrowed_animations(self, ref: ModelRef) -> list[AnimationFile]:
        """The donor's clips (see `_variant_donors`) if the rigs agree: same parents on every shared bone and at least
        BORROW_MIN_SHARED of the donor's joints present. Bones whose bind offset differs (Fire's `foot_base` 22 mm,
        Fairy's `left_arm_01` 27 mm) keep rotations only; the borrower's own outfit bones stay at rest."""
        from ..rigs import compare, trinity_skeleton

        donor = next((r for r in self.trainers() if r.id == ref.extras["clip_donor"]), None)
        if donor is None:
            return []
        sa, sb = self.model_source(ref), self.model_source(donor)
        a = trinity_skeleton(sa.read, sa.exists, ref.extras.get("stem", ref.id))
        b = trinity_skeleton(sb.read, sb.exists, donor.extras.get("stem", donor.id))
        if a is None or b is None:
            return []
        cmp = compare(a, b)
        bones = [p for p in cmp.parent_mismatch if not re.search(r"_mesh(_lod\d+)?$", p)]
        if bones or len(cmp.shared) < self.BORROW_MIN_SHARED * cmp.b_joints:
            return []
        ia, ib = {j.name: j for j in a.joints}, {j.name: j for j in b.joints}
        diff = frozenset(n for n in cmp.shared
                         if np.linalg.norm(np.subtract(ia[n].translation, ib[n].translation)) > self.BORROW_BIND_TOL)
        files = super().animations(replace(ref, anim_folder=donor.anim_folder))
        for af in files:
            af.rotation_only = diff or None
        ref.extras["borrowedFrom"] = donor.id
        return files

    def _body_animations(self, plan: dict) -> list[AnimationFile]:
        """Generic trainers: body clips from motion_cc_base (own folder, shared personality sets, single clips).

        The trainer's own full set (Hiker, Chef...) or else its first shared set keeps bare clip names; the other
        shared sets are alternatives with the same indexes, tagged " [pos_f]" (index + k * 100000, like ZA cc_*).
        """
        zips = self.zips(self.TRAINER_ANIM_ZIPS)
        out: list[AnimationFile] = []
        if plan.get("own"):
            out += _folder_clips(ZipSource(zips, f"{MOTION_CC_ROOT}/{plan['own']}/"))
        has_full_own = any(af.name.startswith("defaultwait") for af in out)
        for k, folder in enumerate(plan.get("sets", []), start=1 if has_full_own else 0):
            tag = folder.split("_", 1)[1]
            out += _folder_clips(ZipSource(zips, f"{MOTION_CC_ROOT}/{folder}/"), index_offset=k * 100000, group=tag if k else "")
        for ref in plan.get("clips", []):
            folder, stem = ref.split("/", 1)
            out += _folder_clips(ZipSource(zips, f"{MOTION_CC_ROOT}/{folder}/"), only=stem)
        return out


def _folder_clips(src: ZipSource, index_offset: int = 0, group: str = "", only: str | None = None) -> list[AnimationFile]:
    out = []
    # own folders mix the trainer's clips with a few copied set clips (tr0510_pc: cc0011_cla_f_*): trainer's first
    for n in sorted(src.list(), key=lambda n: (os.path.basename(n).startswith("cc"), n)):
        if "/" in n or not n.endswith(".tranm") or (only and n[: -len(".tranm")] != only):
            continue
        parsed = clip_name_from_file(n)
        if not parsed:
            continue
        name, idx = parsed
        cfg = n[: -len(".tranm")] + ".tracm"
        out.append(
            AnimationFile(
                name=name, index=idx + index_offset, data=src.read(n), config=src.read(cfg) if src.exists(cfg) else None,
                group=group, skip_bones=BODY_SKIP_BONES,
            )
        )
    return out


def merge_layers(layers: list[AnimationFile], body: list[AnimationFile]) -> list[AnimationFile]:
    """Body clips + the motion_tr layer clips (face/cloth). A layer clip with the same index and name as a body clip
    (motion_tr `04505_art01_loop` = cloth, motion_cc_base `04505_art01_loop` = body) becomes that clip's `extra`
    (the importer adds the bones the body clip lacks); first body clip per (index, name) wins."""
    out: list[AnimationFile] = []
    by_key: dict[tuple[int, str], AnimationFile] = {}
    for af in body:
        key = (af.index, af.name)
        if key not in by_key:
            by_key[key] = af
            out.append(af)
    for af in layers:
        b = by_key.get((af.index, af.name))
        if b is None:
            out.append(af)
        elif af.data:
            b.extra = af.data
            b.config = b.config or af.config
    return out
