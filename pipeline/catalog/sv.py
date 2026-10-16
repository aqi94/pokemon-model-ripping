"""Catalog of Pokémon Scarlet/Violet models inside the ripped SV-* zips (same Trinity formats as Z-A)."""

from __future__ import annotations

import re

from .za import DEFAULT_DATA_DIR, ZaCatalog  # noqa: F401  (DEFAULT_DATA_DIR re-exported for the CLI)


class SvCatalog(ZaCatalog):
    GAME = "sv"
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
    CC_MODEL_ZIPS: list[str] = []  # SV's modular NPCs (model_vr) are not catalogued

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
        return refs

    def trainers(self):
        return self._with_motion(super().trainers())

    def objects(self):
        return self._with_motion(super().objects())
