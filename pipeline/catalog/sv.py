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
    TRAINER_ANIM_ZIPS: list[str] = []  # no SV trainer motion data was ripped
    TRAINER_SHARED_TEX_ZIPS = ["SV-TrainersSharedTex.zip"]
    POKE_RE = re.compile(r"^pokemon/data/(pm\d{4})/(pm\d{4}_\d{2}_\d{2})/\2\.trmdl$")
    TRAINER_RE = re.compile(r"^chara/model_(?:uq|tr)/(tr\d{4}_[^/]+)/(tr\d{4}_\d{2})\.trmdl$")
    OBJECT_RE = re.compile(r"^chara/model_ob/(ob\d{4}_([^/]+))/(ob\d{4}_\d{2})\.trmdl$")
    POKE_ROOT = "pokemon/data"
    TRAINER_ANIM_ROOT = ""
    CC_MODEL_ZIPS: list[str] = []  # SV's modular NPCs (model_vr) are not catalogued
