"""Catalog of Pokémon Legends: Arceus models (Trinity formats, same as Z-A / SV, different folder layout).

LA-Poke.zip : bin/pokemon/pm####/pm####_FF_VV/mdl/pm####_FF_VV.{trmdl,trmsh,trmbf,trskl,trmtr,trmmt,*.bntx}
LA-PokeAnim : bin/pokemon/pm####/pm####_FF_VV/anm/pm####_FF_VV_#####_<name>.{tranm,tracm,traef}
(`bin/chara/data/pm/...` holds 5 older duplicates of some models; the `bin/pokemon` tree is the one used.)
"""

from __future__ import annotations

import os
import re

from ..importers.za.importer import FolderSource, import_model
from .za import DEFAULT_DATA_DIR, ModelRef, ZaCatalog  # noqa: F401  (DEFAULT_DATA_DIR re-exported for the CLI)

# LA-Trainers.7z / LA-Misc.7z extracted once with py7zr (`trainers/chara/data/{tr,cc,pc}`, `misc/chara/data/item`).
LA_EXTRACT_DIR = os.environ.get("POKEMODELS_LA_DATA") or os.path.join(os.path.expanduser("~"), "Documents", "pokemon_raw_data", "la")
TRAINER_DIR_RE = re.compile(r"^(tr\d{4}_\d{2})_(.+)$")
ITEM_DIR_RE = re.compile(r"^item_(\d{3})$")

# LA's seven new species are stored as pm1001..pm1007 = national dex 899..905 (Wyrdeer .. Enamorus).
LA_NEW_SPECIES_BASE = 1000 - 898
HISUIAN_DEX = {58, 59, 100, 101, 157, 211, 215, 503, 549, 570, 571, 628, 705, 706, 713, 724}  # `_41` suffix
UNOWN_LETTERS = [chr(ord("A") + i) for i in range(26)] + ["!", "?"]


# LA numbers some Gen 6/7 species by an internal id (verified by rendering each model).
LA_INTERNAL_TO_DEX = {749: 712, 751: 713, 763: 704, 764: 705, 765: 706, 767: 700, 841: 722, 842: 723, 843: 724}


def dex_of(species: str) -> int:
    n = int(species[2:])
    if n >= 1001:
        return n - LA_NEW_SPECIES_BASE
    return LA_INTERNAL_TO_DEX.get(n, n)


def form_label(dex: int, form: str) -> str:
    """Best-effort label for a `FF_VV` suffix (`00_00` = default)."""
    if form == "00_00":
        return ""
    ff, vv = form.split("_")
    if dex == 201 and ff.isdigit() and 11 <= int(ff) <= 38:
        return UNOWN_LETTERS[int(ff) - 11]
    if dex == 550 and form == "13_41":
        return "White-Striped"
    parts = []
    if vv == "41":
        parts.append("Hisuian" if dex in HISUIAN_DEX else "variant 41")
    elif vv != "00":
        parts.append(f"variant {vv}")
    if ff == "71":
        parts.append("Noble")
    elif ff == "01":
        parts.append("Female")
    elif ff != "00":
        parts.append(f"form {ff}")
    return ", ".join(parts)


class LaCatalog(ZaCatalog):
    GAME = "la"
    POKE_MODEL_ZIPS = ["LA-Poke.zip"]
    POKE_ANIM_ZIPS = ["LA-PokeAnim.zip"]
    TRAINER_MODEL_ZIPS: list[str] = []
    TRAINER_ANIM_ZIPS: list[str] = []
    TRAINER_SHARED_TEX_ZIPS: list[str] = []
    POKE_RE = re.compile(r"^bin/pokemon/(pm\d{4})/(pm\d{4}_\d{2}_\d{2})/mdl/\2\.trmdl$")
    TRAINER_RE = re.compile(r"^$a")  # never matches
    OBJECT_RE = re.compile(r"^$a")
    POKE_ROOT = "bin/pokemon"
    TRAINER_ANIM_ROOT = ""
    CC_MODEL_ZIPS: list[str] = []

    def pokemon(self) -> list[ModelRef]:
        refs: dict[str, ModelRef] = {}
        for z in self.zips(self.POKE_MODEL_ZIPS):
            for n in z.namelist():
                m = self.POKE_RE.match(n)
                if not m:
                    continue
                species, mid = m.group(1), m.group(2)
                dex = dex_of(species)
                form = mid[7:]
                base = self.names.get(str(dex), species)
                label = form_label(dex, form)
                refs[mid] = ModelRef(
                    game=self.GAME,
                    kind="pokemon",
                    id=mid,
                    folder=f"{self.POKE_ROOT}/{species}/{mid}/mdl/",
                    display_name=f"{base} ({label or form})" if form != "00_00" else base,
                    species=species,
                    form=form,
                    anim_folder=f"{self.POKE_ROOT}/{species}/{mid}/anm/",
                    extras={"dex": dex},
                )
        return [refs[k] for k in sorted(refs)]

    def animations(self, ref: ModelRef):
        # `anm/<model>.tranm` (no 5-digit index) is a template shared across models, not a named clip.
        return [a for a in super().animations(ref) if a.index != 99999]

    # -- extracted 7z content (trainers, items): plain folders, no animation data ----------------
    extract_dir = LA_EXTRACT_DIR

    def trainers(self) -> list[ModelRef]:
        root = os.path.join(self.extract_dir, "trainers", "chara", "data", "tr")
        refs = []
        for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            m = TRAINER_DIR_RE.match(d)
            if not m or not os.path.isfile(os.path.join(root, d, d + ".trmdl")):
                continue
            refs.append(
                ModelRef(
                    game=self.GAME,
                    kind="trainer",
                    id=m.group(1),
                    folder=os.path.join(root, d),
                    display_name=m.group(2).replace("_", " ").title(),
                    species=m.group(1)[:6],
                    extras={"folder": d, "stem": d},
                )
            )
        return refs

    def objects(self) -> list[ModelRef]:
        root = os.path.join(self.extract_dir, "misc", "chara", "data", "item")
        refs = []
        for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            if not ITEM_DIR_RE.match(d) or not os.path.isfile(os.path.join(root, d, d + ".trmdl")):
                continue
            refs.append(
                ModelRef(
                    game=self.GAME,
                    kind="object",
                    id=d,
                    folder=os.path.join(root, d),
                    display_name=f"Item {d[5:]}",
                    species=d,
                    extras={"folder": d, "stem": d},
                )
            )
        return refs

    def import_ref(self, ref: ModelRef, with_anim: bool = True):
        if ref.kind == "pokemon":
            return super().import_ref(ref, with_anim)
        model = import_model(FolderSource(ref.folder), ref.extras["stem"], ref.display_name, kind=ref.kind, game=ref.game)
        model.id = ref.id
        return model
