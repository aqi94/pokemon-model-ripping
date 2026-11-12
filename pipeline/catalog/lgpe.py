"""Catalog of Pokemon Let's Go Pikachu/Eevee models.

Sources (read in place): `LGPE-Poke.zip` (models + textures) and `LGPE-PokeAnim.zip` (clips), both with a
`bin/` root:

* `bin/pokemon/pm####_FF/mdl/pm####_FF.gfbmdl` (+ `_rare.gfbmdl`, `.bntx`) and `.../anm/pm####_FF_<clip>.gfbanm`
  (FF: 00 base, 01 female, 51/52 mega (X/Y), 61 Alolan). 209 models incl. `pm0000` (the egg).
* `bin/chara/data/bd/bd####_00_<name>/{mdl,anm}`: the partner Pikachu / Eevee (`bd0001`, `bd0002`).
  `*_col` folders are collision helpers (skipped).
* `bin/chara/data/pd/pd####_00_<name>/mdl`: player accessories (cap, hat, glasses ...), kind "object".
* Trainers: `LGPE-Trainers.7z` extracted to `pokemon_raw_data/lgpe/trainers` (see `trainers()`).
* Field maps / props: `LGPE-FieldMaps.7z` extracted to `pokemon_raw_data/lgpe/fieldmaps` (`field/<folder>/<stem>.gfbmdl`
  + DDS beside it; `field/buildmodel` = props, every other folder = maps). See `maps()` / `props()`.
* Story objects (Poke Balls, berries, gadgets): `LGPE-Misc.7z` extracted to `pokemon_raw_data/lgpe/misc`
  (`chara/data/ob/ob####_VV_<name>/`, laid out like the trainer folders), kind "object".
"""

from __future__ import annotations

import json
import os
import re
import zipfile

from ..importers.lgpe.field import import_field_model
from ..importers.lgpe.importer import LgpeAnimation, clip_name, import_model
from .za import DEFAULT_DATA_DIR, ModelRef, _load_names  # noqa: F401  (DEFAULT_DATA_DIR re-exported for the CLI)

POKE_ZIP = "LGPE-Poke.zip"
ANIM_ZIP = "LGPE-PokeAnim.zip"
_RAW = os.path.join(os.path.expanduser("~"), "Documents", "pokemon_raw_data", "lgpe")
DEFAULT_TRAINER_DIR = os.environ.get("POKEMODELS_LGPE_TRAINERS") or os.path.join(_RAW, "trainers")
DEFAULT_FIELD_DIR = os.environ.get("POKEMODELS_LGPE_FIELD") or os.path.join(_RAW, "fieldmaps")
DEFAULT_MISC_DIR = os.environ.get("POKEMODELS_LGPE_MISC") or os.path.join(_RAW, "misc")
PROP_FOLDERS = {"buildmodel"}
OB_DIR_RE = re.compile(r"^(ob\d{4})_(\d{2})_([a-z0-9]+)$")
# Display names only (no behaviour): Kanto town numbers as the game orders them, `roadNNN` = Route NN.
TOWN_NAMES = {1: "Pallet Town", 2: "Viridian City", 3: "Pewter City", 4: "Cerulean City", 5: "Vermilion City", 6: "Lavender Town",
              7: "Celadon City", 8: "Fuchsia City", 9: "Cinnabar Island", 10: "Indigo Plateau", 11: "Saffron City"}
OB_NAMES = {"monsterball": "Poké Ball", "masterball": "Master Ball", "ultraball": "Ultra Ball", "greatball": "Great Ball", "premierball": "Premier Ball", "cherishball": "Cherish Ball"}


def field_label(stem: str, folder: str) -> str:
    """`town001_00` -> "Pallet Town", `road002_1_00` -> "Route 2 (1)", others: the file stem."""
    m = re.match(r"^(town|road)(\d{3})(?:_(\d))?_00$", stem)
    if m:
        n = int(m.group(2))
        base = TOWN_NAMES.get(n, f"Town {n}") if m.group(1) == "town" else f"Route {n}"
        return base + (f" ({m.group(3)})" if m.group(3) else "")
    return stem if folder in PROP_FOLDERS else f"{stem} ({folder})"

POKE_RE = re.compile(r"^bin/pokemon/(pm\d{4})_(\d{2})/mdl/(pm\d{4}_\d{2})\.gfbmdl$")
BUDDY_RE = re.compile(r"^bin/chara/data/bd/(bd\d{4})_(\d{2})_([a-z]+)/mdl/(bd\d{4}_\d{2})\.gfbmdl$")
ACCESSORY_RE = re.compile(r"^bin/chara/data/pd/(pd\d{4})_(\d{2})_([a-z]+)/mdl/(pd\d{4}_\d{2})\.gfbmdl$")

TRAINER_DIR_RE = re.compile(r"^(tr\d{4})_(\d{2})_([a-z0-9]+)$")
TRAINER_NAMES = {"hero": "Player (boy)", "heroine": "Player (girl)", "rival": "Rival", "rocketcheif": "Rocket Chief"}

# Internal numbers that are not National Dex numbers
SPECIAL_NAMES = {0: "Egg", 891: "Meltan", 892: "Melmetal"}
BUDDY_DEX = {"bd0001": 25, "bd0002": 133}
BUDDY_NAMES = {"bd0001": "Partner Pikachu", "bd0002": "Partner Eevee"}


def form_label(form: str, forms_of_species: set[str]) -> str:
    """Display prefix/suffix for a form id. 01 = female, 51/52 = Mega (X/Y when both exist), 61 = Alolan."""
    if form == "51":
        return "Mega X" if "52" in forms_of_species else "Mega"
    return {"01": "Female", "52": "Mega Y", "61": "Alolan"}.get(form, f"form {form}")


class LgpeCatalog:
    GAME = "lgpe"

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, trainer_dir: str = DEFAULT_TRAINER_DIR, field_dir: str = DEFAULT_FIELD_DIR, misc_dir: str = DEFAULT_MISC_DIR):
        self.data_dir = data_dir
        self.trainer_dir = trainer_dir
        self.field_dir = field_dir
        self.misc_dir = misc_dir
        self._zips: dict[str, zipfile.ZipFile | None] = {}
        self._index: dict[str, dict[str, zipfile.ZipFile]] = {}
        self.names = _load_names()

    # -- archives -------------------------------------------------------------------------------
    def zip(self, name: str) -> zipfile.ZipFile | None:
        if name not in self._zips:
            path = os.path.join(self.data_dir, name)
            self._zips[name] = zipfile.ZipFile(path) if os.path.isfile(path) else None
        return self._zips[name]

    def names_in(self, name: str) -> set[str]:
        z = self.zip(name)
        if z is None:
            return set()
        if name not in self._index:
            self._index[name] = {n for n in z.namelist() if not n.endswith("/")}
        return self._index[name]

    # -- discovery ------------------------------------------------------------------------------
    def _dex_name(self, dex: int) -> str:
        return SPECIAL_NAMES.get(dex) or self.names.get(str(dex), f"pm{dex:04d}")

    def pokemon(self) -> list[ModelRef]:
        found: dict[str, tuple[str, str, str]] = {}
        for n in self.names_in(POKE_ZIP):
            m = POKE_RE.match(n)
            if m and m.group(3) == f"{m.group(1)}_{m.group(2)}":
                found[m.group(3)] = (m.group(1), m.group(2), n)
        forms: dict[str, set[str]] = {}
        for sp, ff, _ in found.values():
            forms.setdefault(sp, set()).add(ff)
        refs = []
        for mid in sorted(found):
            sp, ff, path = found[mid]
            dex = int(sp[2:])
            base = self._dex_name(dex)
            if ff == "00":
                display = base
            elif ff in ("51", "52", "61"):
                display = f"{form_label(ff, forms[sp])} {base}"
            else:
                display = f"{base} ({form_label(ff, forms[sp])})"
            folder = path[: path.rindex("/") + 1]
            refs.append(
                ModelRef(
                    game=self.GAME,
                    kind="pokemon",
                    id=mid,
                    folder=folder,
                    display_name=display,
                    species=sp,
                    form=ff,
                    anim_folder=f"bin/pokemon/{mid}/anm/",
                    extras={"dex": dex},
                )
            )
        # partner Pikachu / Eevee (player companions with their own rig and clips)
        for n in sorted(self.names_in(POKE_ZIP)):
            m = BUDDY_RE.match(n)
            if not m or m.group(4) != f"{m.group(1)}_{m.group(2)}":
                continue
            sp, ff, name, mid = m.groups()
            folder = n[: n.rindex("/") + 1]
            refs.append(
                ModelRef(
                    game=self.GAME,
                    kind="pokemon",
                    id=mid,
                    folder=folder,
                    display_name=BUDDY_NAMES.get(sp, f"Partner {name.title()}"),
                    species=sp,
                    form=ff,
                    anim_folder=folder[: -len("mdl/")] + "anm/",
                    extras={"dex": BUDDY_DEX.get(sp)},
                )
            )
        return refs

    def objects(self) -> list[ModelRef]:
        """Player accessories (pd####): cap, hat, glasses ..."""
        refs = []
        for n in sorted(self.names_in(POKE_ZIP)):
            m = ACCESSORY_RE.match(n)
            if not m or m.group(4) != f"{m.group(1)}_{m.group(2)}":
                continue
            sp, ff, name, mid = m.groups()
            refs.append(
                ModelRef(
                    game=self.GAME,
                    kind="object",
                    id=mid,
                    folder=n[: n.rindex("/") + 1],
                    display_name=name.title(),
                    species=sp,
                    form=ff,
                    extras={"item": name},
                )
            )
        return refs

    def trainers(self) -> list[ModelRef]:
        """Trainer / NPC models from the extracted `LGPE-Trainers.7z` (`chara/data/tr/tr####_VV_<name>/`).

        The rip holds models and DDS textures only: no trainer animations, so these import as static rigged models.
        """
        root = os.path.join(self.trainer_dir, "chara", "data", "tr")
        if not os.path.isdir(root):
            return []
        refs = []
        for d in sorted(os.listdir(root)):
            m = TRAINER_DIR_RE.match(d)
            if not m or not os.path.isfile(os.path.join(root, d, f"{d}.gfbmdl")):
                continue
            sp, var, name = m.groups()
            label = TRAINER_NAMES.get(name) or name.replace("_", " ").title()
            if var != "00":
                label += f" ({var})"
            refs.append(
                ModelRef(
                    game=self.GAME,
                    kind="trainer",
                    id=f"{sp}_{var}",
                    folder=os.path.join(root, d) + os.sep,
                    display_name=label,
                    species=sp,
                    form=var,
                    extras={"folder": d},
                )
            )
        return refs

    def story_objects(self) -> list[ModelRef]:
        """`LGPE-Misc.7z chara/data/ob/ob####_VV_<name>/` (Poke Balls, berries, gadgets): static rigged props."""
        root = os.path.join(self.misc_dir, "chara", "data", "ob")
        if not os.path.isdir(root):
            return []
        refs = []
        for d in sorted(os.listdir(root)):
            m = OB_DIR_RE.match(d)
            if not m or not os.path.isfile(os.path.join(root, d, f"{d}.gfbmdl")):
                continue
            sp, var, name = m.groups()
            label = OB_NAMES.get(name) or name.replace("_", " ").title()
            refs.append(
                ModelRef(game=self.GAME, kind="object", id=f"{sp}_{var}", folder=os.path.join(root, d) + os.sep, display_name=label + (f" ({var})" if var != "00" else ""),
                         species=sp, form=var, extras={"folder": d, "item": name})
            )
        return refs

    def _field(self) -> list[ModelRef]:
        root = os.path.join(self.field_dir, "field")
        if not os.path.isdir(root):
            return []
        found: list[tuple[str, str]] = []
        for folder in sorted(os.listdir(root)):
            fdir = os.path.join(root, folder)
            if os.path.isdir(fdir):
                found += [(folder, f[: -len(".gfbmdl")]) for f in sorted(os.listdir(fdir)) if f.endswith(".gfbmdl")]
        counts: dict[str, int] = {}
        for _, stem in found:
            counts[stem] = counts.get(stem, 0) + 1
        refs = []
        for folder, stem in found:
            mid = stem if counts[stem] == 1 else f"{folder}_{stem}"  # file stems are unique except `sea_00`
            kind = "prop" if folder in PROP_FOLDERS else "map"
            refs.append(
                ModelRef(game=self.GAME, kind=kind, id=mid, folder=os.path.join(root, folder) + os.sep, display_name=field_label(stem, folder),
                         species=folder, form="00", extras={"stem": stem, "area": folder})
            )
        return refs

    def maps(self) -> list[ModelRef]:
        return [r for r in self._field() if r.kind == "map"]

    def props(self) -> list[ModelRef]:
        return [r for r in self._field() if r.kind == "prop"]

    def all(self) -> list[ModelRef]:
        return self.pokemon() + self.trainers() + self.objects() + self.story_objects() + self._field()

    def by_kind(self, kind: str | None) -> list[ModelRef]:
        if kind == "object":
            return self.objects() + self.story_objects()
        return {"pokemon": self.pokemon, "trainer": self.trainers, "map": self.maps, "prop": self.props}.get(kind or "", self.all)()

    def find(self, model_id: str) -> ModelRef | None:
        return next((r for r in self.all() if r.id == model_id), None)

    # -- import ---------------------------------------------------------------------------------
    def animations(self, ref: ModelRef) -> list[LgpeAnimation]:
        z = self.zip(ANIM_ZIP)
        if z is None or not ref.anim_folder:
            return []
        out = []
        for n in sorted(self.names_in(ANIM_ZIP)):
            if n.startswith(ref.anim_folder) and n.endswith(".gfbanm") and "/" not in n[len(ref.anim_folder) :]:
                out.append(LgpeAnimation(name=clip_name(n, ref.id), data=z.read(n)))
        return out

    def import_ref(self, ref: ModelRef, with_anim: bool = True):
        if ref.kind in ("map", "prop"):
            folder = ref.folder

            def read_field(n: str) -> bytes:
                with open(os.path.join(folder, n), "rb") as f:
                    return f.read()

            return import_field_model(read_field, lambda n: os.path.isfile(os.path.join(folder, n)), f"{ref.extras['stem']}.gfbmdl", ref.id, ref.display_name, ref.kind)
        if ref.kind == "trainer" or (ref.kind == "object" and "folder" in ref.extras):
            folder = ref.folder

            def read_file(n: str) -> bytes:
                with open(os.path.join(folder, n), "rb") as f:
                    return f.read()

            return import_model(
                read=read_file,
                exists=lambda n: os.path.isfile(os.path.join(folder, n)),
                model_id=ref.id,
                display_name=ref.display_name,
                kind=ref.kind,
                game=ref.game,
                model_file=f"{ref.extras['folder']}.gfbmdl",
            )
        z = self.zip(POKE_ZIP)
        names = self.names_in(POKE_ZIP)
        return import_model(
            read=lambda n: z.read(ref.folder + n),
            exists=lambda n: (ref.folder + n) in names,
            model_id=ref.id,
            display_name=ref.display_name,
            kind=ref.kind,
            game=ref.game,
            animations=self.animations(ref) if with_anim else [],
        )
