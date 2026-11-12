"""Catalog of Pokémon UNITE models in the AssetStudio export (`Pokémon UNITE Assets/3D Assets`).

- Pokémon: the holowear lobby showcases `Scenes/pref_sce_lobby_<dex4><skin2>_<theme>_lv<N>/` each embed the skinned
  Pokémon wearing that holowear (meshes `rig_lob_<dex>*`: body `dy`/`body`, outfit `<dex><skin>`, props) among
  backdrop cards. One model per scene, highest LOD file. Id `pm<dex>_<skin>_<theme>`.
- Battle Pokémon (separate Drive share, mirrored to `~/Documents/pokemon_raw_data/unite/Pokemon`, env
  `POKEMODELS_UNITE_POKEMON`): `3D Models/Battle Models/pref_char_bat_<dex4><skin2>_<name>_lv3/` (164 of 852
  folders hold an FBX; animated), see `unite_battle.py`. Id `pm<dex>_<skin>`.
- Standalone lobby holowears: `3D Models/Lobby Models/pref_char_lob_*` (114 FBX, idle clip); they
  replace the scene copy of the same holowear (id `pm<dex>_<skin>_<theme>`), the others get `pm<dex>_<skin>_lob`.
- Wild/objective Pokémon: the Marcy4000/Unite project's `Assets/Model/Pokemons` (`UniteCatalog.marcy`, game-named FBX
  with clips), only species/skins the Drive share lacks.
- Loose `Other/pref_char_*` FBXs without a Battle/Lobby counterpart (`UniteCatalog.others`): the low-detail
  Rayquaza scene rig (superseded by Marcy's full Rayquaza).
- Trainers: `NPCs/Lv3/pref_npc_lob_<NNN>_<name>_lv3/` (the four lobby NPCs).
"""

from __future__ import annotations

import difflib
import os
import re

from ..importers.unite.importer import attach_fx_donors, colour_particles, fire_effects, import_model as import_unite_model
from .unite_battle import FOLDER_RE, NPC_REL, POKEMON_DIR, battle_refs, char_refs, form_token, skin_keys, texture_index
from .za import ModelRef, _load_names

DEFAULT_DATA_DIR = os.environ.get("POKEMODELS_UNITE_DATA") or os.path.join(
    os.path.expanduser("~"), "Documents", "MEGA downloads", "Pokémon UNITE Assets", "3D Assets")

# Sparse clone of github.com/Marcy4000/Unite (`git sparse-checkout set --cone Assets/Model/Pokemons`): the wild Pokémon
MARCY_DIR = os.environ.get("POKEMODELS_UNITE_MARCY") or os.path.join(
    os.path.expanduser("~"), "Documents", "pokemon_raw_data", "unite", "marcy", "Assets", "Model", "Pokemons")

LOBBY_RE = re.compile(r"^pref_sce_lobby_(\d{4})(\d{2})_(.+?)_lv(\d)$")
NPC_RE = re.compile(r"^pref_npc_lob_(\d{3})_(.+?)_lv(\d)$")


def _theme_label(theme: str) -> str:
    theme = re.sub(r"^professor(?=[a-z])", "professor_", theme)  # npc001 `professorphorus`
    return " ".join(w[:1].upper() + w[1:] for w in theme.replace("_", " ").split())


class UniteCatalog:
    GAME = "unite"

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, pokemon_dir: str = POKEMON_DIR, marcy_dir: str = MARCY_DIR):
        self.data_dir = data_dir
        self.pokemon_dir = pokemon_dir
        self.marcy_dir = marcy_dir
        self.names = _load_names()

    def _fbx(self, rel_dir: str) -> str | None:
        d = os.path.join(self.data_dir, rel_dir)
        fbxs = [f for f in os.listdir(d) if f.lower().endswith(".fbx")] if os.path.isdir(d) else []
        return f"{rel_dir}/{fbxs[0]}" if fbxs else None

    def pokemon(self) -> list[ModelRef]:
        battle, lobby = self.battle(), self.lobby()
        extra = self.marcy(battle + lobby)
        return battle + lobby + extra + self.others(battle + lobby + extra)

    def marcy(self, known: list[ModelRef]) -> list[ModelRef]:
        """Wild/objective Pokémon from the Marcy4000/Unite project (`Assets/Model/Pokemons/<Folder>/`, sparse clone,
        env `POKEMODELS_UNITE_MARCY`): the game's own `pref_char_bat_<dex><skin>_<name>_lv3.fbx` (credited to the same
        ripper, decrypted rip) with its clips as FBX stacks. Only game-named files are read (the project also holds
        renamed/fan-made FBXs: `Flygon/flygon.fbx`, `reference.fbx`...) and only dex + skin + kind that the Drive
        share lacks. Bare `Pokemons/pref_char_bat_*` copies have no clips: per name the largest file wins. The project's
        folder name is the label (`Gigantamax Pikachu`, `Delibird Tower Defense`)."""
        root = self.marcy_dir
        have = {(r.extras.get("dex"), r.extras.get("skin", r.form), bool(r.extras.get("battle"))) for r in known}
        best: dict[str, tuple[int, str, str, re.Match]] = {}
        for dp, _, fs in os.walk(root) if os.path.isdir(root) else []:
            for f in fs:
                m = FOLDER_RE.match(f[:-4]) if f.lower().endswith(".fbx") else None
                if not m:
                    continue
                dex, skin = int(m.group(2)), re.sub(r"[^0-9A-Za-z]", "", m.group(3) or "00")
                if (dex, skin, m.group(1) == "bat") in have:
                    continue
                path = os.path.join(dp, f)
                size = os.path.getsize(path)
                label = os.path.basename(dp)
                stem = f[:-4].lower()
                if stem not in best or size > best[stem][0] or (size == best[stem][0] and label.startswith("pref_")):
                    best[stem] = (size, path, label, m)
        refs: list[ModelRef] = []
        ids = {r.id for r in known}
        seen_files: dict[tuple, str] = {}
        for stem, (size, path, label, m) in sorted(best.items()):
            dex, skin = int(m.group(2)), re.sub(r"[^0-9A-Za-z]", "", m.group(3) or "00")
            species = self.names.get(str(dex), f"#{dex}")
            token = form_token(m.group(4) or "", species)
            mid = f"pm{dex:04d}_{skin}" + (f"_{token}" if token else "")
            if mid in ids:
                continue
            ids.add(mid)
            # the folder label unless it is just a (mis)spelling of the species (`Vespiqueen`)
            plain = label.startswith("pref_") or difflib.SequenceMatcher(None, label.lower(), species.lower()).ratio() > 0.85
            display = species + (f" ({token})" if token else "") if plain else label
            refs.append(ModelRef(
                game=self.GAME, kind="pokemon", id=mid, folder=path, display_name=display, species=f"pm{dex:04d}",
                form=skin,
                extras={"dex": dex, "skin": skin, "holowear": f"{dex:04d}{skin}", "battle": m.group(1) == "bat",
                        "standalone": True, "folder_name": stem, "source": "marcy"},
            ))
        return refs

    def others(self, known: list[ModelRef]) -> list[ModelRef]:
        """Loose character FBXs that AssetStudio wrote into `Other/` (`pref_char_<bat|lob>_<dex><skin>_<name>_lv<n>`),
        kept only when no Battle/Lobby model has the same dex + skin + kind. The rip's only wild/objective Pokémon is
        one of these (`pref_char_bat_038400_rayquaza_scene_lv0`, a low-detail rig without clips); the other wild
        Pokémon are absent from every source (ledger `rip:wildPokemon`)."""
        root = os.path.join(self.data_dir, "Other")
        have = {(r.extras.get("dex"), r.extras.get("skin", r.form), bool(r.extras.get("battle"))) for r in known}
        ids = {r.id for r in known}
        best: dict[tuple, tuple[int, str, re.Match]] = {}
        for folder in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            m = FOLDER_RE.match(folder)
            if not m or not self._fbx(f"Other/{folder}"):
                continue
            dex, skin = int(m.group(2)), re.sub(r"[^0-9A-Za-z]", "", m.group(3) or "00")
            key = (dex, skin, m.group(1) == "bat", (m.group(4) or "").lower())
            if (dex, skin, key[2]) in have:
                continue
            lv = int(m.group(5))
            if key not in best or lv > best[key][0]:
                best[key] = (lv, folder, m)
        refs = []
        for (dex, skin, is_bat, _), (lv, folder, m) in sorted(best.items()):
            species = self.names.get(str(dex), f"#{dex}")
            token = form_token(m.group(4) or "", species)
            mid = f"pm{dex:04d}_{skin}" + (f"_{token}" if token else "") + ("" if is_bat else "_lob")
            if mid in ids:
                continue
            ids.add(mid)
            display = species + (f" ({token})" if token else "") + ("" if is_bat else " (lobby)")
            refs.append(ModelRef(
                game=self.GAME, kind="pokemon", id=mid, folder=self._fbx(f"Other/{folder}"), display_name=display,
                species=f"pm{dex:04d}", form=skin,
                extras={"dex": dex, "skin": skin, "holowear": f"{dex:04d}{skin}", "battle": is_bat, "standalone": True,
                        "lod": lv, "folder_name": folder, "source": "Other"},
            ))
        return refs

    def lobby(self) -> list[ModelRef]:
        """Lobby holowears: the standalone `Lobby Models` FBX when there is one (animated idle, intact material layers),
        else the copy embedded in a showcase scene. A standalone model that replaces a scene copy keeps the scene's id
        and theme, and still reads the scene's `Materials/*.json` (the standalone folders have none)."""
        scenes = self.scenes()
        themes = {r.extras["holowear"]: _theme_label(r.extras["theme"]) for r in scenes}
        standalone = {r.extras["holowear"]: r for r in char_refs(self.pokemon_dir, "lob", themes)}
        used: set[str] = set()
        out = []
        for r in scenes:
            hw = r.extras["holowear"]
            st = standalone.get(hw)
            if st is not None and hw not in used:
                used.add(hw)
                scene_dir = os.path.dirname(r.folder)
                r = ModelRef(game=r.game, kind=r.kind, id=r.id, folder=st.folder, display_name=r.display_name,
                             species=r.species, form=r.form,
                             extras={**st.extras, "theme": r.extras["theme"], "materialsFrom": scene_dir,
                                     "sceneFolder": r.folder, "sceneMeshes": r.extras["meshes"]})
            out.append(r)
        out += [r for hw, r in sorted(standalone.items()) if hw not in used]
        return out

    def scenes(self) -> list[ModelRef]:
        """Holowears embedded in the lobby showcase scenes (one per scene, highest LOD)."""
        root = os.path.join(self.data_dir, "Scenes")
        best: dict[tuple[str, str, str], tuple[int, str]] = {}
        for folder in os.listdir(root) if os.path.isdir(root) else []:
            m = LOBBY_RE.match(folder)
            if m:
                key = (m.group(1), m.group(2), m.group(3).lower())
                lv = int(m.group(4))
                if key not in best or lv > best[key][0]:
                    best[key] = (lv, folder)
        refs = []
        for (dex, skin, theme), (lv, folder) in sorted(best.items()):
            fbx = self._fbx(f"Scenes/{folder}")
            if not fbx:
                continue
            n = int(dex)
            species = self.names.get(str(n), f"#{n}")
            refs.append(ModelRef(
                game=self.GAME, kind="pokemon", id=f"pm{dex}_{skin}_{theme}", folder=fbx,
                display_name=f"{species} · {_theme_label(theme)} (lobby)", species=f"pm{dex}", form=skin,
                extras={"dex": n, "holowear": f"{dex}{skin}", "theme": theme, "lod": lv,
                        "meshes": rf"^(?:rig_)?lob_{dex}"},  # `rig_lob_0025..`, also `lob_009409_lv3` (3 scenes)
            ))
        return refs

    def battle(self) -> list[ModelRef]:
        themes = {r.extras["holowear"]: _theme_label(r.extras["theme"]) for r in self.scenes()}
        return battle_refs(self.pokemon_dir, themes)

    def trainers(self) -> list[ModelRef]:
        root = os.path.join(self.data_dir, "NPCs", "Lv3")
        refs = []
        for folder in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            m = NPC_RE.match(folder)
            fbx = self._fbx(f"NPCs/Lv3/{folder}") if m else None
            if fbx:
                # the Pokémon share's `3D Models/NPC Models/<folder>/Materials` holds some of their material JSONs
                mats = os.path.join(self.pokemon_dir, NPC_REL, folder)
                extras = {"materialsFrom": mats} if os.path.isdir(os.path.join(mats, "Materials")) else {}
                # the MEGA FBX names its clip stacks but holds no curves; the Marcy4000/Unite copy of the same
                # prefab has them (idle, talk, welcome, congratulations...): sampled onto this rig by bone path
                donor = self._marcy_file(folder + ".fbx")
                if donor:
                    extras["baseClips"] = donor
                refs.append(ModelRef(game=self.GAME, kind="trainer", id=f"npc{m.group(1)}_{m.group(2).lower()}",
                                     folder=fbx, display_name=_theme_label(m.group(2)), species=f"npc{m.group(1)}",
                                     extras=extras))
        return refs

    def _marcy_file(self, name: str) -> str | None:
        if not hasattr(self, "_marcy_files"):
            self._marcy_files = {f.lower(): os.path.join(dp, f) for dp, _, fs in os.walk(self.marcy_dir) for f in fs
                                 if f.lower().endswith(".fbx")} if os.path.isdir(self.marcy_dir) else {}
        return self._marcy_files.get(name.lower())

    def objects(self) -> list[ModelRef]:
        return []

    def all(self) -> list[ModelRef]:
        return self.pokemon() + self.trainers()

    def find(self, model_id: str) -> ModelRef | None:
        return next((r for r in self.all() if r.id == model_id), None)

    def import_ref(self, ref: ModelRef, with_anim: bool = True):
        model = import_unite_model(
            os.path.join(self.data_dir, ref.folder), ref.id, ref.display_name, kind=ref.kind,
            meshes=ref.extras.get("meshes"),
            texture_index=texture_index(self.data_dir, self.pokemon_dir) if ref.extras.get("standalone") else None,
            material_dirs=[os.path.join(self.data_dir, ref.extras["materialsFrom"])] if ref.extras.get("materialsFrom") else (),
            scene_fbx=os.path.join(self.data_dir, ref.extras["sceneFolder"]) if ref.extras.get("sceneFolder") else None,
            base_clips=ref.extras.get("baseClips") if with_anim else None, dex=ref.extras.get("dex"),
        )
        # only effect parts left (Zacian Pirate keeps a background card) counts as no character mesh
        has_body = any(not model.materials[p.material].extras.get("effect") for mm in model.meshes for p in mm.primitives)
        if not has_body and ref.extras.get("sceneFolder"):
            # standalone file without a usable character mesh (Zacian Pirate: only a zero-scaled `1(Clone)` and an
            # effect card): fall back to the copy embedded in the showcase scene
            model = import_unite_model(os.path.join(self.data_dir, ref.extras["sceneFolder"]), ref.id, ref.display_name,
                                       kind=ref.kind, meshes=ref.extras["sceneMeshes"])
            model.extras.setdefault("warnings", []).append("standalone lobby FBX has no character mesh; used the scene copy")
        if not model.meshes:
            raise RuntimeError(f"{ref.id}: no character mesh left after filtering")
        colour_particles(model, ref.extras.get("dex"))
        # a form folder (`0006_51`: Mega Charizard X) uses HOME's form of the same number
        keys = skin_keys(ref.extras.get("folder_name") or "")
        form = ref.extras.get("skin") if keys and keys[1] is None and keys[0][2].startswith("form") else None
        fires = fire_effects(model, ref.extras.get("dex"), form)
        if fires:
            model.extras["fireMaterials"] = fires
        attach_fx_donors(model, self.data_dir)
        if not with_anim:
            model.clips, model.default_clip = [], None
        return model
