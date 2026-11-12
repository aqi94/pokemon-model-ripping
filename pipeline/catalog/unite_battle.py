"""UNITE character models from `3D Assets/Pokemon/3D Models/3D Models/`.

- `Battle Models/pref_char_bat_<dex4><skin2>_<name>_lv3/`: 852 folders, 164 with an FBX (19-31 clips each).
- `Lobby Models/pref_char_lob_<dex4><skin2>_<name>_lv3/`: 615 folders, 114 with an FBX (holowear lobby models, one
  `idle` clip each, material layers intact; 20 of them are also embedded in a lobby showcase scene).
Facts (surveyed 2026-11-01): folders without an FBX are texture-only skins (AssetStudio wrote every asset once, so a
skin that reuses another folder's mesh/textures has no FBX of its own). There are no `Materials/*.json`; the FBX itself
names each material's DiffuseColor / Bump texture (`unite.fbx.material_textures`), and the PNGs live in whichever folder
AssetStudio wrote them first, so textures are resolved through a global name index over the whole tree (`texture_index`,
passed to `unite.importer.import_model`).
"""

from __future__ import annotations

import os
import re

from .za import ModelRef, _load_names

# The Pokémon folder is a separate Google Drive share (not part of the MEGA rip), mirrored file by file with
# `tools/drive_mirror.py` (the Drive zip download lost ~80% of the files).
POKEMON_DIR = os.environ.get("POKEMODELS_UNITE_POKEMON") or os.path.join(
    os.path.expanduser("~"), "Documents", "pokemon_raw_data", "unite", "Pokemon")
BATTLE_REL = os.path.join("3D Models", "Battle Models")
LOBBY_REL = os.path.join("3D Models", "Lobby Models")
NPC_REL = os.path.join("3D Models", "NPC Models")
FOLDER_RE = re.compile(r"^pref_char_(bat|lob)_(?:[a-z]_)?(\d{4})_?(\d{2}|\d{2}[A-Za-z]|\d[a-z]\d{2})?(?:_([A-Za-z0-9_]+?))?_lv(\d)$")


def battle_dir(pokemon_dir: str = POKEMON_DIR) -> str:
    return os.path.join(pokemon_dir, BATTLE_REL)


def form_token(name: str, species: str) -> str:
    """The form/mode part of a folder's name: its tokens minus the species name, a plain `00` and the female `F` names
    (`11_morpeko` -> `11`, `aegislash_shield` -> `shield`, `box` -> `box`, `00_hoopa` -> ``, `mewtwo` -> ``)."""
    sp = re.sub(r"[^a-z0-9]", "", species.lower())
    toks = [t for t in (name or "").split("_") if t]
    keep = [t for t in toks if t.lower() != sp and t != "00" and not (t.endswith("F") and t[:-1].isalpha() and t[:-1].lower() == sp)]
    if keep and len(keep) == len(toks) and not any(c.isdigit() for c in "".join(keep)):
        # a name with no species token at all is the species' own spelling (`ninetales` vs a typo), unless it is
        # unlike every sibling: only a single short word that is not a species name counts as a form (`box`)
        return keep[0].lower() if len(keep) == 1 and len(keep[0]) <= 6 and keep[0].lower() not in _ALL_SPECIES else ""
    return "_".join(k.lower() for k in keep)


_ALL_SPECIES = {re.sub(r"[^a-z0-9]", "", v.lower()) for v in _load_names().values()}


def skin_keys(folder: str) -> tuple[tuple, tuple] | None:
    """(this folder's key, its form's skin-00 key) for matching a skin to the folder holding its base clip set. The
    key = (prefix, dex, skin, name) with the name compared case-insensitively (`035900_Absol` is the base of
    `035902_absol`); the base keeps the skin's form letter (`089204A` -> `089200A`, Urshifu's styles have different
    sets) and a folder without a skin field is skin 00 (`0870_falinks`). Clip tags alone cannot pick it: both Urshifu
    styles tag their base clips `0892`."""
    m = FOLDER_RE.match(folder)
    if not m:
        return None
    prefix = (re.match(r"^pref_char_(?:bat|lob)_([a-z])_", folder) or [None, ""])[1]
    skin = m.group(3) or "00"
    name = (m.group(4) or "").lower()
    # `0006_51_charizard`, `015000_51_mewtwo`'s `_51`: a separate form (Mega X), not a skin of 00. Forms keep their own
    # clips (Mega Charizard Y shares 22 of 58 joints with 0006_00).
    if re.match(r"^pref_char_(?:bat|lob)_(?:[a-z]_)?\d{4}_\d", folder):
        return (prefix, m.group(2), "form" + skin, name), None
    if not skin[:2].isdigit():
        return (prefix, m.group(2), skin, name), (prefix, m.group(2), skin, name)
    return (prefix, m.group(2), skin, name), (prefix, m.group(2), "00" + skin[2:], name)


def char_refs(pokemon_dir: str = POKEMON_DIR, which: str = "bat", themes: dict[str, str] | None = None) -> list[ModelRef]:
    """One ModelRef per `pref_char_<which>_*` folder that holds an FBX. `themes` = holowear code (dex+skin) -> label
    from the lobby showcase scenes. Battle id `pm<dex>_<skin>[F]`; lobby id `pm<dex>_<skin>_lob`."""
    root = os.path.join(pokemon_dir, BATTLE_REL if which == "bat" else LOBBY_REL)
    names = _load_names()
    entries = []
    for folder in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        m = FOLDER_RE.match(folder)
        if not m or m.group(1) != which:
            continue
        fbxs = [f for f in os.listdir(os.path.join(root, folder)) if f.lower().endswith(".fbx")]
        if not fbxs:
            continue
        dex, skin, name = m.group(2), m.group(3) or "00", m.group(4) or ""
        tag = folder[len(f"pref_char_{which}_"):]
        prefix = re.match(r"^([a-z])_", tag)
        sk = re.sub(r"[^0-9A-Za-z]", "", skin)
        mid = f"pm{dex}_{sk}"
        if name.endswith("F") and name[:-1].isalpha():
            mid += "F"
        if prefix:
            mid = f"{prefix.group(1)}_{mid}"
        entries.append((folder, fbxs[0], dex, sk, mid, form_token(name, names.get(str(int(dex)), ""))))
    # Form/mode folders share dex + skin and differ only by a name token: Morpeko `087700_11`/`_12` (Full Belly /
    # Hangry), Mega Mewtwo `_51`/`_52`, Aegislash `_shield`/`_sword`, Dragapult `box`, Hoopa `_12`. Keyed by dex + skin
    # alone, 21 lobby models were dropped (one form replaced the other in `UniteCatalog.lobby`) and battle forms got `_2`.
    # The token joins the id only where folders collide, so ids that never collided (Zacian's `_12`) stay as they were.
    groups: dict[str, int] = {}
    for e in entries:
        groups[e[4]] = groups.get(e[4], 0) + 1
    # A skin FBX carries only the clips authored for it; the rest of its controller is the form's skin-00 set
    # (`importer.inherit_base_clips`). Battle models only: lobby holowears have a single `idle`.
    keyed = {k[0]: f for f in (e[0] for e in entries) if (k := skin_keys(f))}
    refs: list[ModelRef] = []
    seen: set[str] = set()
    for folder, fbx, dex, sk, mid, token in entries:
        keys = skin_keys(folder)
        donor = keyed.get(keys[1]) if keys and keys[1] and which == "bat" else None
        base_fbx = None
        if donor and donor != folder:
            fb = [f for f in os.listdir(os.path.join(root, donor)) if f.lower().endswith(".fbx")]
            base_fbx = os.path.join(root, donor, fb[0]) if fb else None
        mode = token if groups[mid] > 1 else ""
        if mode:
            mid += f"_{mode}"
        if which == "lob":
            mid += "_lob"
        base, n = mid, 2
        while mid in seen:
            mid, n = f"{base}_{n}", n + 1
        seen.add(mid)
        species = names.get(str(int(dex)), f"#{int(dex)}")
        label = (themes or {}).get(dex + sk) or f"skin {sk}"
        display = species if sk == "00" else f"{species} · {label}"
        if mode:
            display += f" ({mode})"
        if which == "lob":
            display += " (lobby)"
        refs.append(ModelRef(
            game="unite", kind="pokemon", id=mid, folder=os.path.join(root, folder, fbx),  # absolute: outside the catalog's data_dir
            display_name=display, species=f"pm{dex}", form=sk,
            extras={"dex": int(dex), "skin": sk, "holowear": dex + sk + (f"_{mode}" if mode else ""), "battle": which == "bat",
                    "standalone": True, "mode": mode or None, "folder_name": folder,
                    **({"baseClips": base_fbx} if base_fbx else {})},
        ))
    return refs


def battle_refs(pokemon_dir: str = POKEMON_DIR, themes: dict[str, str] | None = None) -> list[ModelRef]:
    return char_refs(pokemon_dir, "bat", themes)


_TEX_INDEX: dict[str, dict[str, str]] = {}


def texture_index(data_dir: str, pokemon_dir: str = POKEMON_DIR) -> dict[str, str]:
    """lower-case PNG file name -> path: the Pokémon share first (character exports), then every folder of the MEGA
    `3D Assets` (a few battle textures were only written into lobby scene folders or `Fx/`)."""
    key = data_dir + "|" + pokemon_dir
    if key not in _TEX_INDEX:
        idx: dict[str, str] = {}
        bases = [pokemon_dir] + ([os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir))] if os.path.isdir(data_dir) else [])
        for base in bases:
            for dp, _, fs in os.walk(base):
                for f in fs:
                    if f.lower().endswith(".png"):
                        idx.setdefault(f.lower(), os.path.join(dp, f))
        _TEX_INDEX[key] = idx
    return _TEX_INDEX[key]
