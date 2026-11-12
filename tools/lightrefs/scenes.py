"""Scene taxonomy for lighting references, per game. Scene ids are stable: they become preset ids (`za_<scene>`),
folder names (`refs/za/<scene>/`) and the sidecar `scene` field.

Fit profiles live in fit_profiles.json (shared with tools/fit_scene.mjs). Seeds are starting environments for the
fitter: the viewer's presets by id, or the old uncalibrated guesses kept here under `SEED_ENVS`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Scene:
    id: str
    label: str
    time_of_day: str          # day | dusk | night | any
    weather: str              # sunny | overcast | rain | none
    indoor: bool
    light_type: str           # sun | sky | street+neon | floodlight | moon | window | window+lamp | fluorescent | lamp
    profile: str              # key into fit_profiles.json
    seeds: tuple[str, ...]    # preset ids / SEED_ENVS keys
    candidates: tuple[str, ...] = ()   # likely subjects "id:name"
    queries: tuple[str, ...] = ()      # YouTube search hints / chapter keywords
    hints: str = ""


# Uncalibrated guesses that used to ship as presets; kept as fit seeds only (shared with fit_scene.mjs).
with open(os.path.join(HERE, "seed_envs.json"), encoding="utf-8") as _f:
    SEED_ENVS: dict[str, dict] = {k: v for k, v in json.load(_f).items() if not k.startswith("_")}

_NPCS = (
    "tr0001_00:Taunie", "tr0002_00:Urbain", "tr0003_00:Hotel Manager", "tr0005_00:President", "tr0006_00:Secretary",
    "tr0009_00:Grisham", "tr0010_00:Griselle", "tr0013_00:Detective", "tr0015_00:Canari", "tr0016_00:Tarragon",
    "tr0017_00:Ivor", "tr0018_00:Gwynn", "tr0019_00:Jacinthe", "tr0020_00:Lebanne", "tr0021_00:Corbeau",
    "tr0022_00:Philippe", "tr0023_00:Researcher",
)
_POKEMON = ("pm0025_00_00:Pikachu", "pm0133_00_00:Eevee", "pm0001_00_00:Bulbasaur", "pm0006_00_00:Charizard",
            "pm0448_00_00:Lucario", "pm0094_00_00:Gengar", "pm0053_00_00:Persian")

ZA_SCENES: list[Scene] = [
    Scene("lumiose_day_sunny", "Lumiose, sunny day", "day", "sunny", False, "sun", "outdoor", ("studio", "day"),
          _NPCS + _POKEMON, ("Lumiose City daytime", "main mission", "wild zone"), "hard sun, blue sky, warm bounce off stone"),
    Scene("lumiose_day_overcast", "Lumiose, overcast", "day", "overcast", False, "sky", "outdoor", ("overcast", "studio"),
          _NPCS + _POKEMON, ("overcast", "cloudy"), "flat grey sky light, no hard shadows"),
    Scene("lumiose_rain", "Lumiose, rain", "day", "rain", False, "sky", "outdoor", ("overcast",),
          _NPCS + _POKEMON, ("rain", "Sliggoo", "Goodra"), "wet streets, desaturated, rain streaks: box only dry-looking flat areas"),
    Scene("lumiose_dusk", "Lumiose, dusk", "dusk", "sunny", False, "sun", "dusk", ("sunset", "day"),
          _NPCS + _POKEMON, ("evening", "sunset"), "low warm sun, long shadows, purple sky"),
    Scene("lumiose_night_street", "Lumiose, night street", "night", "none", False, "street+neon", "night", ("za_battle_zone_night",),
          _NPCS + _POKEMON, ("night", "Lumiose night"), "street lamps and shop signs; cool ambient with warm pools"),
    Scene("battle_zone_night", "Battle Zone, night", "night", "none", False, "floodlight", "night", ("za_battle_zone_night",),
          _NPCS + _POKEMON, ("Z-A Royale", "battle zone", "Rogue Mega"), "red laser walls, floodlit ground, cool sky"),
    Scene("wild_zone_day", "Wild Zone, day", "day", "sunny", False, "sun", "outdoor", ("studio", "day"),
          _POKEMON + ("tr0001_00:Taunie", "tr0002_00:Urbain"), ("wild zone", "catching"), "parks and rooftops in daylight"),
    Scene("wild_zone_night", "Wild Zone, night", "night", "none", False, "moon", "night", ("za_battle_zone_night",),
          _POKEMON, ("wild zone night", "Noibat"), "moonlit park, blue ambient"),
    Scene("indoor_hotel", "Hotel Z, window light", "day", "none", True, "window", "indoor", ("za_indoor_hotel",),
          ("tr0003_00:Hotel Manager", "tr0013_00:Detective", "tr0001_00:Taunie", "tr0002_00:Urbain"), ("Hotel Z", "lobby", "room"), "soft daylight through windows, warm interior bounce"),
    Scene("indoor_cafe", "Cafe, window light", "day", "none", True, "window+lamp", "indoor", ("za_indoor_hotel", "indoorWarm"),
          _NPCS, ("cafe", "restaurant", "shop"), "mixed window light and warm lamps"),
    Scene("indoor_lab", "Lab / Pokemon Center, fluorescent", "any", "none", True, "fluorescent", "indoor", ("indoorCool",),
          ("tr0023_00:Researcher", "tr0005_00:President", "tr0006_00:Secretary", "tr0001_00:Taunie"), ("Quasartico", "lab", "Pokemon Center"), "even cool-white ceiling light, little shadow"),
    Scene("indoor_night_lamp", "Indoor, night lamps", "night", "none", True, "lamp", "indoor", ("indoorWarm",),
          _NPCS, ("night", "hotel room", "indoor night"), "warm lamp light, dark windows"),
    Scene("sewers", "Lumiose sewers", "any", "none", True, "lamp", "indoor", ("indoorCool", "indoorWarm"),
          _NPCS + _POKEMON, ("sewers", "underground", "Lumiose Sewers"), "dim, greenish/warm lamps, strong falloff"),
]

# UNITE: no light rig in any readable file (docs/ledger/unite.json lighting:rig), so presets come from footage.
# Lobby/result shots use the `_lob` models (lobby clips: `idle`), battle shots the battle models (`idle`, `run`).
_UNITE_LOBBY = ("pm0025_00_lob:Pikachu", "pm0003_00_lob:Venusaur", "pm0006_00_lob:Charizard",
                "pm0009_00_lob:Blastoise", "pm0094_00_lob:Gengar", "pm0700_00_lob:Sylveon")
_UNITE_BATTLE = ("pm0025_00:Pikachu", "pm0006_00:Charizard", "pm0448_00:Lucario",
                 "pm0094_00:Gengar", "pm0143_00:Snorlax", "pm0700_00:Sylveon", "pm0133_00:Eevee")

UNITE_SCENES: list[Scene] = [
    Scene("lobby", "UNITE lobby / holowear showcase", "any", "none", True, "studio", "indoor", ("studio",),
          _UNITE_LOBBY, ("Pokemon UNITE holowear preview", "Pokemon UNITE lobby", "unite battle prep"),
          "Pokemon shown large on the lobby stage or the holowear/Pokemon preview screen; avoid UI overlays"),
    Scene("battle_theia", "UNITE battle, Theia Sky Ruins (day)", "day", "sunny", False, "sun", "outdoor", ("studio", "day"),
          _UNITE_BATTLE, ("Pokemon UNITE Theia Sky Ruins gameplay", "Pokemon UNITE ranked gameplay no commentary"),
          "top-down battle camera; pick frames where the Pokemon is large (zoomed replays, spawn area) and not in an attack effect"),
    Scene("battle_auroma", "UNITE battle, Auroma Park", "day", "sunny", False, "sun", "outdoor", ("studio", "day"),
          _UNITE_BATTLE, ("Pokemon UNITE Auroma Park gameplay",),
          "Auroma Park quick/unite battles; same framing rules as battle_theia"),
    Scene("result", "UNITE results / MVP screen", "any", "none", True, "studio", "indoor", ("studio",),
          _UNITE_LOBBY + _UNITE_BATTLE, ("Pokemon UNITE MVP screen", "Pokemon UNITE results screen"),
          "post-match winners/MVP pose; the Pokemon fills the frame; check which model (lobby or battle) matches"),
]

SCENES: dict[str, dict[str, Scene]] = {"za": {s.id: s for s in ZA_SCENES}, "unite": {s.id: s for s in UNITE_SCENES}}


def load_profiles() -> dict:
    with open(os.path.join(HERE, "fit_profiles.json"), encoding="utf-8") as f:
        return json.load(f)


def get_scene(game: str, scene_id: str) -> Scene:
    try:
        return SCENES[game][scene_id]
    except KeyError:
        raise SystemExit(f"unknown scene {game}/{scene_id}; known: {', '.join(SCENES.get(game, {}))}") from None


def preset_id(game: str, scene_id: str) -> str:
    return f"{game}_{scene_id}"
