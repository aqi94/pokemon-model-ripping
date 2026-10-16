"""Light rig -> viewer env mapping (pipeline/lighting), on synthetic rigs; no game data needed."""

import os
import re

from pipeline.importers.za.trlgt import Light, LightRig
from pipeline.lighting.catalog import merge
from pipeline.lighting.presets import presets_ts
from pipeline.lighting.rig_to_env import DEFAULT_ENV, apply_delta, closest_kelvin, rig_to_env

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _master() -> LightRig:
    return LightRig([
        Light("Global", "Global", None, None, None, {"PfxTonemapMode": 2.0, "PfxToeStrength": 0.085, "CbCrScale": 0.9, "ContrastVol": 1.0}),
        Light("GILightCommon", "GILightCommon", None, None, None, {"Intensity": 1.5, "Intensity5": 2.0},
              {"Color5": (1.0, 1.0, 1.0, 1.0), "Color0": (0.5, 0.7, 1.0, 1.0)}),
        Light("DirectionalMain", "Directional", (1, 1, 1), (-0.7, 3.0, 0.0), (0, 0, 0), {"Intensity": 1.0}, {"Color": (1, 1, 1, 1)}),
        Light("dir_5", "Selectable", None, None, None, {"CategoryLabel": 5, "Intensity": 5.0, "ClampMax": 5.0}, {"Color": (0.865, 0.85, 0.9, 0.0)}),
        Light("rim_5", "Rim", None, None, None, {"CategoryLabel": 5, "Power": 3.0}, {"Color": (0.54, 0.74, 1.0, 0.2)}),
    ])


def test_master_maps_to_studio_levels():
    env, prov = rig_to_env(_master(), 5, rig_name="master")
    assert env["key"] == 1.2 and env["ambient"] == 1.9
    assert abs(env["elevation"] - 40.1) < 0.1 and env["azimuth"] == DEFAULT_ENV["azimuth"]
    assert env["toneMapping"] == "hable" and env["hable"]["toeStrength"] == 0.085
    assert env["saturation"] == 0.9 and env["back"] == 0.2
    assert env["keyColor"].startswith("#") and env["skyColor"] == "#ffffff"
    assert prov.key_angles[1] == 40.1 and set(DEFAULT_ENV) <= set(env)


def test_merge_overrides_only_listed_params():
    area = LightRig([Light("Global", "Global", None, None, None, {"CbCrScale": 1.1}),
                     Light("dir_5", "Selectable", None, None, None, {"Intensity": 5.9}, {"Color": (1.0, 0.93, 0.97, 0.0)})])
    rig = merge(_master(), area)
    assert rig.get("Global").floats["PfxTonemapMode"] == 2.0 and rig.get("Global").floats["CbCrScale"] == 1.1
    env, _ = rig_to_env(rig, 5, master=_master())
    assert env["key"] == round(1.2 * 5.0 / (0.9 * 5.0), 3)  # 5.9 clamped to ClampMax 5, colour max 1.0 vs 0.9
    assert env["saturation"] == 1.1


def test_weather_delta_floats_add_and_colours_shift_hsv():
    delta = LightRig([Light("dir_5", "Selectable", None, None, None, {"Intensity": -0.5}, {"Color": (0.0, 0.0, 0.0, 0.0)}),
                      Light("GILightCommon", "GILightCommon", None, None, None, {}, {"Color0": (0.0, -1.0, 0.0, 0.0)})])
    rig = apply_delta(_master(), delta)
    assert rig.get("dir_5").floats["Intensity"] == 4.5
    c0 = rig.get("GILightCommon").colors["Color0"]
    assert abs(c0[0] - c0[1]) < 1e-6 and abs(c0[1] - c0[2]) < 1e-6 and abs(c0[0] - 1.0) < 1e-6  # desaturated, value kept
    env, _ = rig_to_env(rig, 5, master=_master())
    assert env["key"] == round(1.2 * 4.5 / 5.0, 3)


def test_closest_kelvin():
    assert closest_kelvin((1.0, 1.0, 1.0)) == 6500
    assert closest_kelvin((1.0, 0.6, 0.3)) < 4000
    assert closest_kelvin((0.7, 0.8, 1.0)) > 7000


def test_presets_ts_shape():
    env, prov = rig_to_env(_master(), 5, rig_name="master")
    ts = presets_ts([{"id": "sv_x", "label": "X", "game": "sv", "env": env, "provenance": prov}], 5)
    assert "export const SV_PRESETS: LightingPreset[]" in ts and "id: 'sv_x'" in ts and "game: 'sv'" in ts
    assert "import type { LightingPreset }" in ts and "hable: {" in ts


def test_default_env_matches_viewer():
    src = open(os.path.join(ROOT, "viewer", "src", "scene", "environment.ts"), encoding="utf-8").read()
    m = re.search(r"export const DEFAULT_ENV: Environment = \{(.*?)\n\};", src, re.S)
    assert m, "DEFAULT_ENV not found"
    keys = set(re.findall(r"(\w+):", m.group(1)))
    assert keys == set(DEFAULT_ENV), keys ^ set(DEFAULT_ENV)
