"""tools/lightrefs: scene taxonomy, timecodes, sidecar/regions validation, preset generation (offline)."""

import json
import os
import re

from tools.lightrefs import build_presets as bp
from tools.lightrefs.scenes import SCENES, SEED_ENVS, load_profiles, preset_id
from tools.lightrefs.sidecar import format_timecode, make_sidecar, parse_timecode, validate_regions, validate_sidecar

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_scene_ids_unique_and_profiles_exist():
    prof = load_profiles()
    for game, scenes in SCENES.items():
        assert len(scenes) == len({s.id for s in scenes.values()})
        for s in scenes.values():
            assert s.profile in prof, s.id
            assert all(k in prof[s.profile] for k in ("knobs", "lock"))
            for seed in s.seeds:
                assert seed in SEED_ENVS or seed == "studio" or seed.startswith("za_"), (s.id, seed)
            assert preset_id(game, s.id) == f"{game}_{s.id}"


def test_parse_timecode():
    assert parse_timecode("83") == 83
    assert parse_timecode("1:23") == 83
    assert parse_timecode("1:23.5") == 83.5
    assert parse_timecode("01:02:03.25") == 3723.25
    assert parse_timecode(12.5) == 12.5
    assert format_timecode(3723.25) == "01:02:03.250"


def _sidecar():
    return make_sidecar(slug="indoor_hotel_manager_01", scene="indoor_hotel", game="za",
                        video={"id": "abc", "url": "https://youtu.be/abc", "platform": "switch2"}, timestamp=1234.25,
                        image={"file": "indoor_hotel_manager_01.png", "width": 1920, "height": 1080, "scaledFrom": [3840, 2160]},
                        tags={"timeOfDay": "day", "weather": "none", "location": "Hotel Z lobby", "indoor": True, "lightType": "window"},
                        subject={"kind": "trainer", "model": "tr0003_00", "name": "Hotel Manager", "confidence": "high"},
                        clip_guess="speak01_loop", context="cutscene")


def test_sidecar_roundtrip_and_validation():
    sc = _sidecar()
    assert validate_sidecar(sc) == []
    assert sc["timecode"] == "00:20:34.250"
    bad = json.loads(json.dumps(sc))
    bad["scene"] = "nope"; bad["subject"]["model"] = "x"; bad["tags"]["weather"] = "hail"
    errs = validate_sidecar(bad)
    assert any("unknown scene" in e for e in errs) and any("model" in e for e in errs) and any("weather" in e for e in errs)


def test_regions_validation():
    sc = _sidecar()
    reg = {"schema": 1, "model": "za/tr0003_00", "clip": "speak01_loop", "frame": 12, "screenshot": "indoor_hotel_manager_01.png",
           "width": 1920, "height": 1080, "camera": {"fov": 35, "position": [0, 1.4, 2], "target": [0, 1.3, 0]},
           "regions": [{"name": "skin", "box": [10, 10, 50, 50], "renderBox": [12, 12, 52, 52]}]}
    assert validate_regions(reg, sc) == []
    reg["regions"].append({"name": "bad", "box": [50, 50, 10, 10]})
    reg["model"] = "za/tr0001_00"
    errs = validate_regions(reg, sc)
    assert any("inverted" in e for e in errs) and any("differs from the sidecar subject" in e for e in errs)


def test_presets_ts_and_default_env_match_viewer():
    src = open(os.path.join(ROOT, "viewer", "src", "scene", "environment.ts"), encoding="utf-8").read()
    m = re.search(r"export const DEFAULT_ENV: Environment = \{(.*?)\n\};", src, re.S)
    assert set(re.findall(r"(\w+):", m.group(1))) == set(bp.DEFAULT_ENV) == set(bp.ENV_KEYS)
    env = dict(bp.DEFAULT_ENV, ambient=2.5, toneMapping="agx")
    ts = bp.presets_ts("za", [{"id": "za_indoor_hotel", "label": "Hotel Z", "game": "za", "env": env, "comment": "test", "shots": []}])
    assert "export const ZA_PRESETS: LightingPreset[]" in ts and "id: 'za_indoor_hotel'" in ts and "toneMapping: 'agx'" in ts
    assert "import type { LightingPreset }" in ts
