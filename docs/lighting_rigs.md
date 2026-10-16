# Scene light rigs (`.trlgt`) and the viewer's lighting presets

Game Freak's Trinity engine (Scarlet/Violet, Legends: Arceus, Legends: Z-A) stores scene lighting as `.trlgt`
FlatBuffers. This note records what the files contain, what the SV dump gives us, how `pipeline/lighting`
maps a rig onto the viewer's `Environment`, and which parts of that mapping are inferred rather than verified.

## Where the files are

| Game | Location | In our data? |
|---|---|---|
| SV | `romfs/env/master/master/master.trlgt`, `env/offset/weather_*.trlgt` (10), `env/fel/*` (133 areas/interiors), `env/spl/*` (10 cutscene + 11 camera filters), `field_graphic/terrain_env/ter_XX_YY.trlgt` (81), `field_graphic/indoor_shieldmap/*` (4) | yes: `SV-Everything.7z`, extracted once into `~/Documents/pokemon_raw_data/sv/env` (env `POKEMODELS_SV_ENV`) |
| Z-A | `light/master/master/master.trlgt`, `light/offset/offset_{clear_sunny,cloudy,rain,mist,rainbow}.trlgt`, `light/env/env_t1_*` (405 Lumiose areas and interiors), `light/spl/spl_ev_dNN_cNN.trlgt` (282 cutscene cuts) | **no**: none of the 16 `ZA-*.zip`; `ZA-Field.zip`'s `field/env/env/ik_t1_depth_N_M.trlgt` are `OverviewHeightMapData` map cameras. Needs a full RomFS dump. |

Scene files `world/scene/parts/env/field_env/env_areaNN_streaming_*.trscn` bind point-light rigs to areas
(`trinity_LightApplierComponent` + rig paths); game scripts swap rigs at runtime (e.g. `env/spl/spl_07`).

## File format (verified on the SV files; schema in PokeDocs `SV/Flatbuffers/field/trlgt.fbs`)

Root `{0 res_0 u32, 1 lights}`; light `{0 type, 1 name, 2 transform, 3 float_params [{name, f32}],
4 vec4_params [{name, Vec4}], 5 string_params [{map_name, map_file}], 6 u32 (4 or 8, unknown)}`.
The schema calls field 2 a `BoundMatrix` of three Vec4 rows, but the file stores three Vec3 fields laid out
like a `.trskl` SRT: scale, rotation (euler XYZ, radians), translation. A directional light travels along its
rotated local -Z: master's sun `rot (-0.7, 3.0, 0)` comes from 40 deg above the horizon; `fel_interior`'s from 58.5 deg.
Reader: `pipeline/importers/za/trlgt.py` (`parse_trlgt`, `Light.direction()/angles()`, `dump_rig`).

Light types in a full rig (`master`, `fel_interior`, `fel_raidbattle`, `spl_NN`):

| Type | Content |
|---|---|
| `Global` | post-process: `PfxTonemapMode` 2 + `PfxToeStrength/ToeLength/ShoulderStrength/ShoulderLength/ShoulderAngle/TonemapGamma` (Hable's filmic curve parameters), `Gamma`, `ContrastMid/Vol`, `CbCrScale`, `CbOffset/CrOffset`, bloom, eye adaptation (`TMMin/MaxBrightness`), DOF, SSR, `AOEnable0..7`, `AOColor0..7` |
| `Sky` | sun size/intensity/colour, moon pitch/yaw, Rayleigh/Mie scattering (day and `SkyNight*`), gradient colours, clouds, light shafts, weather wetness |
| `Fog` | distance/height/horizon/directional fog |
| `ProbeMain` | diffuse/specular probe intensity + colour, `probemain_diffuse/specular.bntx` (6-face cube, 64x64 BC6H signed float) |
| `GILightCommon` | `Intensity` + per-category `Color0..7` / `Intensity0..7`, light-map weights |
| `DirectionalMain` | the sun: colour, intensity, shadow cascades (direction in the SRT) |
| `Ambient` | animated noise (`Speed`, `NoiseScale`, `Strength`), not a light |
| `dir_0..7` "Selectable" | per-category key light: `CategoryLabel`, `Intensity` (0.08 .. 5.9), `ClampMax`, `Color`, self-shadow, `WeatherWetParams*` |
| `rim_0..7` | per-category rim light: `Color`, `SubColor` (alpha = strength, 0 .. 0.8), `Power`, `Decay` |
| `DirectionalForWater`, `RefSpecular`, `ShadowTransform` | water light, reflection specular, shadow frame |

Area rigs are **partial overrides** of `master` (`fel_interior.Global` only lists Gamma/Contrast/CbCr/Tonemap/Bloom
keys; `fel_a_sch_class` is a `ProbeMain` alone; `fel_picnic` only a sun direction and `CbCrScale 1.1`), merged
per light type and parameter (`pipeline/lighting/catalog.merge`). `fel_pkcenter`, `fel_c01`, `*_pointlight`
are sets of `Point_*` / `Spot_*` / `IrradianceVolume_*` local lights (position, `SourceRadius`, colour with
intensity in alpha, cone angles) applied on top of the area rig.

Weather rigs (`env/offset/weather_*`) are **deltas**: floats add (`dir_6 Intensity -2` on master's 4), colours
are HSV offsets (every sky gradient carries `(0, -1, -0.05, 0)` = fully desaturated and slightly darker,
`GILightCommon Color0 (0, -0.71, 0.55)` = greyer and brighter). Inferred from the value pattern (`apply_delta`).

**Category link (verified):** `.trmtr` materials carry `ints.CategoryLabel` (SV: Pikachu body 6, trainer skin 5;
also `CastShadow`, `EnableAdditionalLight`, `LightReceiveType`, `PointLightIndex`), so a material is lit by
`dir_<cat>`, `rim_<cat>` and `GILightCommon.Color<cat> x Intensity x Intensity<cat>`.

**Time of day is not tabulated.** The rigs hold one state; the sky has day and `SkyNight*` scattering terms and
the sun/moon angles, so the day/night cycle is computed from the clock in code. Night presets cannot be derived
from the files (screenshot calibration remains the route for them).

## Mapping to the viewer (`pipeline/lighting/rig_to_env.py`, v1)

| Rig | Viewer `Environment` | Status |
|---|---|---|
| `dir_<cat>.Color x Intensity` (clamped) | `key` = Studio 1.2 x (value / master's value), `keyColor` (explicit), `keyTemp` closest Kelvin | ratio verified, absolute level unknown |
| `DirectionalMain` SRT | `elevation`; the world azimuth is recorded in the preset comment, `azimuth` stays at Studio (characters face anywhere) | verified direction decode |
| `GILightCommon.Intensity x Intensity<cat> x Color<cat>` (x diffuse probe hemispheres when decodable) | `ambient` = Studio 1.9 x ratio to master, `skyColor`, `ground` | inferred |
| `rim_<cat>.Color` (alpha) | `back` = alpha x max channel, `backColor`, `rim` = back x Power/3 | inferred |
| `Global.PfxTonemapMode 2` + `Pfx*` | `toneMapping: 'hable'` + `hable{}` (viewer/src/scene/hable.ts reproduces the curve) | verified parameter set; curve port from Hable's reference code |
| `Global.CbCrScale`, `ContrastVol` | `saturation`, `contrast` | inferred |
| `TMMin/MaxBrightness` (eye adaptation) | nothing: `exposure` stays 1 | the engine auto-exposes; absolute brightness is **not** in the data |
| `Sky`, `Fog`, point lights, AO colours, SSS/`RefSpecular` | nothing | not representable with two directional lights + a hemisphere |

Consequences: presets built this way are faithful in light **colours, direction elevation, key:ambient:rim
ratios, tone curve and grade**, but their overall brightness is anchored to the viewer's Studio level and the
per-category split (trainers vs Pokémon) is reduced to one category per preset (`--category 5`). For category 5
the weather offsets only trim the key by 10% and recolour the sky, so storm/mist/snow collapse into the cloudy preset.

## Commands

```
./.venv/Scripts/python.exe -m pipeline --game sv lights list [--group area]
./.venv/Scripts/python.exe -m pipeline --game sv lights dump fel_interior [--merge]
./.venv/Scripts/python.exe -m pipeline --game sv lights export --all        # out/sv/lights/<rig>.json + probe cube PNGs
./.venv/Scripts/python.exe -m pipeline --game sv lights presets [--category 5]   # viewer/src/scene/svPresets.ts (+ .json)
node tools/render_models.mjs --out out/sv_lights/<preset> --preset <preset> sv/tr0200_00 ...   # then tools/lighting_sheet.py
```
