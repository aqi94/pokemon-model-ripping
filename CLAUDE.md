# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline that converts ripped Pokémon game models into one common format (rigged glTF with every animation as a named clip, plus a `manifest.json`), the Blender-community formats derived from it (FBX for Unreal, OBJ+MTL, PNG textures, laid out like the `Adaman/` sample from models.spriters-resource.com), and a modelviewer.lol-style web viewer that reads the common format. Every game gets its own importer under `pipeline/importers/`; nothing downstream of an importer may know which game an asset came from.

Current scope: Pokémon Legends: Z-A (`ZA-*` zips). Later: LA, SwSh, LGPE (`.gfbmdl/.gfbanm`) as `pipeline/importers/gfbmdl/` against the same `common.model` IR.

## Commands

```bash
# Python (venv at .venv, Python 3.14)
./.venv/Scripts/python.exe -m pytest -q                       # all tests (fixture-based, <1 s)
./.venv/Scripts/python.exe -m pytest pipeline/tests/test_bntx.py -q
./.venv/Scripts/python.exe -m pipeline list --kind pokemon    # models found in the zips
./.venv/Scripts/python.exe -m pipeline convert --id pm0001_00_00 --id tr0018_00   # -> out/za/<id>/{model.glb,manifest.json}
./.venv/Scripts/python.exe -m pipeline convert --all --kind pokemon --resume       # batch, resumable (out/convert_log.jsonl)
./.venv/Scripts/python.exe -m pipeline convert --all --kind object                 # the 27 Poké Ball models (held items)
./.venv/Scripts/python.exe -m pipeline export --id pm0001_00_00                     # -> out/blender/za/<id>/{.fbx,.obj,.mtl,*.png}
./.venv/Scripts/python.exe -m pipeline thumbs [--id ...] [--force]                 # thumb.webp per model via Blender (~4 s each), then reindex
./.venv/Scripts/python.exe -m pipeline index                                        # rebuild out/index.json from manifests

# glTF validation (tools/node_modules/gltf-validator; pass Uint8Array.from(buffer), a Node Buffer fails)
node -e "const v=require('./tools/node_modules/gltf-validator');v.validateBytes(Uint8Array.from(require('fs').readFileSync('out/za/pm0001_00_00/model.glb'))).then(r=>console.log(r.issues.numErrors))"

# Blender headless checks (Store build; see "Blender" below)
./.venv/Scripts/python.exe -c "from pipeline.blender_runner import inspect; print(inspect('out/za/pm0001_00_00/model.glb','out/za/pm0001_00_00/inspect','walk01_loop'))"

# Viewer (viewer/): npm install; npm run dev (http://localhost:5173, serves ../out at /models); npm run build (tsc + vite)
```

Viewer is a single page: `/` opens the last viewed model, the sidebar's Model section switches models in place (history.pushState; options/lighting persist), the thumbnail grid is at `/browse`. Index `thumbnail` paths are relative to `out/` (served at `/models/<thumbnail>`). PNG exports are always transparent (`ViewerScene.pngDataUrl`, also used by `tools/render_models.mjs`, `tools/match_lighting.mjs --out`, `tools/contact_sheet.py`, and the Blender preview scripts).

Deployment assumption (user decision): the viewer runs locally and serves models/animations from the local `out/` folder (`npm run dev` / `npm run preview` map `/models` to `../out`). Do not spend effort on CDN hosting, meshopt/Draco compression or GLB size budgets unless that changes.

Source data: zips in `~/Documents/MEGA downloads/PokéModels` (env `POKEMODELS_DATA`) (override with `--data`). Never copy zips into the repo; `ZipSource` reads members in place. `out/` is git-ignored and can reach tens of GB; ask before a full `--all` run.

## Architecture

```
pipeline/common/model.py      IR: Model, Skeleton/Joint, Mesh/Primitive, Material, TextureImage, Clip/Track, Variant
pipeline/common/gltf_writer.py write_glb(): GLB with skin, PNG textures, KHR_materials_variants, animations
pipeline/common/manifest.py   manifest.json = the viewer's contract (variantGroups, clips, parts, stats, props)
                              props = attach bones (`*_attach`) + clips that animate their scale; the viewer parents a
                              held item (kind "object": the 27 balls) to that bone with an identity transform.
                              Baked materials also carry the un-layered albedo: glTF materials[i].extras.rawBaseColorTexture
                              (viewer Options > Albedo: Baked / Raw albedo / Off), and extras.shading = the game's
                              IkCharacter stylised-shading params (halfLambertBias, shadowColor/Strength/Shift, rim*,
                              specularIntensity, occlusionStrength, normalHeight, unlit) used by the viewer's "Game"
                              shading mode (viewer/src/scene/gameShading.ts patches MeshStandardMaterial via onBeforeCompile),
                              and extras.shadowColorTexture = the shadow-side colour baked from ShadowingColorLayerN
                              (x ShadowStrength, darkened/saturated) that the Game shader cross-fades to by light.
                              Baked texture keys include a hash of the layer colours: materials can share albedo+mask
                              but differ in colours (Lucario torso vs legs, Persian body vs ear tips).
pipeline/importers/za/fb.py   schema-free FlatBuffers table reader (field index -> value); fb.dump() for exploring
pipeline/importers/za/trinity.py  parse_trmdl/trmsh/trmbf/trskl/trmtr/trmmt/tranm + vertex decoding + quaternion unpack
pipeline/importers/za/importer.py import_model(Source, id, ...) -> Model; bakes layer colours; FolderSource/ZipSource
pipeline/importers/bntx/      Switch BNTX -> RGBA (Tegra block-linear deswizzle, texture2ddecoder for BC/ASTC)
pipeline/catalog/za.py        ZaCatalog: which zips/folders hold which model + its animations; pokemon_names.json
pipeline/cli.py               list / convert / export / index
pipeline/blender_runner.py    runs blender/*.py headless and reads back a result JSON
blender/export_formats.py     GLB -> FBX (all actions) + OBJ/MTL + PNG, Adaman naming (_NNN_<mesh>_shape[_lodN])
blender/inspect_glb.py        GLB -> counts + preview PNG renders (front/side/mid-clip)
viewer/                       Vite + React + three.js; reads only out/index.json, manifest.json, model.glb
```

Viewer lighting (`viewer/src/scene/environment.ts`) uses the standard game-engine model rather than ad-hoc knobs: light colour as Kelvin (key / back / sky), key + back (rim) directional lights + sky/ground hemisphere ambient, exposure + tone mapping (linear / ACES / AgX / Khronos neutral) + contrast/saturation grade, bundled as named `LIGHTING_PRESETS` (Options > Lighting). `studio` = the old defaults (VS screenshots, unchanged). In the Game shader the first directional light is the key (toon band, shadow colour on its far side, drives the ambient tint); later lights are back/rim lights that only add light on their lit side. The material rim terms add a view-dependent glow independent of the lights, so `rim` is per scene: it was what made Emma's dark hair look grey indoors (fit drove it to 0).
Calibrate a preset against a screenshot with `node tools/match_lighting.mjs samples/<x>_regions.json --preset <id> [--fit] [--out png]` (dev server running): the regions file holds model/clip/frame/camera and matching boxes on the screenshot and the render, and the script prints per-region CIE76 dE and the best env. `indoorWindow` was fitted to `samples/detective_handclap01_*` (mean dE 4.9 vs 13.4 for studio); coordinate descent gets stuck easily, so try several `--env` starting points.

Conventions the IR enforces: Y-up metres; quaternions (x,y,z,w); `Joint.inverse_bind` is a column-vector 4x4 (the Trinity file stores row-vector 4x3, so the importer transposes); `Skeleton.palette[i]` is the joint index that vertex joint index `i` refers to (Trinity rig/JointInfo index, not node index); textures are (H,W,4) uint8 top row first; UVs are glTF convention (top-left origin). Trinity UVs are bottom-left origin, so the ZA importer flips V and negates tangent handedness; verified visually on Pikachu (as-is puts the back stripes on the arms).

## Z-A format facts (verified on real files, not just docs)

- All `.tr*` files are FlatBuffers without identifiers; `.trmbf` too. Field indexes match the Scarlet/Violet layouts used by gftool/ChicoEevee's importer. `fb.dump()` prints a heuristic tree for new file types.
- `.trmdl` -> mesh files (LOD0 first, `_lodN.trmsh`), skeleton, material files. `.trmsh` -> shapes with vertex declarations (usage enum: 1 pos, 2 nrm, 3 tangent, 6 uv, 7 joints, 8 weights; formats 51 f32x3, 43 f16x4, 48 f32x2, 22 u8x4, 39 u16x4) and parts (index range + material). `.trmbf[i]` -> index/vertex byte vectors for shape i.
- `.trskl` nodes carry SRT (euler XYZ radians) + parent + `joint_info` (rig index); `joints[rig]` has the inverse bind 4x3. Pokémon have ~110 nodes / ~90 rig joints; trainers ~190.
- `.trmtr` material: shader `IkCharacter`; texture slots `BaseColorMap/NormalMap/OcclusionMap/LayerMaskMap/...`; colours `BaseColorLayer1..4` are blended over the (near-white) albedo using the `_lym` mask channels R,G,B,A = layers 1..4 (`importer.bake_layers`). `_rare.trmtr` = shiny: same textures, different layer colours. `.trmmt` lists variant sets ("normal", "rare").
- `.tranm`: info (loop, frame count, fps=60) + per-bone tracks; each channel is a union: 1 fixed, 2 one key per frame, 3 keyed u16 frames, 4 keyed u8 frames. Rotations are 48-bit packed quaternions (3x15-bit in [-pi/4, pi/4], 2-bit index of the omitted component, sign bit). Keyed tracks may repeat frames; importer dedupes and decimates.
- Textures: BC7 (albedo, lym), BC5 (normals), BC4 (ao/masks), all block-linear. Mip 0 only.
- Animation naming `<model>_<5-digit index>_<name>.tranm`; the index is stable across species (00000 defaultwait, 00010 defaultidle, 00030 walk, 00100 run...). Pokémon: median 59 clips. Indexes 10000/20000/30000+ are whole alternate sets reusing the same names (Pikachu, Charizard); the importer suffixes them ` [setN]` and sets `Clip.group`.
- Layer masks can be larger than the albedo (Charizard: 1024 vs 512). Resize them per channel (`importer.resize_rgba`): Pillow's RGBA resize premultiplies alpha and zeroes the colour channels where alpha is 0, which silently produced white Pokémon.
- UV rules (all verified against the models.spriters-resource.com preview of Bulbasaur): V origin is bottom-left (flip for glTF); each material's parts sit in their own integer V tile (body_a 0..1, body_b 1..2, cheeks 2..3: drop the tile); the material's `UVScaleOffset` (su, sv, ou, ov) is applied first, typically (2,1,0,0) with a `mirror` U sampler meaning the texture stores half of a symmetric layout. The importer bakes mirrored textures to full width (mirror-concatenated) so every consumer gets plain repeat UVs, but ONLY when that axis has scale >= 2 (a mirror sampler with scale 1 never leaves [0,1]: Pikachu ears/tail, Charizard eyes); the normal map keeps its own scale via `KHR_texture_transform`. Sampler fields are `.trmtr` material field 3, entries parallel to the texture list, wrap enum 0 repeat / 1 clamp / 6 mirror / 7 mirror-once (treated as clamp: identity inside [0,1]; eyes use it). A V scale of 2 on a clamped axis (Pikachu cheeks) is simply scale-then-tile-drop.
- Layers are applied in order, each covering the previous (lerp by its mask channel), not averaged: Pikachu's cheek mask has layers 1 and 2 both fully on and the cheek is layer 2 (red); averaging gave orange.
- Layer colours (`BaseColorLayerN`) are linear-light floats; the albedo is sRGB. Bake in linear space (`srgb_to_linear` / `linear_to_srgb`) or everything comes out dark and dull (lime bulb became dark green).
- Eye glints are emission layer 5 masked by `HighlightMaskMap` (`EnableHighlight`, `EmissionIntensityLayer5`); baked as an emissive texture.
- Layer enable: a layer with alpha > 0 is on; with alpha 0 it is on unless its colour is one of the tool's placeholder swatches (red 1,0,0 / blue .014,.177,.981 / green .102,1,0 / cyan .023,.972,.875), see `importer._layer_strength`. Tarragon's eyelashes (layer 1, near-black) and brows (layer 2) have alpha 0 but are real. Mask alpha channels are often fully on, so placeholders must be filtered.
- (Superseded detail) `BaseColorLayerN` alpha is the layer's enable/strength: 1.0 on Pokémon, 0.0 on trainer clothing/skin (which ship a black placeholder `sh_black_msk` whose decoded alpha is 1). The bake multiplies each mask channel by that alpha, otherwise skin turns teal. Skin tone comes from the material's `BaseColor` factor times the albedo.
- Shadow colour = layer colour x `ShadowingColorLayerN`, undiluted (verified: Griselle's bow albedo x layer 4 = the in-game near-black navy). `ShadowStrength` is the shader's blend toward it, and `ShadowingGIGain` tints ambient light toward it everywhere (viewer "Ambient tint" slider). Screenshots carry scene light (daylight sky turns grey cloth blue), so match them with the Environment sliders, not the bake.
- Trainer clothing colours: base layers are white/disabled and the albedo is the lit colour, but `ShadowingColorLayerN` still carry the garment hues (Griselle: blue-grey jacket, black bow); the shadow bake follows the mask wherever a shadow layer exists even when base layers are disabled.
- Trainer faces: `ColorTableMap` is `ColorTableDivideNumber` zones wide (1, 2 or 4); top half = each zone's lit tint, bottom half = its shadow tint. The face `_lym` channels R,G,B,A are layers 1..4 and `BaseColorIndexN` (1-based) picks layer N's zone (`importer.apply_color_table`); every face has index1 = 1 = skin (R covers ~87%), A = mouth. Picking the zone from R's value (old rule) always chose the last zone: Mable (tr0023, 4 zones) got a black face, 2-zone faces got no skin tint (Tarragon pale instead of tan, Grisham/Griselle paler than their hands). Lit = albedo x top cell (base colour), shadow = albedo x bottom cell (shadow texture).
- Layer colours replace the material `BaseColor` tint instead of stacking on it: `bake_layers(base=BaseColor)` starts unmasked texels at albedo x BaseColor and lerps each layer to albedo x its colour; BaseColor is baked and the glTF factor set to 1. Grisham's hands carry the skin tone in BaseColor and layers 1-2, and multiplying both made them redder than his face.
- Transparency comes from the material's blend type (`.trmtr` alpha type), not `EnableAlphaTest`: `Blend` (glasses lenses: Mable, Corbeau tr0021, Pa Boss ring; Munna), `BlendPreMultiAlpha` (30 Pokémon materials, e.g. Lampent's glass globe), `Add` (glow/energy effects). The shader is usually `IkCharacterTransparent`; opacity is the albedo alpha (Mable's lenses 0.30, verified against the models.spriters-resource rip in `samples/`). Blended materials export as glTF BLEND even when alpha test is also on (it was MASK, which made lenses opaque); `extras.blend` keeps the game mode and `ViewerScene.applyGameBlend` maps Add -> additive, BlendPreMultiAlpha -> premultipliedAlpha.
- Not yet read (see `python -m pipeline.audit_materials`): face vertex sub-layers (`EnableVertexSubLayer`, `BaseColorSubLayerN`, vertex colours are dropped), the hue-shift shading terms (`HueShiftBias`, `MidArea*`/`DarkArea*`, `SaturationPower`), `BaseColorDarkness`, emission/metallic/specular layers. `tools/skin_check.py` compares face vs body skin per trainer.
- Mesh visibility per clip comes from `.tracm` (`trinity.parse_tracm_visibility`: root[1] shapes, shape[5] -> [2] union of fixed / per-frame bits / keyed bits). Clips carry `visibility` in the manifest; rest visibility = the default clip's frame-0 value, else the majority across clips (Griselle's cup, Bulbasaur's vines, Gengar's long tongue start hidden).
- Trainer face meshes carry vertex colours used as shader sub-layer masks (`EnableVertexSubLayer`); never export them as COLOR_0 or the face renders black. The importer drops vertex colours.
- Props: trainer skeletons have `right_attach` / `left_attach` bones (children of the hands); clips scale them 0.01/1 to hide/show a held item. Poké Ball models are `ZA-TrainersNPC.zip ik_chara/model_ob/ob02NN_00_<name>ball/` (0204 = Poké Ball, 0208 = Nest Ball, 27 kinds), ~7.6 cm, pivot at centre, halves skinned to `upper_parts`/`lower_parts`; open/close clips only exist for ob0204 in `ZA-Demo.zip ik_demo/ee/hokaku/`. Which ball a trainer holds is game logic, not data (needs a manual trainer → ball table).
- Remaining shading differences vs the game (ear-base crease, jaw fold on Pikachu) come from vertex normals under PBR lighting; verified identical with the normal map removed. Not a data bug.
- Trainer materials reference shared textures as `../../share/...`; `ZipSource` resolves relative paths across all zips it was given, so trainer sources include `ZA-TrainersSharedTex*`. Player trainer animations live per outfit part under `ik_chara/motion_pc/<part>/` with repeated names but different data; unique NPCs (`model_uq/tr####_*`) are standalone models animated from `motion_uq/<same folder>/`.
- Zip map: `ZA-Poke(+DLC)` models at `ik_pokemon/data/pm####/pm####_##_##/`, `ZA-PokeAnim(+DLC)` same path; trainers in `ZA-TrainersNPC` (`model_uq`, `model_ob`), `ZA-TrainersNPCCommon` (generic NPC bases + face/dress/hair parts), `ZA-TrainersPlayer` (`model_pc` parts + `model_pc_base/model/p0_base.trskl`), `ZA-TrainersSharedTex` (`ik_chara/share/...`). DLC zips add folders, never replace.

## Modular generic NPCs (`cc_<base>`, ZA)

`ZA-TrainersNPCCommon.zip`: `model_cc_base/<b>/<b>_base.trskl` is a skeleton only (bu kid, dm adult M, em stocky M, fm fit M, gm tall M, rv rival); `model_cc_vr/{face,hrs,drs}/<b>_<kind>NNNN_VV_<name>/` are the swappable parts (face / hair / outfit; VV = colourway). `ZaCatalog.cc_models()` yields one model per base (`cc_dm` ...), built by `importers/za/composite.py` with every part as separate meshes tagged `extras {group, option}`; the manifest gets `partGroups` and the viewer's "Appearance" section shows one option per group (default = `NNNN_00` of each).

- Vertex joint indices address a combined palette: base rig at 0..73, each part's own bones at that part's `trskl.palette_offset` (face 164.., outfit 204..). Parts list their bones as root nodes whose local transform is relative to a base bone (face -> `head`, skirt -> `waist`, found numerically from the bind pose). The composite gives each part its own palette positions and remaps primitive joints, so different dresses with same-offset skirt bones do not collide.
- Clips: `motion_cc_base/<b>/<set>/` (first set bare names, others ` [tag]`, `Clip.group`). Per-part cloth/face expression anims (`motion_cc_vr`) are not imported. `df`/`ff`/`gf` (female) parts and clips exist but their base skeletons are not in the rip, so no female composite.
- Female prefixes `df`/`gf`/`ff` have parts and clips but no base skeleton in the rip; they bind to the same rig as `dm`/`gm`/`fm` (`CC_SKELETON_DONORS`; verified: identical joint indices 0..73, clip first-frame translations equal to 3 dp, shipped `tr0031_00_co_member_f` bind positions identical to `dm_base`). `ff` has no `motion_cc_base/ff`, so it uses the `fm` clips. Models `cc_df/cc_gf/cc_ff`.
- Colour comes from `ColorTableMap` (`*_90_alb`, `ColorTableDivideNumber` = 1..32 zones, top row lit / bottom shadow) with `_lym` channels as layers 1..4 and `BaseColorIndexN` picking zones; `LayerMaskScaleN` = 0 disables a layer (mask alpha is often fully on). Modular materials (`import_model(skin_tones=True)`) bake masked-only (untinted where no channel covers; skin keeps layer 1 as base). Nearly everything is palette-driven: outfit tops/bottoms/body/accessories, hair, eyes, eyebrows, eyewear, skin. The per-NPC indices are runtime game data (no table in the rip), the trmtr values are authored per-part defaults (not always 1..4).
- Skin: `body_skin`/`face_skin` share zones 1..4 (deep..pale), exposed as a `skin` variant group (Tone 1..4, default 3) so head and body match; the body's authored `BaseColor` tint is dropped. Other palettes: material `extras.palette` (raw albedo + mask textures, zone colours, default indices) lets the viewer recolour at runtime (`viewer/src/scene/palette.ts`, Appearance section swatch rows per layer); manifest `partGroups[].options[].palettes` describes them. Eyes use masked-only + layer scales (iris only; the old whole-texture tint made them black). The layer mask ships as two opaque images (`palette.mask` RGB, `palette.maskA` alpha as grey): canvas reads premultiply alpha and zero RGB where A=0, which made every recolour fall back to the near-white raw albedo ("model goes white").
- `_VV` duplicates (`dm_drs0000_00/01/02`) are slightly different designs (different vertex counts), not recolours. Real colourways are `.trmmt` sets `tex_00/tex_01` (14 outfits, 2 faces, 2 hairs; swap albedo/normal/mask + shift indices), not exposed yet; `holo*` sets are Holocast effects (skipped).
- The five crowd models' `motion_uq/tr05xx` folders hold vertex-animation textures (`.trvat` + `_body/_body_normal/_body_tangent.bntx`), not bone clips.
- Accessory meshes (glasses/beard: material `obj*`) default hidden (`accessory: true` in manifest parts). The older `tr0566/0570/0575/0578/0581` models are single-mesh crowd silhouettes with one 512x256 albedo: low-res at the source.

## Scarlet & Violet (`--game sv`)

Same Trinity formats as Z-A, so `SvCatalog` (`pipeline/catalog/sv.py`) is a `ZaCatalog` subclass that only overrides zip lists, regexes and path roots; the importer, glTF writer and viewer are shared. Zips: `SV-Poke(+DLC1/2)`, `SV-PokeAnim(+DLC1/2)`, `SV-TrainersNPC/NPCCommon/DLC1/DLC2/Player`, `SV-Misc`, `SV-TrainersSharedTex`.

```bash
./.venv/Scripts/python.exe -m pipeline --game sv list --kind pokemon     # 976 models (incl. forms)
./.venv/Scripts/python.exe -m pipeline --game sv convert --id pm0025_00_00 --id tr0200_00 --id ob0001_00   # -> out/sv/<id>/
```

- Paths drop the `ik_` prefix: `pokemon/data/pm####/pm####_FF_VV/`, `chara/model_uq|model_tr/tr####_<name>/tr####_NN.trmdl`, `chara/model_ob/ob####_<name>/ob####_NN.trmdl`. Pokémon clips sit next to the model id in `SV-PokeAnim` exactly like ZA (`ANIM_RE` works unchanged; 10000/20000 sets become ` [setN]`).
- Materials: shaders `SSS`, `EyeClearCoat` (Pokémon) and `Standard`/`Hair`/`Transparent` (trainers). Slots mostly match ZA (`BaseColorMap`, `LayerMaskMap`, `BaseColorLayerN`, `LayerMaskScaleN`, `NormalMap`) so `bake_layers` works as-is (Pikachu cheeks/ear tips/shiny verified); the AO slot is `AOMap` (importer falls back to it). SV is PBR, so non-`IkCharacter` materials get `extras.pbr = true` and no `extras.shading`. Not read yet: `RoughnessMap` (`_rgn`), `SSSMaskMap`/`SubsurfaceColor`, `BaseColorSubLayerN`, `HairFlowMap`.
- No SV trainer/object animation data was ripped: trainers and objects convert as static rigged models (0 clips). `model_vr` (NPC variants) and `model_pc` (player parts) are not catalogued yet (modular part folders: `p0_<part>####_<name>/p0_<part>####_NN_<part>NN.trmdl`).
- `SV-Poke` has one stray `pm0000_00_00_body_a_alb.dds`; model `pm9999_00_00` and `pm0000_00_00` are placeholders.

## Blender

Blender 5.2.2 LTS is the Microsoft Store build: `%LOCALAPPDATA%\Microsoft\WindowsApps\blender-launcher.exe -b --python <script> -- <args>`; `blender.exe` under WindowsApps is access-denied and the launcher prints nothing, so every script writes a result JSON (see `blender_runner.run_script`). Collada export does not exist in Blender 5 (DAE is out of scope). The factory scene may contain a stray object; scripts delete pre-existing objects after import. FBX export uses `bake_anim_use_all_actions` (NLA-strip mode silently skips muted tracks).

## Pokémon GO (`--game go`)

Source: `~/Documents/pokemon_raw_data/go/pogo_assets/3D Assets` (env `POKEMODELS_GO_DATA`) (override with `--data`), binary FBX 7300 + PNG per folder, no Unity animation assets. Scope so far: trainers only = `Characters/*` (29 folders, ids `go_<folder>`, `out/go/<id>/`). `pipeline/catalog/go.py` lists them; `pipeline/importers/go/importer.py` reads the FBX through headless Blender (`blender/dump_fbx.py` writes `dump.npz` + JSON already in glTF space), then `model_from_dump` builds the IR (testable without Blender, see `test_go_pipeline.py`).

```bash
./.venv/Scripts/python.exe -m pipeline --game go list --kind trainer
./.venv/Scripts/python.exe -m pipeline --game go convert --all --kind trainer      # ~5 s each
./.venv/Scripts/python.exe -m pipeline --game go thumbs
```

- FBX is cm under an armature/empty with scale 0.01 and a 90 deg X rotation. The dump folds that scale into bone translations only (bases stay unit scale) and puts the Z-up -> Y-up rotation on root joints; non-root locals are the file's parent-relative locals. Characters come out ~1.0 m tall (the file's own scale), not 1.7 m.
- Clips are Blender actions named `<armature>|<clip>|Base Layer`, sampled every frame (scene fps 25 or 60 depending on the file) and decimated; tracks equal to rest are dropped. Only 17/29 characters carry curves (the others import as a static rigged T-pose); `Prop1` empty actions are ignored. Loop = name contains idle/loop.
- Materials: base colour texture is resolved from the FBX material by file name next to the FBX. `AvatarMale/FemaleV2` share one untextured material across parts, so each part gets its own material and `<m|f>_hair_default.png` / `<m|f>_skin_default.png` chosen by part name; skin is greyscale (tone comes from `*_skin_mask.png`, not applied yet); eyes/clothing are not assembled (clothing lives in `Clothing/`, not imported).
- Animation also exists in `Addressable Assets/` FBXs only as empty stacks; the 246 `.bundle` files (UnityFS) are unexamined and may hold real AnimationClips (UnityPy spike).

## Brilliant Diamond / Shining Pearl (`--game bdsp`)

Unity 2019.4.27f1 AssetBundles (UnityFS, unencrypted) read with UnityPy; `pipeline/importers/bdsp/importer.py`, `pipeline/catalog/bdsp.py`. Sources are `BDSP-Trainers.7z` / `BDSP-Poke.7z` in the rip folder; extract them once with `py7zr` into `~/Documents/pokemon_raw_data/bdsp/{trainers,poke}` (env `POKEMODELS_BDSP_DATA`) (the catalog reads the extracted bundles; `cab_index.json` there maps CAB names to files). Scope so far: trainers (`trainers/Characters/persons/{battle,field}/<id>`, `objects/ob####_##`); Pokémon (`BDSP-Poke`: `battle|field|common/pm####_FF_VV`, `*/animations/`, `statue/`) are not imported yet.

```bash
./.venv/Scripts/python.exe -m pipeline --game bdsp list --kind trainer
./.venv/Scripts/python.exe -m pipeline --game bdsp convert --id tr0001_00 --id pc0001_00 --id fc0001_00
```

- A `tr*` bundle is self-contained: Transform hierarchy, SkinnedMeshRenderers (+ Mesh, Materials, ASTC Texture2Ds) and generic AnimationClips. `fc*` (player field models) point into other bundles (meshes in `fc0001_11`, shared `pc_parts`): `BdspCatalog.import_ref` loads the externals' bundles too (`externals()` -> `cab_index`), `_load` returns only the primary bundle's objects and PPtrs are `deref()`ed (materials/textures/meshes may live in a dependency). The shader bundle (`cab-1dc8...`) is not in the rip; shader properties are read from the Material only.
- Unity is left-handed; characters face +Z in both, so the importer mirrors X (pos/normal x, quat (x,-y,-z,w), tangent (-x,y,z,-w), matrices S M S, reversed winding) and flips V.
- Clips are generic (Transform bindings by CRC32 of the bone path relative to the root, root name excluded; attribute 1 pos / 2 quat / 3 scale, typeID 114 = MonoBehaviour float curves, ignored). `m_MuscleClip.m_Clip.data`: streamed (time f32, count, count x [curve idx, 4 coeff f32, value = last]) then dense then constant curves; slots are assigned to bindings in order. The first streamed frame is stamped -FLT_MAX (initial value): clamp to 0. Keys are taken at their values and linearly interpolated (cubic slopes ignored).
- Materials: `_MainTex`=`_col`, `_BumpTex`=`_nor`, `_MaskTex`=`_msk`, `_ComplexTex`=`_uti` (unread). `_col` is near-greyscale; colour = albedo x tint from the mask (`bake_tint`): zones R=`_PrimaryColor`, G=`_SecondaryColor`, B=`_SkinColor` (black = untinted); body/face materials use `_SkinColor` on the R/B regions (their `_PrimaryColor` is a cool shadow tint, applying it turned faces blue); the hair shader (`_LIGHTING_HAIR`) uses `_PrimaryColor` on the whole texture. Verified visually on Cynthia (tr0001: black coat, gold fur, blonde), the cowgirl and the player (pc0001). Raw albedo goes to `raw_base_color`. `_ALPHABLEND_ON` + real alpha -> BLEND.
- Player models are modular: `pc0001_00` is the base, `pc0001_10..22` are further part bundles (not yet composed into an Appearance group). `fc0001_00` carries hair1/2, shoes, bicycle, surf mount ... as separate meshes; inactive GameObjects / disabled SMRs import with `Mesh.visible = False`.
