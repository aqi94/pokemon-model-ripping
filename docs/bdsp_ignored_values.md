# BDSP trainer materials: Unity values the importer/viewer ignore or approximate

Audit only (no code changed). Data: all `Material` objects in `~/Documents/pokemon_raw_data/bdsp/trainers/Characters/{persons/battle (917 mats, 124 bundles), objects (222 mats, 128 bundles), persons/field (966 mats, 161 bundles)}` = 2105 materials. The shader bundle is not in the rip, so the shader's own constants (Cull, exact formulas) are unknown; only material values are audited. Every material points at the same shader family (`_Lighting` 0 = standard, 2 = hair); 8 materials are the Pokemon shader (`_PokemonStencil`: Bonsly dolls, tr1021 Pokemon bodies).

## What we read today

- Importer (`pipeline/importers/bdsp/importer.py::_material`): textures `_MainTex/_BumpTex/_MaskTex/_ComplexTex (decoded, unused)/_Col0Tex/_NormalMapTex`; keywords `_ENVMAP_ON`, `_LIGHTING_HAIR`, `_ALPHABLEND_ON`; floats `_Reflectivity` (only to scale the `_ENVMAP_ON` boost down below 0.15) and `_ColorBaseU/V`; colours `_PrimaryColor/_SecondaryColor/_SkinColor` (tint bake).
- All other colours are copied to `extras.unity.colors`, but the viewer reads only `extras.unity.keywords` (`reflection.ts::isReflective`). All other floats are dropped.
- No `emissive`, `alpha_cutoff` or `double_sided` is ever set; vertex colours are not exported.
- Non-`_ENVMAP_ON` materials get viewer default shading (specularIntensity 0.04, rim 0). `_ENVMAP_ON` ones get a fixed 3.0 / env 1.5 (0.6 / 0.4 if the albedo is dark), scaled by reflectivity/0.3 only when reflectivity < 0.15.

## Summary (sorted by risk; risk is a guess from the data unless stated)

| # | Property | Now | Data range | Risk |
|---|---|---|---|---|
| 1 | `_SpecularIntensity` / `_SpecularCosinePower` / `_SpecularColor` on non-env materials | default 0.04, white | 0..16 / 0.1..256; 561 of 2105 mats not (2.0, 10.0); 428 mats non-white colour | high |
| 2 | `_EmissionColor` x `_EmissionColorIntensity` (+ `_uti` R mask) | ignored | intensity 0..16, 830 mats > 0; 336 with colour alpha > 0 | high |
| 3 | `_Reflectivity` > 0.3 and `_ReflectionBias` on `_ENVMAP_ON` | fixed 1.5 / roughness 0.2 | refl 0..0.92; bias 0..5 (blur mip) | med-high |
| 4 | `_ALPHATEST_ON` / `_Cutoff` / `_AlphaClip` | cutout not applied unless also `_ALPHABLEND_ON` | 64 mats; 8 pure-cutout in battle/obj | med |
| 5 | Vertex colours (grey RGB + alpha) | dropped | 135 of 189 battle meshes vary | med (meaning unknown) |
| 6 | Rim: `_RimColor/_RimColorShadow/_RimPower/_RimScale/_RimStrength/_RimBlendWeight` | none | RimColor black on 492 mats (all eyes, some face/body) | med |
| 7 | `_ComplexTex` (`_uti`) channels | decoded, unused | R sparse (eye glints), G sparse (cloth/hair), B ~1 | med (meaning is a guess) |
| 8 | Blend state: `_Mode 1`, `_SrcBlend 5`, `_DstBlend 10`, `_ZWrite 0`, queue 3000 | ordinary BLEND with depth write | 34 mats (28 with queue 3000 + zwrite 0) | med |
| 9 | Hair anisotropy `_AnisLevelH/V`, blue `_SpecularColor` (0.7,0.8,1.0) | none | 9 mats H != 1, 1 mat V = 11.12; 153 blue spec colours | low-med |
| 10 | `_MaskShadowBlendWeight/_MaskShadowExponential`, `_Subsurface*` | none | blend 0 / 0.5 / 1 on 712 / 300 / 1083 mats; subsurface almost all default | low-med |
| 11 | `_FresnelPower` | fixed | 0.1..8, 4.0 on 1678 | low |
| 12 | `_Color` tint, `_ColorIntensity` | ignored | `_Color` non-white on 6 mats; intensity 0.9..1.1 on 27 | low |
| 13 | `_BumpScale` | ignored (1.0) | 0..1; 58 mats != 1 | low |
| 14 | Shooting light `_ShootingLightColor/_Offset` | none | constant (0.2,0.15,0.1) on 346 mats | low (constant) |
| 15 | Cull / double sided | `double_sided` never set | not in material (shader state) | unknown |

## Details

### 1. Specular on non-env materials
Now: only `_ENVMAP_ON` materials get a specular boost; everything else uses the viewer default (0.04), so `_SpecularIntensity` and `_SpecularColor` are lost. Data: `_SpecularIntensity` 2.0 (1539 mats), 0.75 (234, hair), 4.0 (50), 7.0 (38), 1.0 (29), up to 16; power 10 (1536), 50 (246), 3, 40, up to 256. Non-env with intensity >= 4: 108 mats / 104 models. Strongest: all Poke Ball objects `ob0201..ob0213` (`ball` 7.0 / 40), held-ball materials `tr0042/0043/1033/1048/1081 ball`, `tr0047_00 scarf` 16 / 256 (plus `_AnisLevelH` 0.38, `_AnisLevelV` 11.12), `pc0002_18 metal 2` 12.7 / 3 and `metal` 7.5 / 3, `tr2008_00 acce` 8 / 35, `tr1053_00 wear` 5.5, `ob0008_00 presentbox` 5. `_SpecularColor` is non-white on 428 mats: (0.7,0.8,1.0) x163 (hair, 67 in battle; acce 7), (0.43,0.41,0.59) x54, (1.0,0.85,0.5) x36. Hair uses 0.75 / 50 on 102 of ~110 battle hair mats. Risk high for balls and bright props (matte versus the game). Evidence: the same fields caused the Construction Worker helmet over-shine in the opposite direction. One line: map `_SpecularIntensity/_CosinePower/_SpecularColor` into `extras.shading` for all materials.

### 2. Emission
Now: nothing. `_EmissionColorIntensity`: 0 on 1275 mats; 1 x270, 2 x258, 3 x173, then up to 16. Eyes (`Leye/Reye`) carry 2-3 (146 at 3.0, 90 at 2.0) with colour (0.81,0.87,0.90, a=1), which matches `_uti` R being a sparse eye-glint mask (R mean 0.006, max 0.64-0.67 on pc0001_00 / tr0040_00 / tr0047_00 eyes), so catchlights are probably emissive in game (guess). Strongest non-eye: `ob1004_00 light` 12, `tr0039_00 wear2` 10 and `wear` 8.5 (yellow 1.0,0.72,0.05), `tr1055_00 lens` 9.5, `pc0001_11 / pc0002_11 / ob1003_00 light` 8, `ob1009_09 badge1..8` 6 (coloured), `ob1008_00 light` 4. Colour alpha is 0 on 682 and 1 on 235 battle mats (meaning unknown, maybe an enable). `_EmissionOnTime/_EmissionOffTime` (0.2 / 0.4 on 16 mats) suggest blinking. Risk high for lamps, badges, tr0039 (dull instead of glowing); eyes medium. One line: set `emissive_factor` = colour x intensity (masked by `_uti` R) for materials with intensity > 0.

### 3. Reflectivity above 0.3, ReflectionBias
`_ENVMAP_ON` is on 238 mats, identical to `_EnvironmentMapEnable` 1. Reflectivity on those: 0.2 x74, 0.05 x68, 0.1 x24, 0.3 x23, 0.4 x13, 0.5 x5, plus `tr1046_00 glass` 0.92, `tr0047_00 metal` 0.6, `tr1029_00 metal` 0.5, `tr0051_00 metal` 0.4 (battle: 65 of 73 are >= 0.15, so they keep the fixed 3.0 / 1.5). The importer only reduces for < 0.15, so a 0.92 mirror and a 0.2 metal get the same 1.5. `_ReflectionBias` (blur mip) is 0 on 91, 0.77 on 62, 3 on 34, 4 on 23, 5 on 7 (tr0001 acce 3, tr0018/0031 metal 4, tr0040 metal 3, ob1008 base 5, ob1020 pearls 5, ob1009 badgecase 3). The game blurs the reflection, we use fixed roughness 0.2, so these look too mirror-like. `_FresnelPower` on env mats: 4 x118, 2 x66, 8 x14, 0.6 x7 (tr1046 glass). Non-env mats with reflectivity >= 0.4: pc0002_10/15/22 `wear2` 0.5, ob1015_00 `ball3` 1.0, ob1016_00 `record` 0.4 (no env map because the keyword is absent; unknown whether the game reflects). Risk med-high. One line: scale envIntensity by reflectivity both ways and roughness by `_ReflectionBias`.

### 4. Alpha test
64 `_ALPHATEST_ON` mats (queue 2450, `_AlphaClip` 1); 51 of them also carry `_ALPHABLEND_ON` and go through BLEND. Pure cutout in battle/objects (opaque with texture holes in our output, because only `_ALPHABLEND_ON` flips alpha mode): tr0030_00 wear, tr1055_00 wear/ski, tr1056_00 wear/ski, tr1068_00 wear, tr1072_00 wear, tr1088_00 wear (`_Cutoff` 0.5). 32 of the 64 have `_Cutoff` 0.0 (glass/shadow/cap; behave like plain blend). Risk med. One line: set alpha_mode MASK with `_Cutoff` for `_ALPHATEST_ON` without `_ALPHABLEND_ON`.

### 5. Vertex colours
Dropped. 186 of 189 battle meshes carry a colour channel; 135 vary (RGB grey, std up to ~0.3; alpha also varies on some, e.g. pc0001_10/_11/_14/_17, std 17-35 of 255). Looks like baked occlusion/shade (`*_baseSkin`), alpha possibly a mask. Unknown how the shader uses it. Risk med; the only evidence is that the data varies. One line: export as COLOR_0 only after checking it against a game screenshot.

### 6. Rim
Unity rim values are not translated (viewer rim default 0 for these materials). `_RimColor` white on 1552, black on 492 (all 220 eyes, 8 face, 7 body, 3 glass, 2 ball: rim off), others pastel ((0.95,0.76,0.76), (0.65,0.79,1.0)). `_RimPower` 8 (1548), 4, 1.5 (226, broad), up to 32; `_RimScale` 0.5 (1210), 1.0, 0.25, up to 4.12 (46 mats >= 1.5, e.g. ob1008_00, ob1015_00); `_RimStrength` 8 on 2020, others 0.5..32; `_RimBlendWeight` 0.3 (1311), 0.5, 0.4. `USE_RIM_MASK_TEX` only on tr1021 Body00. Risk med: silhouettes lack the soft rim glow; broad-rim (power <= 1.5, 260 mats) ones most. One line: feed `_RimColor/_RimPower/_RimScale` into `extras.shading.rim*`.

### 7. `_ComplexTex` (`*_uti`) channels
Stats over 672 battle and 133 object textures: A = 1 everywhere; B mean 0.91 (nearly 1, dips to 0.35-0.6 in creases: looks like AO / shade multiplier); R is 0 in 508 textures and sparse on eyes (glint) so likely an emission mask; G is 0 in 305 textures, mean 0.1-0.2 with peaks to 1.0 on cloth, hats, shoes and hair so likely a specular / gloss mask. All guesses from statistics and from correlation with `_EmissionColorIntensity` (R max > 0.5 in 126 of 197 battle textures whose material has intensity > 0). Not verified in game. Risk med. One line: verify channel meaning with a render before using.

### 8. Blend modes
`_Mode 1` on 34 mats (`_SrcBlend 5`, `_DstBlend 10` = standard alpha blend); 28 of them are queue 3000 with `_ZWrite 0` (ob1008_00 x3, tr1062_00 / fc1062_00 x2, pc0002_00/11/13 shadow, tr0030 / tr0062 ...). Another 715 `_ALPHABLEND_ON` mats use `_Mode 0` (opaque queue, alpha used as texture blend). No additive (`_DstBlend 1`) exists. We use depth-writing BLEND for all, so draw-order artefacts on glasses and shadow planes are possible. Risk med. One line: disable depthWrite when `_ZWrite 0`.

### 9-14. Smaller items
- Hair anisotropy: `_AnisLevelH` 0.15 on shoes/wear of pc0002_00/14/19/20, tr0018_00 wear/acce, 0.38 tr0047_00 scarf, 0.2 tr1052_00 wear2; `_AnisLevelV` 11.12 only on tr0047_00 scarf. `_AnisotropicLightingEnable` is 0 everywhere it exists.
- `_MaskShadowBlendWeight` 0 (712) / 0.5 (300) / 1 (1083), 0.7-0.8 on 5 mats; `_MaskShadowExponential` 4 on 2083, 16 on 4. `_SubsurfaceBlendWeight` < 1 on 5 mats (ob1008, tr0047, tr0048, ob1015); `_SubsurfaceAdditionalColor` non-black on 12 mats (ob1008_00 x5, ob1015_00, ob1020_00, tr0047_00, ob1019_00, fc2017_00).
- `_Color` multiplies albedo in Unity and is ignored: 0.5 grey on ob2000_00 box/box_icon; gold (0.67,0.55,0.19) on tr2009 / fc2009 / fc2015 / fc2023 `metal`. `_ColorIntensity` 0.9 x16, 0.95 x8, 1.1 x3 (ob1009_00 x8, tr0044_00, fc0044_00).
- Pokemon-shader mats (tr1021_00 bodies, ob1017/1018 and pc_parts `Eye_*`): `_ConstantColorSd0` shadow tints (e.g. 0.875,0.827,0.702), `_Layer1*`, `_EmissionMaskUse`, `_SpecularScale/_SpecularPower` 0.8..10; only `_ColorBaseU/V` are read.
- `_BumpScale` 0.5 on 39 mats, 0.8 on 9, 0 on 10 (ob1009_09 x8 at 0.5 / 0), 59 mats have no `_BumpTex`.
- `_Cull`: only `_CullMode 2` (back) on the 8 Pokemon-shader mats; otherwise Cull is shader state and not in the data. `double_sided` is always False, so a two-sided piece (hair card, cape, skirt) would show one side only if the game draws it double-sided.
- Shooting light (`_ShootingLightColor` (0.2,0.15,0.1), `_ShootingLightOffset` (5,-5,0)) is identical on all 346 mats that have it (a fixed fill light for battle shots); not modelled.

## Models to eyeball first

1. `tr0047_00` (scarf spec 16 / 256 plus anisotropy, metal refl 0.6 bias 4)
2. `tr1046_00` (glass reflectivity 0.92)
3. `ob0201_00 .. ob0213_00` and balls held by `tr0042/0043/1033/1048/1081` (spec 7 / 40, no highlight today)
4. `tr0039_00` (emission 10 / 8.5 yellow), `tr1055_00` (lens emission 9.5, cutout wear/ski), `tr1056_00`
5. `ob1009_09` (8 emissive badges, bump 0.5 / 0), `ob1008_00`, `ob1004_00`, `ob1003_00`, `pc0001_11` / `pc0002_11` (`light` emission 8-12)
6. `pc0002_18` (metal spec 12.7 / 3), `tr2008_00 acce`, `tr1053_00 wear`
7. `tr1021_00` (Pokemon-shader bodies, rim mask, `_ConstantColorSd0`), `ob1017_00`, `ob1018_00`
8. `tr1062_00`, `tr0030_00` (blend + zwrite 0 glass / shadow, pure cutout wear)
9. `tr1052_00` guitar (spec 8, env), `tr1033_00` / `tr1051_00` (`_LIGHTING_PHONE`)
10. Any trainer's eyes (emission 2-3 and `_uti` R glint mask)

## How this was found

Run with `.venv/Scripts/python.exe -I` (UnityPy is in the venv). Scratch copies of the scripts live in `C:\Users\andyq\AppData\Local\Temp\claude\C--Users-andyq-Documents-pokemon-model-ripping\ae989b5c-7374-453b-8314-723b097f50c5\scratchpad\` (`scan_mats.py`, `summ.py`, `q2.py`, `q4.py`, `scan_uti.py`, `scan_vc.py`). Each run takes 2-15 s per folder.

Step 1, dump all materials (`python -I scan_mats.py battle.json <dir>`; dir = folder of bundle files, e.g. `~/Documents/pokemon_raw_data/bdsp/trainers/Characters/persons/battle`, `.../objects`, `.../persons/field`):

```python
import sys, os, json, UnityPy
out, dirs = sys.argv[1], sys.argv[2:]
res = []
for d in dirs:
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isfile(p): continue
        env = UnityPy.load(p)
        for o in env.objects:
            if o.type.name != "Material": continue
            m = o.read(); pr = m.m_SavedProperties
            res.append({"model": name, "mat": m.m_Name, "shader_pid": (m.m_Shader.file_id, m.m_Shader.path_id),
              "kw": list(m.m_ShaderKeywords) if isinstance(m.m_ShaderKeywords, list) else str(m.m_ShaderKeywords).split(),
              "floats": {k: float(v) for k, v in pr.m_Floats},
              "colors": {k: [float(c.r), float(c.g), float(c.b), float(c.a)] for k, c in pr.m_Colors},
              "tex": {k: (v.m_Texture.path_id != 0) for k, v in pr.m_TexEnvs},
              "cull_tags": dict(getattr(m, "stringTagMap", {}) or {}), "queue": getattr(m, "m_CustomRenderQueue", None)})
json.dump(res, open(out, "w"))
```

Step 2, per-property distributions and deviants:

```python
import json, collections
rows = [r for f in ("battle.json", "obj.json", "field.json") for r in json.load(open(f))]
fl = collections.defaultdict(list)
for r in rows:
    for k, v in r["floats"].items(): fl[k].append(v)
for k, v in sorted(fl.items()):
    print(k, len(v), min(v), max(v), collections.Counter(round(x, 3) for x in v).most_common(5))
# colours: same with tuple(round(x, 3) for x in rgba); keywords: Counter(k for r in rows for k in r["kw"])
# deviants: collections.Counter(r["model"] for r in rows if r["floats"]["_SpecularIntensity"] >= 4).most_common(10)
```

Step 3, `_uti` channel stats (for each material's `_ComplexTex`, `t = v.m_Texture.deref().read()`):

```python
import numpy as np
a = np.asarray(t.image.convert("RGBA"), np.float32) / 255.0
stats = {c: (a[..., i].mean(), np.percentile(a[..., i], 99), a[..., i].min(), a[..., i].max()) for i, c in enumerate("RGBA")}
# correlate with m_Floats["_EmissionColorIntensity"] of the owning material
```

Step 4, vertex colours:

```python
from UnityPy.helpers.MeshHelper import MeshHandler
mh = MeshHandler(o.read()); mh.process()
a = np.asarray(mh.m_Colors, np.float32).reshape(len(mh.m_Colors), -1)   # varying if a.std(axis=0).max() > 0.02
```

Compare against the code with `grep -n "m_Floats\|m_Colors\|_ENVMAP\|_Reflectivity" pipeline/importers/bdsp/importer.py` and `grep -rn "unity" viewer/src`.

Limits: the field-person bundles (`persons/field`) are included in the counts although the request was battle + objects (per-set counts are noted where it matters); the shader bundle is absent so Cull / ZTest / exact formulas are unknown; items marked "guess" come from statistics only.
