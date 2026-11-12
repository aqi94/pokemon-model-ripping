"""Let's Go Pikachu/Eevee field maps and field props: `field/<area>/*.gfbmdl` + DDS -> static common `Model`.

Same `.gfbmdl` container as the Pokemon/trainers (see `gfbmdl.py`), but a different shader family (`Field*Shader*`).
Verified on `field/area02` (14 maps) and `field/buildmodel` (100 props), all 360 field files surveyed:

* Vertices are already in model (world) space, centimetres, Y up. Root[7] groups map each mesh to a bone, but every
  bone a group references has an identity local transform (215 of 20606 bones are not identity: door hinges,
  `dendo_machine` Pokemon slots, propellers; none of them carries a group), so the meshes are written unparented.
  `build_field_model` still applies a referenced bone's world matrix should one ever be non-identity.
* Vertex layout: pos f32x3, normal f16x4, tangent f16x4, UV0 (f32 or f16), optional UV1, UV2 (usage 5), colour u8x4,
  joints/weights only on a few animated props (enc_grass, windmill): those import unskinned at rest.
* Texture slots carry samplers {1 wrap U, 2 wrap V} with the SwSh enum (0 repeat on colour maps, 1 clamp on the toon
  tables, 3 border on `DepthBuffer`). Colour maps tile far outside [0, 1] (ground UV0 spans ~11 units per mesh),
  so the base colour keeps the source UVs and a REPEAT sampler (no texture baking).
* Material field 20 = named render states: `CullMode` (2 = two-sided foliage; field 7 is the same flag inverted),
  `BlendMode` (= field 6; 5 water/glass/premultiplied, 1 alpha -> BLEND, 2 shadow casters), `DiscardEnable` (1025 =
  alpha cut-out at `DiscardValuie`), and the UV set each texture samples (`Tex0N_UV`, `UV_tex0N`, `UVSet0`...).

The field shaders are multi-texture (ground = 2 ground + 2 grass layers blended by masks; cliffs = cliff strip + ground
cap + border mask; trees = silhouette + colour ramp + mask). Only one colour layer per material is exported (see
`BASE_SLOTS`); the others are listed in `extras.fieldLayers` so a later blend shader can use them. Rules that are
hypotheses rather than read values are recorded in `docs/ledger/lgpe.json` (`field:*`).
"""

from __future__ import annotations

import re
import struct
from typing import Callable

import numpy as np

from ...common.model import Material, Mesh, Model, Primitive, TextureImage
from .gfbmdl import GfMaterial, Gfbmdl, parse_gfbmdl
from .importer import CM, _quat_to_mat, decode_dds
from ..za import trinity as T

# Base-colour slot per material, first present wins; its UV set is read from the material's render states (field 20),
# trying the listed keys in order (names differ per shader: cliffs/rocks `Tex00_UV` = ground 0, `Tex01_UV` = rock 1;
# grass `UVSet0`; trees `UV_tex01`...), else the default. Slot names are the engine's shader inputs.
BASE_SLOTS: list[tuple[str, tuple[str, ...], int]] = [
    ("Texture01", ("Tex01_UV", "UV_tex01", "UVset", "UV"), 0),  # objects, roofs, trees, tall grass, waves
    ("TextureMap01", ("UVSet0", "UV_tex0"), 0),  # grass shaders, windows
    ("CliffTex01", ("Tex01_UV",), 1),  # cliffs: the rock strip (UV1 runs along the cliff, U 0..22)
    ("Rock_tex", ("Tex01_UV",), 1),  # rocks
    ("GroundTex01", ("UV_tex0", "Tex00_UV"), 0),  # ground: the first ground layer
    ("Ground00Tex01", ("Ground00UVset",), 0),  # battle ground (skydome)
]
ON = 1025  # render-state toggle value for "on" (0 = off)
# FieldWaterShader01 has no albedo: `WaterNetTex01/02` are caustic-line overlays (black with white lines) tinted by
# `WaterNetTex0N_Color`; the water body is the `Base_Color1` (..`Base_Color2`) colour, used as the base factor.
# Texture slots that are not colour layers (lookup tables, screen-space or shadow inputs).
NON_COLOUR_SLOTS = re.compile(
    r"^(DepthBuffer|ShadowToonTable|lightToonTable|LightProjMap|LightTblTex|NormalMap|ParallaxMap|ParallaxTexture\d+|"
    r"HighlightMap|HightlightMap|Hilight|Ground\d\dHighlightMap|EmissionMap|Reflection_tex|sky_tex|Mask_texture)$"
)
# Shaders whose vertex RGB is data rather than a colour multiplier: tree vertex colours are ~0 RGB with the alpha
# varying (wind/ramp weights?; FieldTreeShader04/05 mean RGB 0.33/0.48). Everywhere else the vertex colour is a baked
# light/shadow tint (white..teal (46,123,110), mean 0.7-0.9) and is written as COLOR_0.
DATA_VERTEX_COLOUR = re.compile(r"^FieldTreeShader")
# Shadow casters: no textures, `SkipMainRendering` (drawn only into the shadow map).
SHADOW_ONLY = re.compile(r"^Field\w*ShadowOnly|^FieldForestShadowShader")
BLEND_MODES = {5: "BLEND", 1: "BLEND"}


def _uv_transform(gm: GfMaterial, slot: str) -> tuple[float, float, float, float]:
    """`TexNN_Scale_U/V` + `TexNN_Translate_U/V` for `Texture0N`/`TextureMap0N` slots (NN = N). Identity elsewhere."""
    m = re.match(r"^Texture(?:Map)?0(\d)$", slot)
    if not m:
        return (1.0, 1.0, 0.0, 0.0)
    k = f"Tex0{m.group(1)}_"
    f = gm.floats
    return (f.get(k + "Scale_U", 1.0) or 1.0, f.get(k + "Scale_V", 1.0) or 1.0, f.get(k + "Translate_U", 0.0), f.get(k + "Translate_V", 0.0))


def _ramp_tree(gm: GfMaterial, tex: "_Tex") -> tuple[np.ndarray, str] | None:
    """Ramp-shaded foliage (FieldTreeShader04/05, every material with this slot layout): `Texture01` is a black
    silhouette, `Texture03` the white leaf shapes (alpha = cut-out) and `Texture02` a vertical colour ramp (W x 4W,
    lit green at the top fading to transparent). The game picks the ramp row by lighting; without that shader the base
    colour is `Texture03` x the ramp's alpha-weighted mean colour (hypothesis, ledger `field:rampTree`)."""
    t2, t3 = gm.textures.get("Texture02"), gm.textures.get("Texture03")
    if not t2 or not t3:
        return None
    ramp, mask = tex.get(t2), tex.get(t3)
    if ramp is None or mask is None or ramp.shape[0] < 2 * ramp.shape[1]:
        return None
    a = ramp[..., 3:4].astype(np.float32) / 255.0
    tint = (ramp[..., :3].astype(np.float32) * a).reshape(-1, 3).sum(0) / max(float(a.sum()), 1e-6)
    out = mask.copy()
    out[..., :3] = (mask[..., :3].astype(np.float32) * tint / 255.0).clip(0, 255).astype(np.uint8)
    return out, f"{t3}_x_{t2}"


def _bake_window(gm: GfMaterial, img: np.ndarray) -> np.ndarray:
    """FieldWindowShader (house/shop/gym/Center windows, lamp glass, the Center and Gym door glass): the glass colour is in
    the material's `Color01` and the `TextureMap01` image is added on top of it. House/shop windows: `Color01` is pale sky
    blue (0.63, 0.79, 1.0) and the texture a dark grey mask (max 0.4-0.8: frame vignette, highlights), which used to export
    alone and render near-black panes. Center/Gym door leaves: `Color01` is near-black and the texture is the coloured
    Poke Ball logo. colour = screen(Color01, texture) = Color01 + (1 - Color01) x texture (hypothesis, ledger
    `field:window`): the same as the WP 2.5c sum where either is dark (house panes, door logos), but a bright mask (the Mart
    window body is 0.25-0.3 with 0.8 streaks) no longer saturates the whole pane to white. Opaque, like the source's
    `BlendMode` 0; `Color02`, `Min/Max/Intensity` and the cloud projection (`LightProjMap`) are not drawn."""
    c1 = gm.colors.get("Color01")
    if c1 is None:
        return img
    c = np.asarray(c1, np.float32)
    rgb = c + (1.0 - c) * (img[..., :3].astype(np.float32) / 255.0)
    out = np.empty_like(img)
    out[..., :3] = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def _layer(gm: GfMaterial, tex: "_Tex", slot: str, uv: int) -> dict | None:
    """Reference to one extra colour/mask layer for `extras.fieldBlend` & co: the texture is registered so the glTF writer
    stores it (an image + texture with this layer's sampler) and the FBX export writes it as a PNG beside the model."""
    name = gm.textures.get(slot)
    if not name or name.startswith("dummy") or tex.get(name) is None:
        return None
    tex.register(name, tex.get(name))
    return {"texture": name, "uv": int(uv), "wrap": list(gm.samplers.get(slot, ("repeat", "repeat")))}


def field_blend(gm: GfMaterial, tex: "_Tex", base: str | None) -> dict:
    """Engine-neutral recipe of the field shaders' layer blending (WP 2.5d; hypotheses, ledger `field:blend*`), keyed by the
    shader-input slots a material uses (survey of all 360 field models: 523 cliff/rock, 136 ground, 299 lawn, 77 roof
    materials), never by model. A renderer computes, per pixel:

        under = base colour, or lerp(base, base2, noise.r | vertex alpha)            ("base2", "base2By")
        over  = lerp(over[0], over[1], noise.r)                                       ("over", "noise")
        colour = lerp(under, over, mask.a) * mask.rgb                                 ("mask")
        cut-out alpha = cutout.a (alpha test at `cutoff`)                             ("cutout")

    * Cliffs / rocks (`BorderTex` present): the rock strip (base, UV1) is capped by the two ground layers (`GroundTex01/02`,
      UV `Tex00_UV`) varied by the `BlendTex` noise; `BorderTex` is a trim sheet of leafy-edged bands on UV `GroundBlend`
      (2): its alpha is ~1 on cliff tops and ~0.3 on faces (measured over the area02 cliffs), its RGB a soft edge shadow.
    * Ground (`GrassTex01` present): soil (`GroundTex01/02`, noise `GrassBlendTex`) under grass (`GrassTex01/02`), the
      `BlendTex` trim sheet on UV `UV_tex02` (2) says where the grass is (alpha 1 = grass: 0.94 of Pallet's main lawn).
    * Lawn cards (`green_blend` present, FieldGrassShader01/02/05): `TextureMap01/02` varied by `green_blend`; the leafy
      card outline is `green_hikari`'s alpha on UV `UVSet01` (1) when `DiscardEnable` (cut at `DiscardValuie`: inside at
      ~75 % of the area02 card centroids, outside at the edge vertices), or the `alpha01` layer on UV0
      (FieldGrassShader05); `light_line` (diagonal streaks, `scroll_U/V`) is a moving highlight.
    * Roofs (`Texture02` with `TexBlendType` 1 and `VertexColor` on, FieldDefaultShader): `Texture02` is a sun-bleached copy
      of the tiles, blended in by the vertex alpha (0 on most of the roof, ramping to 1 at eaves/ridges).
    * Pre-shaded pines (`Texture01` + `Texture02` + `DirlightColor`, FieldTreeShader02): colour += Texture02 x lightColor
      ("base2By": "lightAdd").
    * Blended lawn decals without discard (`rockmask01`): `green_hikari`'s alpha on UV1 is the opacity ("opacity").
    * Tall-grass tufts (FieldGrassShader04): `Texture03` is the light line."""
    r = gm.render
    t = gm.textures
    out: dict = {}

    def put(key: str, slot: str, uv: int) -> dict | None:
        ref = _layer(gm, tex, slot, uv)
        if ref is not None:
            out[key] = ref
        return ref

    if t.get("BorderTex"):
        uv = r.get("Tex00_UV", 0)
        over = [x for x in (_layer(gm, tex, "GroundTex01", uv), _layer(gm, tex, "GroundTex02", uv)) if x]
        if over and put("mask", "BorderTex", r.get("GroundBlend", 2)):
            out["over"] = over
            put("noise", "BlendTex", uv)
    elif t.get("GrassTex01"):
        uv = r.get("UV_tex0", 0)
        if base == "GroundTex01" and put("base2", "GroundTex02", uv):
            out["base2By"] = "noise"
        over = [x for x in (_layer(gm, tex, "GrassTex01", uv), _layer(gm, tex, "GrassTex02", uv)) if x]
        if over and put("mask", "BlendTex", r.get("UV_tex02", 2)):
            out["over"] = over
        put("noise", "GrassBlendTex", uv)
    elif t.get("green_blend"):
        uv = r.get("UVSet0", 0)
        if base == "TextureMap01" and put("base2", "TextureMap02", uv):
            out["base2By"] = "noise"
            put("noise", "green_blend", uv)
        if r.get("DiscardEnable") == ON and put("cutout", "green_hikari", r.get("UVSet01", 1)) is None:
            put("cutout", "alpha01", 0)  # FieldGrassShader05 (grass_s04): the blade shape
        if "cutout" in out:
            out["cutoff"] = float(gm.floats.get("DiscardValuie") or 0.5)
        elif r.get("BlendMode", gm.blend) in BLEND_MODES:
            put("opacity", "green_hikari", r.get("UVSet01", 1))  # blended decal (rockmask01: soil under rocks)
        if put("lightLine", "light_line", 0):
            out["lightLineScroll"] = [float(gm.floats.get("scroll_U", 0.0)), float(gm.floats.get("scroll_V", 0.0))]
    elif gm.shader.startswith("FieldGrassShader") and t.get("Texture03"):
        # Tall-grass tufts (FieldGrassShader04, 51 materials): Texture03 is the same diagonal-streak texture the lawn shader
        # names `light_line` (grass08_3_com in both) -> the same moving highlight. Texture02 (a lighter copy of the tufts
        # on UV1) stays unused.
        if put("lightLine", "Texture03", 0):
            out["lightLineScroll"] = [1.0, 1.0]
    elif base == "Texture01" and r.get("TexBlendType") == 1 and r.get("VertexColor") == ON:
        if put("base2", "Texture02", r.get("Tex02_UV", 0)):
            out["base2By"] = "vertexAlpha"
    elif base == "Texture01" and t.get("Texture02") and not t.get("Texture03") and "DirlightColor" in gm.colors:
        # Pre-shaded foliage (FieldTreeShader02 pines, 97 materials): Texture01 is the colour with purple-grey shade baked
        # under each tier, Texture02 a grey light mask (0.6 default, white tips, black under the tiers). The game lights it
        # with the green `DirlightColor`; drawn alone the shade reads navy. colour += Texture02 x DirlightColor (hypothesis,
        # ledger `field:treeLightAdd`).
        if put("base2", "Texture02", r.get("UV_tex01", 0)):
            out["base2By"] = "lightAdd"
            out["lightColor"] = [float(x) for x in gm.colors["DirlightColor"]]
    if "noise" in out and "base2" not in out and "over" not in out:
        out.pop("noise")
    return out


def field_ramp(gm: GfMaterial, tex: "_Tex") -> dict:
    """Ramp-shaded foliage (FieldTreeShader04/05, 133 materials): the colour ramp `Texture02` is addressed by UV `UVSet01` (1),
    whose V follows the crown height (corr. 0.90-0.98 with height and 0.96 with the up-facing normal over area02's trees), so
    the game shades each leaf card by where it sits in the crown, not by one tint. colour = lerp(silhouette, leaf x ramp,
    ramp.a) with leaf = `Texture03`, silhouette = `Texture01` (near-black, same alpha), both on UV `UV_tex01` (hypothesis,
    ledger `field:rampTreeUV1`). The baked base colour (leaf x mean ramp) stays for renderers without the ramp."""
    r = gm.render
    out: dict = {}
    for key, slot, uv in (("leaf", "Texture03", r.get("UV_tex01", 0)), ("ramp", "Texture02", r.get("UVSet01", 1)),
                          ("silhouette", "Texture01", r.get("UV_tex01", 0))):
        ref = _layer(gm, tex, slot, uv)
        if ref is not None:
            out[key] = ref
    return out if "leaf" in out and "ramp" in out else {}


def field_water(gm: GfMaterial, tex: "_Tex") -> dict:
    """Water layers (FieldWaterShader01 sea/ponds, FieldWaterShader02 shore foam; 64 materials). Sea: the body is
    `Base_Color1/2`, `WaterNetTex01/02` are white caustic-line nets tinted `WaterNetTex0N_Color`, `WaterLightTex01/02` big
    light blobs (`WaterLight_Color`), `sky_tex` a horizon gradient. Their scroll speeds are animated (the static `texNN_U/V`
    are 0), so a renderer picks its own drift. Foam: `Texture01/02` (a scalloped band in alpha) with their `TexNN_Translate`
    offsets. Opacity is the vertex alpha (0.5..0.9 towards the open sea)."""
    r = gm.render
    uv = r.get("UV", 0)
    out: dict = {}
    nets = [x for x in (_layer(gm, tex, "WaterNetTex01", uv), _layer(gm, tex, "WaterNetTex02", uv)) if x]
    if nets:
        out["net"] = nets
        out["netColor"] = [list(gm.colors.get(f"WaterNetTex0{i}_Color", (1.0, 1.0, 1.0))) for i in (1, 2)]
        lights = [x for x in (_layer(gm, tex, "WaterLightTex01", uv), _layer(gm, tex, "WaterLightTex02", uv)) if x]
        if lights:
            out["light"] = lights
            out["lightColor"] = list(gm.colors.get("WaterLight_Color", (1.0, 1.0, 1.0)))
        sky = _layer(gm, tex, "sky_tex", uv)
        if sky:
            out["sky"] = sky
        out["colors"] = [list(gm.colors[k]) for k in ("Base_Color1", "Base_Color2") if k in gm.colors]
        return out
    foam = [x for x in (_layer(gm, tex, "Texture01", r.get("Tex01_UV", 0)), _layer(gm, tex, "Texture02", r.get("Tex02_UV", 0))) if x]
    if gm.shader.startswith("FieldWaterShader") and len(foam) == 2:
        out["foam"] = foam
        out["foamOffset"] = [[float(gm.floats.get(f"Tex0{i}_Translate_U", 0.0)), float(gm.floats.get(f"Tex0{i}_Translate_V", 0.0))] for i in (1, 2)]
    return out


def base_slot(gm: GfMaterial) -> tuple[str, int] | None:
    for slot, keys, uv in BASE_SLOTS:
        if gm.textures.get(slot):
            return slot, next((gm.render[k] for k in keys if k in gm.render), uv)
    return None


class _Tex:
    def __init__(self, load: Callable[[str], np.ndarray | None]):
        self.load = load
        self.cache: dict[str, np.ndarray | None] = {}
        self.textures: dict[str, TextureImage] = {}
        self.missing: set[str] = set()

    def get(self, name: str) -> np.ndarray | None:
        if name not in self.cache:
            img = self.load(name) if name else None
            self.cache[name] = img
            if img is None and name:
                self.missing.add(name)
        return self.cache[name]

    def register(self, key: str, rgba: np.ndarray) -> str:
        if key not in self.textures:
            self.textures[key] = TextureImage(key=key, rgba=np.ascontiguousarray(rgba, np.uint8))
        return key


def convert_field_material(gm: GfMaterial, tex: _Tex) -> tuple[Material | None, int]:
    """-> (material or None when the material is not drawn, UV set of its base colour)."""
    if SHADOW_ONLY.match(gm.shader) or gm.switches.get("SkipMainRendering"):
        return None, 0
    mat = Material(name=gm.name)
    mat.roughness_factor = 1.0
    mat.extras["shader"] = gm.shader
    # CullMode 2 = none (foliage cards); field 7 encodes the same (1 = CullMode 0, 0 = CullMode 2) when field 20 is absent
    mat.double_sided = gm.render["CullMode"] == 2 if "CullMode" in gm.render else gm.cull == 0
    blend = gm.render.get("BlendMode", gm.blend)
    layers = {s: t for s, t in gm.textures.items() if t and not NON_COLOUR_SLOTS.match(s) and not t.startswith("dummy")}
    pick = base_slot(gm)
    uv_set = 0
    if pick is not None:
        slot, uv_set = pick
        img = tex.get(gm.textures[slot])
        key = gm.textures[slot]
        ramp_tree = _ramp_tree(gm, tex)
        if ramp_tree is not None:
            img, key = ramp_tree
            slot = "Texture03"
            mat.extras["rampTree"] = True
        if img is not None and gm.shader == "FieldWindowShader":
            img = _bake_window(gm, img)
            mat.extras["glass"] = True
        if img is not None:
            mat.base_color = tex.register(key, img)
            mat.wrap["base_color"] = gm.samplers.get(slot, ("repeat", "repeat"))
            mat.uv_sets["base_color"] = uv_set
            tr = _uv_transform(gm, slot)
            if tr != (1.0, 1.0, 0.0, 0.0):
                mat.uv_transform["base_color"] = tr
            holes = int(img[..., 3].min()) < 250
            discard = gm.render["DiscardEnable"] == ON if "DiscardEnable" in gm.render else gm.floats.get("DiscardValuie", 0.0) > 0
            if blend in BLEND_MODES:
                mat.alpha_mode = "BLEND"
            elif holes and discard:  # only `DiscardEnable` materials cut out: the bark (`stem01_com`) keeps other data in alpha
                mat.alpha_mode = "MASK"
                # `DiscardValuie` (sic) is the shader's alpha-discard threshold (0.85 on leaves/flowers, 0.47 on grass)
                mat.alpha_cutoff = float(gm.floats.get("DiscardValuie") or 0.5)
        layers.pop(slot, None)
        if mat.extras.get("rampTree"):
            layers.pop("Texture02", None)
            ramp = field_ramp(gm, tex)
            if ramp:
                mat.extras["fieldRamp"] = ramp
        blend_layers = field_blend(gm, tex, pick[0])
        if blend_layers:
            mat.extras["fieldBlend"] = blend_layers
            if "cutout" in blend_layers and mat.alpha_mode == "OPAQUE":
                # the card outline lives in another layer (the base texture has no holes): an alpha-tested material
                mat.alpha_mode = "MASK"
                mat.alpha_cutoff = blend_layers["cutoff"]
    water = field_water(gm, tex) if gm.shader.startswith("FieldWater") else {}
    if water:
        mat.extras["fieldWater"] = water
    if mat.base_color is None:
        col = gm.colors.get("Base_Color1") or gm.colors.get("Color01") or gm.colors.get("Color")
        if col is not None:
            mat.base_color_factor = (*(float(c) for c in col), 1.0)
        if blend in BLEND_MODES:
            mat.alpha_mode = "BLEND"
            mat.base_color_factor = (*mat.base_color_factor[:3], 0.75)  # FITTED: water opacity by eye on area02 sea_00, 2026-10-20
    if gm.textures.get("EmissionMap") and gm.floats.get("Emissive_Intensity", 0.0) > 0:
        em = tex.get(gm.textures["EmissionMap"])
        if em is not None:
            mat.emissive = tex.register(gm.textures["EmissionMap"], em)
            mat.uv_sets["emissive"] = 0
            c = gm.colors.get("EmissionColor", (1.0, 1.0, 1.0))
            mat.emissive_factor = tuple(float(min(1.0, x * gm.floats["Emissive_Intensity"])) for x in c)
    if layers:
        mat.extras["fieldLayers"] = dict(sorted(layers.items()))
    return mat, uv_set


def _bone_worlds(g: Gfbmdl) -> list[np.ndarray]:
    worlds: list[np.ndarray] = []
    for i, b in enumerate(g.bones):
        local = np.eye(4)
        local[:3, :3] = _quat_to_mat(T.euler_xyz_to_quat(*b.euler)) @ np.diag(b.scale)
        local[:3, 3] = np.asarray(b.translation, np.float64)
        p = b.parent if 0 <= b.parent < i else -1
        worlds.append(local if p < 0 else worlds[p] @ local)
    return worlds


def build_field_model(
    g: Gfbmdl,
    load_texture: Callable[[str], np.ndarray | None],
    model_id: str,
    display_name: str,
    kind: str,
    game: str = "lgpe",
) -> Model:
    tex = _Tex(load_texture)
    materials: dict[str, Material] = {}
    mat_names: list[str | None] = []
    uv_of: list[int] = []
    for gm in g.materials:
        mat, uv = convert_field_material(gm, tex)
        uv_of.append(uv)
        if mat is None:
            mat_names.append(None)
            continue
        name, n = mat.name, 2
        while name in materials:
            name = f"{gm.name}.{n}"
            n += 1
        mat.name = name
        materials[name] = mat
        mat_names.append(name)

    worlds = _bone_worlds(g)
    bone_of_mesh = {gr.mesh: gr.bone for gr in g.groups}
    meshes: list[Mesh] = []
    used: dict[str, int] = {}
    skipped_tris = 0
    for mi, gmesh in enumerate(g.meshes):
        bi = bone_of_mesh.get(mi, -1)
        xf = worlds[bi] if 0 <= bi < len(worlds) else np.eye(4)
        identity = np.allclose(xf, np.eye(4), atol=1e-5)
        prims: list[Primitive] = []
        for poly in gmesh.polygons:
            mname = mat_names[poly.material] if poly.material < len(mat_names) else None
            if mname is None or len(poly.indices) == 0:
                skipped_tris += len(poly.indices) // 3
                continue
            uniq, inv = np.unique(poly.indices.astype(np.int64), return_inverse=True)
            pos = gmesh.positions[uniq].astype(np.float64)
            nrm = gmesh.normals[uniq].astype(np.float64) if gmesh.normals is not None else None
            if not identity:
                pos = pos @ xf[:3, :3].T + xf[:3, 3]
                if nrm is not None:
                    nrm = nrm @ np.linalg.inv(xf[:3, :3])
                    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8)
            # every UV set the mesh has, V flipped (gfbmdl UVs are bottom-left origin); TEXCOORD_n = source set n
            uvs = []
            for k in range(3):
                if k not in gmesh.uv_sets:
                    break
                uv = gmesh.uv_sets[k][uniq]
                uvs.append(np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1).astype(np.float32))
            mat = materials[mname]
            if mat.uv_sets.get("base_color", 0) >= len(uvs):  # slot's UV set missing on this mesh: fall back to UV0
                mat.uv_sets["base_color"] = 0
            colors = None
            if gmesh.colors is not None and not DATA_VERTEX_COLOUR.match(mat.extras.get("shader", "")):
                c = gmesh.colors[uniq].astype(np.float32).copy()
                if mat.alpha_mode != "BLEND" and (mat.extras.get("fieldBlend") or {}).get("base2By") != "vertexAlpha":
                    c[:, 3] = 1.0  # opaque/cut-out meshes (trees...) carry other data in the vertex alpha; roofs keep it
                    # as the Texture01 -> Texture02 blend weight (`field_blend`), opaque renderers ignore it otherwise
                # blended meshes (water, wave foam strips: alpha 0.3..1 towards the shore) keep it as opacity (hypothesis)
                # all-black RGB on a whole primitive = unpainted (sea rocks: texture grey, vertex RGB 0), not a black tint
                if (c < 0.999).any() and c[:, :3].max() > 0.0:
                    colors = c
            prims.append(
                Primitive(
                    material=mname,
                    indices=inv.astype(np.uint32),
                    positions=(pos * CM).astype(np.float32),
                    normals=nrm.astype(np.float32) if nrm is not None else None,
                    uvs=uvs,
                    colors=colors,
                )
            )
        if not prims:
            continue
        base = g.bones[bi].name if 0 <= bi < len(g.bones) and g.bones[bi].name else f"mesh{mi}"
        n = used.get(base, 0)
        used[base] = n + 1
        meshes.append(Mesh(name=base if n == 0 else f"{base}.{n}", primitives=prims))

    model = Model(
        id=model_id,
        game=game,
        display_name=display_name,
        kind=kind,
        skeleton=None,
        meshes=meshes,
        materials=materials,
        textures=tex.textures,
    )
    lo = np.min([p.positions.min(0) for m in meshes for p in m.primitives], axis=0) if meshes else np.zeros(3)
    hi = np.max([p.positions.max(0) for m in meshes for p in m.primitives], axis=0) if meshes else np.zeros(3)
    model.extras.update(
        {
            "bounds": {"min": [round(float(x), 3) for x in lo], "max": [round(float(x), 3) for x in hi]},
            "sizeMetres": [round(float(x), 2) for x in (hi - lo)],
            "vertices": int(sum(len(p.positions) for m in meshes for p in m.primitives)),
            "shadowOnlyTriangles": skipped_tris,
        }
    )
    if tex.missing:
        model.extras["missingTextures"] = sorted(tex.missing)
    return model


def import_field_model(read: Callable[[str], bytes], exists: Callable[[str], bool], model_file: str, model_id: str, display_name: str, kind: str) -> Model:
    """`read`/`exists` resolve names in the model's folder (`<stem>.gfbmdl`, `<texture>.dds`)."""
    g = parse_gfbmdl(read(model_file))

    def load(name: str) -> np.ndarray | None:
        if not exists(f"{name}.dds"):
            return None
        try:
            return decode_dds(read(f"{name}.dds"))
        except (ValueError, NotImplementedError, struct.error):
            return None

    return build_field_model(g, load, model_id, display_name, kind)
