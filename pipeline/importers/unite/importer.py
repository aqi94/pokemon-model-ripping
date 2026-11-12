"""Pokémon UNITE importer: AssetStudio FBX + PNG + Unity material JSON -> common Model.

Geometry, skin and clips come from the GO path (headless Blender, `go.importer.dump_fbx` / `model_from_dump`); this
module replaces the materials with the ones described by the sibling `Materials/<material>.json` (Unity
`m_SavedProperties`, the same shape HOME/BDSP read from bundles).

Facts (verified on the rip, 2026-11-01):
- FBX 7300, UnitScaleFactor 1 (= cm) but the data is in metres: imported at scale 100 (Pikachu 0.51 m with its
  `reference` bone scale 0.322; Blastoise 1.1 m).
- Some meshes lost their per-triangle material layer in the export (`fbx.geometries` -> materials_ok False); Blender
  would put every triangle on material 0, so such models are refused instead of guessed.
- `_MixMap` = `t_mro[e]_*`: R metallic, G roughness, B occlusion, A (mroe) unread (emission mask?). Hypothesis in
  `docs/ledger/unite.json`.
- Every Pokémon lobby material is opaque (`_AlphaClip` 0, `_SrcBlend` 1, `_DstBlend` 0); `_Cull` 0 = two-sided. The
  outfit albedo alpha is mostly 0 on opaque materials, so it is not opacity (unread).
"""

from __future__ import annotations

import copy
import json
import functools
import os
import re

import numpy as np
from PIL import Image

from ...common.model import Clip, Joint, Material, Mesh, Model, TextureImage, Track, Variant
from ..go.importer import dump_fbx, model_from_dump
from ..za.importer import categorize, decimate
from . import alpha as alpha_rules
from . import emission as emission_rules
from . import families as unite_families
from . import fbx as fbxr
from . import fbx_anim
from . import fx as unite_fx
from . import shading as unite_shading
from . import matcap_terms
from . import outline as outline_rules
from . import player_shaders as ps

FBX_SCALE = 100.0  # metres stored under a cm UnitScaleFactor (see module doc)


class LostMaterialLayer(RuntimeError):
    """The FBX lost which triangles use which material; importing it would paint everything with material 0."""


def read_unity_material(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        j = json.load(f)
    sp = j.get("m_SavedProperties") or {}
    tex = {}
    for slot, env in (sp.get("m_TexEnvs") or {}).items():
        t = (env or {}).get("m_Texture") or {}
        if t.get("Name") and not t.get("IsNull"):
            tex[slot] = {"name": t["Name"], "scale": (env["m_Scale"]["X"], env["m_Scale"]["Y"]), "offset": (env["m_Offset"]["X"], env["m_Offset"]["Y"])}
    colors = {k: (v["r"], v["g"], v["b"], v["a"]) if "r" in v else tuple(v.values()) for k, v in (sp.get("m_Colors") or {}).items()}
    return {
        "name": j.get("m_Name") or j.get("Name") or os.path.splitext(os.path.basename(path))[0],
        "shader": ((j.get("m_Shader") or {}).get("Name")) or "",
        "family": unite_families.family_of(((j.get("m_Shader") or {}).get("Name")) or ""),
        "textures": tex,
        "floats": dict(sp.get("m_Floats") or {}),
        "colors": colors,
    }


def material_jsons(folder: str) -> dict[str, dict]:
    mdir = os.path.join(folder, "Materials")
    out = {}
    if os.path.isdir(mdir):
        for fn in os.listdir(mdir):
            if fn.lower().endswith(".json"):
                info = read_unity_material(os.path.join(mdir, fn))
                out[os.path.splitext(fn)[0].lower()] = info
    return out


def _png(folder: str, name: str, index: dict[str, str] | None = None) -> str | None:
    """Texture `name` (no extension) beside the FBX, else through `index` (lower-case file name -> path): AssetStudio
    writes every asset once, into the first folder that uses it, so skins/evolutions find theirs elsewhere."""
    p = os.path.join(folder, name + ".png")
    if os.path.isfile(p) and _readable(p):
        return p
    q = (index or {}).get(name.lower() + ".png")
    return q if q and _readable(q) else None


@functools.lru_cache(maxsize=None)
def _readable(path: str) -> bool:
    """False for a file Pillow cannot decode: it then counts as missing (`missingTextures`) instead of failing the model
    (Marcy4000/Unite's Bellibolt `t_n_bat_0939bodyb_lv3.png` is 840 bytes without a PNG header)."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001
        return False


def _stem(file_name: str) -> str:
    return re.sub(r"\.(png|tga|psd|jpg)$", "", file_name, flags=re.I)


MATCAP_TEX = re.compile(r"^t_matcap_", re.I)
ENV_MATCAP_TEX = re.compile(r"^t_mipmap\d*\b", re.I)
MASK_TEX = re.compile(r"^t_m_", re.I)  # `_FurTex`/`_FurLengthTex` masks (lobby JSONs), never a colour map
MIX_TEX = re.compile(r"^t_mroe?_", re.I)
# Unity slots that hold a matcap image in the lobby JSONs (block survey 2026-11-01: `_MatTex` 38, `_MatcapTex` 34,
# `_MatCapTex` 20, `_RampTex` 13 (+ real ramps), `_MatMap` 3; `ghostGlass` uses `_RampMap`)
# `_RampTex` / `_RampMap` are not matcaps (Cloth `_IFRAMP` ramp, ghostGlass's glass image): removed 2026-11-09 (WP-D)
MATCAP_SLOTS = ("_MatCapTex", "_MatcapTex", "_MatTex", "_MatMap")


material_token = fbxr.material_token
names_material = fbxr.names_material


def _tex_entry(name: str) -> dict:
    return {"name": _stem(name), "scale": (1.0, 1.0), "offset": (0.0, 0.0)}


def pick_matcap(names: list[str], material: str) -> str | None:
    """The matcap among `names` that belongs to `material`: the one named with its id (`t_matcap_lob_014320a` for
    `mat_lob_014320a_lv3`), else the only/first one."""
    caps = [_stem(n) for n in names if MATCAP_TEX.match(n)]
    if not caps:
        return None
    tok = material_token(material)
    return next((c for c in caps if c.lower().endswith(tok) or tok + "_" in c.lower() or c.lower().endswith(tok + "_lv3")), caps[0])


def info_from_fbx(slots: dict[str, str], folder: str, index: dict[str, str] | None = None,
                  block: list[str] | None = None, material: str = "") -> dict:
    """Material info from the FBX (no `Materials/*.json`: battle models, NPCs): the linked DiffuseColor/Bump plus the
    material's texture block (`fbx.material_blocks`: its unlinked mro(e), matcap, outline maps). The mro(e) map falls
    back to the albedo's sibling by name (`t_d_X` -> `t_mro_X` / `t_mroe_X`: a texture another material introduced
    first is not in this block). A DiffuseColor link to a matcap (Litwick line `fireB`) is the matcap, not the albedo;
    a link to a `t_m_` mask (NPC hair) is replaced by the block's `t_d_` texture when there is one."""
    block = block or []
    tex = {}
    for slot, prop in (("_MainTex", "DiffuseColor"), ("_BumpMap", "Bump"), ("_BumpMap", "NormalMap")):
        if slots.get(prop) and slot not in tex:
            tex[slot] = _tex_entry(slots[prop])
    d = tex.get("_MainTex", {}).get("name", "")
    as_diffuse = False
    if MATCAP_TEX.match(d):
        tex["_MatCapTex"] = tex.pop("_MainTex")
        d = ""
        as_diffuse = True  # the FBX links the matcap as DiffuseColor (Litwick-line fire): a colour source, not glass (WP-D)
    elif MASK_TEX.match(d):
        tex.pop("_MainTex")
        own = [n for n in block if n.lower().startswith("t_d_")]
        if own:
            tex["_MainTex"] = _tex_entry(own[0])
        d = tex.get("_MainTex", {}).get("name", "")
    if d.lower().startswith("t_d_"):
        for cand in ("t_mro_" + d[4:], "t_mroe_" + d[4:]):
            if _png(folder, cand, index):
                tex["_MixMap"] = _tex_entry(cand)
                break
    mix = next((n for n in block if MIX_TEX.match(n)), None)
    if mix and "_MixMap" not in tex:
        tex["_MixMap"] = _tex_entry(mix)
    cap = pick_matcap(block, material) if "_MatCapTex" not in tex else None
    if cap:
        tex["_MatCapTex"] = _tex_entry(cap)
    env = next((n for n in block if ENV_MATCAP_TEX.match(n)), None)
    if env:  # the Cloth/basepbr shaders' generic environment matcap (`_MatcapTex` = `t_mipmap03` in the lobby JSONs)
        tex["_MatcapTex"] = _tex_entry(env)
    if not slots and block:
        # no link at all (Fx materials: AssetStudio links only _MainTex/_BumpMap, an Fx shader's colour is `_Tex1`): the
        # colour texture is the first non-character texture of its own block, one named after it first (`_Tex1` sorts
        # first among an Fx material's m_TexEnvs in 87% of the scene JSONs). INFERRED: ledger `fbx:materialBlocks`.
        cands = [n for n in block if not fbxr.CHARACTER_KINDS.match(n) and _stem(n).lower() not in fbxr.GLOBAL_TEXTURES]
        col = next((n for n in cands if names_material(n, material)), cands[0] if cands else None)
        if col:
            tex["_Tex1"] = _tex_entry(col)
    kind = "battle" if re.search(r"(^|_)bat(_|\d)", material or "", re.I) and not re.search(r"(^|_)lob", material or "", re.I) else "lobby"
    return {"name": "", "shader": "", "textures": tex, "floats": {}, "colors": {}, "source": "fbx", "block": list(block),
            "family": unite_families.infer_family(material, block, kind), "familyInferred": True,
            **({"matcapAsDiffuse": True} if as_diffuse else {})}


def _blender_name(name: str) -> str:
    """Blender suffixes duplicate datablock names (.001)."""
    return re.sub(r"\.\d{3}$", "", name)


def _load_outline_map(name: str, folder: str, textures: dict, index: dict[str, str] | None, missing: list[str] | None) -> str | None:
    """`_OutlineMap` (`t_vc_*`) as raw RGBA bytes under its own key (WP-G; the shader reads it unconverted, in gamma space)."""
    path = _png(folder, name, index)
    if not path:
        if missing is not None:
            missing.append(name)
        return None
    key = name + "~vc"
    if key not in textures:
        textures[key] = TextureImage(key=key, rgba=np.array(Image.open(path).convert("RGBA"), np.uint8), srgb=False)
    return key


def build_material(name: str, info: dict, folder: str, textures: dict[str, TextureImage],
                   index: dict[str, str] | None = None, missing: list[str] | None = None) -> Material:
    mat = Material(name=name, roughness_factor=1.0)
    fl, tx = info["floats"], info["textures"]

    def load(slot: str, srgb: bool, opaque: bool = False) -> str | None:
        t = tx.get(slot)
        path = _png(folder, t["name"], index) if t else None
        if not path:
            if t and missing is not None:
                missing.append(t["name"])
            return None
        key = t["name"] if opaque or not srgb else t["name"] + "~a"  # alpha-keeping copy apart from the alpha-forced one
        if key not in textures:
            rgba = np.array(Image.open(path).convert("RGBA"), np.uint8)
            if opaque:
                rgba[..., 3] = 255
            textures[key] = TextureImage(key=key, rgba=rgba, srgb=srgb)
        return key

    # Alpha mode from the shader data (`alpha.decide`, ledger `material:transparency`); `basepbrGlass` (trainer eyewear
    # JSONs) is marked by its shader name only (its floats say opaque): its lens albedo's alpha (0.07-0.30) is the opacity.
    ad = alpha_rules.decide(info["shader"], fl)
    if info["shader"] and re.search(r"basepbrGlass", info["shader"]):
        ad = {**ad, "mode": "BLEND", "source": "basepbrGlass (shader name)"}
    if ad["mode"] == "OPAQUE" and fl.get("_AlphaClip"):
        ad = {**ad, "mode": "MASK", "cutoff": float(fl.get("_Cutoff", 0.5)), "source": "_AlphaClip"}
    see_through = ad["mode"] == "BLEND"
    clip = ad["mode"] == "MASK"
    mat.base_color = load(color_slot(tx) or "_MainTex", True, opaque=not (see_through or clip))
    if mat.base_color and see_through and textures[mat.base_color].rgba[..., 3].min() >= 250             and not re.search(r"Bubble|Glass|actorbase", info["shader"] or ""):
        # blended by the data but the albedo is solid: nothing shows through, keep it opaque (no sorting artefacts)
        see_through = False
        ad = {**ad, "mode": "OPAQUE", "source": ad["source"] + "; albedo alpha solid"}
        mat.base_color = load(color_slot(tx) or "_MainTex", True, opaque=True)
    if see_through:
        mat.alpha_mode = "BLEND"
        if ad["alpha_factor"] != 1.0:
            mat.base_color_factor = (*mat.base_color_factor[:3], ad["alpha_factor"])
    elif clip:
        mat.alpha_mode, mat.alpha_cutoff = "MASK", ad["cutoff"]
    mat.double_sided = fl.get("_Cull") == 0.0
    mat.normal = load("_BumpMap", False)

    mix = tx.get("_MixMap")
    mix_path = _png(folder, mix["name"], index) if mix else None
    if mix_path:
        key = mix["name"] + "_orm"
        if key not in textures:
            m = np.array(Image.open(mix_path).convert("RGBA"), np.uint8)
            # glTF ORM: R occlusion, G roughness, B metallic  <-  mro: R metallic, G roughness, B occlusion
            orm = np.stack([m[..., 2], m[..., 1], m[..., 0], np.full(m.shape[:2], 255, np.uint8)], -1)
            textures[key] = TextureImage(key=key, rgba=orm, srgb=False)
        mat.metallic_roughness = key
        mat.occlusion = key
        mat.metallic_factor = 1.0
    else:
        mat.metallic_factor = 0.0
        mat.roughness_factor = 0.8

    mat.extras = {"unity": unity_extras(info), "unite": {"alpha": {"mode": ad["mode"], "cutoff": ad["cutoff"], "source": ad["source"]}}}
    outline_rules.add_outline(mat.extras, info, lambda n: _load_outline_map(n, folder, textures, index, missing))  # WP-G
    if see_through:
        mat.extras["blend"] = "Blend"
    if mix_path:
        # WP-C: the mro(e) alpha is emission (or a Ramp/Matcap mask); occlusion B is applied per `ao` (emission.py)
        unite = mat.extras.setdefault("unite", {})
        unite["maskKind"] = emission_rules.mask_kind(info)
        ao = emission_rules.ao_params(info)
        if ao:
            unite["ao"] = ao
        em = emission_rules.emission_params(info)
        if em:
            ek = f"{mix['name']}_emis_{mat.base_color or 'white'}"
            if ek not in textures:
                base = textures[mat.base_color].rgba if mat.base_color in textures else None
                textures[ek] = TextureImage(key=ek, rgba=emission_rules.bake_emission(
                    base, np.array(Image.open(mix_path).convert("RGBA"), np.uint8)[..., 3]), srgb=True)
            mat.emissive = ek
            mat.emissive_factor, strength = emission_rules.factors(em)
            if strength:
                mat.extras["khrEmissiveStrength"] = strength
            unite["emission"] = {**em, "source": "mixmap.a x albedo"}
    # WP-D: "matcap" is four separate shader terms (matTex / rampLight / env), not one metal reflection; each is written
    # with its real constants by `matcap_terms.lit_terms` and drawn by the viewer (scene/matcap.ts). A metallic texel has
    # no diffuse and shows only what it reflects: the reflection is the `env` term (the matcap standing in for the scene
    # probe, which is not in the rip), so a material without one keeps its colour (metal factor 0). The old 1/luminance
    # gain (FITTED floor 0.15) and `extras.matcap` / `matcapMode` / `matcapGain` are no longer written; the viewer still
    # reads them from GLBs converted before 2026-11-09. Ledger `material:matcapMetal`.
    terms = matcap_terms.lit_terms(info, folder, index, textures, _png, name)
    if terms:
        mat.extras.setdefault("unite", {}).update(terms)
    if mix_path and "env" not in terms:
        mat.metallic_factor = 0.0  # roughness + occlusion from the ORM map still apply
    return mat


# ---- LobbyPlayer shaders (lobby NPCs): colour maths of the game's own shaders, see `player_shaders` ----------------

PLAYER_COLOURS_LINEAR = True  # `[HDR]` colour properties are read as linear light (ledger `shader:lobbyPlayerColourSpace`)


def _load_float(folder: str, tx: dict, slot: str, index: dict[str, str] | None, missing: list[str] | None,
                srgb: bool) -> np.ndarray | None:
    """Texture of `slot` as float (H, W, 4) 0..1, decoded from sRGB when `srgb` (alpha stays raw), None when absent."""
    t = tx.get(slot)
    if not t:
        return None
    path = _png(folder, t["name"], index)
    if not path:
        if missing is not None:
            missing.append(t["name"])
        return None
    a = np.array(Image.open(path).convert("RGBA"), np.float32) / 255.0
    if srgb:
        a[..., :3] = ps.srgb_to_linear(a[..., :3])
    return a


def build_player_material(name: str, kind: str, info: dict, folder: str, textures: dict[str, TextureImage],
                          index: dict[str, str] | None = None, missing: list[str] | None = None) -> Material:
    """Base colour / roughness / emission baked per the game's `LobbyPlayer` shader (`player_shaders`). The RampLight
    emission needs the second UV set and is added later by `bake_ramp_emission`."""
    tx, fl, colors = info["textures"], info["floats"], info["colors"]
    main = _load_float(folder, tx, "_MainTex", index, missing, True)
    mix = _load_float(folder, tx, "_MixMap", index, missing, False) if kind != "ramp" else None
    bump = _load_float(folder, tx, "_BumpMapNr", index, missing, False)
    sizes = [a.shape[:2] for a in (main, mix, bump) if a is not None]
    h, w = max(sizes, key=lambda s: s[0] * s[1]) if sizes else (4, 4)
    if main is None:  # shader default "white"
        main = np.ones((h, w, 4), np.float32)
    main = ps.fit(main, h, w)
    if mix is not None:
        mix = ps.fit(mix, h, w)
    bump_a = ps.fit(bump[..., 3:4], h, w)[..., 0] if bump is not None else None
    rgb, alpha, emission = ps.player_albedo(kind, main[..., :3], mix, bump_a, colors, fl, PLAYER_COLOURS_LINEAR)

    mat = Material(name=name, roughness_factor=1.0, metallic_factor=0.0)
    mat.double_sided = fl.get("_Cull") == 0.0
    out = np.empty((h, w, 4), np.uint8)
    out[..., 3] = 255
    mat.extras = {"unity": unity_extras(info), "playerShader": kind}
    key = f"{name}_{kind}"
    if kind == "eye":
        # unlit: the output colour is the whole look. Stored as emission (hue-preserving, HDR peak kept as a strength)
        # over a black base so lighting does not touch it
        peak = float(max(rgb.max(), 1.0))
        out[..., :3] = np.round(ps.linear_to_srgb(rgb / peak) * 255.0)
        textures[key] = TextureImage(key=key, rgba=out, srgb=True)
        mat.base_color_factor = (0.0, 0.0, 0.0, 1.0)
        mat.emissive = key
        mat.emissive_factor = (1.0, 1.0, 1.0)
        if peak > 1.0:
            mat.extras["khrEmissiveStrength"] = round(peak, 4)
        mat.extras["unlit"] = True
        return mat
    out[..., :3] = np.round(ps.linear_to_srgb(rgb) * 255.0)
    if alpha is not None:
        out[..., 3] = np.round(np.clip(alpha, 0.0, 1.0) * 255.0)
        cutoff = float(fl.get("_Cutoff", 0.5))
        if float(alpha.min()) < cutoff:
            mat.alpha_mode, mat.alpha_cutoff = "MASK", cutoff
    textures[key] = TextureImage(key=key, rgba=out, srgb=True)
    mat.base_color = key
    if emission is not None:
        ep = float(max(emission.max(), 1.0))
        ek = key + "_emissive"
        eo = np.empty((h, w, 4), np.uint8)
        eo[..., :3] = np.round(ps.linear_to_srgb(emission / ep) * 255.0)
        eo[..., 3] = 255
        textures[ek] = TextureImage(key=ek, rgba=eo, srgb=True)
        mat.emissive, mat.emissive_factor = ek, (1.0, 1.0, 1.0)
        if ep > 1.0:
            mat.extras["khrEmissiveStrength"] = round(ep, 4)
    rough = ps.roughness_of(kind, ps.fit(main[..., 3:4], h, w)[..., 0], mix, colors, fl)
    rk = key + "_orm"
    orm = np.empty((h, w, 4), np.uint8)
    orm[..., 0] = 255
    orm[..., 1] = np.round(rough * 255.0)
    orm[..., 2] = 0  # metallic not baked: a glTF metal texel has no diffuse and nothing here to reflect
    orm[..., 3] = 255
    textures[rk] = TextureImage(key=rk, rgba=orm, srgb=False)
    mat.metallic_roughness = rk
    if bump is not None:
        nk = tx["_BumpMapNr"]["name"]
        if nk not in textures:
            n = np.round(bump * 255.0).astype(np.uint8)
            n[..., 3] = 255
            textures[nk] = TextureImage(key=nk, rgba=n, srgb=False)
        mat.normal = nk
    return mat


def uv_layers(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mesh Model name -> (first UV set, second UV set) per control point, in the file's convention, for meshes whose
    two layers are both direct per-vertex (`ByVertice`). The MEGA FBX keeps one layer; the Marcy4000 copy of an NPC
    keeps the second (the RampLight glow mask lives there)."""
    top = {n.name: n for n in fbxr.parse(path)}
    objs, conns = top.get("Objects"), top.get("Connections")
    if objs is None or conns is None:
        return {}
    geoms = {g.props[0]: g for g in objs.all("Geometry") if len(g.props) > 2 and g.props[2] == "Mesh"}
    models = {c.props[0]: fbxr.object_name(c) for c in objs.all("Model") if len(c.props) > 2 and c.props[2] == "Mesh"}
    out = {}
    for c in conns.children:
        p = c.props
        if not (len(p) >= 3 and p[0] == "OO" and p[1] in geoms and p[2] in models):
            continue
        layers = geoms[p[1]].all("LayerElementUV")
        if len(layers) < 2:
            continue
        arrs = []
        for L in layers[:2]:
            if str(L.child("MappingInformationType").props[0]) != "ByVertice" or L.child("UVIndex") is not None:
                break
            arrs.append(np.asarray(L.child("UV").props[0], np.float64).reshape(-1, 2))
        if len(arrs) == 2 and len(arrs[0]) == len(arrs[1]):
            out[models[p[2]]] = (arrs[0], arrs[1])
    return out


UV_MATCH_DECIMALS = 5


def second_uv_for(prim, layers: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, float]:
    """Second UV per vertex of a dumped primitive: vertices are matched to the file's control points by their first UV
    (Blender keeps the file's floats). Returns (uv1 (V, 2) with NaN rows where unmatched, matched share)."""
    uv0, uv1 = layers
    key: dict[tuple[float, float], int] = {}
    for i, (u, v) in enumerate(np.round(uv0, UV_MATCH_DECIMALS)):
        key.setdefault((float(u), float(v)), i)
    mine = np.round(np.stack([prim.uvs[0][:, 0], 1.0 - prim.uvs[0][:, 1]], 1).astype(np.float64), UV_MATCH_DECIMALS)
    ids = np.array([key.get((float(u), float(v)), -1) for u, v in mine])
    out = np.full((len(ids), 2), np.nan)
    ok = ids >= 0
    out[ok] = uv1[ids[ok]]
    return out, float(ok.mean()) if len(ids) else 0.0


def bake_ramp_emission(model: Model, name: str, info: dict, folder: str, index: dict[str, str] | None,
                       uv_source: str | None, textures: dict[str, TextureImage]) -> str | None:
    """RampLight emission (static frame t = 0) into an emissive texture on UV0. Returns a warning when it cannot."""
    mat = model.materials.get(name)
    if mat is None:
        return None
    if not uv_source or not os.path.isfile(uv_source):
        return f"{name}: RampLight glow needs the second UV set, not in this FBX (no donor copy)"
    ramp = _load_float(folder, info["textures"], "_RampMap", index, None, True)
    mixmap = _load_float(folder, info["textures"], "_MixMap", index, None, False)
    if ramp is None:
        return f"{name}: RampLight ramp texture missing"
    layers = uv_layers(uv_source)
    main = textures.get(mat.base_color) if mat.base_color else None
    h, w = (main.height, main.width) if main else (1024, 1024)
    mask = mixmap[..., 0] if mixmap is not None else np.ones((4, 4), np.float32)  # "white" default
    acc = np.zeros((h, w, 3), np.float32)
    cover = np.zeros((h, w), bool)
    used = False
    for mesh in model.meshes:
        lay = layers.get(_blender_name(mesh.name))
        if lay is None:
            continue
        for p in mesh.primitives:
            if p.material != name:
                continue
            uv1, share = second_uv_for(p, lay)
            if share < 0.99:
                return f"{name}: only {share:.0%} of {mesh.name}'s vertices matched the donor FBX's second UV set"
            uv1 = np.nan_to_num(uv1, nan=2.0)
            rgb, cov = ps.ramp_emission(p.uvs[0].astype(np.float64), uv1, p.indices, ramp[..., :3], mask,
                                        info["floats"], (h, w))
            acc[cov] = rgb[cov]
            cover |= cov
            used = True
    if not used:
        return f"{name}: no mesh of the donor FBX matches this material's meshes"
    acc = ps.dilate(acc, cover)
    peak = float(max(acc.max(), 1.0))
    rk = name + "_ramp_emissive"
    eo = np.empty((h, w, 4), np.uint8)
    eo[..., :3] = np.round(ps.linear_to_srgb(acc / peak) * 255.0)
    eo[..., 3] = 255
    textures[rk] = TextureImage(key=rk, rgba=eo, srgb=True)
    mat.emissive, mat.emissive_factor = rk, (1.0, 1.0, 1.0)
    if peak > 1.0:
        mat.extras["khrEmissiveStrength"] = round(peak, 4)
    mat.extras["rampLight"] = {"time": 0.0, "coverage": round(float(cover.mean()), 4)}
    return None


def matcap_slot(tx: dict) -> str | None:
    """The texture slot holding this material's matcap: a matcap slot (`MATCAP_SLOTS`), but a `_RampTex`/`_RampMap`
    only when it names a `t_matcap_` image (it also holds real ramps)."""
    for s in MATCAP_SLOTS:
        if s in tx and (not s.startswith("_Ramp") or MATCAP_TEX.match(tx[s]["name"])):
            return s
    return None


def unity_extras(info: dict) -> dict:
    return {
        "shader": info["shader"] or None,
        "source": info.get("source", "json"),
        "textures": {k: v["name"] for k, v in info["textures"].items()},
        "floats": info["floats"],
        "colors": {k: list(v) for k, v in info["colors"].items()},
    }


# Colour texture slots: lit shaders `_MainTex`, `PGame/Battlefield/Unlit` `_BaseMap`, the `Fx/` shaders `_Tex1`.
COLOR_SLOTS = ("_MainTex", "_BaseMap", "_Tex1", "_tex1")


def color_slot(tx: dict) -> str | None:
    return next((k for k in COLOR_SLOTS if k in tx), None)


def _srgb_to_linear(c) -> np.ndarray:
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


CUTOUT_SHARE = 0.2  # a silhouette mask has at least this share of empty (< 0.05) AND of solid (> 0.95) texels


def _solid_alpha(path: str) -> bool:
    """The PNG's own alpha is (nearly) solid, so its grey is the mask (noise_06: 0.973..1.0); a streak mask with real
    transparency (Skeledirge's `body_01_wzb`) is not a noise sheen."""
    im = np.asarray(Image.open(path).convert("RGBA"))
    g = im[..., :3].max(-1)
    # ... and its grey varies (a flat grey card is a plain translucent mask) over the whole tile: a tiling noise has
    # few empty texels (noise_06: 5th percentile 0.06), a soft silhouette many (Charizard 16's `basegate_object_01`: 53%)
    return (float((im[..., 3] > 242).mean()) >= 0.95 and int(np.ptp(g)) > 32
            and float((g < 13).mean()) < CUTOUT_SHARE)


def _grey(rgba: np.ndarray) -> bool:
    rgb = rgba[..., :3].astype(np.int16)
    return int((rgb.max(-1) - rgb.min(-1)).max()) <= 8


def is_cutout(path: str) -> bool:
    """A mask that cuts a shape out of its card: enough empty and enough solid texels (alpha when it varies, else the
    grey value). Flame silhouettes `fire_01/76/78` pass (0.39-0.52 empty, 0.42-0.47 solid); noise/distortion maps
    (`motion_03` 0.02/0.00, `noise_06/09`), glow layers (`X_xin`, `X_tou`) and sparse blob masks (`fire_02`, 0.10 solid)
    do not."""
    a = np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255.0
    m = a[..., 3] if a[..., 3].min() < 0.98 else a[..., :3].max(-1)
    return float(np.mean(m < 0.05)) >= CUTOUT_SHARE and float(np.mean(m > 0.95)) >= CUTOUT_SHARE


def build_effect_material(name: str, info: dict, folder: str, textures: dict[str, TextureImage],
                          index: dict[str, str] | None = None, missing: list[str] | None = None) -> Material | None:
    """An effect part attached to the character (Ponyta's mane, tail flames, holowear glow/aura cards): unlit
    (emissive), blended, two-sided unless `_Cull` says otherwise. Colour = the colour slot x `_Color1`/`_TintColor`.
    Alpha: the texture's own alpha when it has one; a grey opaque texture (most `t_fx_*` masks) is its own alpha; an
    opaque colour texture is drawn additively. `Fx/` `_BlendMode` 3 = additive, else alpha blend (INFERRED from the
    enum's use: 3 on 1,218 glow/trail materials, 1 on 720 cards; docs/ledger/unite.json `fxBlendMode`). Returns None
    when there is no colour texture (nothing to draw)."""
    fl, tx = info["floats"], info["textures"]
    fxp = unite_fx.plan(info)  # WP-E: the Fx shader's formula (JSON materials only)
    slot = fxp["colorSlot"] if fxp and fxp["colorSlot"] in tx else color_slot(tx)
    path = _png(folder, tx[slot]["name"], index) if slot else None
    if path and info.get("source") == "fbx" and not is_cutout(path):
        # the guessed `_Tex1` (first texture of the block) cuts nothing: a tiling noise/distortion map (Mega Charizard
        # X's tail: `t_fx_0004_motion_03`, drawn as grey wisps). The mask is the block's first real silhouette
        # (`t_fx_0006_51charizard_fire_78_wpx`, the base flame's `fire_01` shape). INFERRED: property names are not in
        # the FBX; ledger `fx:cutoutMask`.
        for n in (_stem(b) for b in info.get("block") or []):
            q = _png(folder, n, index)
            if q and q != path and is_cutout(q):
                tx = {**tx, slot: _tex_entry(n)}
                info = {**info, "textures": tx, "maskFromBlock": n}
                path = q
                break
    if not path:
        if slot and missing is not None:
            missing.append(tx[slot]["name"])
        return None
    # A character atlas (`t_d_*`: Rapidash's mane links its body atlas) carries a mask in alpha, not opacity (as on
    # every UNITE character material): drawn opaque-alpha. Only effect textures (`t_fx_*`, `Fx/` `_Tex1`) blend by
    # their alpha; a grey opaque effect mask is its own alpha.
    atlas = tx[slot]["name"].lower().startswith("t_d_")
    baked = bool(fxp and unite_fx.needs_bake(fxp))
    key = tx[slot]["name"] + "_fx" + ("_" + unite_fx.key_suffix(fxp, info) if baked else "")
    if key not in textures:
        rgba = np.array(Image.open(path).convert("RGBA"), np.uint8).copy()
        rgb = rgba[..., :3].astype(np.int16)
        if atlas:
            rgba[..., 3] = 255
        elif float((rgba[..., 3] > 242).mean()) >= 0.95 and (rgb.max(-1) - rgb.min(-1)).max() <= 8:
            # a grey mask whose alpha is (nearly) solid is its own alpha: `t_fx_0001_noise_06` ships alpha 0.973-1.0,
            # which the old min >= 250 test kept as a solid card (Mega Charizard X's `wenli` blob)
            rgba[..., 3] = rgba[..., :3].max(-1)
        if baked:
            others = {}
            for k in fxp["roles"]:
                q = _png(folder, tx[k]["name"], index) if k in tx and k != slot else None
                if q:
                    others[k] = np.array(Image.open(q).convert("RGBA"), np.uint8)
            rgba = unite_fx.bake(fxp, rgba, others)
        textures[key] = TextureImage(key=key, rgba=rgba, srgb=True)
    has_alpha = textures[key].rgba[..., 3].min() < 250
    mode = fl.get("_BlendMode")
    # FBX-only material whose colour slot holds a tiling grey noise and no silhouette was found in its block: a noise
    # sheen layer (wenli 纹理 "texture"; 200 of 267 `*wenli*` scene JSON materials are additive noise/pattern Fx), drawn
    # additively at its grey value, never as a solid flame (ledger `fx:wenliSheen`)
    sheen = bool(info.get("source") == "fbx" and fxp is None and not atlas and not info.get("maskFromBlock")
                 and not is_cutout(path) and _grey(textures[key].rgba) and _solid_alpha(path))
    premult = fxp is not None and mode == 2.0  # Src1/Dst10 (ledger `fxBlendMode`)
    # an opaque character atlas is solid paint, never additive light (Crocalor's two-tone flame atlas, manes)
    additive = mode == 3.0 or (mode is None and not has_alpha and not atlas) or sheen
    tint = np.ones(4)
    for k in ("_Color1", "_TintColor"):
        if k in info["colors"]:
            tint = np.asarray(info["colors"][k][:4], np.float64)
            break
    if fxp is not None:
        # the Fx shaders ignore `_Color1.a` (fragment of m_1P_Base: alpha = Tex1.a * vc.a * _Alpha1) and multiply
        # rgb by `_Emission1`; only three shaders read the colour alpha (`unite_fx.COLOR_ALPHA_SHADERS`)
        alpha = float(tint[3] if unite_fx.uses_color_alpha(info) else 1.0) * float(fl.get("_Alpha1", 1.0))
        tint = np.concatenate([tint[:3] * fxp["emission"], tint[3:]])
    else:
        alpha = float(tint[3]) * float(fl.get("_Alpha1", 1.0))
    mat = Material(name=name, roughness_factor=1.0, metallic_factor=0.0)
    mat.base_color = key
    mat.emissive = key
    mat.base_color_factor = (0.0, 0.0, 0.0, float(np.clip(alpha, 0.0, 1.0)))
    rgb = np.clip(tint[:3], 0.0, None)
    peak = float(rgb.max())
    if peak > 1.0:  # HDR tint (glTF emissiveFactor is [0, 1]): keep the hue, record the strength
        rgb = rgb / peak
    mat.emissive_factor = tuple(float(v) for v in rgb)
    mat.alpha_mode = "BLEND"
    mat.double_sided = fl.get("_Cull", 0.0) == 0.0
    st = info["colors"].get("_Dtex_st" if fxp and fxp["family"] == "overlay3" else "_tex1_st")
    if st is not None and tuple(st[:4]) != (1.0, 1.0, 0.0, 0.0):
        for k in ("base_color", "emissive"):
            mat.uv_transform[k] = (float(st[0]), float(st[1]), float(st[2]), float(st[3]))
    mat.extras = {"unity": unity_extras(info), "blend": "Add" if additive else ("BlendPreMultiAlpha" if premult else "Blend"),
                  "effect": True}
    if fxp is not None:
        mat.extras.setdefault("unite", {})["fx"] = unite_fx.extras(fxp, info)
    if info.get("maskFromBlock"):
        mat.extras["maskFromBlock"] = info["maskFromBlock"]
    if sheen:
        mat.extras["noiseSheen"] = True
    if atlas and not additive and mode is None:  # an explicit Fx _BlendMode (JSON) keeps its blending
        mat.alpha_mode = "OPAQUE"  # depth-tested and depth-writing like the body it belongs to
        mat.extras.pop("blend")
        mat.extras["solidEffect"] = True
    if peak > 1.0:
        mat.extras["emissiveStrength"] = round(peak, 4)
    return mat


ARTIST_TAG = re.compile(r"_[a-z]{2,3}$")  # the effect artist's initials the rip appends to fx names (`_hb`, `_wpx`, `_gp`)


def effect_key(material: str) -> str:
    """A material name without its artist tag and case: `mat_fx_rig_bat_0911_04_hb` and the lobby copy's
    `mat_fx_rig_bat_0911_04` are the same authored material (the lobby export lost the tag)."""
    return ARTIST_TAG.sub("", _blender_name(material).lower())


def sibling_fbxs(folder: str) -> list[str]:
    """Other FBXs of the same species (dex) in the Pokémon share: every battle/lobby folder whose name carries the same
    `pref_char_<bat|lob>_<dex4>` (own folder first excluded)."""
    m = re.match(r"pref_char_(?:bat|lob)_(\d{4})", os.path.basename(folder.rstrip("/\\")))
    if not m:
        return []
    root = os.path.dirname(os.path.dirname(os.path.abspath(folder)))
    out = []
    for kind in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        base = os.path.join(root, kind)
        if not os.path.isdir(base):
            continue
        for d in sorted(os.listdir(base)):
            if re.match(rf"pref_char_(?:bat|lob)_{m.group(1)}", d) and os.path.join(base, d) != os.path.abspath(folder):
                out += [os.path.join(base, d, f) for f in os.listdir(os.path.join(base, d)) if f.lower().endswith(".fbx")]
    return out


def borrow_effect_info(material: str, folder: str, index: dict[str, str] | None, cache: dict) -> tuple[dict, str, str, np.ndarray | None] | None:
    """A texture-less effect material's textures from the same species' other FBXs (battle <-> lobby <-> other skins):
    a material of the same name (artist tag ignored, `effect_key`) that does have its texture in the rip. Returns
    (info, folder of the donor FBX, donor material name, the donor mesh's mean vertex colour or None) when every candidate that resolves to a texture on disk agrees
    on the colour texture; None when none does or they disagree (a near-miss: reported by the caller)."""
    key = effect_key(material)
    found: dict[str, tuple] = {}
    for fbx in sibling_fbxs(folder):
        if fbx not in cache:
            try:
                cache[fbx] = (fbxr.material_textures(fbx), fbxr.material_blocks(fbx))
            except Exception:  # noqa: BLE001
                cache[fbx] = ({}, {})
        links, blocks = cache[fbx]
        for name in set(links) | set(blocks):
            if effect_key(name) != key:
                continue
            fo = os.path.dirname(fbx)
            info = info_from_fbx(links.get(name, {}), fo, index, blocks.get(name), name)
            slot = color_slot(info["textures"])
            if slot and _png(fo, info["textures"][slot]["name"], index):
                col = None
                if (fbx, "vc") not in cache:
                    cache[(fbx, "vc")] = (fbxr.mesh_vertex_colours(fbx), fbxr.mesh_materials(fbx))
                vc, slots = cache[(fbx, "vc")]
                for mesh, mats in slots.items():
                    if name in mats and mesh in vc:
                        col = vc[mesh]
                        break
                found.setdefault(info["textures"][slot]["name"], (info, fo, name, col))
    return next(iter(found.values())) if len(found) == 1 else None


GLASS_ALPHA_MAX = 250  # a glass image has coverage in alpha; chrome/metal matcaps are fully opaque (255)


def _translucent(path: str) -> bool:
    a = np.asarray(Image.open(path).convert("RGBA"))[..., 3]
    return int(a.min()) < GLASS_ALPHA_MAX


def matcap_of(info: dict, folder: str, index: dict[str, str] | None, material: str) -> str | None:
    """Matcap image of a glass material: `_RampMap` (lobby `ghostGlass`), else, for an FBX material, a translucent
    image named after it: `t_matcap_[bat_|lob_]<dex><skin><letter>[_lv3]`, letterless as a fallback (Gengar skin 01
    `009401a` -> `t_matcap_009401_lv3`, Snorlax skin 20 `014320b` -> `t_matcap_bat_014320b`), or a translucent
    non-albedo texture in its own block (Snorlax 20 lobby's helmet: `t_d_lob_014320`, a glass sphere with alpha
    coverage like Gengar's). HYPOTHESIS: ledger `matcapGlass`."""
    t = info["textures"].get("_RampMap")
    if t:
        return _png(folder, t["name"], index)
    if matcap_only(info):  # the matcap is the material's only colour source (Litwick line `fireA/B`)
        return _png(folder, info["textures"][matcap_slot(info["textures"])]["name"], index)
    if info.get("source") != "fbx":
        return None
    albedo = info["textures"].get("_MainTex", {}).get("name", "").lower()
    for n in info.get("block") or []:  # the material's own textures first
        if _stem(n).lower() == albedo or MIX_TEX.match(n) or n.lower().startswith(("t_n_", "t_vc_", "t_fx_")):
            continue
        p = _png(folder, _stem(n), index)
        if p:
            with Image.open(p) as im:
                square = im.width == im.height <= 256
            if square and _translucent(p):
                return p
    m = re.search(r"_(bat|lob)_(\d{6})([a-z]?)_lv\d", material) or re.search(r"()_(\d{6})([a-z]?)_lv\d", material)
    if m:
        kind = m.group(1) + "_" if m.group(1) else ""
        for pre in dict.fromkeys((kind, "", "bat_", "lob_")):  # its own kind's prefix first
            for body in ([m.group(2) + m.group(3)] if m.group(3) else []) + [m.group(2)]:
                for cand in (f"t_matcap_{pre}{body}_lv3", f"t_matcap_{pre}{body}"):
                    p = _png(folder, cand)
                    if p and _translucent(p):
                        return p
    return None


GLASS_SHADER = re.compile(r"ghostGlass", re.I)


def matcap_only(info: dict) -> bool:
    """A material whose only colour source is a matcap (the FBX links the matcap as DiffuseColor: Litwick, Lampent,
    Chandelure flames). Drawn by the matcap, like `ghostGlass` (alpha = the matcap's own)."""
    tx = info["textures"]
    return color_slot(tx) is None and matcap_slot(tx) is not None
# FITTED: mean albedo alpha under a material's own UVs below which an FBX-only material without a normal map is glass,
# on the 127 DiffuseColor-only materials of the matcap folders (selects Snorlax 20 lobby 0 and Gengar 01 lobby 0; next
# lowest 37 = Blaziken 06 bodyb), 2026-11-01. Ledger `matcapGlass`.
CLEAR_ALBEDO_ALPHA = 16


def clear_albedo(info: dict, prim, folder: str, index: dict[str, str] | None) -> bool:
    """An FBX-only character material that links an albedo but no normal map, and whose albedo is transparent where
    its own triangles sample it: the glass slot of `ghostGlass`-like helmets (Gengar Space's lobby helmet has this
    exact FBX signature and its JSON says `ghostGlass`). Character albedo alpha is otherwise a cloth mask, so the
    missing normal map is part of the rule."""
    tx = info["textures"]
    if info.get("source") != "fbx" or "_MainTex" not in tx or "_BumpMap" in tx or not prim.uvs:
        return False
    path = _png(folder, tx["_MainTex"]["name"], index)
    if not path:
        return False
    a = np.asarray(Image.open(path).convert("RGBA"))[..., 3]
    h, w = a.shape
    uv = np.asarray(prim.uvs[0], np.float64) % 1.0
    vals = a[np.clip((uv[:, 1] * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)]
    return float(vals.mean()) < CLEAR_ALBEDO_ALPHA


def build_glass_material(name: str, info: dict, matcap: str, textures: dict[str, TextureImage],
                         folder: str = "", index: dict[str, str] | None = None) -> Material:
    """Glass (Gengar Space's helmet `ghostGlass`, battle `actorglass`): the albedo under the glass is black with alpha =
    the glass thickness; colour and alpha come from the matcap and the view angle, transparent face-on and opaque at
    grazing angles (`matcap_terms.glass_terms`, drawn by the viewer from `extras.unite.glass`). A material whose only
    colour is a matcap linked as DiffuseColor (Litwick-line fire, `matcapAsDiffuse`) keeps the older matcap-alpha look (`extras.matcap`): its shader
    is not a glass one. Flat fallback = the matcap's mean colour where it is visible and its mean alpha."""
    key = os.path.splitext(os.path.basename(matcap))[0]
    if key not in textures:
        textures[key] = TextureImage(key=key, rgba=np.array(Image.open(matcap).convert("RGBA"), np.uint8), srgb=True)
    rgba = textures[key].rgba.astype(np.float64) / 255.0
    vis = rgba[..., 3] > 20 / 255
    rgb = rgba[..., :3][vis].mean(0) if vis.any() else rgba[..., :3].reshape(-1, 3).mean(0)
    mat = Material(name=name, roughness_factor=0.1, metallic_factor=0.0)
    mat.base_color_factor = (*(float(v) for v in _srgb_to_linear(rgb)), float(rgba[..., 3].mean()))
    mat.alpha_mode = "BLEND"
    mat.double_sided = info["floats"].get("_Cull") == 0.0
    mat.extras = {"unity": unity_extras(info), "blend": "Blend", "glass": True}
    if info.get("matcapAsDiffuse"):
        mat.extras["matcap"] = key
        return mat
    # the shader's `main` is the albedo (alpha = thickness) or, with no texture, black with alpha 0
    main = info["textures"].get("_MainTex")
    path = _png(folder, main["name"], index) if main else None
    if path:
        mk = main["name"]
        if mk not in textures:
            textures[mk] = TextureImage(key=mk, rgba=np.array(Image.open(path).convert("RGBA"), np.uint8), srgb=True)
        mat.base_color, mat.base_color_factor = mk, (1.0, 1.0, 1.0, 1.0)
    else:
        mat.base_color_factor = (0.0, 0.0, 0.0, 0.0)
    mat.extras.setdefault("unite", {})["glass"] = matcap_terms.glass_terms(info, key, name)
    return mat


# Shader families that never shade the character itself (surveyed over every lobby-scene material, 2026-11-01): move
# effects (`Fx/`, `Fx2/`, also used for glow cards skinned to the Pokémon), backdrop cards (`Unlit/`, `LobbyScene/`).
# Character materials come in many families (`LobbyPokemon/`, `Lobby/m_lob_basepbr*`, `Hidden/Lobby/...Lengcy`,
# `Battle/m_bat_actorbase`), so this is a deny rule.
NON_CHARACTER_SHADER = re.compile(r"/Fx\d?/|^Unlit/|/LobbyScene/")


def is_character_shader(shader: str) -> bool:
    return not NON_CHARACTER_SHADER.search(shader)


# Without a JSON there is no shader name; effect materials are the ones named with an `fx` token or painted with an
# effect texture (all 40 such materials across the 164 battle FBXs: `mat_fx_*`, `mat_bat_000912fx_glow_lv3`,
# `mat_bat_009411fxa_lv3`, `mat_lob_fx_0146_*`; textures `t_fx_*`, `t_lob_fx_*`).
FX_MATERIAL = re.compile(r"(?:^|_|\d)fx(?:[a-z]?_|[a-z]?$|\d)", re.I)
FX_TEXTURE = re.compile(r"^t_(?:lob_)?fx_", re.I)


def is_character_material(name: str, info: dict) -> bool:
    """JSON: by shader family. FBX-only: a normal-map link means a lit character shader whatever the name (0 of 739
    `Fx/` scene materials have `_BumpMap`, 833 character ones do: Ponyta/Rapidash `hari_fx` manes); so does an atlas
    albedo whose own block holds an outline map (`t_vc_`, lit shaders only: lobby Rapidash `haria/harib_fx`). Else
    an `fx` token in the name or an effect colour texture means an effect."""
    if info.get("shader"):
        return is_character_shader(info["shader"])
    tx = info["textures"]
    d = (tx.get(color_slot(tx) or "_MainTex") or {}).get("name", "")
    if "_BumpMap" in tx:
        return True
    if d.lower().startswith("t_d_") and any(n.lower().startswith("t_vc_") for n in info.get("block") or []):
        return True
    return not (FX_MATERIAL.search(name) or FX_TEXTURE.match(d))


BACKDROP_SHADER = re.compile(r"^Unlit/|/LobbyScene/")


def material_role(name: str, info: dict) -> str:
    """"backdrop" (scene cards: `Unlit/`, `LobbyScene/` shaders), "effect" (`Fx/` shaders, `fx`-named materials,
    `t_fx_` albedos: drawn as unlit blended parts attached to the character) or "character"."""
    if info.get("shader") and BACKDROP_SHADER.search(info["shader"]):
        return "backdrop"
    return "character" if is_character_material(name, info) else "effect"


# Effect prefabs only attacks play (Dhelmise `pref_fx_Dhelmise_natk1_atk1`); idle prefabs
# (`pref_fx_0004_battlefire_lv3`, `pref_fx_Gastly_idle_lv3`) stay visible.
ATTACK_PREFAB = re.compile(r"^pref_fx_.*(?:atk|skill|buff)", re.I)


def ancestors(scene: "fbx_anim.FbxScene", name: str) -> list[str]:
    node = next((n for n in scene.nodes.values() if n.name == name and n.kind == "Mesh"), None)
    out = []
    k = node.parent if node else 0
    while k in scene.nodes:
        out.append(scene.nodes[k].name)
        k = scene.nodes[k].parent
    return out


OVERLAY_SHARE = 0.9  # fraction of a primitive's vertices that coincide with another, textured primitive's


def overlay_shells(model: Model) -> list[tuple[str, str]]:
    """Primitives without a colour texture whose vertices (>= 90%, within 1e-5 m) are another textured primitive's:
    an overlay pass drawn on the same surface (Xmas Pikachu's `ClothTwinkle` sparkle over the coat). Imported opaque
    they z-fight the surface they decorate. Returns (mesh, material) pairs."""
    textured = [p.positions for m in model.meshes for p in m.primitives if model.materials[p.material].base_color]
    if not textured:
        return []
    ref = set(map(tuple, np.round(np.concatenate(textured) / 1e-5).astype(np.int64).tolist()))
    out = []
    for mesh in model.meshes:
        for p in mesh.primitives:
            mat = model.materials[p.material]
            if mat.base_color or mat.extras.get("glass") or not len(p.positions):
                continue
            q = np.round(p.positions / 1e-5).astype(np.int64).tolist()
            if np.mean([tuple(v) in ref for v in q]) >= OVERLAY_SHARE:
                out.append((mesh.name, p.material))
    return out


# FITTED: share of directions (from the body centre) in which an effect must reach beyond the body to count as a shell
# around it; surveyed on the 58 effect primitives the old bbox test flagged: none above 0.85 (Solgaleo's coincident
# shell), Ponyta/Rapidash manes 0.42-0.58, the rest <= 0.35. 2026-11-01.
ENCLOSE_MIN = 0.9
SHELL_REACH = 0.98  # an effect reaching 98% as far as the body in a direction covers it there (coincident shells)


def _directions() -> np.ndarray:
    """The 26 directions of a 3x3x3 grid around the centre (unit vectors)."""
    d = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], float)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def enclosure(effect: np.ndarray, body: np.ndarray, centre: np.ndarray) -> float:
    """Share of the 26 directions from `centre` in which the effect reaches out as far as the body (>= SHELL_REACH of it:
    a shell lying ON the body, Solgaleo's glow or Gastly skin 11's aura, counts; cones of cos > 0.9; a direction the
    body does not reach counts when the effect does)."""
    dirs = _directions()

    def reach(pts):
        v = pts - centre
        r = np.linalg.norm(v, axis=1)
        u = v / np.maximum(r[:, None], 1e-9)
        cos = u @ dirs.T  # (N, 26)
        return np.where(cos > 0.9, r[:, None], -1.0).max(axis=0)

    e, b = reach(effect.astype(np.float64)), reach(body.astype(np.float64))
    return float(((e > 0) & (e >= SHELL_REACH * b)).mean())


def _flag_enclosing_effects(model: Model) -> None:
    """Effect parts that wrap the body like a shell (beyond it in >= ENCLOSE_MIN of all directions) get `behindBody`, so
    the viewer paints the body over them (as HOME's smoke, ZA's gas). The old test (bounds contain the body centre) also
    caught manes and tufts attached on one side, which were then drawn behind the head."""
    solid = [p.positions for m in model.meshes for p in m.primitives if not model.materials[p.material].extras.get("effect")]
    if not solid:
        return
    allp = np.concatenate(solid)
    centre = (allp.min(0) + allp.max(0)) / 2
    for m in model.meshes:  # per mesh: one shell may be split over several materials (Solgaleo's glow: 07 + 08)
        fx = [p for p in m.primitives if model.materials[p.material].extras.get("effect") and len(p.positions)]
        if fx and enclosure(np.concatenate([p.positions for p in fx]), allp, centre) >= ENCLOSE_MIN:
            for p in fx:
                model.materials[p.material].extras["behindBody"] = True


def _uv_islands(indices: np.ndarray, uv: np.ndarray) -> list[np.ndarray]:
    """Vertex-index groups of the connected triangle islands of one primitive."""
    parent = np.arange(len(uv))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for t in indices.reshape(-1, 3):
        r = find(int(t[0]))
        for v in t[1:]:
            parent[find(int(v))] = r
    used = np.unique(indices)
    roots = np.array([find(int(v)) for v in used])
    return [used[roots == r] for r in np.unique(roots)]


def _pow2_cells(extent: float) -> int:
    """Grid divisions (1, 2, 4 or 8) whose cell still holds an island of this UV extent."""
    for n in (8, 4, 2):
        if extent <= 1.0 / n + 0.01:
            return n
    return 1


# FITTED: share of the sheet's drawn pixels a cell border may cut through and still count as a border (Lapras 2x4 sheet:
# 0; the wrong 2x8 split cuts every eye), surveyed over the UNITE eye/mouth sheets 2026-11-01.
GRID_CUT_MAX = 0.002


def _grid_cuts_content(rgba: np.ndarray, cols: int, rows: int) -> bool:
    """True when a cell border of a cols x rows grid runs through drawn content (pixels unlike the sheet's background;
    the pure-red unused-cell marker is not content). A real cell grid never cuts an expression in two."""
    H, W = rgba.shape[:2]
    flat = rgba.reshape(-1, rgba.shape[-1])
    keys, counts = np.unique(flat, axis=0, return_counts=True)
    bg = keys[counts.argmax()].astype(int)
    content = np.abs(rgba.astype(int) - bg).sum(-1) > 40
    rgb = rgba[..., :3].astype(int)
    red = (rgb[..., 0] > 240) & (rgb[..., 1] < 20) & (rgb[..., 2] < 20)  # the marker is near-red, not pure
    near = red.copy()
    for d in (1, 2):  # its anti-aliased edge is a blend with the background (box dilation, corners included)
        near[:, d:] |= red[:, :-d]
        near[:, :-d] |= red[:, d:]
    red = near.copy()
    for d in (1, 2):
        near[d:] |= red[:-d]
        near[:-d] |= red[d:]
    content &= ~near
    total = content.sum()
    if total == 0:
        return False
    cut = 0
    for k in range(1, cols):
        x = k * W // cols
        cut += content[:, max(x - 1, 0):x + 1].any(1).sum()
    for k in range(1, rows):
        y = k * H // rows
        cut += content[max(y - 1, 0):y + 1, :].any(0).sum()
    return cut / max(H + W, 1) > GRID_CUT_MAX


def _atlas_grid(rgba: np.ndarray, cols0: int, rows0: int) -> tuple[int, int]:
    """Finest grid no larger than (cols0, rows0) whose borders cut no drawn content. The UV island only bounds the cell
    size from below (an eye mesh may fill a fraction of its cell), so the sheet itself decides the grid."""
    cands = sorted(((c, r) for c in (1, 2, 4, 8) for r in (1, 2, 4, 8) if c <= cols0 and r <= rows0),
                   key=lambda cr: (-cr[0] * cr[1], -cr[0]))
    for c, r in cands:
        if not _grid_cuts_content(rgba, c, r):
            return c, r
    return cols0, rows0


def eye_cell_shifts(islands: list[tuple[float, float, float, float]], rgba: np.ndarray) -> tuple[int, int, list[tuple[int, int]]]:
    """Expression atlas of an eye material. `islands` = UV boxes (u0, v0, u1, v1; glTF space, V down) of the eye
    shapes. UNITE eye sheets are grids of cells (2 columns of left/right eye, one expression per row, or 4 columns of
    two pairs); the mesh sits on one expression. Grid = power-of-two divisions that hold the largest island; the
    expression is the block of cells the islands occupy (span), and the alternatives are the same block shifted by whole
    spans (so a left eye never lands in a right-eye column), skipping blocks that leave the sheet, are blank, or are the
    sheet's pure-red "unused" marker. Returns (cols, rows, [(dc, dr) shifts in cells, excluding (0, 0)]).
    Surveyed over 1,147 UNITE eye materials (docs/ledger/unite.json `eyeAtlas`)."""
    cols = _pow2_cells(max(b[2] - b[0] for b in islands))
    rows = _pow2_cells(max(b[3] - b[1] for b in islands))
    cols, rows = _atlas_grid(rgba, cols, rows)
    cells = [(int(np.clip(np.floor((b[0] + b[2]) / 2 * cols), 0, cols - 1)), int(np.clip(np.floor((b[1] + b[3]) / 2 * rows), 0, rows - 1))) for b in islands]
    c0, c1 = min(c for c, _ in cells), max(c for c, _ in cells)
    r0, r1 = min(r for _, r in cells), max(r for _, r in cells)
    sc, sr = c1 - c0 + 1, r1 - r0 + 1
    H, W = rgba.shape[:2]

    def block(c: int, r: int) -> np.ndarray:
        return rgba[r * H // rows:(r + sr) * H // rows, c * W // cols:(c + sc) * W // cols]

    home = block(c0, r0)
    out: list[tuple[int, int]] = []
    seen = [home]
    for r in range(r0 % sr, rows - sr + 1, sr):
        for c in range(c0 % sc, cols - sc + 1, sc):
            if (c, r) == (c0, r0):
                continue
            b = block(c, r)
            rgb = b[..., :3].astype(int)
            if (rgb == (255, 0, 0)).all(-1).mean() > 0.9:
                continue  # the sheet's unused-cell marker (Ninetales lobby: 96% pure red, the rest is edge noise)
            if (np.abs(rgb - np.median(rgb.reshape(-1, 3), axis=0)).sum(-1) > 40).mean() < 0.002:
                continue  # blank
            if any(b.shape == s.shape and np.array_equal(b, s) for s in seen):
                continue
            seen.append(b)
            out.append((c - c0, r - r0))
    return cols, rows, out


def atlas_variants(model: Model, token: str = "eye", group: str = "eyes", label: str = "Eyes") -> int:
    """Add a variant group (`eyes`, `mouth`) for the materials whose name has `token` and that have an expression atlas
    (the rip has no eye/mouth texture animation: FBX clips only hold bone T/R/S, see the ledger), so the viewer can
    offer the sheet's other expressions. Options offset the base colour map by whole cells (KHR_texture_transform);
    the first option is the authored expression."""
    n_opts = 0
    groups: dict[str, list[tuple[str, int, int]]] = {}
    for name, mat in model.materials.items():
        if token not in name.lower() or not mat.base_color:
            continue
        boxes = []
        for mesh in model.meshes:
            for p in mesh.primitives:
                if p.material == name and p.uvs:
                    uv = p.uvs[0]
                    for isl in _uv_islands(p.indices, uv):
                        lo, hi = uv[isl].min(0), uv[isl].max(0)
                        boxes.append((float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])))
        if not boxes:
            continue
        cols, rows, shifts = eye_cell_shifts(boxes, model.textures[mat.base_color].rgba)
        if shifts:
            groups[name] = [(name, dc / cols, dr / rows) for dc, dr in shifts]
    if not groups:
        return 0
    n = max(len(v) for v in groups.values())
    model.variants.append(Variant(name=f"{group}_0", materials={}, group=group, label="Default", default=True, group_label=label))
    for i in range(n):
        mats = {}
        for name, opts in groups.items():
            if i < len(opts):
                m = copy.deepcopy(model.materials[name])
                m.name = name
                m.uv_transform["base_color"] = (1.0, 1.0, opts[i][1], opts[i][2])
                mats[name] = m
        model.variants.append(Variant(name=f"{group}_{i + 1}", materials=mats, group=group, label=f"{label} {i + 1}", group_label=label))
        n_opts += 1
    return n_opts


JAW_BONE = re.compile(r"jaw", re.I)
# FITTED: a jaw that stays within 25 deg of its bind pose for a whole clip is the game's slight idle/run mouth-open,
# which reads as a gaping mouth (the lobby copy has no jaw track, so it rests closed at the bind pose); attacks and
# roars open 30-60 deg. Surveyed on 234 battle models with animated jaws (64 hold >= 5 deg open in idle; Ninetales
# 8-19 deg), 2026-11-01.
JAW_REST_MAX_DEG = 25.0
# FITTED: a jaw track whose keys all stay this close to the closed pose already rests closed (Gengar idle/run/stun hold
# -8.6..-9.6 deg from bind while its closed pose is ~-9), so it is kept as authored, 2026-11-02.
JAW_CLOSED_TOL = 3.0


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of (x, y, z, w) quaternions; b may be (N, 4)."""
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)


def _qinv(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], np.float64)


def _rotvecs(d: np.ndarray) -> np.ndarray:
    """(N, 4) unit quaternions -> (N, 3) rotation vectors (radians), shortest arc."""
    d = np.where(d[:, 3:4] < 0, -d, d)
    n = np.linalg.norm(d[:, :3], axis=1)
    ang = 2 * np.arctan2(n, d[:, 3])
    return d[:, :3] * (ang / np.maximum(n, 1e-12))[:, None]


def rest_jaws(model: Model) -> int:
    """Close the mouth at rest: jaw rotation tracks that never open more than JAW_REST_MAX_DEG past the jaw's CLOSED
    pose are an idle/run mouth-open artefact. The closed pose is not always the bind pose: Ninetales' bind is closed and
    its idle holds 8-19 deg open, but Gengar's bind is 9-19 deg open and every clip holds the jaw shut below it.

    Per jaw bone, over all keys of all clips: signed angle s along the dominant hinge axis, flipped so the side whose
    extreme lies farther from the median opens; closed s_c = min(0, 5th percentile) (0 = bind). Without a key opening
    more than JAW_REST_MAX_DEG past it (lobby models with one idle clip) the open side is unknown and nothing changes.
    A track within JAW_CLOSED_TOL of closed is kept; one that stays within JAW_REST_MAX_DEG is dropped (closed = bind)
    or held at the authored closed key. Writes extras.jaws (slider: closed -> widest open) and extras.jawClosed.
    Returns the number of tracks dropped or replaced."""
    sk = model.skeleton
    if sk is None:
        return 0
    rest = {j.name: np.asarray(j.rotation, np.float64) for j in sk.joints if JAW_BONE.search(j.name)}
    keys: dict[str, list[np.ndarray]] = {}
    for c in model.clips:
        for t in c.tracks:
            if t.kind == "rotation" and t.joint in rest:
                keys.setdefault(t.joint, []).append(np.asarray(t.values, np.float64).reshape(-1, 4))
    jaws, closed_info = [], {}
    plan: dict[str, tuple[np.ndarray, np.ndarray, float, np.ndarray | None]] = {}  # bone -> (bind^-1, axis, s_c, closed q)
    for bone, arrs in keys.items():
        inv = _qinv(rest[bone])
        q_all = np.concatenate(arrs)
        v = _rotvecs(_qmul(inv, q_all))
        if not np.any(np.linalg.norm(v, axis=1) > 1e-6):
            continue
        axis = np.linalg.svd(v, full_matrices=False)[2][0]
        s = np.degrees(v @ axis)
        if s[int(np.argmax(np.abs(s)))] < 0:  # SVD sign is arbitrary: start from the key farthest from bind
            axis, s = -axis, -s
        med = float(np.median(s))
        if med - s.min() > s.max() - med + 1e-3:
            axis, s = -axis, -s
        s_c = min(0.0, float(np.percentile(s, 5)))
        if s.max() - s_c <= JAW_REST_MAX_DEG:
            continue  # no clip opens this jaw: open side unknown, keep everything as authored
        q_c = None
        if s_c < 0:
            q_c = q_all[int(np.argmin(np.abs(s - s_c)))]
        plan[bone] = (inv, axis, s_c, q_c)
        closed_info[bone] = round(s_c, 1)
        base = q_c if q_c is not None else rest[bone]
        d = _qmul(_qinv(base), q_all[int(np.argmax(s))])
        if d[3] < 0:
            d = -d
        jaws.append({"bone": bone, "open": [round(float(x), 6) for x in d], "degrees": round(float(s.max() - s_c), 1)})
    if jaws:
        model.extras["jaws"] = jaws
        model.extras["jawClosed"] = closed_info
    changed = 0
    for c in model.clips:
        keep = []
        for t in c.tracks:
            if t.kind == "rotation" and t.joint in plan:
                inv, axis, s_c, q_c = plan[t.joint]
                s = np.degrees(_rotvecs(_qmul(inv, np.asarray(t.values, np.float64).reshape(-1, 4))) @ axis) - s_c
                if s.max() > JAW_CLOSED_TOL and s.max() <= JAW_REST_MAX_DEG:
                    changed += 1
                    if q_c is None:
                        continue  # closed = bind: drop the track
                    t = Track(t.joint, "rotation", np.zeros(1, np.float32), q_c.astype(np.float32)[None])
            keep.append(t)
        c.tracks = keep
    if changed:
        model.extras["jawRestTracks"] = changed
    return changed


def clean_clip_name(name: str) -> str:
    """`ani_atk1_bat_002501` -> `atk1`, `ani_idle_lob_003` -> `idle` (AssetStudio clip = Unity AnimationClip name)."""
    return re.sub(r"_(?:bat|lob)_\w+$", "", re.sub(r"^ani_", "", name)) or name


def clip_names(stacks: list[str]) -> list[str]:
    """Clean names for a file's clip stacks. Two stacks can clean to one name: Morpeko's Hangry file also carries the
    Full Belly `ani_run_bat_0877_12` beside its own `ani_run_bat_0877_11` (32 of 1,485 models). The stack with the
    file's own tag (its `idle`'s, else the most common) keeps the bare name, the others get ` [tag]`."""
    cleaned = [clean_clip_name(n) for n in stacks]
    tags = [(re.search(r"_(?:bat|lob)_(\w+)$", n) or [None, ""])[1] for n in stacks]
    idle = [t for c, t in zip(cleaned, tags) if c == "idle"]  # idle is always the file's own mode (Morpeko 11/12)
    own = idle[0] if idle else (max(set(tags), key=tags.count) if tags else "")
    out = []
    for n, c, t in zip(stacks, cleaned, tags):
        dup = cleaned.count(c) > 1
        mine = t == own or not any(t2 == own for c2, t2 in zip(cleaned, tags) if c2 == c)
        out.append(f"{c} [{t}]" if dup and t and not mine else c)
    for i, n in enumerate(out):  # still equal (same tag twice): number them
        if out.count(n) > 1:
            out[i] = f"{n} [{out[:i].count(n) + 1}]" if out[:i].count(n) else n
    return out


BIND_TOL = 1e-3  # m (+0.5% of the mesh size): dumped vertices vs the FBX bind space, beyond this the FBX skeleton is not trusted


def _reset_top(scene: "fbx_anim.FbxScene") -> None:
    """Express the scene in its top-level node's space (a prefab placed at the origin): Charizard's root Null sits at
    x -1 m, Ninetales skin 11's is turned 161 deg. Its transform is removed from the scene and the binds."""
    tops = [n for n in scene.nodes.values() if n.parent == 0]
    if len(tops) != 1:
        return
    root_w = fbx_anim.world_rest(scene, tops[0].id)
    if np.allclose(root_w, np.eye(4), atol=1e-9):
        return
    tops[0].t, tops[0].r, tops[0].s = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    tops[0].pre = tops[0].post = (0.0, 0.0, 0.0)
    inv_root = np.linalg.inv(root_w)
    for n in scene.nodes.values():
        if n.bind is not None:
            n.bind = inv_root @ n.bind
    for own in scene.mesh_cluster_binds.values():
        for bid, (tl, w) in list(own.items()):
            own[bid] = (inv_root @ tl, w)


def _match_nodes(sk, scene: "fbx_anim.FbxScene") -> list:
    """FBX node of each joint (None = no node of that name): by name, below the parent joint's node when same-named
    nodes exist (Unity binds curves by path; the tail flame prefab repeats `Bone_Tail06`)."""
    by_name: dict[str, list] = {}
    for n in scene.nodes.values():
        if n.kind in ("LimbNode", "Null", "Root"):
            by_name.setdefault(n.name, []).append(n)

    def bind_of(n):
        return n.bind if n.bind is not None else fbx_anim.world_rest(scene, n.id)

    nodes: list = []
    for j in sk.joints:
        blender_world = np.linalg.inv(np.asarray(j.inverse_bind, np.float64))
        cands = by_name.get(_blender_name(j.name)) or []
        if not cands:
            nodes.append(None)
            continue
        if j.parent >= 0 and nodes[j.parent] is not None:
            pid = nodes[j.parent].id

            def under(c, pid=pid):
                k = c.parent
                while k in scene.nodes:
                    if k == pid:
                        return True
                    k = scene.nodes[k].parent
                return False

            cands = [c for c in cands if under(c)] or cands
        if len(cands) > 1:  # prefer a node that actually deforms vertices, then the one nearest the joint's bind
            cands = sorted(cands, key=lambda c: (c.id not in scene.clustered,
                                                 np.linalg.norm(bind_of(c)[:3, 3] - blender_world[:3, 3])))
        nodes.append(cands[0])
    return nodes


def _above(sk, nodes: list, scene: "fbx_anim.FbxScene") -> list:
    """Per joint: the static rest transform of what sits above it in the FBX but is not its parent joint (identity
    almost always). None for a joint without a node, or whose parent joint has none."""
    out: list = []
    for i, n in enumerate(nodes):
        p = sk.joints[i].parent
        if n is None or (p >= 0 and nodes[p] is None):
            out.append(None)
            continue
        top = nodes[p].id if p >= 0 else 0
        if n.parent == top:
            out.append(np.eye(4))
        elif p < 0:
            out.append(fbx_anim.world_rest(scene, n.parent))
        else:
            out.append(np.linalg.inv(fbx_anim.world_rest(scene, top)) @ fbx_anim.world_rest(scene, n.parent))
    return out


def _sample_clips(sk, nodes: list, above: list, scene: "fbx_anim.FbxScene", stacks: set[str] | None = None) -> list[Clip]:
    """The scene's clip stacks (or only `stacks`) sampled at their key rate onto the skeleton's joints (`nodes` =
    each joint's FBX node, `above` from `_above`). Tracks equal to the joint's rest are left out."""
    clips = []
    for ci, (stack, channels) in enumerate(sorted(scene.stacks.items())):
        if stacks is not None and stack not in stacks:
            continue
        times, fps = fbx_anim.clip_times(channels)
        tracks: list[Track] = []
        for i, (j, n) in enumerate(zip(sk.joints, nodes)):
            ch = channels.get(n.id) if n is not None and above[i] is not None else None
            if not ch:
                continue
            tt = fbx_anim.sample_channel(n, ch, "T", times)
            rr = fbx_anim.sample_channel(n, ch, "R", times)
            ss = fbx_anim.sample_channel(n, ch, "S", times)
            if np.allclose(above[i], np.eye(4), atol=1e-9):
                q = np.array([fbx_anim.quat_xyzw(fbx_anim.rotation_part(r, n.pre, n.post)) for r in rr])
            else:
                trs = [fbx_anim.decompose(above[i] @ fbx_anim.local_matrix(t, r, sc, n.pre, n.post)) for t, r, sc in zip(tt, rr, ss)]
                tt = np.array([a for a, _, _ in trs]); q = np.array([b for _, b, _ in trs]); ss = np.array([c for _, _, c in trs])
            for k in range(1, len(q)):  # shortest path for interpolation
                if np.dot(q[k], q[k - 1]) < 0:
                    q[k] = -q[k]
            rest = {"translation": np.asarray(j.translation), "rotation": np.asarray(j.rotation), "scale": np.asarray(j.scale)}
            for kind, vals in (("translation", tt), ("rotation", q), ("scale", ss)):
                vals = vals.astype(np.float32)
                flat = np.abs(vals - vals[0]).max() < 1e-5
                if flat and (np.abs(vals[0] - rest[kind]).max() < 1e-5 or (kind == "rotation" and np.abs(vals[0] + rest[kind]).max() < 1e-5)):
                    continue
                if flat:
                    tracks.append(Track(j.name, kind, times[:1].astype(np.float32), vals[:1]))
                else:
                    f, v = decimate(times.astype(np.float32), vals, 1e-4)
                    tracks.append(Track(j.name, kind, f.astype(np.float32), v.astype(np.float32)))
        clips.append(Clip(name=stack, tracks=tracks, frame_rate=fps, frame_count=len(times),
                          loop=bool(re.search(r"idle|loop", stack.lower())), index=ci, category=categorize(stack), source=stack))
    return clips


BASE_BIND_SHARE = 0.9  # share of the donor's keyed bones that must exist (by path) on the skin's rig


def _typo_variant(own: str, base: str) -> bool:
    """`own` is `base` misspelled: same digits in the same order and one adjacent-letter transposition, or one letter
    more/less (`akt1`/`atk1`, `spll1`/`spell1`, `sepll1b1`/`spell1b1`). A changed digit (`atk1`/`atk2`) or a replaced
    letter never counts."""
    a, b = own.lower(), base.lower()
    if a == b or re.findall(r"\d+", a) != re.findall(r"\d+", b):
        return False
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        return len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]
    if abs(len(a) - len(b)) != 1:
        return False
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1:] == short and long_[i].isalpha() for i in range(len(long_)))


def merge_base_clips(own: list[str], base: list[str]) -> tuple[list[str], dict[str, str]]:
    """Which base clips a skin inherits. `own`/`base` are clean clip names (`clean_clip_name`). A skin's own clip
    replaces the base clip of the same name (Unity AnimatorOverrideController), or of the name it misspells. Returns
    (base names to add, {overridden base name: own name} for the misspelled ones)."""
    own_set = set(own)
    add, aliases = [], {}
    for b in base:
        if b in own_set:
            continue
        typo = next((o for o in own if o not in base and _typo_variant(o, b)), None)
        if typo:
            aliases[b] = typo
        else:
            add.append(b)
    return add, aliases


def inherit_base_clips(model: Model, donor_fbx: str) -> dict | None:
    """UNITE skins are override controllers: a skin FBX carries only the clips authored for it, tagged with the skin
    (`ani_born_bat_002513`), and plays the species' base set (tagged `0025`, in the form's skin-00 FBX) for the rest.
    Pikachu 13 ships 4 clips and no `idle`, Eevee 18 only `score`. Samples the donor's clips the skin lacks onto the
    skin's joints (matched by name below the same parent, as Unity binds curves by path) with their values verbatim
    (Unity applies curve values absolutely). Returns the record for `extras.inheritedClips`, or None."""
    sk = model.skeleton
    try:
        scene = fbx_anim.parse(donor_fbx)
    except Exception as e:  # noqa: BLE001
        model.extras.setdefault("warnings", []).append(f"base clips not read ({os.path.basename(donor_fbx)}): {e}")
        return None
    _reset_top(scene)
    stacks = sorted(scene.stacks)
    names = clip_names(stacks)
    # the donor's own set: a stack that cleans to an alternate (` [tag]`, another mode's) is not part of it
    base = {n: s for s, n in zip(stacks, names) if " [" not in n}
    own = [clean_clip_name(c.name) for c in model.clips]
    add, aliases = merge_base_clips(own, list(base))
    if not add:
        return None
    nodes = _match_nodes(sk, scene)
    above = _above(sk, nodes, scene)
    keyed = {nid for s in add for nid in scene.stacks[base[s]]}
    bound = {n.id for n, a in zip(nodes, above) if n is not None and a is not None}
    limbs = {nid for nid in keyed if scene.nodes.get(nid) is not None and scene.nodes[nid].kind == "LimbNode"}
    # A skin may lack an accessory subtree of the base (Fletchinder 02 has no folded-wing `*WingClose*_ACC` chain
    # under `Bip001 Spine`); Unity ignores curves whose path is missing. Such subtrees hang below a bound bone and are
    # left out of the share, which then measures whether the rig itself is the base's.
    skin_names = {_blender_name(j.name) for j in sk.joints}

    def accessory(nid: int) -> bool:
        k = nid
        while k in scene.nodes and k not in bound:
            if scene.nodes[k].name in skin_names:
                return False  # same-named bone at another path: a different rig, not a missing accessory
            k = scene.nodes[k].parent
        return k in bound

    absent = {nid for nid in limbs - bound if accessory(nid)}
    rig = limbs - absent
    share = len(rig & bound) / len(rig) if rig else 1.0  # nothing but absent parts keyed: no rig to disagree with
    folder = os.path.basename(os.path.dirname(donor_fbx))
    if share < BASE_BIND_SHARE:
        model.extras.setdefault("warnings", []).append(
            f"base clips of {folder} not inherited: only {share:.0%} of their keyed bones are on this rig")
        return None
    clips = _sample_clips(sk, nodes, above, scene, {base[s] for s in add})
    # clips that only move parts this skin lacks (Scizor 03/04: `jw*` keys the base's `Bone_weapon00N` only)
    empty = [clean_clip_name(c.name) for c in clips if not c.tracks]
    clips = [c for c in clips if c.tracks]
    add = [a for a in add if a not in empty]
    if not clips:
        return None
    start = max((c.index for c in model.clips), default=-1) + 1
    for k, c in enumerate(clips):
        c.index = start + k
    model.clips += clips
    return {"from": folder, "clips": sorted(add), "boundBones": round(share, 3),
            **({"absentBones": sorted(scene.nodes[n].name for n in absent)} if absent else {}),
            **({"overriddenAliases": aliases} if aliases else {}),
            **({"skippedClips": sorted(empty)} if empty else {})}


def add_emitter_joints(model: Model, scene: "fbx_anim.FbxScene") -> list[str]:
    """Empty nodes between an idle effect prefab and the nearest joint become (non-skinning) joints, so the effect
    follows their animation: Charmeleon's and Charizard's `Bone_Tail06` is a Null (no vertex weights, so not in
    Blender's armature) that every clip rotates, and the tail flame prefab hangs from it; pinned to `Bone_Tail05` at
    rest the flame pointed straight back instead of up the curled tail. `apply_fbx_skeleton` then gives them the FBX's
    TRS and curves like any joint. Returns the names added."""
    sk = model.skeleton
    if sk is None or not sk.joints:
        return []
    names = {_blender_name(j.name): i for i, j in enumerate(sk.joints)}
    added = []
    for nid in _emitter_nodes(scene):
        chain, k = [], scene.nodes[nid].parent
        while k in scene.nodes and scene.nodes[k].name not in names:
            n = scene.nodes[k]
            if n.kind == "Mesh" or n.name.lower().startswith("pref_fx_"):
                chain = None
                break
            chain.append(n)
            k = n.parent
        if not chain or k not in scene.nodes:
            continue
        parent = names[scene.nodes[k].name]
        for n in reversed(chain):
            ib = np.linalg.inv(fbx_anim.world_rest(scene, n.id))
            sk.joints.append(Joint(name=n.name, parent=parent, inverse_bind=ib.astype(np.float32)))
            parent = names[n.name] = len(sk.joints) - 1
            added.append(n.name)
    return added


def apply_fbx_skeleton(model: Model, scene: "fbx_anim.FbxScene", placements: dict[str, list] | None = None) -> str | None:
    """Replace Blender's joints (TRS, inverse binds) and clips with the FBX's own values (see `fbx_anim`). Joints keep
    Blender's names/order/parents so the skin weights stay valid. `placements` = mesh name -> the 4x4 matrix the
    Blender dump placed that mesh's vertices with (`dump_fbx.py` meshes[].matrix). Returns None when applied, else
    why not."""
    sk = model.skeleton
    if sk is None or not sk.joints:
        return "no skeleton"
    _reset_top(scene)
    nodes = _match_nodes(sk, scene)
    missing = [j.name for j, n in zip(sk.joints, nodes) if n is None]
    if missing:
        return f"joint {missing[0]} not in the FBX"
    # Blender's bone positions are not compared: its armature drops the rig's `reference` scale (Pikachu 0.745, so
    # `Bip001` sat 0.24 m off) while the vertices come out right. What the FBX inverse binds need is that the
    # vertices are in the FBX bind space; check exactly that.
    # Blender places some meshes differently from the FBX's Lcl chain (it derives the mesh matrix from AssetStudio's
    # odd cluster `Transform`): Charizard dumped at x -2 m, Ninetales skin 11 turned twice, Snorlax 1.275x, Scyther
    # skin 07 raised. Each mesh is re-placed exactly: its FBX rest world (root-local) @ inverse(Blender's matrix).
    # Then every skinned vertex must coincide with the FBX geometry (per vertex, not bounds: dropped effect
    # primitives such as Ponyta's flame mane shrink the dump).
    fbx_verts = fbx_anim.bind_space_vertices(scene)
    mesh_nodes = {n.name: n for n in scene.nodes.values() if n.kind == "Mesh"}
    fixes: dict[str, np.ndarray] = {}
    for mesh in model.meshes:
        node = mesh_nodes.get(_blender_name(mesh.name))
        m = (placements or {}).get(mesh.name)
        if node is not None and m is not None:
            f = fbx_anim.world_rest(scene, node.id) @ np.linalg.inv(np.asarray(m, np.float64))
            if not np.allclose(f, np.eye(4), atol=1e-6):
                fixes[mesh.name] = f
    checked = 0
    for mesh in model.meshes:
        ref = fbx_verts.get(_blender_name(mesh.name))
        prims = [p.positions for p in mesh.primitives if p.joints is not None and len(p.positions)]
        if ref is None or not len(ref) or not prims:
            continue
        f = fixes.get(mesh.name, np.eye(4))
        pos = np.concatenate(prims).astype(np.float64) @ f[:3, :3].T + f[:3, 3]
        tol = BIND_TOL + 0.005 * np.linalg.norm(ref.max(axis=0) - ref.min(axis=0))
        frac = fbx_anim.matched_fraction(pos, ref, tol)
        if frac < 0.99:
            return f"mesh {mesh.name}: only {frac:.0%} of its vertices lie in the FBX bind space"
        checked += 1
    if not checked:
        return "no skinned mesh found in the FBX to verify the bind space"
    worlds = {n.id: fbx_anim.world_rest(scene, n.id) for n in nodes}
    above = _above(sk, nodes, scene)
    for i, (j, n) in enumerate(zip(sk.joints, nodes)):
        if np.allclose(above[i], np.eye(4), atol=1e-9):
            j.translation = tuple(float(v) for v in n.t)
            j.rotation = fbx_anim.quat_xyzw(fbx_anim.rotation_part(n.r, n.pre, n.post))
            j.scale = tuple(float(v) for v in n.s)
        else:
            j.translation, j.rotation, j.scale = fbx_anim.decompose(above[i] @ fbx_anim.local_matrix(n.t, n.r, n.s, n.pre, n.post))
        bm = n.bind if n.bind is not None else worlds[n.id]
        # a bone scaled to 0 at rest (hidden, Unity convention) has no inverse; it skins nothing (Dragapult skin 05), so
        # undoing its position is enough (and a valid glTF affine matrix)
        if abs(np.linalg.det(bm)) > 1e-12:
            ib = np.linalg.inv(bm)
        else:
            ib = np.eye(4)
            ib[:3, 3] = -bm[:3, 3]
        j.inverse_bind = ib.astype(np.float32)

    model.clips = _sample_clips(sk, nodes, above, scene)
    scene.joint_nodes = [n.id for n in nodes]
    # only now that everything succeeded (a failure above keeps Blender's skeleton AND its dump)
    for mesh in model.meshes:
        fix = fixes.get(mesh.name)
        if fix is None:
            continue
        for p in mesh.primitives:
            _transform_primitive(p, fix)
    if fixes:
        model.extras.setdefault("warnings", []).append(
            "re-placed meshes from Blender's matrix to the FBX rest: " + ", ".join(sorted(fixes)))
    return None


def bind_rigid_skins(model: Model, scene: "fbx_anim.FbxScene") -> list[str]:
    """Meshes whose skin clusters all lack weights ride the first cluster's bone at weight 1 (Unity: a renderer
    without blend weights follows its bone 0). Blender's dump put them on joint 0 (the static root) instead, so
    Lucario's staff (`Bone001` under the right hand, scaled 0.01 by clips except while swung) hovered beside the
    body in every clip. Surveyed: 99 such meshes in 80 of 1,508 FBX. Returns the meshes rebound."""
    sk = model.skeleton
    if sk is None or not scene.rigid_skins:
        return []
    by_name: dict[str, int] = {}
    for i, j in enumerate(sk.joints):
        by_name.setdefault(_blender_name(j.name), i)
    pal = {ji: k for k, ji in enumerate(sk.palette)}
    done = []
    for mesh in model.meshes:
        nid = scene.rigid_skins.get(_blender_name(mesh.name))
        ji = by_name.get(scene.nodes[nid].name) if nid is not None else None
        if ji is None or ji not in pal:
            continue
        for p in mesh.primitives:
            if p.joints is None:
                continue
            p.joints = np.zeros_like(p.joints)
            p.joints[:, 0] = pal[ji]
            p.weights = np.zeros_like(p.weights)
            p.weights[:, 0] = 1.0
        done.append(mesh.name)
    return done


def _transform_primitive(p, m: np.ndarray) -> None:
    """Move a primitive's positions/normals/tangents by the 4x4 `m` (column vectors)."""
    m3, nm = m[:3, :3], np.linalg.inv(m[:3, :3]).T
    p.positions = (p.positions @ m3.T + m[:3, 3]).astype(p.positions.dtype)
    if p.normals is not None:
        nn = p.normals @ nm.T
        p.normals = (nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-8)).astype(p.normals.dtype)
    if p.tangents is not None:
        t = p.tangents.copy()
        t[:, :3] = t[:, :3] @ m3.T
        t[:, :3] /= np.maximum(np.linalg.norm(t[:, :3], axis=1, keepdims=True), 1e-8)
        p.tangents = t.astype(p.tangents.dtype)


MESH_BIND_TOL = 1e-4  # max |difference| between a mesh's own cluster bind and the bone's kept bind (m / matrix units)


def _bind_only_by(scene: "fbx_anim.FbxScene", mesh, sk, pos: int) -> bool:
    """True when no other mesh's clusters bind this palette joint (its kept bind is this mesh's own)."""
    nid = scene.joint_nodes[sk.palette[pos]]
    me = _blender_name(mesh.name)
    return not any(nid in binds for name, binds in scene.mesh_cluster_binds.items() if name != me)


def apply_mesh_binds(model: Model, scene: "fbx_anim.FbxScene") -> list[str]:
    """Skin every mesh with its OWN clusters' binds (Unity: bindposes per SkinnedMeshRenderer). The skeleton keeps one
    bind per bone; a mesh whose clusters disagree on a bone it uses would be torn (Miraidon skin 03's body clusters
    include its mesh node's +1.5 m offset, the outfit's on the same bones do not). Unity places a vertex at
    bone(t) @ inv(TL_own) @ p; glTF at bone(t) @ inv(K) @ p', so p' = R p with R = K @ inv(TL_own). When R is the
    same for every bone the mesh uses it is baked into the vertices (one skin); otherwise the mesh gets its own
    inverse binds (`Mesh.inverse_binds`, an extra glTF skin). Returns "<mesh> (baked|own skin)" per mesh changed."""
    sk = model.skeleton
    if sk is None or not scene.joint_nodes or not scene.mesh_cluster_binds:
        return []
    done = []
    for mesh in model.meshes:
        own = scene.mesh_cluster_binds.get(_blender_name(mesh.name))
        if not own:
            continue
        used: set[int] = set()  # palette positions with weight
        for p in mesh.primitives:
            if p.joints is not None and p.weights is not None:
                used |= set(np.unique(p.joints[p.weights > 0]).tolist())
        corr: dict[int, np.ndarray] = {}  # palette position -> R
        for pos in sorted(used):
            nid = scene.joint_nodes[sk.palette[pos]]
            if nid not in own:
                continue
            k = np.linalg.inv(np.asarray(sk.joints[sk.palette[pos]].inverse_bind, np.float64))
            tl = own[nid][0]
            if abs(np.linalg.det(tl)) < 1e-12 or np.abs(k - tl).max() <= MESH_BIND_TOL:
                continue
            corr[pos] = k @ np.linalg.inv(tl)
        if not corr:
            continue
        rs = list(corr.values())
        # bones only this mesh binds (its own cluster set their kept bind, so they need no correction yet) can take
        # the same R: Fuecoco skin 01's tuft node has scale 1 and its clusters carry the 1/40 instead (TransformLink
        # scale 35.7 vs 0.9); its feeler bones are the tuft's alone, only the shared head disagreed, so the tuft got an
        # own skin with vertices 40x too large in bind space. Rebinding them as R @ TL keeps every vertex where it was.
        solo = [pos for pos in sorted(used) if pos not in corr and _bind_only_by(scene, mesh, sk, pos)]
        if (len(corr) + len(solo) == len(used) and all(np.abs(r - rs[0]).max() <= MESH_BIND_TOL for r in rs)):
            for p in mesh.primitives:
                _transform_primitive(p, rs[0])
            for pos in solo:
                j = sk.joints[sk.palette[pos]]
                k = np.linalg.inv(np.asarray(j.inverse_bind, np.float64))
                j.inverse_bind = np.linalg.inv(rs[0] @ k).astype(np.asarray(j.inverse_bind).dtype)
            done.append(f"{mesh.name} (baked" + (f", {len(solo)} own bones rebound)" if solo else ")"))
        else:
            mesh.inverse_binds = {pos: np.linalg.inv(own[scene.joint_nodes[sk.palette[pos]]][0]).astype(np.float32)
                                  for pos in corr}
            done.append(f"{mesh.name} (own skin, {len(corr)} of {len(used)} bones)")
    return done


def tint_effects_by_vertex_colour(model: Model, colours: dict[str, np.ndarray],
                                  borrowed: dict[str, np.ndarray] | None = None) -> list[str]:
    """Effect materials (textured, FBX-only: no JSON `_Color1`) take their mesh's mean vertex colour as the colour of
    their greyscale mask (Unity Fx shaders multiply by the vertex colour; the masks alone rendered white: Charmander's
    tail flame, Mega Charizard Y's). Linear values, used as the emissive factor (alpha x the colour's alpha). A material
    on several meshes takes the first mesh's colour. Ledger `material:effectVertexColour`."""
    done = []
    for mesh in model.meshes:
        c0 = colours.get(_blender_name(mesh.name).split("~")[0])
        for p in mesh.primitives:
            mat = model.materials[p.material]
            # a mesh without a vertex colour layer takes the donor mesh's (`borrow_effect_info`): same authored material
            c = c0 if c0 is not None else (borrowed or {}).get(mat.name)
            if c is None:
                continue
            u = mat.extras.get("unity") or {}
            if (not mat.extras.get("effect") or mat.extras.get("missingTexture") or mat.extras.get("solidEffect")
                    or u.get("source") != "fbx" or mat.name in done):
                continue
            rgb = np.clip(c[:3], 0.0, 1.0)
            mat.emissive_factor = tuple(float(v) for v in rgb)
            mat.base_color_factor = (*mat.base_color_factor[:3], float(mat.base_color_factor[3] * np.clip(c[3], 0.0, 1.0)))
            mat.extras["vertexColour"] = [round(float(v), 4) for v in c]
            done.append(mat.name)
    return done


ALT_STATE_SHARE = 0.9  # both ways: the copy's vertices on the earlier body, and the earlier body's on the copy
COINCIDE_TOL = 3e-3  # m: vertices this close are the same surface (Morpeko's lobby decal sits within 0.21 mm)


def _body_like(model: Model, mesh) -> bool:
    return not all(model.materials[p.material].extras.get("effect") for p in mesh.primitives)


def hide_alternate_states(model: Model) -> list[str]:
    """A whole BODY (non-effect) mesh that repeats an earlier body mesh, both ways (>= ALT_STATE_SHARE of its vertices
    within COINCIDE_TOL of that mesh, and that mesh's vertices of it), is an alternate state of that body which game
    code switches on (Solgaleo's `dyFx04` alternate-texture body copy; no FBX visibility flag or clip track drives it).
    Imported hidden, listed in Model Parts. Effect meshes repeating each other are stacked layers of one effect (kept);
    partial overlaps are not copies (Cinderace lobby's teeth: 2% within 1 mm of the body; the round-2 voxel test at
    0.2% of the model size, ~9 mm, hid them). Returns the meshes hidden."""
    bodies: list[np.ndarray] = []
    out = []
    for mesh in model.meshes:
        if not mesh.primitives or not _body_like(model, mesh):
            continue
        pos = np.concatenate([p.positions for p in mesh.primitives]).astype(np.float64)
        if mesh.visible and not mesh.extras.get("overlayDecal") and len(pos) >= 50:
            for ref in bodies:
                if (fbx_anim.matched_fraction(pos, ref, COINCIDE_TOL) >= ALT_STATE_SHARE
                        and fbx_anim.matched_fraction(ref, pos, COINCIDE_TOL) >= ALT_STATE_SHARE):
                    mesh.visible = False
                    mesh.extras["alternateState"] = True
                    out.append(mesh.name)
                    break
            if not mesh.visible:
                continue
        bodies.append(pos)
    return out


DECAL_CLEAR_SHARE = 0.5  # share of the decal's colour texture that is transparent (alpha < 8): a cut-out sheet


def flag_overlay_decals(model: Model, folder: str, index: dict[str, str] | None) -> list[str]:
    """A body mesh lying on ONE primitive of another mesh, both ways (>= 90% of its vertices within COINCIDE_TOL of that
    primitive, and the primitive's of it), whose colour texture is mostly transparent, is a cut-out decal over that
    surface: Morpeko's `bodyb` cheek sheet (82% alpha 0, four rows of cheek-colour discs; the colour row is switched
    by game code, not in the rip) over its eye slot. Character albedo alpha is forced opaque elsewhere, which turned
    the sheet into a khaki face over the eyes (and z-fought in the lobby, where it is exactly coplanar). Drawn MASK at
    0.5 with the texture's real alpha, hidden by default (the eye sheet already paints the default cheeks). Surveyed
    over 1,504 FBX: the 4 Morpeko `bodyb` meshes. Returns the decal meshes."""
    out = []
    for mesh in model.meshes:
        if not mesh.visible or not mesh.primitives or not _body_like(model, mesh):
            continue
        pos = np.concatenate([p.positions for p in mesh.primitives]).astype(np.float64)
        under = None
        for other in model.meshes:
            if other is mesh:
                continue
            for q in other.primitives:
                qp = q.positions.astype(np.float64)
                if (fbx_anim.matched_fraction(pos, qp, COINCIDE_TOL) >= ALT_STATE_SHARE
                        and fbx_anim.matched_fraction(qp, pos, COINCIDE_TOL) >= ALT_STATE_SHARE):
                    under = q.material
                    break
            if under:
                break
        if not under:
            continue
        mats = {p.material for p in mesh.primitives}
        paths = [_png(folder, (model.materials[m].extras.get("unity") or {}).get("textures", {}).get("_MainTex", ""), index)
                 for m in mats]
        if not paths or not all(paths):
            continue
        alphas = [np.asarray(Image.open(pth).convert("RGBA"))[..., 3] for pth in paths]
        if min(float((a < 8).mean()) for a in alphas) < DECAL_CLEAR_SHARE:
            continue
        for m, pth in zip(mats, paths):
            mat = model.materials[m]
            key = mat.base_color
            if key in model.textures:  # restore the real alpha (build_material forced it opaque)
                rgba = model.textures[key].rgba.copy()
                rgba[..., 3] = np.asarray(Image.open(pth).convert("RGBA"))[..., 3]
                model.textures[key] = TextureImage(key=key, rgba=rgba, srgb=model.textures[key].srgb)
            mat.alpha_mode, mat.alpha_cutoff = "MASK", 0.5
            mat.extras["decal"] = True
        mesh.visible = False
        mesh.extras["overlayDecal"] = under
        out.append(mesh.name)
    return out


SIBLING_COVERAGE = 0.9  # share of a slot's UV samples that must land on drawn texels of the sibling's colour texture


def _drawn(rgba: np.ndarray) -> np.ndarray:
    return (rgba[..., 3] >= 128) & (rgba[..., :3].max(-1) >= 16)


def borrow_coverage(prim, tex: "TextureImage | None") -> float:
    """Share of a primitive's UV samples that land on drawn texels of a (borrowed) effect texture; 0 without UVs."""
    if tex is None or not prim.uvs or not len(prim.uvs[0]):
        return 0.0
    uv = np.asarray(prim.uvs[0], np.float64) % 1.0
    h, w = tex.rgba.shape[:2]
    return float(_drawn(tex.rgba[np.clip((uv[:, 1] * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)]).mean())


def flag_sparse_effects(model: Model) -> list[str]:
    """FBX-only blended effect materials whose own texture is drawn on fewer than `SIBLING_COVERAGE` of their own UV
    samples carry a streak/flicker overlay, not the part's colour (Skeledirge battle flames: `t_fx_rig_lob_0911_body_01_wzb`
    2-4%, the head bird's `t_fx_lob_0911_lv3_02` 22%; footage shows solid flames). Flagged `sparseTexture` (the share), so
    `home_parts` may colour them from the species' HOME counterpart when its match gates pass (ledger
    `fx:sparseEffectTexture`). Noise sheens, Fx-JSON materials and additive glows are not flagged."""
    out = []
    for mesh in model.meshes:
        for p in mesh.primitives:
            mat = model.materials[p.material]
            ex = mat.extras
            if (not ex.get("effect") or ex.get("noiseSheen") or ex.get("solidEffect") or ex.get("missingTexture")
                    or ex.get("unite", {}).get("fx") or ex.get("blend") == "Add" or mat.alpha_mode != "BLEND"
                    or (ex.get("unity") or {}).get("source") != "fbx" or "sparseTexture" in ex):
                continue
            cov = borrow_coverage(p, model.textures.get(mat.emissive or mat.base_color))
            if p.uvs and cov < SIBLING_COVERAGE:
                ex["sparseTexture"] = round(cov, 3)
                out.append(mat.name)
    return out


def _uv_coverage(prim, path: str) -> float:
    uv = np.asarray(prim.uvs[0], np.float64) % 1.0
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    h, w = rgba.shape[:2]
    return float(_drawn(rgba[np.clip((uv[:, 1] * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)]).mean())


def donor_sibling_colour(mesh, prim, material: str, folder: str, index: dict[str, str] | None,
                         cache: dict) -> tuple[dict, str, str] | None:
    """`sibling_colour` across skins: the same mesh (by name) in another FBX of the species that carries this material
    beside a textured slot. Crocalor skin 01's head flame `rig_bat_0910_fx_lv3` is one texture-less slot
    (`mat_fx_rig_bat_0911_01_hb`); skin 00's identical mesh (same positions and UVs) splits it into that slot and
    `mat_bat_0910_fx_lv3` = `t_d_bat_0910_fx_lv3`, the yellow/orange atlas. Same `SIBLING_COVERAGE` gate on this
    primitive's own UVs. Returns ({slot: texture entry}, donor folder, donor material) or None."""
    if not prim.uvs or not len(prim.uvs[0]):
        return None
    me = _blender_name(mesh.name).split("~")[0]
    key = effect_key(material)
    for fbx in sibling_fbxs(folder):
        if fbx not in cache:
            try:
                cache[fbx] = (fbxr.material_textures(fbx), fbxr.material_blocks(fbx))
            except Exception:  # noqa: BLE001
                cache[fbx] = ({}, {})
        if (fbx, "vc") not in cache:
            try:
                cache[(fbx, "vc")] = (fbxr.mesh_vertex_colours(fbx), fbxr.mesh_materials(fbx))
            except Exception:  # noqa: BLE001
                cache[(fbx, "vc")] = ({}, {})
        links, blocks = cache[fbx]
        mats = cache[(fbx, "vc")][1].get(me) or []
        if not any(effect_key(n) == key for n in mats):
            continue
        fo = os.path.dirname(fbx)
        for name in mats:
            if effect_key(name) == key:
                continue
            info = info_from_fbx(links.get(name, {}), fo, index, blocks.get(name), name)
            slot = color_slot(info["textures"])
            path = _png(fo, info["textures"][slot]["name"], index) if slot else None
            if path and _uv_coverage(prim, path) >= SIBLING_COVERAGE:
                return {slot: dict(info["textures"][slot])}, fo, name
    return None


def sibling_colour(mesh, prim, info_of, folder: str, index: dict[str, str] | None) -> dict | None:
    """A texture-less effect slot whose UVs land on drawn content of the colour texture of another slot of the SAME mesh
    is the other half of that atlas (Crocalor's head flame: `_01_hb` samples 94% of the orange half of
    `t_d_bat_0910_fx_lv3`, its sibling the yellow half). Returns {slot: texture entry} or None."""
    if not prim.uvs or not len(prim.uvs[0]):
        return None
    uv = np.asarray(prim.uvs[0], np.float64) % 1.0
    for q in mesh.primitives:
        if q is prim:
            continue
        info = info_of(_blender_name(q.material))
        slot = color_slot(info["textures"])
        path = _png(folder, info["textures"][slot]["name"], index) if slot else None
        if not path:
            continue
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        h, w = rgba.shape[:2]
        hit = _drawn(rgba[np.clip((uv[:, 1] * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)])
        if float(hit.mean()) >= SIBLING_COVERAGE:
            return {slot: dict(info["textures"][slot])}
    return None


def particle_emitters(scene: "fbx_anim.FbxScene") -> list[str]:
    """Idle effect prefabs (`pref_fx_*`, not attack/skill/buff ones) whose subtree holds no mesh: Unity ParticleSystems,
    which AssetStudio's FBX export does not write (only their GameObjects, as empty Nulls). Plain Charizard's tail flame
    (`pref_fx_0004_battlefire_lv3`), Gastly's gas (`pref_fx_Gastly_idle_lv3`). Ledger `fbx:particleSystems`."""
    return sorted({scene.nodes[i].name for i in _emitter_nodes(scene)})


def prefab_key(name: str) -> str:
    """`pref_fx_0004_lobbyfire_atk_lv3 (1)` -> `pref_fx_0004_lobbyfire_atk`: Unity duplicate suffixes and the `_lv3`
    level tag dropped, so a prefab matches its copies nested in other prefabs."""
    return re.sub(r"(?:_lv\d)?$", "", re.sub(r"\s*\(\d+\)$", "", name.strip())).lower()


def nested_donor_prefabs() -> dict[str, str]:
    """prefab_key of a prefab nested inside a donor prefab -> that donor prefab (`unite_fx_donors.json` `nested`):
    plain Charmander's `pref_fx_0004_battlefire_lv3` holds the particle prefab `pref_fx_0004_lobbyfire_atk` beside its
    `CharmanderTail` mesh flame, and every lobby Charizard keeps only `pref_fx_0004_lobbyfire_atk_lv3` (empty Nulls)."""
    try:
        with open(FX_DONORS, encoding="utf-8") as f:
            table = json.load(f)["prefabs"]
    except OSError:
        return {}
    out: dict[str, str] = {}
    for outer, ds in table.items():
        for d in ds:
            for n in d.get("nested") or []:
                out.setdefault(prefab_key(n), outer)
    return out


def _emitter_nodes(scene: "fbx_anim.FbxScene") -> list[int]:
    nested = nested_donor_prefabs()
    kids: dict[int, list[int]] = {}
    for n in scene.nodes.values():
        kids.setdefault(n.parent, []).append(n.id)

    def has_mesh(nid: int) -> bool:
        return scene.nodes[nid].kind == "Mesh" or any(has_mesh(k) for k in kids.get(nid, []))

    out = []
    for n in scene.nodes.values():
        # an `atk`-named prefab that a donor's idle prefab nests is part of the idle effect (lobby Charizard's tail
        # flame `pref_fx_0004_lobbyfire_atk_lv3` on Bone_Tail06)
        if n.name.lower().startswith("pref_fx_") and (not ATTACK_PREFAB.search(n.name) or prefab_key(n.name) in nested)                 and not has_mesh(n.id):
            par = scene.nodes.get(n.parent)
            if par is None or not par.name.lower().startswith("pref_fx_"):  # outermost prefab only
                out.append(n.id)
    return sorted(out, key=lambda i: scene.nodes[i].name)


def own_smoke_prefabs(scene: "fbx_anim.FbxScene", slots: dict[str, list[str]], links: dict, dex: int | None) -> list[int]:
    """Idle effect prefabs of THIS file that hold meshes whose materials have no texture link in the FBX, for a
    species whose HOME effect is smoke: Gastly skin 10's `pref_fx_Gastly_idle_lv3` (`mesh huo1` / `mesh huo2`). They
    are drawn exactly like the borrowers of the same prefab (`attach_fx_donors` with this file as its own donor:
    camera-facing masked disc) and the ordinary mesh copies are removed (user decision 2026-11-02, scoped to the gas;
    plain Charmander's own `CharmanderTail` flame stays an ordinary mesh)."""
    kind = None
    if dex:
        try:
            with open(HOME_FX, encoding="utf-8") as f:
                sp = json.load(f)["species"].get(str(int(dex)))
            kind = sp["effects"][0]["kind"] if sp and sp["effects"] else None
        except OSError:
            pass
    if kind != "smoke":
        return []
    kids: dict[int, list[int]] = {}
    for n in scene.nodes.values():
        kids.setdefault(n.parent, []).append(n.id)
    out = []
    for n in scene.nodes.values():
        if not n.name.lower().startswith("pref_fx_") or ATTACK_PREFAB.search(n.name):
            continue
        par = scene.nodes.get(n.parent)
        if par is not None and par.name.lower().startswith("pref_fx_"):
            continue
        names, stack = [], list(kids.get(n.id, []))
        while stack:
            k = stack.pop()
            stack += kids.get(k, [])
            if scene.nodes[k].kind == "Mesh":
                names.append(scene.nodes[k].name)
        if not names:
            continue
        unlinked = all(not links.get(mat.lower()) for nm in names for mat in slots.get(nm, []) or [""])
        if unlinked:
            out.append(n.id)
    return out


def particle_records(scene: "fbx_anim.FbxScene", model: Model, nodes: list[int] | None = None) -> list[dict]:
    """Where each empty emitter sits, for the viewer's particle stand-in: the nearest ancestor that is a joint of the
    final skeleton (after pruning: Charizard battle's unweighted `Bone_Tail06` is gone, so the flame hangs off
    `Bone_Tail05` at its rest offset) and the emitter's rest transform relative to it. Emitter subtrees carry no
    curves in any clip (surveyed 2026-11-01), so a static child of that bone is exact. `scale` = the largest world
    scale in the emitter's subtree relative to the emitter (`huo 1` 1.3 x `CTFIre (5)` 1.35 on Charizard battle):
    the prefab's own sizing, kept as data."""
    joints = {j.name for j in (model.skeleton.joints if model.skeleton else [])}
    kids: dict[int, list[int]] = {}
    for n in scene.nodes.values():
        kids.setdefault(n.parent, []).append(n.id)
    out = []
    for nid in (_emitter_nodes(scene) if nodes is None else nodes):
        bone = scene.nodes[nid].parent
        while bone in scene.nodes and scene.nodes[bone].name not in joints:
            bone = scene.nodes[bone].parent
        if bone not in scene.nodes:
            continue
        w_e = fbx_anim.world_rest(scene, nid)
        t, q, s = fbx_anim.decompose(np.linalg.inv(fbx_anim.world_rest(scene, bone)) @ w_e)
        own = float(np.linalg.norm(w_e[:3, :3], axis=0).mean()) or 1.0
        sub, stack, inner = 1.0, list(kids.get(nid, [])), []
        while stack:
            k = stack.pop()
            stack += kids.get(k, [])
            if scene.nodes[k].name.lower().startswith("pref_fx_"):
                inner.append(scene.nodes[k].name)
            sub = max(sub, float(np.linalg.norm(fbx_anim.world_rest(scene, k)[:3, :3], axis=0).mean()) / own)
        out.append({"prefab": scene.nodes[nid].name, "bone": scene.nodes[bone].name,
                    "translation": [round(v, 6) for v in t], "rotation": [round(v, 6) for v in q],
                    "scale": [round(v, 6) for v in s], "subtreeScale": round(sub, 4),
                    **({"nested": sorted(set(inner))} if inner else {})})
    return out


FX_DONORS = os.path.join(os.path.dirname(__file__), "..", "..", "catalog", "unite_fx_donors.json")


def attach_fx_donors(model: Model, data_dir: str) -> int:
    """Borrow an empty emitter's geometry from a skin whose copy of the SAME prefab kept its meshes
    (`unite_fx_donors.json`, `tools/gen_unite_fx_donors.py`): Gastly skin 10's `pref_fx_Gastly_idle_lv3` holds the gas
    discs `mesh huo1` / `mesh huo2` that skins 00/01/09/11 only keep as empty Nulls. Each record gains `donor` = the donor
    model id and per mesh its rest transform relative to the prefab node (`matrix`, 4x4 row-major), raw geometry
    (polar-UV disc: U around, V centre -> rim) and its material's texture block in AssetStudio's write order (the first
    is the mask, `t_fx_0092_heibai_03_tjj`), added to `model.textures` as `fx:<file>` and listed in
    `model.extras["fxTextures"]` (the writer stores them, the viewer draws the meshes). Every emitter kind (the
    Charizard-line tail flame borrows plain Charmander's `CharmanderTail`). Per mesh also: `maskChannel` ("a" when the
    first texture's alpha varies, else "r": AssetStudio's grey masks are opaque), `vertexColour` (mean RGBA of the
    mesh's colour layer, Unity Fx shaders multiply by it; None = none) and `behindBody` (the mesh's rest bounds contain
    the donor body's centre, like `_flag_enclosing_effects`). User decision 2026-11-01: nothing that is not in the data
    (no invented shapes or motion). Returns the number of records given a donor."""
    recs = list(model.extras.get("particles") or [])
    if not recs:
        return 0
    try:
        with open(FX_DONORS, encoding="utf-8") as f:
            table = json.load(f)["prefabs"]
    except OSError:
        return 0
    done = 0
    for r in recs:
        donors = [d for d in table.get(r["prefab"], []) if (d["model"] == model.id) == bool(r.get("own"))]
        via = None
        if not donors and not r.get("own"):
            # the emitter's prefab is nested in a donor prefab (lobby Charizard's `lobbyfire_atk` inside
            # `battlefire`): borrow the outer prefab's meshes, placed as they sit relative to the nested copy
            # (lobby Charizard 02/16 wrap it once more: `pref_fx_0004_fire` > `lobbyfire_atk_lv3`)
            nest = nested_donor_prefabs()
            hit = next((n for n in [r["prefab"], *r.get("nested", [])] if prefab_key(n) in nest), None)
            outer = nest.get(prefab_key(hit)) if hit else None
            donors = [d for d in table.get(outer, []) if d["model"] != model.id] if outer else []
            via, inner_name = outer, hit
        if not donors:
            continue
        d = donors[0]
        path = d["fbx"] if os.path.isabs(d["fbx"]) else os.path.join(data_dir, d["fbx"])
        scene = fbx_anim.parse(path)
        if via:
            outer_node = next((n for n in scene.nodes.values() if n.name == via), None)
            pref = None
            if outer_node is not None:
                stack = [outer_node.id]
                while stack and pref is None:
                    k = stack.pop()
                    for c in scene.nodes.values():
                        if c.parent == k:
                            if prefab_key(c.name) == prefab_key(inner_name):
                                pref = c
                                break
                            stack.append(c.id)
        else:
            pref = next((n for n in scene.nodes.values() if n.name == r["prefab"]), None)
        if pref is None:
            continue
        inv = np.linalg.inv(fbx_anim.world_rest(scene, pref.id))
        geo = fbxr.mesh_geometry(path, set(d["meshes"]))
        vcols = fbxr.mesh_vertex_colours(path)
        body = [v for v in fbx_anim.bind_space_vertices(scene).values() if len(v)]
        centre = None
        if body:
            allv = np.concatenate(body)
            centre = (allv.max(0) + allv.min(0)) / 2
        slots = fbxr.mesh_materials(path)
        blocks = fbxr.material_blocks(path)
        folder = os.path.dirname(path)
        meshes = []
        for name in d["meshes"]:
            node = next((n for n in scene.nodes.values() if n.name == name and n.kind == "Mesh"), None)
            if node is None or name not in geo:
                continue
            mat = (slots.get(name) or [""])[0]
            keys = []
            for fn in blocks.get(mat, []):
                src = os.path.join(folder, fn)
                if not os.path.isfile(src):
                    continue
                key = "fx:" + fn
                if key not in model.textures:
                    model.textures[key] = TextureImage(key=key, rgba=np.asarray(Image.open(src).convert("RGBA")), srgb=False)
                keys.append(key)
            gm = geo[name]
            # Rest placement: skinned meshes (CharmanderTail rides the flame's own Bone_Tail06-08 chain; its bind-space
            # vertices sit 1.9 m to the side) are posed by their clusters at rest (linear blend of bone rest world x
            # inverse TransformLink); unskinned ones by their node. Baked into the positions, relative to the prefab.
            w_node = fbx_anim.world_rest(scene, node.id)
            vb = gm["positions"].astype(np.float64) @ w_node[:3, :3].T + w_node[:3, 3]
            skin = fbxr.mesh_skin_weights(path, name)
            binds = scene.mesh_cluster_binds.get(name, {})
            if skin:
                acc = np.zeros_like(vb)
                wsum = np.zeros(len(vb))
                pt = gm["points"]
                for bid, (ix, ww) in skin.items():
                    tl = binds.get(bid, (None,))[0]
                    if tl is None:
                        continue
                    M = fbx_anim.world_rest(scene, bid) @ np.linalg.inv(tl)
                    wp = np.zeros(int(pt.max()) + 1)
                    wp[ix[ix <= pt.max()]] = ww[ix <= pt.max()]
                    wv = wp[pt]
                    acc += wv[:, None] * (vb @ M[:3, :3].T + M[:3, 3])
                    wsum += wv
                ok = wsum > 1e-6
                vb[ok] = acc[ok] / wsum[ok, None]
            vw = vb
            vb = vw @ inv[:3, :3].T + inv[:3, 3]
            m = np.eye(4)
            chan = "r"
            if keys:
                a = model.textures[keys[0]].rgba[..., 3]
                chan = "a" if int(a.max()) - int(a.min()) > 8 else "r"
            behind = bool(centre is not None and np.all(vw.min(0) <= centre) and np.all(centre <= vw.max(0)))
            vc = vcols.get(name)
            ft = flame_coordinate(vb, np.zeros(3))  # the prefab origin sits on the emitter's joint
            meshes.append({"name": name, "flameT": None if ft is None else [round(float(v), 4) for v in ft],
                           "noiseDir": noise_direction(gm["uvs"], ft), "material": mat, "textures": keys, "maskChannel": chan,
                           "vertexColour": None if vc is None else [round(float(v), 4) for v in vc], "behindBody": behind,
                           # per vertex too: the Charmander flame runs orange at its root (G 0.27) to red at the tip
                           # (G 0.075), which the mean flattened
                           "vertexColours": None if gm.get("colours") is None
                           else [round(float(v), 3) for v in gm["colours"][:, :3].reshape(-1)],
                           "matrix": [round(float(v), 6) for v in m.reshape(-1)],
                           "positions": [round(float(v), 5) for v in vb.reshape(-1)],
                           "uvs": None if gm["uvs"] is None else [round(float(v), 5) for v in gm["uvs"].reshape(-1)],
                           "indices": [int(v) for v in gm["indices"]]})
        if meshes:
            r["donor"] = {"model": d["model"], "meshes": meshes, **({"via": via} if via else {})}
            fx = model.extras.setdefault("fxTextures", [])
            fx += [k for mm in meshes for k in mm["textures"] if k not in fx]
            done += 1
    return done


HOME_FX = os.path.join(os.path.dirname(__file__), "..", "..", "catalog", "unite_home_fx.json")
HOME_FX_NOISE = os.path.join(os.path.dirname(HOME_FX), "unite_home_fx")


def noise_direction(uvs, flame_t) -> float:
    """+1 / -1: which way HOME's V scroll runs along this flame. HOME's shader is not in the rip, so the sign of a
    `*UVTranslate` speed is ambiguous; it is chosen so noise features travel root -> tip (as flames do in every
    reference), from the correlation of the mesh's V with its along-flame coordinate (INFERRED, ledger `fx:homeMotion`).
    Read in the viewer as: noise V offset = -direction x speed x t."""
    if uvs is None or flame_t is None:
        return 1.0
    v = np.asarray(uvs, np.float64).reshape(-1, 2)[:, 1]
    t = np.asarray(flame_t, np.float64).reshape(-1)
    if len(v) != len(t) or np.ptp(v) < 1e-6 or np.ptp(t) < 1e-6:
        return 1.0
    return 1.0 if float(np.corrcoef(v, t)[0, 1]) >= 0 else -1.0


def home_noise_layers(model: Model, effect: dict) -> list[dict]:
    """HOME's two scrolling noise layers of a fire core (`unite_home_fx.json` `noise`: file, Unity UV scale, speed in UV/s)
    as GLB textures (`homefx:<file>`, grey, linear). The viewer moves them at HOME's speeds inside the UNITE flame's own
    silhouette (ledger `fx:homeMotion`). Returns [{"key", "scale", "speed"}]."""
    out = []
    for n in effect.get("noise") or []:
        key = f"homefx:{n['file']}"
        if key not in model.textures:
            path = os.path.join(HOME_FX_NOISE, n["file"])
            if not os.path.isfile(path):
                continue
            g = np.asarray(Image.open(path).convert("L"), np.uint8)
            model.textures[key] = TextureImage(key=key, rgba=np.dstack([g, g, g, np.full_like(g, 255)]), srgb=False)
        out.append({"key": key, "scale": n.get("scale", [1.0, 1.0]), "speed": n.get("speed", [0.0, 0.0])})
    return out


def colour_particles(model: Model, dex: int | None) -> int:
    """Give each emitter record (`extras.particles`) the same species' HOME effect colours (`unite_home_fx.json`, from
    `tools/gen_unite_fx_colors.py`; user decision 2026-11-01: flame colours are the same across media). The first HOME
    core material of the species is used; emitters of species without one stay uncoloured (the viewer skips them).
    Returns the number coloured."""
    recs = model.extras.get("particles") or []
    if not recs or not dex:
        return 0
    try:
        with open(HOME_FX, encoding="utf-8") as f:
            table = json.load(f)["species"]
    except OSError:
        return 0
    sp = table.get(str(int(dex)))
    if not sp or not sp["effects"]:
        return 0
    e = sp["effects"][0]
    noise = home_noise_layers(model, e) if e["kind"] == "fire" else []
    for r in recs:
        r.update({"kind": e["kind"], "source": f"home:{e['material']}", "baseColor": e["baseColor"],
                  "layerColor": e["layerColor"], "lerpProfile": e.get("lerpProfile")})
        if noise:
            r.update({"noise": noise, "noiseNorm": e.get("noiseNorm", 1.0), "noiseMean": e.get("noiseMean", 0.5), "loop": e.get("loop")})
    if noise:
        fx = model.extras.setdefault("fxTextures", [])
        fx += [n["key"] for n in noise if n["key"] not in fx]
    return len(recs)


def home_fire(dex: int | None, form: str | None = None) -> dict | None:
    """The HOME flame core (`unite_home_fx.json`) for a species, or for one of its forms (`6_51` = HOME `pm0006_51_00`,
    Mega Charizard X: UNITE's `0006_51` folder uses HOME's form number). None when HOME has no fire core for it."""
    if not dex:
        return None
    try:
        with open(HOME_FX, encoding="utf-8") as f:
            table = json.load(f)["species"]
    except OSError:
        return None
    sp = table.get(f"{int(dex)}_{form}" if form else str(int(dex)))
    e = next((e for e in (sp or {}).get("effects", []) if e.get("kind") == "fire"), None)
    if not e:
        return None
    return {"source": f"home:{e['material']}", "baseColor": e["baseColor"], "layerColor": e["layerColor"],
            "lerpProfile": e.get("lerpProfile"), "noise": e.get("noise"), "noiseNorm": e.get("noiseNorm"),
            "noiseMean": e.get("noiseMean"), "loop": e.get("loop")}


def flame_coordinate(positions: np.ndarray, root_point) -> np.ndarray | None:
    """Per-vertex coordinate along a flame, 0 at the root to 1 at the tip (glTF attribute `_FLAMET`). The axis is the
    positions' principal axis; the root is the end nearest `root_point` (the emitter joint: the prefab origin for donor
    meshes, the shallowest weighted joint for a model's own flame). A flame is a long thin card/shell, so the principal
    axis is its length. None for degenerate input."""
    pos = np.asarray(positions, np.float64)
    if len(pos) < 3:
        return None
    c = pos.mean(0)
    axis = np.linalg.svd(pos - c, full_matrices=False)[2][0]
    s = (pos - c) @ axis
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-6:
        return None
    s0 = float((np.asarray(root_point, np.float64) - c) @ axis)
    t = (s - lo) / (hi - lo)
    return (t if abs(lo - s0) <= abs(hi - s0) else 1.0 - t).astype(np.float32)


def _own_flame_root(model: Model, p) -> np.ndarray | None:
    """Rest position of the shallowest joint that carries weight on this primitive (a flame skinned to a tail chain
    hangs from its first joint); None when the primitive is unskinned."""
    sk = model.skeleton
    if sk is None or p.joints is None or p.weights is None:
        return None
    used = {int(j) for j, w in zip(p.joints.reshape(-1), p.weights.reshape(-1)) if w > 0.01}
    best = None
    for pal in used:
        if pal >= len(sk.palette):
            continue
        ji = sk.palette[pal]
        jt = sk.joints[ji]
        if jt.inverse_bind is None:
            continue
        depth, k = 0, jt.parent
        while k >= 0:
            depth, k = depth + 1, sk.joints[k].parent
        if best is None or depth < best[0]:
            best = (depth, np.linalg.inv(np.asarray(jt.inverse_bind, np.float64))[:3, 3])
    return None if best is None else best[1]


def fire_effects(model: Model, dex: int | None, form: str | None = None) -> list[str]:
    """Flame materials drawn like the game's fire (viewer `fireEffect.ts`): blended effect materials whose primitives
    carry vertex colours (Unity Fx shaders multiply the mask by them: the flame's orange root -> red tip, Mega X's
    blue -> white) on a species/form with a HOME fire core. They get `extras.fire` (HOME core colour + ramp) and keep
    their per-vertex colours (COLOR_0); every other primitive drops the dump's colours (lit meshes carry shader data in
    that layer). User feedback 2026-11-02: "orange/yellow hybrid with blurred edges". Returns the materials."""
    core = home_fire(dex, form)
    out: list[str] = []

    def grey_mask(mat) -> bool:
        # the look replaces the texture's colour: only for grey masks, whose colour IS the vertex colour. Moltres's
        # flames are painted (`t_lob_fx_0146moltres_tou`) over white vertex colours and turned white under it.
        t = model.textures.get(mat.base_color) if mat.base_color else None
        if t is None:
            return False
        rgb = t.rgba[..., :3].astype(np.int16)
        return int((rgb.max(-1) - rgb.min(-1)).max()) <= 8

    for mat in model.materials.values():
        # noise sheens of a flame species with no hue of their own (all-white vertex colours: Mega X's `wenli`) take
        # HOME's flame `_BaseColor`, like the white-coloured flames below (`rimFromHome`)
        if core is not None and mat.extras.get("noiseSheen") and min(mat.emissive_factor) > 0.95:
            mat.emissive_factor = tuple(float(v) for v in _srgb_to_linear(core["baseColor"][:3]))
            mat.extras["sheenFromHome"] = True
    for mesh in model.meshes:
        for p in mesh.primitives:
            mat = model.materials[p.material]
            ex = mat.extras
            fire = (core is not None and p.colors is not None and ex.get("effect") and mat.alpha_mode == "BLEND"
                    and not ex.get("missingTexture") and not ex.get("solidEffect") and ex.get("blend") != "Add"
                    and grey_mask(mat))
            if not fire:
                if not ex.get("unite", {}).get("colourFrom"):  # HOME-coloured texture-less parts keep their colours (home_parts)
                    p.colors = None
                continue
            if mat.name not in out:
                ex["fire"] = {k: v for k, v in core.items() if k not in ("noise", "noiseNorm", "noiseMean", "loop")}
                layers = home_noise_layers(model, core)
                if layers:
                    # {"map": {"texture": key}} becomes a glTF texture reference in the writer (`field_refs`)
                    ex["fire"]["noise"] = [{"map": {"texture": n["key"]}, "scale": n["scale"], "speed": n["speed"]} for n in layers]
                    ex["fire"]["noiseNorm"] = core.get("noiseNorm") or 1.0
                    ex["fire"]["noiseMean"] = core.get("noiseMean") if core.get("noiseMean") is not None else 0.5
                    ex["fire"]["loop"] = core.get("loop")
                out.append(mat.name)
            root = _own_flame_root(model, p)
            if root is not None:
                p.flame_t = flame_coordinate(p.positions, root)
            if float(p.colors[:, :3].min()) > 0.95:
                # all-white vertex colours carry no hue (Moltres lobby's head/tail, Charizard 16 lobby): the rim is
                # HOME's `_BaseColor` instead (the fallback the user named), through the emissive factor
                p.colors = None
                ex["fire"]["rimFromHome"] = True
                mat.emissive_factor = tuple(float(v) for v in _srgb_to_linear(core["baseColor"][:3]))
    # Along-flame colour (viewer: HOME's baseColor -> layerColor ramp on `_FLAMET`) needs every primitive of the material to
    # have the coordinate (the shader switches per material); then HOME's pair replaces the per-vertex colours.
    for name in out:
        prims = [p for mesh in model.meshes for p in mesh.primitives if model.materials[p.material].name == name]
        if prims and all(p.flame_t is not None for p in prims):
            model.materials[prims[0].material].extras["fire"]["alongFlame"] = True
            if model.materials[prims[0].material].extras["fire"].get("noise"):
                model.materials[prims[0].material].extras["fire"]["noiseDir"] = noise_direction(
                    np.concatenate([p.uvs[0] for p in prims]) if all(p.uvs for p in prims) else None,
                    np.concatenate([p.flame_t for p in prims]))
            for p in prims:
                p.colors = None
        else:
            for p in prims:
                p.flame_t = None
    return out


def split_missing_texture_parts(model: Model) -> int:
    """Move primitives whose material has no texture in the rip into their own mesh `<mesh>~notex`, hidden by default
    (listed in the viewer's Model Parts). Done after the skin is final so the copy keeps the mesh's joints/binds."""
    out = []
    n = 0
    for mesh in model.meshes:
        miss = [p for p in mesh.primitives if model.materials[p.material].extras.get("missingTexture")]
        if not miss:
            out.append(mesh)
            continue
        ids = {id(p) for p in miss}  # by identity: dataclass equality would compare the numpy arrays
        keep = [p for p in mesh.primitives if id(p) not in ids]
        if keep:
            mesh.primitives = keep
            out.append(mesh)
        out.append(Mesh(name=mesh.name + "~notex", primitives=miss, lod=mesh.lod, visible=False,
                        extras={**mesh.extras, "missingTexture": True}, inverse_binds=mesh.inverse_binds))
        n += len(miss)
    model.meshes = out
    return n


# FITTED (user decision 2026-11-01): lens opacity; the data marks which slots are lenses but their albedo alpha
# (0.92-0.98) does not say how see-through they are. Judged against the Serebii Blastoise skin 02/13 art.
LENS_OPACITY = 0.6


def _albedo_samples(model: Model, material: str, path: str) -> np.ndarray | None:
    """RGBA texels of `path` under every UV of the primitives that use `material` (nearest texel)."""
    uvs = [p.uvs[0] for m in model.meshes for p in m.primitives if p.material == material and p.uvs]
    if not uvs:
        return None
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    h, w = rgba.shape[:2]
    uv = np.concatenate(uvs).astype(np.float64) % 1.0
    return rgba[np.clip((uv[:, 1] * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)]


def flag_lenses(model: Model, folder: str, index: dict[str, str] | None) -> list[str]:
    """Tinted lenses (Blastoise skin 02/13 sunglasses, visors): an FBX-only lit material whose albedo alpha under its
    own UVs is a uniform plateau below opaque (< 5% of samples at 255, range <= 20, 200 <= mean < 253), dark (brightest
    channel mean < 128), while another material on the SAME albedo samples it opaque (>= 90% at 255): the atlas marks
    the lens region in alpha. Surveyed over 4,862 FBX-only materials: 11 materials in 11 models, no known false
    positive (without the dark test, Greninja's `0658bodyc` x5 would be hit, opaque basepbr in its JSON). Drawn BLEND at
    LENS_OPACITY. Ledger `material:lens`."""
    by_albedo: dict[str, list[str]] = {}
    for name, mat in model.materials.items():
        u = (mat.extras.get("unity") or {})
        if u.get("source") == "fbx" and not mat.extras.get("effect") and not mat.extras.get("glass") and mat.alpha_mode == "OPAQUE":
            alb = (u.get("textures") or {}).get("_MainTex")
            if alb:
                by_albedo.setdefault(alb, []).append(name)
    out = []
    for alb, names in by_albedo.items():
        if len(names) < 2:
            continue
        path = _png(folder, alb, index)
        if not path:
            continue
        samples = {n: _albedo_samples(model, n, path) for n in names}
        opaque = {n for n, s in samples.items() if s is not None and len(s) and (s[:, 3] == 255).mean() >= 0.9}
        for n, s in samples.items():
            if s is None or not len(s) or n in opaque or not (opaque - {n}):
                continue
            a = s[:, 3].astype(np.float64)
            if (a == 255).mean() < 0.05 and a.max() - a.min() <= 20 and 200 <= a.mean() < 253 and s[:, :3].mean(0).max() < 128:
                mat = model.materials[n]
                mat.alpha_mode = "BLEND"
                mat.base_color_factor = (*mat.base_color_factor[:3], LENS_OPACITY)
                mat.extras["blend"] = "Blend"
                mat.extras["lens"] = True
                out.append(n)
    return out


# FITTED: cut-out albedo rule on the UNITE rip (scene JSON, standalone lobby and battle FBX materials), 2026-11-08.
# A character albedo whose alpha is mostly solid and bimodal, with the material's own UVs landing on alpha 0, is an
# alpha-tested cut-out (Venusaur Asia's gold beads sit on a clean 0/255 strip); the JSONs do not say (`_AlphaClip`
# is set once in the whole rip). Cloth masks (mostly alpha 0) and atlas padding the UVs never reach stay opaque.
# Ledger `material:cutoutAlbedo`.
CUTOUT_SOLID_MIN = 0.5      # share of texels with alpha > 0.95
CUTOUT_MID_MAX = 0.05       # share of texels strictly between 0.05 and 0.95
CUTOUT_UV_HIT_MIN = 0.002   # share of the material's UV samples (vertices + triangle centroids) on alpha < 0.05


def cutout_stats(alpha: np.ndarray, uvs: list[np.ndarray], indices: list[np.ndarray]) -> tuple[float, float, float]:
    """(solid share, in-between share, share of UV samples on alpha < 0.05) for an albedo alpha channel (uint8)."""
    a = alpha.astype(np.float64) / 255.0
    solid, mid = float((a > 0.95).mean()), float(((a > 0.05) & (a < 0.95)).mean())
    h, w = a.shape
    hits, total = 0, 0
    for uv, idx in zip(uvs, indices):
        uv = np.asarray(uv, np.float64)
        pts = [uv]
        tri = np.asarray(idx, np.int64)[: len(idx) // 3 * 3].reshape(-1, 3)
        if len(tri):
            pts.append(uv[tri].mean(1))
        pts = np.concatenate(pts) % 1.0
        v = a[np.clip((pts[:, 1] * h).astype(int), 0, h - 1), np.clip((pts[:, 0] * w).astype(int), 0, w - 1)]
        hits += int((v < 0.05).sum())
        total += len(v)
    return solid, mid, hits / max(total, 1)


def cutout_passes(stats: tuple[float, float, float]) -> bool:
    solid, mid, hit = stats
    return solid >= CUTOUT_SOLID_MIN and mid < CUTOUT_MID_MAX and hit >= CUTOUT_UV_HIT_MIN


def flag_cutouts(model: Model, folder: str, index: dict[str, str] | None) -> list[str]:
    """Alpha-tested character materials: an OPAQUE lit material (not effect/glass/lens) whose albedo passes
    `is_cutout` under its own primitives' UVs keeps the albedo alpha (a second texture key, the plain one stays
    alpha-forced for cloth-mask materials sharing the file) and becomes MASK at the JSON `_Cutoff` (else 0.5)."""
    out = []
    for name, mat in model.materials.items():
        if mat.alpha_mode != "OPAQUE" or mat.extras.get("effect") or mat.extras.get("glass") or mat.extras.get("lens"):
            continue
        u = mat.extras.get("unity") or {}
        if u.get("source", "json") != "fbx":
            continue  # JSON materials: the shader data decides (`alpha.decide`), ledger `material:cutoutAlbedo`
        if (mat.extras.get("unite") or {}).get("family") in ("fx", "glass", "bubble"):
            continue
        alb = (u.get("textures") or {}).get(color_slot(u.get("textures") or {}) or "_MainTex")
        if not alb or not mat.base_color:
            continue
        path = _png(folder, alb, index)
        prims = [p for m in model.meshes for p in m.primitives if p.material == name and p.uvs]
        if not path or not prims:
            continue
        alpha = np.asarray(Image.open(path).convert("RGBA"))[..., 3]
        if not cutout_passes(cutout_stats(alpha, [p.uvs[0] for p in prims], [p.indices for p in prims])):
            continue
        key = mat.base_color + "~cut"
        if key not in model.textures:
            rgba = np.array(Image.open(path).convert("RGBA"), np.uint8)
            model.textures[key] = TextureImage(key=key, rgba=rgba, srgb=True)
        mat.base_color = key
        mat.alpha_mode = "MASK"
        mat.alpha_cutoff = float((u.get("floats") or {}).get("_Cutoff", 0.5)) if isinstance(u.get("floats"), dict) else 0.5
        mat.extras["cutout"] = True
        mat.extras.setdefault("unite", {})["alpha"] = {"mode": "MASK", "cutoff": mat.alpha_cutoff, "source": "albedo statistics (FBX-only, FITTED)"}
        out.append(name)
    return out


def prune_unused_joints(model: Model, keep_names: set[tuple[str, str]] = frozenset()) -> int:
    """Drop joints that move no vertex of the kept meshes and have no such descendant (the bones of dropped effect
    meshes: Charmander's tail flame prefab brings `0006fire04(Clone)` -> `Bone_Tail06.001` at scale 100, which
    framed the model as a speck). Remaps the skin palette and drops their tracks. Returns the number removed."""
    sk = model.skeleton
    if sk is None:
        return 0
    weighted: set[int] = set()
    for mesh in model.meshes:
        for prim in mesh.primitives:
            if prim.joints is not None and prim.weights is not None:
                weighted |= {sk.palette[k] for k in np.unique(prim.joints[prim.weights > 0]).tolist()}
    # (joint, parent) names that carry an effect emitter: Charmeleon/Charizard's unweighted `Bone_Tail06`, which the
    # idle clips animate and the tail flame prefab hangs from
    for i, j in enumerate(sk.joints):
        par = sk.joints[j.parent].name if j.parent >= 0 else ""
        if (j.name, par) in keep_names:
            weighted.add(i)
    keep = set()
    for j in weighted:
        while j >= 0 and j not in keep:
            keep.add(j)
            j = sk.joints[j].parent
    if not keep or len(keep) == len(sk.joints):
        return 0
    old_ids = [i for i in range(len(sk.joints)) if i in keep]
    new_of = {o: n for n, o in enumerate(old_ids)}
    joints = []
    for o in old_ids:
        j = sk.joints[o]
        j.parent = new_of.get(j.parent, -1)
        joints.append(j)
    pal_pos = {}  # old palette position -> new palette position
    palette = []
    for pos, ji in enumerate(sk.palette):
        if ji in keep:
            pal_pos[pos] = len(palette)
            palette.append(new_of[ji])
    for mesh in model.meshes:
        for prim in mesh.primitives:
            if prim.joints is not None:
                lut = np.array([pal_pos.get(k, 0) for k in range(max(len(sk.palette), int(prim.joints.max()) + 1))], np.uint16)
                prim.joints = lut[prim.joints]
        if mesh.inverse_binds:
            mesh.inverse_binds = {pal_pos[k]: v for k, v in mesh.inverse_binds.items() if k in pal_pos}
    removed = {sk.joints[i].name for i in range(len(sk.joints)) if i not in keep}
    sk.joints, sk.palette = joints, palette
    for c in model.clips:
        c.tracks = [t for t in c.tracks if t.joint not in removed]
    return len(removed)


def import_model(fbx: str, model_id: str, display_name: str, kind: str = "pokemon", meshes: str | None = None,
                 character_only: bool = True, dump_dir: str | None = None, texture_index: dict[str, str] | None = None,
                 material_dirs: tuple[str, ...] | list[str] = (), scene_fbx: str | None = None,
                 base_clips: str | None = None, dex: int | None = None) -> Model:
    """`meshes` = regex on mesh names (scene FBXs); `character_only` drops unskinned backdrop primitives
    (`material_role`); effect parts are kept as unlit blended materials. Material data = `Materials/<name>.json`, else
    the FBX's texture links; textures are looked up beside the FBX, then in `texture_index` (lower-case file name ->
    path). `material_dirs` = other folders whose `Materials/*.json` describe the same materials (a lobby scene holding
    the same holowear); `scene_fbx` = that scene's FBX, whose mesh slots resolve the standalone file's texture-less
    placeholders (`mat_off_Lit`); `base_clips` = the form's skin-00 FBX whose clips fill the ones this skin does not
    override (`inherit_base_clips`)."""
    rx = re.compile(meshes) if meshes else None
    lost = [g["name"] for g in fbxr.geometries(fbx) if not g["materials_ok"] and (rx is None or rx.search(g["name"]))]
    folder = os.path.dirname(os.path.abspath(fbx))
    meta, arrays = dump_fbx(fbx, dump_dir, meshes, scale=FBX_SCALE, prefix="unitefbx_")
    kept = {_blender_name(n) for n in meta.get("kept_meshes") or [m["name"] for m in meta["meshes"]]}
    lost = [n for n in lost if n in kept]
    if lost:
        raise LostMaterialLayer(f"{model_id}: per-triangle materials missing in the FBX for {', '.join(lost)}")

    model = model_from_dump(meta, arrays, folder, model_id, display_name, kind=kind, game="unite", colours=True)
    try:
        scene = fbx_anim.parse(fbx)
    except Exception:  # noqa: BLE001
        scene = None
    infos = {}
    for d in [*material_dirs, folder]:
        infos.update(material_jsons(d))
    links = {k.lower(): v for k, v in fbxr.material_textures(fbx).items()}
    # A lobby holowear's standalone FBX gives some slots a texture-less placeholder (`mat_off_Lit`: bj opera Pikachu's
    # flags); the showcase scene's copy of the same mesh names the real material.
    own_slots = fbxr.mesh_materials(fbx) if scene_fbx else {}
    scene_slots = fbxr.mesh_materials(scene_fbx) if scene_fbx else {}
    if scene_fbx:
        for k, v in fbxr.material_textures(scene_fbx).items():
            links.setdefault(k.lower(), v)
    textures: dict[str, TextureImage] = {}
    materials: dict[str, Material] = {}
    dropped: list[str] = []
    effects: list[str] = []
    replaced: list[str] = []
    missing: list[str] = []
    borrowed: list[str] = []
    notex: list[str] = []
    borrowed_fx: list[str] = []
    donor_cache: dict = {}
    borrowed_colour: dict[str, np.ndarray] = {}
    ramp_pending: dict[str, dict] = {}

    blocks = {k.lower(): v for k, v in fbxr.material_blocks(fbx).items()}

    def info_of(base: str) -> dict:
        return infos.get(base.lower()) or info_from_fbx(links.get(base.lower(), {}), folder, texture_index,
                                                        blocks.get(base.lower()), base)

    for mesh in model.meshes:
        prims = []
        for p in mesh.primitives:
            base = _blender_name(p.material)
            # model_from_dump renames a material it found no texture for to `<mesh>_<material>`
            pre = mesh.name.lower() + "_"
            if base.lower().startswith(pre) and (base.lower()[len(pre):] in links or base.lower()[len(pre):] in infos):
                base = base[len(pre):]
            info = info_of(base)
            if not info["textures"]:
                mname = _blender_name(mesh.name)
                own, alt = own_slots.get(mname, []), scene_slots.get(mname, [])
                k = own.index(base) if base in own else -1
                sub = alt[k] if 0 <= k < len(alt) and len(alt) == len(own) else (alt[0] if len(set(alt)) == 1 else None)
                if sub and sub != base and info_of(sub)["textures"]:
                    replaced.append(f"{mesh.name}/{base} -> {sub}")
                    base, info = sub, info_of(sub)
            role = material_role(base, info)
            if character_only and role == "backdrop" and p.joints is None:
                dropped.append(f"{mesh.name}/{base}")
                continue
            if base not in materials:
                glass = (GLASS_SHADER.search(info["shader"]) or not info["textures"] or matcap_only(info)
                         or clear_albedo(info, p, folder, texture_index))
                matcap = matcap_of(info, folder, texture_index, base) if glass else None
                if matcap:
                    materials[base] = build_glass_material(base, info, matcap, textures, folder, texture_index)
                elif role == "effect":
                    m = build_effect_material(base, info, folder, textures, texture_index, missing)
                    if m is None:
                        sib = sibling_colour(mesh, p, info_of, folder, texture_index)
                        if sib is not None:
                            m = build_effect_material(base, {**info, "textures": {**info["textures"], **sib}},
                                                      folder, textures, texture_index, missing)
                            borrowed.append(f"{mesh.name}/{base} <- {next(iter(sib.values()))['name']}")
                    if m is None and not info["textures"].get("_MainTex"):
                        bor = borrow_effect_info(base, folder, texture_index, donor_cache)
                        if bor is not None:
                            m = build_effect_material(base, {**bor[0], "borrowedFrom": bor[2]}, bor[1], textures,
                                                      texture_index, missing)
                            # a sparse borrowed mask (Skeledirge's `t_fx_lob_0911_lv3_02`: 30% of the lobby bird drawn)
                            # is flagged later by flag_sparse_effects; home_parts replaces it only when HOME matches
                            if m is not None:
                                m.extras["borrowedFrom"] = bor[2]
                                if bor[3] is not None:
                                    borrowed_colour[base] = bor[3]
                                borrowed_fx.append(f"{mesh.name}/{base} <- {bor[2]} ({os.path.basename(bor[1])})")
                    if m is None and not info["textures"].get("_MainTex"):
                        dsib = donor_sibling_colour(mesh, p, base, folder, texture_index, donor_cache)
                        if dsib is not None:
                            m = build_effect_material(base, {**info, "textures": {**info["textures"], **dsib[0]}},
                                                      dsib[1], textures, texture_index, missing)
                            if m is not None:
                                m.extras["borrowedFrom"] = dsib[2]
                                borrowed_fx.append(f"{mesh.name}/{base} <- {dsib[2]} (same mesh, {os.path.basename(dsib[1])})")
                    if m is None:
                        # no colour anywhere in the rip (Fuecoco's tuft, Skeledirge's lobby crown...): kept as a hidden,
                        # untextured part the viewer can switch on (user decision 2026-11-01), not dropped
                        m = Material(name=base, base_color_factor=(0.6, 0.6, 0.6, 1.0), roughness_factor=1.0)
                        m.extras = {"unity": unity_extras(info), "effect": True, "solidEffect": True, "missingTexture": True}
                        notex.append(f"{mesh.name}/{base}")
                    materials[base] = m
                elif not info["textures"]:
                    dropped.append(f"{mesh.name}/{base} (placeholder material without textures)")
                    continue
                elif ps.player_kind(info["shader"]):
                    kind = ps.player_kind(info["shader"])
                    materials[base] = build_player_material(base, kind, info, folder, textures, texture_index, missing)
                    if kind == "ramp":
                        ramp_pending[base] = info
                else:
                    materials[base] = build_material(base, info, folder, textures, texture_index, missing)
            if materials[base].extras.get("effect"):
                effects.append(f"{mesh.name}/{base}")
            p.material = base
            prims.append(p)
        mesh.primitives = prims
        if scene is not None and any(ATTACK_PREFAB.search(x) for x in ancestors(scene, _blender_name(mesh.name))):
            mesh.visible = False
    model.meshes = [m for m in model.meshes if m.primitives]
    model.materials, model.textures = materials, textures
    for base, mat in materials.items():  # WP-A: extras.unite.family on EVERY material (JSON shader, else inferred)
        fi = info_of(base)
        uf = mat.extras.setdefault("unite", {})
        uf["family"] = fi.get("family") or unite_families.family_of(fi.get("shader", ""))
        if fi.get("familyInferred"):
            uf["familyInferred"] = True
        sh = unite_shading.shading_params(fi, uf["family"])  # WP-F: SH / specular / fur-rim / player-shadow constants
        if sh:
            uf["shading"] = sh
    for rname, rinfo in ramp_pending.items():
        why_not = bake_ramp_emission(model, rname, rinfo, folder, texture_index, base_clips, textures)
        if why_not:
            model.extras.setdefault("warnings", []).append(why_not)
    shells = set(overlay_shells(model))
    if shells:
        for mesh in model.meshes:
            mesh.primitives = [p for p in mesh.primitives if (mesh.name, p.material) not in shells]
        model.meshes = [m for m in model.meshes if m.primitives]
        used = {p.material for m in model.meshes for p in m.primitives}
        model.materials = {k: v for k, v in model.materials.items() if k in used}
        dropped += [f"{m}/{mt} (overlay shell)" for m, mt in sorted(shells)]
    _flag_enclosing_effects(model)
    tinted = tint_effects_by_vertex_colour(model, fbxr.mesh_vertex_colours(fbx), borrowed_colour)
    if tinted:
        model.extras["vertexColourEffects"] = tinted
    decals = flag_overlay_decals(model, folder, texture_index)
    if decals:
        model.extras["overlayDecals"] = decals
    alt = hide_alternate_states(model)
    if alt:
        model.extras["alternateStates"] = alt
    lenses = flag_lenses(model, folder, texture_index)
    if lenses:
        model.extras["lensMaterials"] = lenses
    cutouts = flag_cutouts(model, folder, texture_index)
    if cutouts:
        model.extras["cutoutMaterials"] = cutouts
    zero_normals = 0
    for mesh in model.meshes:
        for p in mesh.primitives:
            if p.normals is not None:
                z = np.linalg.norm(p.normals, axis=1) < 1e-6
                if z.any():  # degenerate source vertices; glTF requires unit normals
                    p.normals[z] = (0.0, 1.0, 0.0)
                    zero_normals += int(z.sum())
    warnings = [w for w in model.extras.get("warnings", []) if not w.startswith("mesh ")]  # model_from_dump's lookup
    if dropped:
        warnings.append("dropped primitives: " + "; ".join(dropped))
    if replaced:
        warnings.append("placeholder slots resolved from the showcase scene: " + "; ".join(replaced))
    if borrowed:
        warnings.append("effect colour taken from a textured slot of the same mesh: " + "; ".join(borrowed))
    if borrowed_fx:
        warnings.append("effect textures borrowed from the same species' other FBX (same material name): " + "; ".join(borrowed_fx))
        model.extras["borrowedEffects"] = borrowed_fx
    if notex:
        warnings.append("effect parts with no texture in the rip, imported hidden: " + "; ".join(notex))
    particles = particle_emitters(scene) if scene is not None else []
    if particles:
        model.extras["particlesNotInRip"] = particles
        warnings.append("particle effects not in the rip (empty emitter nodes): " + ", ".join(particles))
    if effects:
        model.extras["effectParts"] = sorted(set(effects))
    if zero_normals:
        warnings.append(f"{zero_normals} zero-length source normals set to +Y")
    if meta.get("zero_scale_meshes"):
        warnings.append("meshes scaled to 0 in the file (hidden), skipped: " + ", ".join(meta["zero_scale_meshes"]))
    if missing:
        warnings.append("textures not in the rip: " + ", ".join(sorted(set(missing))))
        model.extras["missingTextures"] = sorted(set(missing))
    if warnings:
        model.extras["warnings"] = warnings
    else:
        model.extras.pop("warnings", None)
    try:
        if scene is None:
            raise RuntimeError("FBX parse failed")
        add_emitter_joints(model, scene)
        why = apply_fbx_skeleton(model, scene, {m["name"]: m["matrix"] for m in meta["meshes"] if "matrix" in m})
        rigid = bind_rigid_skins(model, scene)
        if rigid:
            model.extras.setdefault("warnings", []).append("weightless skins bound to their first cluster bone: " + ", ".join(rigid))
        if why is None:
            own = apply_mesh_binds(model, scene)
            if own:
                model.extras["meshBinds"] = own
    except Exception as e:  # noqa: BLE001
        why = f"FBX skeleton reader failed: {e}"
    if why:
        model.extras.setdefault("warnings", []).append("kept Blender's skeleton/clips: " + why)
    else:
        model.extras["skeletonSource"] = "fbx"
        if base_clips:  # skin override controller: the form's base set fills the clips the skin does not override
            for c, n in zip(model.clips, clip_names([c.name for c in model.clips])):
                c.name = n
            rec = inherit_base_clips(model, base_clips)
            if rec:
                for c in model.clips[len(model.clips) - len(rec["clips"]):]:
                    c.name = clean_clip_name(c.name)
                model.extras["inheritedClips"] = rec
    flag_sparse_effects(model)
    from pipeline.importers.unite.home_parts import colour_from_home  # texture-less parts: colour from HOME (ledger fx:homeColouredParts)
    colour_from_home(model, dex)
    split_missing_texture_parts(model)
    atlas_variants(model, "eye", "eyes", "Eyes")
    atlas_variants(model, "mouth", "mouth", "Mouth")
    rest_jaws(model)
    own = own_smoke_prefabs(scene, fbxr.mesh_materials(fbx), links, dex) if scene is not None else []
    if own:
        kids: dict[int, list[int]] = {}
        for n in scene.nodes.values():
            kids.setdefault(n.parent, []).append(n.id)
        under, stack = set(), list(own)
        while stack:
            k = stack.pop()
            under.add(scene.nodes[k].name)
            stack += kids.get(k, [])
        gone = [m.name for m in model.meshes if _blender_name(m.name).split("~")[0] in under]
        model.meshes = [m for m in model.meshes if m.name not in gone]
        model.extras.setdefault("warnings", []).append(
            "own effect meshes without texture links drawn from the prefab's data (extras.particles): " + ", ".join(gone))
    hold = set()
    if scene is not None:
        for nid in _emitter_nodes(scene) + list(own):
            b = scene.nodes.get(scene.nodes[nid].parent)
            if b is not None:
                g = scene.nodes.get(b.parent)
                hold.add((b.name, g.name if g is not None else ""))
    pruned = prune_unused_joints(model, hold)
    if pruned:
        model.extras["prunedJoints"] = pruned
    if scene is not None and (particles or own):
        recs = particle_records(scene, model) if particles else []
        for r in particle_records(scene, model, own):
            r["own"] = True
            recs.append(r)
        if recs:
            model.extras["particles"] = recs
    for c, n in zip(model.clips, clip_names([c.name for c in model.clips])):
        c.name = n
    model.clips.sort(key=lambda c: c.name)
    names = [c.name for c in model.clips]
    model.default_clip = "idle" if "idle" in names else next((n for n in names if "idle" in n), names[0] if names else None)
    return model
