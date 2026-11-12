"""Pokémon HOME importer: Unity 2022.3 AssetBundles -> common Model.

The prefab bundle holds the Transform hierarchy, the SkinnedMeshRenderers and Materials, and a controller MonoBehaviour whose
`layers[].states[]` name the clips. Meshes, Texture2Ds and AnimationClips live in dependency bundles that the prefab reaches by
CAB name (`SerializedFile.externals`; `m_FileID` k = externals[k-1]). Hierarchy, skinning, handedness and generic-clip decoding are
Unity's and are shared with the BDSP importer (`..bdsp.importer`); only the material (Game Freak's Pokémon shader: `_Col0Tex` is the
lit albedo, `_L1Col0Tex` the iris layer, as in SwSh/LGPE) and the clip lookup differ.
"""

from __future__ import annotations

import os
import re

import numpy as np

from ...common.model import Clip, Material, Mesh, Model, Primitive, TextureImage, Variant
from ..bdsp.importer import (
    ATTR_POS,
    Hierarchy,
    _clip_curves,
    _build_skeleton,
    _clip_tracks,
    _mat,
    _mesh_primitives,
)
from ..za.importer import categorize


def dependency_paths(path: str, data_dir: str) -> list[str]:
    """Dependency bundle files named by the prefab's AssetBundle object (`mtshaders` is not in the rip)."""
    import UnityPy

    env = UnityPy.load(path)
    primary = next(iter(env.files.values()))
    sf = next(f for f in primary.files.values() if hasattr(f, "objects"))
    ab = next(o for o in sf.objects.values() if o.type.name == "AssetBundle").read()
    out = []
    for d in ab.m_Dependencies:
        p = os.path.join(data_dir, d + ".unity3d")
        if os.path.isfile(p):
            out.append(p)
    return out


def _env(path: str, deps: list[str]):
    import UnityPy

    env = UnityPy.load(path, *deps)
    primary = next(iter(env.files.values()))
    sf = next(f for f in primary.files.values() if hasattr(f, "objects"))
    return env, sf


def _expand(rgba: np.ndarray, sx: int, sy: int, wrap_u: int, wrap_v: int) -> np.ndarray:
    """Bake the material's `_Col0Tex` tiling into the texture so the mesh UVs stay plain 0..1 (the pipeline's convention).

    Unity wrap modes: 0 repeat, 1 clamp, 2 mirror. A (2, 1) scale with a mirror U sampler is the usual 'texture holds half of a symmetric
    layout'; every other scale is expanded the same way (repeat tiles, mirror alternates flipped copies, clamp keeps one copy).
    """

    def axis(a: np.ndarray, n: int, wrap: int, ax: int) -> np.ndarray:
        if n <= 1:
            return a
        if wrap == 2:
            parts = [np.flip(a, axis=ax) if i % 2 else a for i in range(n)]
        else:
            parts = [a] * n
        return np.concatenate(parts, axis=ax)

    return axis(axis(rgba, sx, wrap_u, 1), sy, wrap_v, 0)


def _add_iris(atlas: np.ndarray, iris: np.ndarray, su: float, sv: float, flip_u: bool) -> np.ndarray:
    """Composite the `_L1Col0Tex` layer under the eye atlas' transparent windows.

    The eye mesh's UV extent is one cell of the expression atlas, and the layer is sampled at `uv * scale` (4, 4 on Pokémon eyes; negative =
    mirrored, Gastly's left eye), so one copy of the iris image fills exactly one cell: tile it |su| x |sv| times over the baked atlas.
    The iris is static in game (the atlas offset moves, the layer does not), so it is painted into every cell, where the lid shapes
    of that cell cut their windows. `flip_u`: the layer is mirrored (negative scale) XOR this eye samples a mirrored period of the atlas.
    """
    from PIL import Image

    h, w = atlas.shape[:2]
    nx, ny = max(1, int(round(abs(su)))), max(1, int(round(abs(sv))))
    cell = np.asarray(Image.fromarray(iris).resize((max(1, w // nx), max(1, h // ny)), Image.BILINEAR), np.uint8)
    if flip_u:
        cell = cell[:, ::-1]
    if sv < 0:
        cell = cell[::-1]
    layer = np.tile(cell, (ny, nx, 1))[:h, :w]
    a = atlas[..., 3:4].astype(np.float32) / 255.0
    out = atlas.astype(np.float32)
    out[..., :3] = out[..., :3] * a + layer[..., :3].astype(np.float32) * (1.0 - a)
    out[..., 3] = 255
    return out.astype(np.uint8)


def _texture(ptr, cache: dict, textures: dict[str, TextureImage], suffix: str = "") -> str | None:
    """Texture2D -> RGBA, top row first (UnityPy already flips; verified on Pikachu: ear tips, cheeks and tail line up with the UVs)."""
    ck = (ptr.file_id, ptr.path_id)
    if ck not in cache:
        t = ptr.deref().read()
        rgba = np.array(t.image.convert("RGBA"), np.uint8)
        key = t.m_Name + suffix  # the shiny bundle reuses the base texture names
        textures[key] = TextureImage(key=key, rgba=rgba, srgb=True)
        cache[ck] = key
    return cache[ck]


def _gray(ptr, cache: dict, textures: dict[str, TextureImage], suffix: str, size: tuple[int, int]) -> np.ndarray:
    """R channel of a mask texture as float 0..1, resized to `size` (w, h)."""
    from PIL import Image

    key = _texture(ptr, cache, textures, suffix)
    r = textures[key].rgba[..., 0]
    if (r.shape[1], r.shape[0]) != size:
        r = np.asarray(Image.fromarray(r).resize(size, Image.BILINEAR))
    return r.astype(np.float32) / 255.0


def _effect_texture(m, cache, textures, suffix: str, core_colors: dict | None = None) -> tuple[str, bool] | None:
    """Flame / smoke cards (`FireMask`/`FireCore`, `SmokeMask`/`SmokeCore`): procedural scrolling shaders with no albedo, only grey masks.

    Core: colour = `_BaseColor` lerped toward `_LayerColor` by `_LerpTex` (a vertical gradient), alpha = `_Blend0Tex`. Mask: the outer
    silhouette, alpha = `_Mask0Tex`; it has no colour of its own, so it takes the sibling core's `_BaseColor`. The shader scrolls the masks;
    the bake is their static frame (the viewer has no UV panner for these).
    """
    props = m.m_SavedProperties
    tex = {n: v.m_Texture for n, v in props.m_TexEnvs if v.m_Texture.path_id}
    cols = {n: (float(c.r), float(c.g), float(c.b)) for n, c in props.m_Colors}
    key = f"{m.m_Name}_fx{suffix}"
    if "_Blend0Tex" in tex:
        fl = {n: float(v) for n, v in props.m_Floats}
        size = (128, 128)

        def layer(name: str, i: int) -> np.ndarray:
            """Noise layer tiled by its `_BlendNUVScale`, on a common grid."""
            g = _gray(tex[name], cache, textures, suffix, size)
            return np.tile(g, (max(1, int(round(fl.get(f"_Blend{i}UVScaleV", 1)))), max(1, int(round(fl.get(f"_Blend{i}UVScaleU", 1))))))

        b0 = layer("_Blend0Tex", 0)
        b1 = layer("_Blend1Tex", 1) if "_Blend1Tex" in tex else np.ones_like(b0)
        h = max(b0.shape[0], b1.shape[0]); w = max(b0.shape[1], b1.shape[1])
        from PIL import Image as _I

        def fit(a: np.ndarray) -> np.ndarray:
            return a if a.shape == (h, w) else np.asarray(_I.fromarray((a * 255).astype(np.uint8)).resize((w, h), _I.BILINEAR), np.float32) / 255.0

        # Two noise layers at different tilings: the shader combines them (product, renormalised to its peak) rather than showing one.
        cache[("noise", m.m_Name)] = fit(b0)  # the shell sculpt (`_sculpt_shells`) reuses the first layer as its cloud map
        alpha = fit(b0) * fit(b1)
        alpha = np.clip(alpha / max(float(np.percentile(alpha, 99)), 1e-3), 0, 1)
        base = np.array(cols.get("_BaseColor", (1, 1, 1)), np.float32)
        layer_c = np.array(cols.get("_LayerColor", cols.get("_BaseColor", (1, 1, 1))), np.float32)
        lt = textures[_texture(tex["_LerpTex"], cache, textures, suffix)].rgba if "_LerpTex" in tex else None
        if lt is not None:
            # the gradient sits in R (Charizard) or, when R is flat, in A (Gastly)
            ch = lt[..., 0] if np.ptp(lt[..., 0]) > 8 else lt[..., 3]
            lerp = np.asarray(_I.fromarray(ch).resize((w, h), _I.BILINEAR), np.float32)[..., None] / 255.0
        else:
            lerp = np.zeros((h, w, 1), np.float32)
        rgb = base * (1 - lerp) + layer_c * lerp
    elif "_Mask0Tex" in tex:
        t0 = textures[_texture(tex["_Mask0Tex"], cache, textures, suffix)].rgba
        alpha = t0[..., 0].astype(np.float32) / 255.0
        c = (core_colors or {}).get("_BaseColor") or cols.get("_Color", (1.0, 1.0, 1.0))
        rgb = np.broadcast_to(np.array(c, np.float32), (*alpha.shape, 3))
    else:
        return None
    rgba = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], axis=-1)
    textures[key] = TextureImage(key=key, rgba=(rgba * 255 + 0.5).astype(np.uint8), srgb=True)
    return key, True


def _material(m, cache, textures, suffix: str = "", core_colors: dict | None = None) -> Material:
    props = m.m_SavedProperties
    tex = {name: v for name, v in props.m_TexEnvs}
    fl = {k: float(v) for k, v in props.m_Floats}
    cols = {n: [float(c.r), float(c.g), float(c.b), float(c.a)] for n, c in props.m_Colors}
    mat = Material(name=m.m_Name, roughness_factor=0.8)
    keys = {}
    for unity_name, slot in (("_Col0Tex", "col"), ("_L1Col0Tex", "layer1"), ("_EmissionMaskTex", "emi")):
        t = tex.get(unity_name)
        if t is not None and t.m_Texture.path_id:
            keys[slot] = _texture(t.m_Texture, cache, textures, suffix)
    mat.base_color = keys.get("col")
    if not mat.base_color:
        fx = _effect_texture(m, cache, textures, suffix, core_colors)
        if fx:
            mat.base_color = fx[0]
            mat.alpha_mode = "BLEND"
            mat.double_sided = fl.get("_CullMode", 2.0) == 0.0  # Unity's default is Cull Back
            mat.extras["home"] = {"floats": fl, "colors": cols, "effect": True, "name": m.m_Name}
            return mat
    col = tex.get("_Col0Tex")
    sx = sy = 1
    if mat.base_color and col is not None:
        sx, sy = int(round(col.m_Scale.x)), int(round(col.m_Scale.y))
        if sx > 1 or sy > 1:
            ts = col.m_Texture.deref().read().m_TextureSettings
            key = f"{mat.base_color}_x{sx}y{sy}"
            if key not in textures:
                textures[key] = TextureImage(key=key, rgba=_expand(textures[mat.base_color].rgba, sx, sy, ts.m_WrapU, ts.m_WrapV), srgb=True)
            mat.base_color = key
    mat.extras["home"] = {"floats": fl, "colors": cols, "textures": {k: v for k, v in keys.items() if k != "col"}}
    bu, bv = fl.get("_ColorBaseU", 0.0), fl.get("_ColorBaseV", 0.0)
    if mat.base_color and (bu or bv):
        # Expression atlas: `_ColorBaseU/V` selects the tile (the shader samples atlas(uv + base)); baked as in the BDSP Pokémon shader.
        key = f"{mat.base_color}_base_{bu:g}_{bv:g}"
        if key not in textures:
            t = textures[mat.base_color].rgba
            hh, ww = t.shape[:2]
            textures[key] = TextureImage(key=key, rgba=np.roll(np.roll(t, -int(round(bu * ww / sx)), axis=1), int(round(bv * hh / sy)), axis=0), srgb=True)
        mat.base_color = key
    l1 = tex.get("_L1Col0Tex")
    if mat.base_color and keys.get("layer1") and l1 is not None and fl.get("_Layer1Enable", 0.0) > 0.5:
        # Composited after the meshes are built (`_apply_irises`): the iris orientation depends on whether this eye's UVs fall in a mirrored
        # period of the atlas sampler, which only the mesh knows.
        mat.extras["home"]["iris"] = {"layer": keys["layer1"], "su": float(l1.m_Scale.x), "sv": float(l1.m_Scale.y), "tile_u": float(sx), "base_u": float(bu)}
    if fl.get("_CullMode", 2.0) == 0.0:
        mat.double_sided = True
    return mat


def _find_core_colors(objs, mats: "_Materials") -> None:
    for o in objs:
        if o.type.name == "Material":
            m = o.read()
            if any(n == "_Blend0Tex" for n, _ in m.m_SavedProperties.m_TexEnvs):
                c = dict((n, (float(v.r), float(v.g), float(v.b))) for n, v in m.m_SavedProperties.m_Colors)
                if "_BaseColor" in c:
                    mats.core_colors = {"_BaseColor": c["_BaseColor"]}
                    return


# FITTED: world size of a smoke billboard card = _BillboardScale x vertex colour B x this, on Gastly's gas (the shader that sizes the
# camera-facing sprites is not in the rip), 2026-10-20
BILLBOARD_SIZE = 2.0


def _expand_billboards(smr, prims: list[Primitive], materials_of) -> list[Primitive]:
    """Smoke wisps are 1 cm quads (4 coincident-ish vertices) that the game's billboard shader (`_BillboardScale`) blows up into
    camera-facing sprites: vertex colour RG = the quad corner (0/1), B = its size. A static glTF has no billboards, so each card becomes
    two crossed vertical quads of that size around its centre (visible from every side); the mask's UVs ride along."""
    from UnityPy.helpers.MeshHelper import MeshHandler

    mh = MeshHandler(smr.m_Mesh.deref().read())
    mh.process()
    cols = getattr(mh, "m_Colors", None)
    if cols is None:
        return prims
    cols = np.asarray(cols, np.float32)
    cols = cols / 255.0 if cols.max() > 1.5 else cols
    out = []
    subs = [np.unique(np.asarray(t, np.int64).reshape(-1, 3)) for t in mh.get_triangles() if len(t)]
    for prim, used in zip(prims, subs):
        mat = materials_of(prim.material)
        extent = float(np.ptp(prim.positions.reshape(-1, 4, 3), axis=1).max()) if len(prim.positions) % 4 == 0 and len(prim.positions) else 1.0
        if not mat.extras.get("home", {}).get("effect") or len(prim.positions) % 4 or extent > 0.05:
            out.append(prim)
            continue
        c = cols[used]
        gain = float(mat.extras["home"]["floats"].get("_BillboardScale", 1.0)) * BILLBOARD_SIZE
        pos, uv, nrm, jt, wt, idx = [], [], [], [], [], []
        centre0 = np.zeros(3, np.float32)
        for q in range(0, len(prim.positions), 4):
            sl = slice(q, q + 4)
            ctr = prim.positions[sl].mean(0)
            size = float(c[q, 2]) * gain
            out_dir = ctr - centre0
            out_dir = out_dir / max(float(np.linalg.norm(out_dir)), 1e-6)
            for ax in (np.array([1, 0, 0], np.float32), np.array([0, 0, 1], np.float32)):
                base = len(pos) // 1
                for k in range(4):
                    cx, cy = float(c[q + k, 0]), float(c[q + k, 1])
                    pos.append(ctr + ax * (cx - 0.5) * size + np.array([0, 1, 0], np.float32) * (cy - 0.5) * size)
                    uv.append(prim.uvs[0][q + k])
                    nrm.append(out_dir)
                    jt.append(prim.joints[q + k] if prim.joints is not None else None)
                    wt.append(prim.weights[q + k] if prim.weights is not None else None)
                # corners sit in the source order (0,0) (0,1) (1,0) (1,1); both windings so the card shows from either side
                idx += [base, base + 1, base + 2, base + 1, base + 3, base + 2, base, base + 2, base + 1, base + 1, base + 2, base + 3]
        card = Primitive(material=prim.material, indices=np.asarray(idx, np.uint32), positions=np.asarray(pos, np.float32))
        card.normals = np.asarray(nrm, np.float32)
        card.uvs = [np.asarray(uv, np.float32)]
        if prim.joints is not None:
            card.joints = np.asarray(jt, prim.joints.dtype)
            card.weights = np.asarray(wt, np.float32)
        out.append(card)
    return out


# FITTED: displacement of an enclosing smoke shell = this x weight x (cloud - mean) metres along the normals, on Gastly (the game's cloud-vertex
# shader is not in the rip; ZA's gas shell is displaced the same way by `DisplacementHeight` 0.3 x a cloud map), 2026-10-20
SHELL_SCULPT = 1.0


def _sculpt_shells(meshes: list[Mesh], mats: "_Materials") -> None:
    """A smooth enclosing sphere is the shell ZA sculpts into lumps: push each vertex along its normal by the shell's own cloud noise
    (`_Blend0Tex` sampled at the vertex UV), welded by position so seams and poles do not tear."""
    for me in meshes:
        for p in me.primitives:
            m = mats.materials[p.material]
            noise = mats.cache.get(("noise", m.extras.get("home", {}).get("name", p.material.removesuffix("_rare"))))
            if noise is None or not m.extras.get("behindBody") or p.normals is None or not p.uvs:
                continue
            h, w = noise.shape
            u = np.mod(p.uvs[0][:, 0], 1.0)
            v = np.mod(p.uvs[0][:, 1], 1.0)
            d = noise[np.minimum((v * h).astype(int), h - 1), np.minimum((u * w).astype(int), w - 1)]
            key = np.round(p.positions / 1e-3).astype(np.int64)
            _, inv = np.unique(key, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            mean = np.bincount(inv, weights=d) / np.bincount(inv)
            d = mean[inv] - float(mean.mean())
            # ZA weights the push by vertex colour R: 0.5 on the face side up to 1.0 at the back (the face stays calm, the back billows).
            # glTF +Z is the front, so the weight follows -nz.
            w = 0.5 + 0.5 * np.clip(0.5 - 0.5 * p.normals[:, 2], 0.0, 1.0)
            p.positions = (p.positions + p.normals * (d * w * SHELL_SCULPT)[:, None]).astype(np.float32)


def _apply_irises(meshes: list[Mesh], mats: "_Materials", base_mats: "_Materials | None" = None) -> None:
    """Paint `_L1Col0Tex` irises into the eye atlases (see `_add_iris`), one texture per orientation.

    The eye samples the baked atlas at x = `tile_u * u - _ColorBaseU` (Unity's mirror wrap reflects odd periods, i.e. floor(x) odd),
    where u is the eye mesh's UV; a mirrored eye shows its atlas flipped, so its iris, which the layer places in mesh UV space, must be
    flipped in atlas space too. `base_mats` (the normal bundle) supplies the UVs for the shiny bundle's materials.
    """
    uv_of: dict[str, float] = {}
    for me in (meshes if base_mats is None else []):
        for p in me.primitives:
            if p.uvs:
                uv_of[p.material] = float(np.mean(p.uvs[0][:, 0]))
    if base_mats is not None:
        uv_of = base_mats.uv_u
    mats.uv_u = uv_of
    for name, m in mats.materials.items():
        info = m.extras.get("home", {}).get("iris")
        if not info or not m.base_color:
            continue
        u = uv_of.get(name, uv_of.get(name.removesuffix("_rare"), 0.0))
        # The baked atlas is `roll(expanded, -base * W_orig)`, i.e. E'(u) = T(tile * u - base): same sign here, or the wrong eye is flipped
        # (verified: with the opposite sign Mewtwo/Lucario/Charizard looked cross-eyed).
        x = info["tile_u"] * u - info["base_u"]
        mirrored = int(np.floor(x)) % 2 == 1
        flip = (info["su"] < 0) != mirrored
        key = f"{m.base_color}_iris_{info['layer']}_{int(flip)}"
        if key not in mats.textures:
            mats.textures[key] = TextureImage(key=key, rgba=_add_iris(mats.textures[m.base_color].rgba, mats.textures[info["layer"]].rgba, info["su"], info["sv"], flip), srgb=True)
        m.base_color = key


def _flag_enclosing_effects(meshes: list[Mesh], materials: dict[str, Material]) -> None:
    """Effect cards that wrap the body (Gastly's smoke) are painted UNDER it in game, never across the face: flag them `behindBody`
    (the viewer draws them first without depth writes, like the ZA gas shell). Rule: the effect primitive's bounds contain the centre of
    the solid meshes' bounds."""
    solid = [p.positions for me in meshes for p in me.primitives if not materials[p.material].extras.get("home", {}).get("effect")]
    if not solid:
        return
    allp = np.concatenate(solid)
    centre = (allp.min(0) + allp.max(0)) / 2
    for me in meshes:
        for p in me.primitives:
            m = materials[p.material]
            if m.extras.get("home", {}).get("effect") and (p.positions.min(0) <= centre).all() and (p.positions.max(0) >= centre).all():
                m.extras["behindBody"] = True


class _Materials:
    """Material export for one bundle; duplicate names get a numeric suffix in SMR order, so a shiny bundle names them alike."""

    def __init__(self, suffix: str = ""):
        self.suffix = suffix
        self.uv_u: dict[str, float] = {}
        self.core_colors: dict = {}  # `_BaseColor` of the bundle's flame/smoke core, for its colourless mask sibling
        self.cache: dict = {}
        self.by_ptr: dict = {}
        self.materials: dict[str, Material] = {}
        self.textures: dict[str, TextureImage] = {}

    def __call__(self, ptr) -> str:
        ck = (ptr.file_id, ptr.path_id)
        if ck not in self.by_ptr:
            m = ptr.deref().read()
            mat = _material(m, self.cache, self.textures, self.suffix, self.core_colors)
            name = m.m_Name
            if name in self.materials:
                name = f"{name}.{len(self.materials)}"
            mat.name = name
            self.materials[name] = mat
            self.by_ptr[ck] = name
        return self.by_ptr[ck]


def _controller_clips(objs, sf, env) -> list[tuple[str, object]]:
    """(state name, AnimationClip) of the unmasked layers of the controller MonoBehaviour, in layer order."""
    by_key = {}
    for f in env.files.values():
        for g in getattr(f, "files", {}).values():
            if hasattr(g, "objects"):
                cab = next((k for k, v in f.files.items() if v is g), "").lower()
                for pid, o in g.objects.items():
                    if o.type.name == "AnimationClip":
                        by_key[(cab, pid)] = o
    cabs = [x.path.split("/")[-1].lower() for x in sf.externals]
    out, seen = [], set()
    for o in objs:
        if o.type.name != "MonoBehaviour":
            continue
        try:
            t = o.read_typetree()
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(t, dict) or "layers" not in t:
            continue
        for layer in t["layers"]:
            if layer.get("avatarMask", {}).get("m_PathID"):  # eye / mouth layers: masked overlays, not whole-body clips
                continue
            for s in layer.get("states", []):
                c = s.get("clip", {})
                fid = int(c.get("m_FileID", 0))
                if fid < 1 or fid > len(cabs):
                    continue
                obj = by_key.get((cabs[fid - 1], c.get("m_PathID")))
                if obj is not None and s["name"] not in seen:
                    seen.add(s["name"])
                    out.append((s["name"], obj))
    return out


def _uv_helpers(objs, mats: "_Materials") -> dict[int, tuple[str, float]]:
    """Helper Transform path id -> (exported material name, U divisor).

    The controller MonoBehaviour's `_materialModifiers` bind a helper transform (`CusAnimMat_<Mat>_UV_ColorUVTranslate`) to a material
    (`binders[].renderer` + `materialIndex`); the clips animate the helper's localPosition and the shader reads it as the material's
    `ColorUVTranslate` (the eye/mouth expression atlas row/column). The U divisor is the baked tiling of the material's `_Col0Tex`
    (the texture was expanded, so a U shift of 1 original texture is 1/sx of the baked width).
    """
    smr = {o.path_id: o.read() for o in objs if o.type.name == "SkinnedMeshRenderer"}
    out: dict[int, tuple[str, float]] = {}
    for o in objs:
        if o.type.name != "MonoBehaviour":
            continue
        try:
            t = o.read_typetree()
        except Exception:  # noqa: BLE001
            continue
        for mm in t.get("_materialModifiers", []) if isinstance(t, dict) else []:
            props = [p for p in mm.get("properties", []) if p.get("propertyName") == "ColorUVTranslate"]
            if not props:
                continue
            helper = props[0]["textureTransform"]["m_PathID"]
            for b in mm.get("binders", []):
                r = smr.get(b["renderer"]["m_PathID"])
                ms = list(r.m_Materials) if r is not None else []
                if r is not None and 0 <= b["materialIndex"] < len(ms):
                    m = ms[b["materialIndex"]].deref().read()
                    col = dict(m.m_SavedProperties.m_TexEnvs).get("_Col0Tex")
                    out[helper] = (mats(ms[b["materialIndex"]]), max(1.0, round(col.m_Scale.x)) if col is not None else 1.0)
    return out


# The helper's localPosition is in metres; the shader's ColorUVTranslate is that x 100 (the eye atlas has 4 rows and the blink clips step
# the helper by 0.0025: 0.25 UV per row).
UV_HELPER_GAIN = 100.0


def _uv_helper_offsets(clip, h: Hierarchy, helpers: dict[int, tuple[str, float]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not helpers:
        return {}
    curves = _clip_curves(clip)
    slot = 0
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for g in clip.m_ClipBindingConstant.genericBindings:
        dim = 4 if (g.typeID == 4 and g.attribute == 2) else 3 if g.typeID == 4 else 1
        pid = h.hash.get(int(g.path)) if g.typeID == 4 and g.attribute == ATTR_POS else None
        if pid in helpers:
            comps = [curves.get(slot + k) for k in range(2)]
            if all(c is not None for c in comps):
                name, div = helpers[pid]
                times = np.unique(np.concatenate([c[0] for c in comps]))
                at = lambda c: np.interp(times, c[0], c[1]) if len(c[0]) > 1 else np.full(len(times), c[1][0])  # noqa: E731
                # glTF offsets: +y down the image; U halves with the baked tiling.
                vals = np.stack([at(comps[0]) * UV_HELPER_GAIN / div, at(comps[1]) * UV_HELPER_GAIN], axis=1).astype(np.float32)
                vals = np.where(np.abs(vals) < 1e-6, 0.0, vals).astype(np.float32)
                if np.abs(vals).max() > 1e-5:
                    keep = np.ones(len(times), bool)
                    keep[1:] = np.any(np.abs(np.diff(vals, axis=0)) > 1e-6, axis=1)
                    out[name] = (times[keep].astype(np.float32), vals[keep])
        slot += dim
    return out


def uv_motion(env, sf, clip_re: str = r"_loop01$") -> dict:
    """Constant UV scroll speeds the looping idle clip drives through `_materialModifiers` helpers.

    Effect shaders have no speed property: every `*UVTranslate` (`Blend0/1`, `Mask0/1` on fire/smoke cards, `ColorUVTranslate`
    / `DisplaceUVTranslate` on `_AsFire` paint) is a helper transform whose localPosition the clip ramps linearly and wraps
    (Charizard `loop01`: Blend0 V steps 0.00025 m per 1/60 s frame, wrapping every 1.333 s, i.e. a seamless loop). Speed =
    the median per-second step x `UV_HELPER_GAIN` (the eye helpers' gain), in Unity UV units (V up), per material name.
    Returns {"loop": seconds, "clip": name, "materials": {material: {property: [u, v]}}} or {} when nothing scrolls.
    """
    objs = list(sf.objects.values())
    h = Hierarchy(objs)
    smr = {o.path_id: o.read() for o in objs if o.type.name == "SkinnedMeshRenderer"}
    helpers: dict[int, list[tuple[str, str]]] = {}
    for o in objs:
        if o.type.name != "MonoBehaviour":
            continue
        try:
            t = o.read_typetree()
        except Exception:  # noqa: BLE001
            continue
        for mm in t.get("_materialModifiers", []) if isinstance(t, dict) else []:
            for p in mm.get("properties", []):
                name = p.get("propertyName") or ""
                if not name.endswith("UVTranslate"):
                    continue
                for b in mm.get("binders", []):
                    r = smr.get(b["renderer"]["m_PathID"])
                    ms = list(r.m_Materials) if r is not None else []
                    if r is not None and 0 <= b["materialIndex"] < len(ms):
                        helpers.setdefault(p["textureTransform"]["m_PathID"], []).append((ms[b["materialIndex"]].deref().read().m_Name, name))
    if not helpers:
        return {}
    clip = next((o.read() for o in env.objects if o.type.name == "AnimationClip" and re.search(clip_re, o.peek_name() or "")), None)
    if clip is None:
        return {}
    curves = _clip_curves(clip)
    out: dict[str, dict[str, list[float]]] = {}
    slot = 0
    for g in clip.m_ClipBindingConstant.genericBindings:
        dim = 4 if (g.typeID == 4 and g.attribute == 2) else 3 if g.typeID == 4 else 1
        pid = h.hash.get(int(g.path)) if g.typeID == 4 and g.attribute == ATTR_POS else None
        if pid in helpers:
            speed = []
            for k in range(2):
                c = curves.get(slot + k)
                if c is None or len(c[0]) < 2:
                    speed.append(0.0)
                    continue
                d = np.diff(c[1]) / np.maximum(np.diff(c[0]), 1e-6)
                speed.append(round(float(np.median(d)) * UV_HELPER_GAIN, 4))
            for mat, prop in helpers[pid]:
                out.setdefault(mat, {})[prop] = speed
        slot += dim
    return {"loop": round(float(clip.m_MuscleClip.m_StopTime), 4), "clip": clip.m_Name, "materials": out} if out else {}


def _clips(objs, sf, env, h: Hierarchy, index, skeleton, helpers: dict[int, tuple[str, float]] | None = None) -> list[Clip]:
    clips = []
    for name, o in _controller_clips(objs, sf, env):
        c = o.read()
        rate = float(c.m_SampleRate) or 30.0
        stop = float(c.m_MuscleClip.m_StopTime)
        tracks = [t for t in _clip_tracks(c, h, index, skeleton) if not t.joint.startswith("CusAnimMat_")]
        uv = _uv_helper_offsets(c, h, helpers or {})
        clips.append(
            Clip(
                name=name,
                tracks=tracks,
                uv_offset=uv,
                uv_offset_mode="gltf_step" if uv else "",
                frame_rate=rate,
                frame_count=int(round(stop * rate)) + 1,
                loop=bool(c.m_MuscleClip.m_LoopTime),
                category=categorize(name),
                source=c.m_Name,
            )
        )
    for i, c in enumerate(clips):
        c.index = i
    return clips


def import_bundle(
    path: str,
    model_id: str,
    display_name: str,
    kind: str = "pokemon",
    game: str = "home",
    with_anim: bool = True,
    deps: list[str] | None = None,
    rare_path: str | None = None,
    rare_deps: list[str] | None = None,
) -> Model:
    env, sf = _env(path, deps or [])
    objs = list(sf.objects.values())
    h = Hierarchy(objs)
    smrs = [o.read() for o in objs if o.type.name == "SkinnedMeshRenderer"]

    bones_used: list[int] = []
    bind: dict[int, np.ndarray] = {}
    for smr in smrs:
        mesh = smr.m_Mesh.deref().read()
        for b, bp in zip(smr.m_Bones, mesh.m_BindPose):
            if b.path_id and b.path_id not in bind:
                bind[b.path_id] = _mat(bp)
                bones_used.append(b.path_id)
    skeleton, index = _build_skeleton(h, bones_used, bind)
    palette_of = {pid: i for i, pid in enumerate(bones_used)}

    mats = _Materials()
    _find_core_colors(objs, mats)
    meshes: list[Mesh] = []
    for smr in smrs:
        prims = _expand_billboards(smr, _mesh_primitives(smr, mats, palette_of), lambda n: mats.materials[n])
        if prims:
            go = smr.m_GameObject.read()
            meshes.append(Mesh(name=go.m_Name, primitives=prims, visible=bool(go.m_IsActive) and bool(smr.m_Enabled)))

    _flag_enclosing_effects(meshes, mats.materials)
    _sculpt_shells(meshes, mats)
    _apply_irises(meshes, mats)

    variants: list[Variant] = []
    if rare_path:
        renv, rsf = _env(rare_path, rare_deps or [])
        rmats = _Materials()
        _find_core_colors(list(rsf.objects.values()), rmats)
        for o in rsf.objects.values():
            if o.type.name == "SkinnedMeshRenderer":
                for p in o.read().m_Materials:
                    rmats(p)
        # The shiny prefab names its materials and textures `<name>_rare`; a variant overrides the base material of the same name.
        _apply_irises([], rmats, mats)
        changed = {}
        for k, v in rmats.materials.items():
            base = k.removesuffix("_rare")
            b = mats.materials.get(base)
            if b is None:
                continue
            # Effect textures are baked per bundle (names differ), so compare pixels; keep the base material's flags.
            same = np.array_equal(mats.textures[b.base_color].rgba, rmats.textures[v.base_color].rgba) if b.base_color and v.base_color and b.extras.get("home", {}).get("effect") else v.base_color == b.base_color
            if not same:
                v.name = base
                if "behindBody" in b.extras:
                    v.extras["behindBody"] = True
                changed[base] = v
        if changed:
            mats.textures.update(rmats.textures)
            variants = [
                Variant(name="normal", materials={}, group="coloring", label="Normal", default=True),
                Variant(name="rare", materials=changed, group="coloring", label="Shiny"),
            ]

    clips = _clips(objs, sf, env, h, index, skeleton, _uv_helpers(objs, mats)) if with_anim else []
    default = next((c.name for c in clips if re.search(r"wait|idle", c.name, re.I)), clips[0].name if clips else None)
    return Model(
        id=model_id,
        game=game,
        display_name=display_name,
        kind=kind,
        skeleton=skeleton,
        meshes=meshes,
        materials=mats.materials,
        textures=mats.textures,
        clips=clips,
        default_clip=default,
        variants=variants,
    )
