"""Pokemon Let's Go Pikachu/Eevee importer: `.gfbmdl` + `.bntx` + `.gfbanm` -> common `Model`.

Verified format facts (real LGPE files):

* Units are centimetres (Pikachu bbox 51 high); the IR is metres, so positions and bone translations are x0.01.
* Bones: SRT with euler XYZ radians (q = qZ*qY*qX, same as Trinity), parent index, no stored inverse bind:
  the inverse bind is the inverse of the composed rest pose. Vertex joint indices are indices into the bone
  list directly (always < bone count; there is no per-mesh palette).
* Vertices are interleaved per mesh in the order of the mesh's attribute list; sizes come from the component
  type (0 f32, 1 f16, 3 u8, 8 u8 normalised), see `gfbmdl.py`.
* `.gfbanm` is the same FlatBuffers layout as Z-A `.tranm` (info {loop, frames, fps 30|60}; per bone
  scale/rotation/translation channels, rotations are the 48-bit packed quaternion), so `trinity.parse_tranm`
  reads it unchanged. Translations are centimetres.
* Materials use `PokeDefaultShader`: `Col0Tex` is the lit albedo, `ColSdw0Tex` the shadow tint (unused here).
  `ColorUVScaleU/V` (mostly 2,1) + a mirror sampler mean the texture holds one half of a symmetric layout;
  `ColorUVTranslateU` = 1 picks the unmirrored second half (EyeR). Same rules as Z-A: the mirrored texture is
  baked to full width, each primitive's integer V tile is dropped and V is flipped.
  Eye/mouth atlases hold the expression frames; the default expression is the tile the UVs address.
* Eyes: `Col0Tex` has holes (alpha 0); `Layer1Enable` + `LyCol0Tex` (iris) shows through them, sampled with
  `Layer1UVScaleU/V` (4) + `Layer1BaseU`. The importer composites col over iris into one opaque texture.
* Shiny: `<id>_rare.gfbmdl` has the same meshes with `_rare` textures -> variant group "coloring".
"""

from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ...common.model import Clip, Joint, Material, Mesh, Model, Primitive, Skeleton, TextureImage, Variant
from ..bntx import parse_bntx
from ..za import trinity as T
from ..za.importer import AnimationFile, _convert_clip, categorize
from .gfbmdl import GfMaterial, Gfbmdl, parse_gfbmdl

CM = 0.01

# Shaders that are procedural effects in game (flames, smoke): approximated with a tinted alpha mask.
# slot of the grey mask, tint, opacity scale
EFFECT_SHADERS = {
    "PokeFireCoreShader1": ("Blend0Tex", (1.0, 0.82, 0.30), 1.0),
    "PokeFireMaskShader1": ("Mask0Texture", (1.0, 0.45, 0.12), 0.85),
    "PokeSmokeMaskShader1": ("MaskTexture", (0.45, 0.40, 0.50), 0.7),
    "PokeMegaGangarShader": ("MaskTexture", (0.55, 0.25, 0.70), 0.8),
}
# Helpers that are not renderable character meshes
SKIP_SHADERS = {"CharaCollisionShader"}
DROPPED_MATERIALS = {"Yuka"}  # floor slab under mega Gengar (it sinks into it)


@dataclass
class LgpeAnimation:
    name: str  # clip name, e.g. ba10_waitA01
    data: bytes


def _quat_to_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        np.float64,
    )


def build_skeleton(g: Gfbmdl) -> Skeleton:
    joints: list[Joint] = []
    worlds: list[np.ndarray] = []
    seen: dict[str, int] = {}
    for i, b in enumerate(g.bones):
        name = b.name or f"bone{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        q = T.euler_xyz_to_quat(*b.euler)
        t = tuple(float(x) * CM for x in b.translation)
        local = np.eye(4)
        local[:3, :3] = _quat_to_mat(q) @ np.diag(b.scale)
        local[:3, 3] = t
        parent = b.parent if 0 <= b.parent < i else -1
        world = local if parent < 0 else worlds[parent] @ local
        worlds.append(world)
        joints.append(
            Joint(
                name=name,
                parent=parent,
                translation=t,
                rotation=q,
                scale=tuple(float(x) for x in b.scale),
                inverse_bind=np.linalg.inv(world).astype(np.float32),
                skinning=True,
            )
        )
    return Skeleton(joints=joints, palette=list(range(len(joints))))


# --------------------------------------------------------------------------- textures
def decode_dds(data: bytes) -> np.ndarray:
    """Mip 0 of a DDS file -> (H, W, 4) uint8. Covers the trainer rip: BC1 (DXT1 / DX10 71,72), BC7 (DX10 98,99)
    and uncompressed 32-bit RGB(A) (Pillow lacks the sRGB BC1 DX10 variant)."""
    import texture2ddecoder as t2d

    if data[:4] != b"DDS ":
        raise ValueError("not a DDS file")
    h, w = struct.unpack_from("<II", data, 12)
    pf_flags, fourcc, bits = struct.unpack_from("<I4sI", data, 80)
    rmask, gmask, bmask, amask = struct.unpack_from("<4I", data, 92)
    offset = 128
    codec = None
    if fourcc == b"DX10":
        dxgi = struct.unpack_from("<I", data, 128)[0]
        offset = 148
        codec = {71: "bc1", 72: "bc1", 77: "bc3", 78: "bc3", 98: "bc7", 99: "bc7"}.get(dxgi)
        if codec is None:
            raise NotImplementedError(f"DDS DXGI format {dxgi}")
    elif fourcc == b"DXT1":
        codec = "bc1"
    elif fourcc == b"DXT5":
        codec = "bc3"
    pw, ph = -(-w // 4) * 4, -(-h // 4) * 4
    if codec:
        fn = {"bc1": t2d.decode_bc1, "bc3": t2d.decode_bc3, "bc7": t2d.decode_bc7}[codec]
        raw = np.frombuffer(fn(data[offset:], pw, ph), np.uint8).reshape(ph, pw, 4)
        return raw[:h, :w][..., [2, 1, 0, 3]].copy()
    if bits == 32:
        a = np.frombuffer(data, np.uint8, count=w * h * 4, offset=offset).reshape(h, w, 4)
        shift = {m: i for i, m in enumerate((0xFF, 0xFF00, 0xFF0000, 0xFF000000))}
        out = np.empty((h, w, 4), np.uint8)
        for k, m in enumerate((rmask, gmask, bmask)):
            out[..., k] = a[..., shift[m]]
        out[..., 3] = a[..., shift[amask]] if amask else 255
        return out
    raise NotImplementedError("unsupported DDS pixel format")


def _is_dummy(name: str) -> bool:
    return not name or name.startswith("dummy") or name.startswith("chara_d_") or name == "projection_effect_col"


def _bilinear_wrap(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample `img` (H,W,C) at texture coords u (right) / v (down), both 0..1, with repeat wrapping."""
    h, w = img.shape[:2]
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    x0 %= w
    y0 %= h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    f = img.astype(np.float32)
    top = f[y0, x0] * (1 - fx) + f[y0, x1] * fx
    bot = f[y1, x0] * (1 - fx) + f[y1, x1] * fx
    return top * (1 - fy) + bot * fy


@dataclass
class UvBake:
    """UV transform baked into the primitive UVs (after the mirrored texture was doubled)."""

    su: float = 1.0
    sv: float = 1.0
    tu: float = 0.0
    tv: float = 0.0


class _Ctx:
    def __init__(self, load: Callable[[str], np.ndarray | None]):
        self.load = load
        self.textures: dict[str, TextureImage] = {}
        self.cache: dict[str, np.ndarray | None] = {}

    def tex(self, name: str) -> np.ndarray | None:
        if name not in self.cache:
            self.cache[name] = None if _is_dummy(name) else self.load(name)
        return self.cache[name]

    def register(self, key: str, rgba: np.ndarray, srgb: bool = True) -> str:
        if key not in self.textures:
            self.textures[key] = TextureImage(key=key, rgba=np.ascontiguousarray(rgba, np.uint8), srgb=srgb)
        return key


def _mirror(img: np.ndarray, mirrored: tuple[bool, bool]) -> np.ndarray:
    if mirrored[0]:
        img = np.concatenate([img, img[:, ::-1]], axis=1)
    if mirrored[1]:
        img = np.concatenate([img[::-1], img], axis=0)  # V is flipped after scaling: the mirrored copy goes first
    return img


def _first(f: dict[str, float], *names: str, default: float = 0.0) -> float:
    for n in names:
        if n in f:
            return f[n]
    return default


def colour_uv(f: dict[str, float]) -> tuple[float, float, float, float]:
    """(scale u, scale v, translate u, translate v) of the colour texture. Pokemon: `ColorUVScale*`/`ColorUVTranslate*`
    + `ColorBase*`; trainers: `L0UVTranslate*` + `L0UVBase*`."""
    su = _first(f, "ColorUVScaleU", "L0ScaleU", default=1.0) or 1.0
    sv = _first(f, "ColorUVScaleV", "L0ScaleV", default=1.0) or 1.0
    tu = _first(f, "ColorUVTranslateU", "L0UVTranslateU") + _first(f, "ColorBaseU", "L0UVBaseU")
    tv = _first(f, "ColorUVTranslateV", "L0UVTranslateV") + _first(f, "ColorBaseV", "L0UVBaseV")
    return su, sv, tu, tv


def layer_uv(f: dict[str, float]) -> tuple[float, float, float, float]:
    """Same for layer 1 (the iris): `Layer1UV*`/`Layer1Base*` (Pokemon) or `L1Scale*`/`L1UV*` (trainers)."""
    su = _first(f, "Layer1UVScaleU", "L1ScaleU", default=1.0)
    sv = _first(f, "Layer1UVScaleV", "L1ScaleV", default=1.0)
    tu = _first(f, "Layer1UVTranslateU", "L1UVTranslateU") + _first(f, "Layer1BaseU", "L1UVBaseU")
    tv = _first(f, "Layer1UVTranslateV", "L1UVTranslateV") + _first(f, "Layer1BaseV", "L1UVBaseV")
    return su, sv, tu, tv


def composite_layer(col: np.ndarray, layer: np.ndarray, gm: GfMaterial, uv: UvBake, mirrored: tuple[bool, bool]) -> np.ndarray:
    """Col0 (alpha = cut-out) over the layer-1 texture (iris); returns an opaque RGBA on the mirrored grid.

    The layer lookup follows the game: layer_uv = raw_uv * Layer1Scale + Layer1Base + Translate, with raw_uv
    recovered from the baked-image coordinate through the colour UV transform.
    """
    lsu, lsv, ltu, ltv = layer_uv(gm.floats)
    h, w = col.shape[:2]
    W, H = w * (2 if mirrored[0] else 1), h * (2 if mirrored[1] else 1)
    ub = (np.arange(W, dtype=np.float32) + 0.5) / W
    vb = 1.0 - (np.arange(H, dtype=np.float32) + 0.5) / H  # bottom-left origin
    u_raw = (ub - uv.tu) / uv.su
    v_raw = (vb - uv.tv) / uv.sv
    lu = u_raw * lsu + ltu
    lv = v_raw * lsv + ltv
    uu, vv = np.meshgrid(lu % 1.0, (1.0 - lv) % 1.0)  # image rows run top-down
    it = _bilinear_wrap(layer, uu, vv)
    base = _mirror(col, mirrored).astype(np.float32)
    a = base[..., 3:4] / 255.0
    rgb = base[..., :3] * a + it[..., :3] * (1.0 - a)
    return np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255.0, np.float32)], axis=2).clip(0, 255).astype(np.uint8)


def convert_material(gm: GfMaterial, ctx: _Ctx, tag: str = "") -> tuple[Material | None, UvBake]:
    """GfMaterial -> IR Material + the UV transform its primitives must bake."""
    mat = Material(name=gm.name)
    uv = UvBake()
    if gm.shader in SKIP_SHADERS:
        return None, uv
    f = gm.floats
    mat.extras["shader"] = gm.shader

    if gm.shader in EFFECT_SHADERS:
        slot, tint, strength = EFFECT_SHADERS[gm.shader]
        mask = ctx.tex(gm.textures.get(slot, ""))
        if mask is None:
            return None, uv
        rgba = np.zeros(mask.shape[:2] + (4,), np.uint8)
        rgba[..., :3] = (np.array(tint) * 255).astype(np.uint8)
        rgba[..., 3] = (mask[..., 0].astype(np.float32) * strength).astype(np.uint8)
        mat.base_color = ctx.register(f"{gm.textures[slot]}_fx{tag}", rgba)
        mat.alpha_mode = "BLEND"
        mat.double_sided = True
        mat.extras["effect"] = True
        return mat, uv

    col_name = gm.textures.get("Col0Tex") or gm.textures.get("L0ColTex") or ""
    col = ctx.tex(col_name)
    if col is None:
        return mat, uv  # untextured (flat white) material
    su, sv, tu, tv = colour_uv(f)
    mirror = (su >= 2.0, sv >= 2.0)
    uv = UvBake(
        su / (2.0 if mirror[0] else 1.0),
        sv / (2.0 if mirror[1] else 1.0),
        tu / (2.0 if mirror[0] else 1.0),
        tv / (2.0 if mirror[1] else 1.0),
    )
    layer_name = gm.textures.get("LyCol0Tex") or gm.textures.get("L1ColTex") or ""
    layer = ctx.tex(layer_name) if (gm.switches.get("Layer1Enable") or gm.shader == "CharaEyeShader") else None

    def bake(src: np.ndarray, src_name: str, const: str) -> tuple[str, np.ndarray, str]:
        """Mirror / composite the iris layer / tint one of the colour textures; returns (key, image, alpha mode)."""
        key = f"{src_name}{tag}"
        holes = src[..., 3].min() < 250
        mode = "OPAQUE"
        if layer is not None and holes:
            img = composite_layer(src, layer, gm, uv, mirror)
            key += "_" + layer_name
        else:
            img = _mirror(src, mirror)
            if holes:
                mode = "MASK"
        cc = gm.colors.get(const, (1.0, 1.0, 1.0))
        if max(abs(c - 1.0) for c in cc) > 1e-3:
            img = img.copy()
            img[..., :3] = (img[..., :3].astype(np.float32) * np.clip(np.asarray(cc, np.float32), 0, 4)).clip(0, 255).astype(np.uint8)
            key += "_cc"
        return key + ("_mu" if mirror[0] else "") + ("_mv" if mirror[1] else ""), img, mode

    key, img, mode = bake(col, col_name, "ConstantColor0")
    mat.base_color = ctx.register(key, img)
    if mode == "MASK":
        mat.alpha_mode, mat.alpha_cutoff = "MASK", 0.5
    sdw_name = gm.textures.get("ColSdw0Tex", "")
    sdw = ctx.tex(sdw_name)
    if sdw is not None and sdw.shape == col.shape:
        skey, simg, _ = bake(sdw, sdw_name, "ConstantColorSd0")
        mat.extras["shadow_color"] = ctx.register(skey, simg)  # the viewer's Game shading fades to it on the unlit side
    return mat, uv


# --------------------------------------------------------------------------- meshes
def _prim_uvs(uv0: np.ndarray, bake: UvBake) -> np.ndarray:
    u = uv0[:, 0] * bake.su + bake.tu
    v = uv0[:, 1] * bake.sv + bake.tv
    v = v - math.floor(float(v.mean()))  # each polygon sits in its own integer V tile
    return np.stack([u, 1.0 - v], axis=1).astype(np.float32)


def _norm_weights(w: np.ndarray) -> np.ndarray:
    s = w.sum(axis=1, keepdims=True)
    out = np.where(s > 1e-6, w / np.maximum(s, 1e-6), np.array([1, 0, 0, 0], np.float32))
    return out.astype(np.float32)


def build_meshes(g: Gfbmdl, mat_names: list[str | None], bakes: list[UvBake], skeleton: Skeleton | None) -> list[Mesh]:
    meshes: list[Mesh] = []
    used: dict[str, int] = {}
    for gm in g.meshes:
        for poly in gm.polygons:
            if poly.material >= len(mat_names) or mat_names[poly.material] is None or len(poly.indices) == 0:
                continue
            mname = mat_names[poly.material]
            uniq, inv = np.unique(poly.indices.astype(np.int64), return_inverse=True)
            uvs = [_prim_uvs(gm.uvs[0][uniq], bakes[poly.material])] if gm.uvs else []
            joints = weights = None
            if skeleton and gm.joints is not None and gm.weights is not None:
                weights = _norm_weights(gm.weights[uniq])
                joints = np.where(weights > 0, gm.joints[uniq], 0).astype(np.uint16)  # zero-weight slots point at joint 0
            prim = Primitive(
                material=mname,
                indices=inv.astype(np.uint32),
                positions=(gm.positions[uniq] * CM).astype(np.float32),
                normals=gm.normals[uniq] if gm.normals is not None else None,
                uvs=uvs,
                joints=joints,
                weights=weights,
            )
            n = used.get(mname, 0)
            used[mname] = n + 1
            meshes.append(Mesh(name=mname if n == 0 else f"{mname}.{n}", primitives=[prim]))
    return meshes


# --------------------------------------------------------------------------- animations
ANIM_PREFIX_RE = re.compile(r"^[a-z]{2}\d{4}_\d{2}_?")
PREFIX_CATEGORY = {"ba": "battle", "kw": "emote", "fi": "move", "gg": "emote", "ri": "move"}
MATERIAL_ONLY_RE = re.compile(r"^(eye|mouth)\d+$")  # UV-scroll expression clips: no bone data


def clip_name(filename: str, model_id: str) -> str:
    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem == model_id:
        return "base"
    return ANIM_PREFIX_RE.sub("", stem, count=1) or stem


def lgpe_category(name: str) -> str:
    cat = categorize(name)
    if cat != "other":
        return cat
    return PREFIX_CATEGORY.get(name[:2], "other")


def convert_animation(anim: LgpeAnimation, skeleton: Skeleton, order: int) -> Clip:
    tr = T.parse_tranm(anim.data)  # raises ValueError for material-only files
    names = {j.name for j in skeleton.joints}
    clip = _convert_clip(tr, AnimationFile(name=anim.name, index=order, data=anim.data, group=""))
    tracks = []
    for t in clip.tracks:
        if t.joint not in names:
            continue
        if t.kind == "translation":
            t.values = (t.values * CM).astype(np.float32)
        tracks.append(t)
    clip.tracks = tracks
    clip.category = lgpe_category(anim.name)
    clip.index = order
    clip.loop = bool(tr.looped) or anim.name.endswith("_loop")
    return clip


def pick_default_clip(clips: list[Clip]) -> str | None:
    for pat in (r"^ba10_waitA", r"^ba\d+_wait[A-Z]\d*", r"^fi\d+_wait", r"wait", r"idle"):
        for c in clips:
            if re.search(pat, c.name):
                return c.name
    return clips[0].name if clips else None


# --------------------------------------------------------------------------- model
def import_model(
    read: Callable[[str], bytes],
    exists: Callable[[str], bool],
    model_id: str,
    display_name: str,
    kind: str = "pokemon",
    game: str = "lgpe",
    animations: list[LgpeAnimation] | None = None,
    model_file: str | None = None,
) -> Model:
    """`read`/`exists` resolve names relative to the model's `mdl/` folder (`<id>.gfbmdl`, `<tex>.bntx`)."""
    model_file = model_file or f"{model_id}.gfbmdl"
    g = parse_gfbmdl(read(model_file))

    def load_tex(name: str) -> np.ndarray | None:
        try:
            if exists(f"{name}.bntx"):  # Pokemon: Switch BNTX
                texs = parse_bntx(read(f"{name}.bntx"))
                return texs[0].rgba if texs else None
            if exists(f"{name}.dds"):  # trainers: plain DDS
                return decode_dds(read(f"{name}.dds"))
        except (ValueError, NotImplementedError, OSError, struct.error):
            return None
        return None

    ctx = _Ctx(load_tex)
    skeleton = build_skeleton(g) if g.bones else None
    materials: dict[str, Material] = {}
    mat_names: list[str | None] = []
    bakes: list[UvBake] = []
    for gm in g.materials:
        mat, bake = convert_material(gm, ctx)
        bakes.append(bake)
        if mat is None or gm.name in DROPPED_MATERIALS:
            mat_names.append(None)
            continue
        name, n = mat.name, 2
        while name in materials:
            name = f"{mat.name}.{n}"
            n += 1
        mat.name = name
        materials[name] = mat
        mat_names.append(name)
    meshes = build_meshes(g, mat_names, bakes, skeleton)

    variants: list[Variant] = []
    rare_file = model_file[: -len(".gfbmdl")] + "_rare.gfbmdl"
    if exists(rare_file):
        gr = parse_gfbmdl(read(rare_file))
        over: dict[str, Material] = {}
        if len(gr.materials) == len(g.materials):
            for i, gm in enumerate(gr.materials):
                if mat_names[i] is None:
                    continue
                m2, _ = convert_material(gm, ctx)
                if m2 is not None and m2.base_color != materials[mat_names[i]].base_color:
                    m2.name = mat_names[i]
                    over[m2.name] = m2
        if over:
            variants = [
                Variant(name="normal", materials={}, group="coloring", label="Normal", default=True),
                Variant(name="rare", materials=over, group="coloring", label="Shiny"),
            ]

    clips: list[Clip] = []
    skipped: list[str] = []
    if skeleton and animations:
        for i, a in enumerate(sorted(animations, key=lambda a: (a.name != "base", a.name))):
            if MATERIAL_ONLY_RE.match(a.name):
                skipped.append(a.name)
                continue
            try:
                clips.append(convert_animation(a, skeleton, i))
            except ValueError:
                skipped.append(a.name)
    model = Model(
        id=model_id,
        game=game,
        display_name=display_name,
        kind=kind,
        skeleton=skeleton,
        meshes=meshes,
        materials=materials,
        textures=ctx.textures,
        clips=clips,
        variants=variants,
        default_clip=pick_default_clip(clips),
    )
    if skipped:
        model.extras["materialOnlyClips"] = sorted(skipped)
    apply_rest_visibility(model)
    group_hair_meshes(model)
    if model.default_clip:  # the default clip first: it is the pose most tools show before playing anything
        model.clips.sort(key=lambda c: (c.name != model.default_clip, c.index))
    return model




def apply_rest_visibility(model: Model) -> None:
    """The game hides parts by scaling bones: the default clip's frame 0 is the rest state.

    Bulbasaur's vines (`*FeelerC##`, scale 0.1 on X in every clip) hang straight up in the bind pose; copying
    such constant non-unit scales onto the joints' rest scale gives the hidden state before any clip plays
    (the inverse binds keep the bind pose, so skinning is unchanged).
    """
    clip = next((c for c in model.clips if c.name == model.default_clip), None)
    if clip and model.skeleton:
        by_name = {j.name: j for j in model.skeleton.joints}
        for t in clip.tracks:
            j = by_name.get(t.joint)
            if t.kind == "scale" and j is not None and len(t.times) == 1 and np.abs(t.values[0] - 1.0).max() > 1e-3:
                j.scale = tuple(float(x) for x in t.values[0])


def group_hair_meshes(model: Model) -> None:
    """Partner Pikachu/Eevee ship alternative hairstyles as overlapping `hair*` meshes: expose them as one part group."""
    hair = [m for m in model.meshes if re.fullmatch(r"hair(\.\d+)?", m.name)]
    if len(hair) < 2:
        return
    options = []
    for i, m in enumerate(hair):
        m.extras.update({"group": "hair", "option": f"hair{i + 1}"})
        m.visible = False
        options.append({"id": f"hair{i + 1}", "label": f"Hairstyle {i + 1}", "default": False})
    options.insert(0, {"id": "none", "label": "None", "default": True})
    model.extras.setdefault("part_groups", []).append({"name": "hair", "label": "Hair", "options": options})
