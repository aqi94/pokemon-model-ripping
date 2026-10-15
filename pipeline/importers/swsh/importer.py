"""Sword/Shield Pokemon importer: `.gfbmdl` + `.bntx` + `.gfbanm` -> common Model.

The file is Y up, faces +Z and is in centimetres (Pikachu is ~50 cm with ears), so everything is scaled by 0.01. Bones
carry local SRT (euler XYZ radians); the skin palette is the bones flagged `IsRigged` in file order, and the inverse
bind matrices are the inverses of the bind-pose world matrices (the file ships none).

Materials: `Col0Tex` is the (lit, already coloured) albedo. UVs follow the Z-A rules: V is bottom-left origin, every
material's parts sit in their own integer V tile (dropped), then `ColorUVScale/Translate/Base` apply; the sampler is
usually mirror in U with scale 2, i.e. the texture stores half of a symmetric layout. Like the Z-A importer the textures
are baked (mirror/repeat tiles expanded to cover the used UV range) so the glTF uses plain [0,1] UVs.
"""

from __future__ import annotations

import math
import re
from typing import Callable

import numpy as np

from ...common.model import Clip, Joint, Material, Mesh, Model, Primitive, Skeleton, TextureImage, Track, Variant
from ..za.importer import categorize, decimate
from ..za.trinity import euler_xyz_to_quat
from . import gfbmdl as G
from .gfbanm import Channel, parse_gfbanm

CM = 0.01
REST_TOL = 1e-4
CLIP_TOL = 1e-3
MAX_TILES = 4
MAX_TEX = 4096

CLIP_CATEGORY = {"ba": "battle", "fi": "move", "kw": "emote", "tr": "other", "cm": "emote"}


# ------------------------------------------------------------------ skeleton
def _quat_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _local_matrix(b: G.Bone) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _quat_matrix(euler_xyz_to_quat(*b.rotation)) * np.asarray(b.scale)[None, :]
    m[:3, 3] = np.asarray(b.translation) * CM
    return m


def build_skeleton(model: G.GfbModel) -> Skeleton:
    n = len(model.bones)
    world: list[np.ndarray | None] = [None] * n

    def get_world(i: int) -> np.ndarray:
        if world[i] is None:
            b = model.bones[i]
            loc = _local_matrix(b)
            world[i] = get_world(b.parent) @ loc if 0 <= b.parent < i else loc
        return world[i]  # type: ignore[return-value]

    joints = []
    palette = []
    for i, b in enumerate(model.bones):
        w = get_world(i)
        joints.append(
            Joint(
                name=b.name,
                parent=b.parent if 0 <= b.parent < n else -1,
                translation=tuple(float(v) * CM for v in b.translation),
                rotation=tuple(float(v) for v in euler_xyz_to_quat(*b.rotation)),
                scale=tuple(float(v) for v in b.scale),
                inverse_bind=np.linalg.inv(w).astype(np.float32) if b.rigged else None,
                skinning=b.rigged,
            )
        )
        if b.rigged:
            palette.append(i)
    return Skeleton(joints=joints, palette=palette)


# ------------------------------------------------------------------ textures
def _expand_axis(img: np.ndarray, axis: int, lo: int, hi: int, wrap: str) -> np.ndarray:
    """Tile `img` along `axis` for integer tile indices lo..hi-1 (mirror flips odd tiles)."""
    tiles = []
    for k in range(lo, hi):
        t = img
        if wrap == "mirror" and k % 2:
            t = np.flip(img, axis=axis)
        tiles.append(t)
    return tiles[0] if len(tiles) == 1 else np.concatenate(tiles, axis=axis)


def _axis_plan(t_min: float, t_max: float, wrap: str) -> tuple[int, int, bool]:
    """(lo, hi, baked). Clamped axes are not baked; others bake when the used range spans <= MAX_TILES tiles."""
    if wrap == "clamp":
        return 0, 1, False
    lo = int(math.floor(t_min + 1e-4))
    hi = max(int(math.ceil(t_max - 1e-4)), lo + 1)
    if hi - lo > MAX_TILES:
        return 0, 1, False
    return lo, hi, True


# ------------------------------------------------------------------ materials
class MaterialBuilder:
    def __init__(self, load_tex: Callable[[str], np.ndarray | None]):
        self.load_tex = load_tex
        self.textures: dict[str, TextureImage] = {}
        self._cache: dict[str, np.ndarray | None] = {}

    def tex(self, name: str) -> np.ndarray | None:
        if name not in self._cache:
            try:
                self._cache[name] = self.load_tex(name)
            except Exception:  # noqa: BLE001 - a missing/unsupported texture just drops that slot
                self._cache[name] = None
        return self._cache[name]

    def material(self, mat: G.Material, key: str, ranges: tuple[float, float, float, float] | None) -> Material:
        """`ranges` = (u_min, u_max, v_min, v_max) of the final texture coordinates (image space, top-left origin),
        used to bake mirror/repeat tiles into the albedo."""
        out = Material(name=key, roughness_factor=0.75, double_sided=mat.cull == 0)
        out.extras["game"] = {"shader": mat.shader, "blend": mat.blend}
        col = mat.tex("Col0Tex")
        rgba = self.tex(col.file) if col and col.file else None
        if rgba is not None and ranges is not None:
            wu, wv = col.wrap
            ulo, uhi, ub = _axis_plan(ranges[0], ranges[1], wu)
            vlo, vhi, vb = _axis_plan(ranges[2], ranges[3], wv)
            if ub and (uhi - ulo) * rgba.shape[1] > MAX_TEX:
                ub = False
            if vb and (vhi - vlo) * rgba.shape[0] > MAX_TEX:
                vb = False
            img = rgba
            if ub:
                img = _expand_axis(img, 1, ulo, uhi, wu)
            if vb:
                img = _expand_axis(img, 0, vlo, vhi, wv)
            tkey = col.file if img is rgba else f"{col.file}_u{ulo}_{uhi}_v{vlo}_{vhi}"
            if tkey not in self.textures:
                self.textures[tkey] = TextureImage(tkey, img, True)
            out.base_color = tkey
            out.wrap["base_color"] = ("clamp" if ub else wu, "clamp" if vb else wv)
            out.extras["uv_plan"] = [ulo, uhi, ub, vlo, vhi, vb]
            if rgba[..., 3].min() < 250:
                if mat.blend in (G.BLEND_ALPHA, G.BLEND_PREMUL, G.BLEND_ADD):
                    out.alpha_mode = "BLEND"
                elif mat.alpha_test:
                    out.alpha_mode = "MASK"
        if mat.blend == G.BLEND_ADD:
            out.extras["blend"] = "Add"
        elif mat.blend == G.BLEND_PREMUL:
            out.extras["blend"] = "BlendPreMultiAlpha"
        elif mat.blend == G.BLEND_ALPHA:
            out.extras["blend"] = "Blend"
        return out


def _uv_params(mat: G.Material) -> tuple[float, float, float, float]:
    v = mat.values
    su, sv = v.get("ColorUVScaleU", 1.0), v.get("ColorUVScaleV", 1.0)
    ou = v.get("ColorUVTranslateU", 0.0) + v.get("ColorBaseU", 0.0)
    ov = v.get("ColorUVTranslateV", 0.0) + v.get("ColorBaseV", 0.0)
    return su, sv, ou, ov


def import_gfbmdl(
    data: bytes,
    rare_data: bytes | None,
    load_tex: Callable[[str], np.ndarray | None],
    model_id: str,
    display_name: str,
    kind: str = "pokemon",
    game: str = "swsh",
) -> Model:
    gm = G.parse_gfbmdl(data)
    skel = build_skeleton(gm)
    rare = G.parse_gfbmdl(rare_data) if rare_data else None
    builder = MaterialBuilder(load_tex)
    rare_builder = MaterialBuilder(load_tex) if rare else None

    materials: dict[str, Material] = {}
    rare_materials: dict[str, Material] = {}
    ranges_of: dict[str, tuple] = {}
    mesh_list: list[Mesh] = []

    for mi, mref in enumerate(gm.meshes):
        shape = gm.shapes[mref.shape]
        a = shape.attrs
        pos = a[G.V_POS][:, :3].astype(np.float32) * CM
        nrm = a[G.V_NRM][:, :3].astype(np.float32) if G.V_NRM in a else None
        uv0 = a.get(G.V_UV0)
        jn, wt = a.get(G.V_JOINTS), a.get(G.V_WEIGHTS)
        bone = gm.bones[mref.bone] if 0 <= mref.bone < len(gm.bones) else None
        mname = bone.name if bone else f"mesh{mi}"
        prims: list[Primitive] = []
        for mat_i, idx in shape.polygons:
            if not len(idx):
                continue
            mat = gm.materials[mat_i]
            used, inv = np.unique(idx, return_inverse=True)
            tri = inv.astype(np.uint32)
            u_t = v_img = None
            ranges = None
            if uv0 is not None:
                src = uv0[used][:, :2].astype(np.float64)
                su, sv, ou, ov = _uv_params(mat)
                tile = math.floor(src[:, 1].min() + 1e-4)
                u_t = src[:, 0] * su + ou
                v_img = 1.0 - ((src[:, 1] - tile) * sv + ov)
                ranges = (float(u_t.min()), float(u_t.max()), float(v_img.min()), float(v_img.max()))
            key = mat.name
            n = 1
            while key in materials and ranges_of[key] != ranges:
                key = f"{mat.name}.{n}"
                n += 1
            if key not in materials:
                materials[key] = builder.material(mat, key, ranges)
                ranges_of[key] = ranges
                if rare is not None and rare_builder is not None:
                    rm = rare.materials[mat_i] if mat_i < len(rare.materials) else mat
                    rare_materials[key] = rare_builder.material(rm, key, ranges)
            plan = materials[key].extras.get("uv_plan")
            uv = None
            if u_t is not None:
                if plan:
                    ulo, uhi, ub, vlo, vhi, vb = plan
                    uu = (u_t - ulo) / (uhi - ulo) if ub else u_t
                    vv = (v_img - vlo) / (vhi - vlo) if vb else v_img
                else:
                    uu, vv = u_t, v_img
                uv = np.stack([uu, vv], axis=1).astype(np.float32)
            prim = Primitive(material=key, indices=tri, positions=pos[used], normals=nrm[used] if nrm is not None else None)
            if uv is not None:
                prim.uvs = [uv]
            if jn is not None and wt is not None:
                prim.joints = jn[used].astype(np.uint16)
                w = wt[used].astype(np.float32)
                s = w.sum(axis=1, keepdims=True)
                prim.weights = np.where(s > 0, w / np.maximum(s, 1e-8), np.array([1, 0, 0, 0], np.float32))
            prims.append(prim)
        if prims:
            # effect meshes (fire/smoke shaders) have no albedo: hide them by default
            has_albedo = all(materials[p.material].base_color for p in prims)
            mesh_list.append(Mesh(name=mname, primitives=prims, visible=(bone.visible if bone else True) and has_albedo))

    for m in list(materials.values()) + list(rare_materials.values()):
        m.extras.pop("uv_plan", None)

    variants = []
    if rare is not None and any(rare_materials[k].base_color != materials[k].base_color for k in materials):
        variants = [
            Variant(name="normal", materials={}, group="coloring", label="Normal", default=True),
            Variant(name="rare", materials=rare_materials, group="coloring", label="Shiny"),
        ]
    textures = dict(builder.textures)
    if rare_builder is not None and variants:
        textures.update(rare_builder.textures)
    return Model(
        id=model_id,
        game=game,
        display_name=display_name,
        kind=kind,
        skeleton=skel,
        meshes=mesh_list,
        materials=materials,
        textures=textures,
        variants=variants,
    )


# ------------------------------------------------------------------ animation
def _strict(ch: Channel) -> tuple[np.ndarray, np.ndarray]:
    f, v = ch.frames, ch.values
    if len(f) <= 1:
        return f, v
    keep = np.ones(len(f), bool)
    keep[1:] = f[1:] > np.maximum.accumulate(f)[:-1]
    return f[keep], v[keep]


def _fix_signs(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] *= -1.0
    return out


CLIP_NAME_RE = re.compile(r"^(?P<model>pm\d{4}_\d{2})(?:_(?P<name>.+))?\.gfbanm$")


def clip_from_gfbanm(data: bytes, name: str, skeleton: Skeleton, index: int = 0) -> Clip | None:
    """One clip, or None when it animates no bone (eye/mouth clips only drive material UVs)."""
    anm = parse_gfbanm(data)
    fps = float(anm.frame_rate)
    joints = {j.name: j for j in skeleton.joints}
    tracks: list[Track] = []
    for bt in anm.tracks:
        j = joints.get(bt.bone)
        if j is None:
            continue
        for kind, ch, rest in (
            ("translation", bt.translation, np.asarray(j.translation, np.float32)),
            ("rotation", bt.rotation, np.asarray(j.rotation, np.float32)),
            ("scale", bt.scale, np.asarray(j.scale, np.float32)),
        ):
            if ch is None or not len(ch.frames):
                continue
            f, v = _strict(ch)
            v = v.astype(np.float32)
            if kind == "translation":
                v = v * CM
            if kind == "rotation":
                v = _fix_signs(v)
            if np.abs(v - rest).max() < REST_TOL or (kind == "rotation" and np.abs(v + rest).max() < REST_TOL):
                continue
            if len(f) > 1 and np.abs(v - v[0]).max() < REST_TOL:
                f, v = f[:1], v[:1]
            else:
                f, v = decimate(f, v, CLIP_TOL)
            tracks.append(Track(j.name, kind, f.astype(np.float32) / fps, v))
    if not tracks:
        return None
    m = re.match(r"^([a-z]{2})\d*_", name)
    return Clip(
        name=name,
        tracks=tracks,
        frame_rate=fps,
        frame_count=anm.frame_count,
        loop=anm.looped,
        index=index,
        category=CLIP_CATEGORY.get(m.group(1), categorize(name)) if m else categorize(name),
        source=name,
    )
