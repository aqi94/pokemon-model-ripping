"""Brilliant Diamond / Shining Pearl importer: Unity 2019.4 AssetBundle -> common Model.

A trainer bundle is self-contained: the prefab hierarchy (Transforms), SkinnedMeshRenderers with Mesh + Materials,
Texture2Ds (ASTC) and generic (bone-path) AnimationClips. Pokémon split the same data over several bundles
(`battle/pm*` prefab, `common/pm*` meshes+clips, `common/pm*_VV` colour textures); `import_bundles` takes them all.

Unity is left-handed (+X right, +Y up, +Z forward). Characters face +Z in both Unity and glTF, so the handedness is
fixed by mirroring X: positions/normals negate x, quaternions (x,y,z,w) -> (x,-y,-z,w), tangents (-x,y,z,-w),
triangle winding reverses, 4x4 matrices become S M S with S = diag(-1,1,1,1). UVs are bottom-left origin: V flips.
"""

from __future__ import annotations

import hashlib
import re
import zlib

import numpy as np

from ...common.model import Clip, Joint, Material, Mesh, Model, Primitive, Skeleton, TextureImage, Track
from ..za.importer import categorize, decimate

REST_TOL = 1e-4
CURVE_TOL = 1e-3
# AnimationUtility binding attributes for typeID 4 (Transform)
ATTR_POS, ATTR_ROT, ATTR_SCALE = 1, 2, 3
TEX_SLOTS = {"_MainTex": "col", "_BumpTex": "nor", "_MaskTex": "msk", "_ComplexTex": "uti",
             "_Col0Tex": "col", "_NormalMapTex": "nor"}  # the last two: the Pokémon shader of props such as Poké Fan F's Bonsly dolls


def _load(path: str, deps: list[str] | None = None) -> list:
    """Objects of the bundle at `path`. `deps` are loaded alongside so PPtrs into other bundles (shared meshes of the
    player's field models) resolve, but only the primary bundle's own objects are returned."""
    import UnityPy

    env = UnityPy.load(path, *(deps or []))
    primary = next(iter(env.files.values()))
    sf = next(f for f in primary.files.values() if hasattr(f, "objects"))
    return list(sf.objects.values())


def externals(path: str) -> list[str]:
    """CAB names (lower case) of the bundles the one at `path` points into."""
    import UnityPy

    env = UnityPy.load(path)
    primary = next(iter(env.files.values()))
    sf = next(f for f in primary.files.values() if hasattr(f, "objects"))
    return [x.path.split("/")[-1].lower() for x in sf.externals]


# ---------------------------------------------------------------- coordinate conversion
def _p(v) -> tuple[float, float, float]:
    return (-float(v.x), float(v.y), float(v.z))


def _q(q) -> tuple[float, float, float, float]:
    return (float(q.x), -float(q.y), -float(q.z), float(q.w))


def _mat(m) -> np.ndarray:
    a = np.array([[getattr(m, f"e{r}{c}") for c in range(4)] for r in range(4)], np.float64)
    s = np.diag([-1.0, 1.0, 1.0, 1.0])
    return (s @ a @ s).astype(np.float32)


# ---------------------------------------------------------------- hierarchy
class Hierarchy:
    """Transform tree of one bundle: names, parents, TRS and CRC32 path hashes (animation binding keys)."""

    def __init__(self, objs):
        self.tr = {o.path_id: o.read() for o in objs if o.type.name == "Transform"}
        self.name = {pid: t.m_GameObject.read().m_Name for pid, t in self.tr.items()}
        self.parent = {pid: (t.m_Father.path_id if t.m_Father and t.m_Father.path_id in self.tr else 0) for pid, t in self.tr.items()}
        self.roots = [pid for pid in self.tr if not self.parent[pid]]
        self.order: list[int] = []  # parents before children
        children: dict[int, list[int]] = {}
        for pid in self.tr:
            children.setdefault(self.parent[pid], []).append(pid)
        stack = list(reversed(self.roots))
        while stack:
            pid = stack.pop()
            self.order.append(pid)
            stack.extend(reversed(children.get(pid, [])))
        self.path: dict[int, str] = {}
        for pid in self.order:
            par = self.parent[pid]
            self.path[pid] = self.name[pid] if not par else self.path[par] + "/" + self.name[pid]
        # Unity hashes the path relative to the animated root (the root GameObject's own name is not included).
        self.hash: dict[int, int] = {}
        for pid in self.order:
            p = self.path[pid]
            self.hash[zlib.crc32(p.split("/", 1)[1].encode()) if "/" in p else 0] = pid


def _build_skeleton(h: Hierarchy, bones_used: list[int], bind: dict[int, np.ndarray]) -> tuple[Skeleton, dict[int, int]]:
    index = {pid: i for i, pid in enumerate(h.order)}
    names: dict[int, str] = {}
    seen: set[str] = set()
    for pid in h.order:
        n = h.name[pid]
        if n in seen:
            n = h.path[pid].replace("/", ".")
        seen.add(n)
        names[pid] = n
    used = set(bones_used)
    joints = []
    for pid in h.order:
        t = h.tr[pid]
        joints.append(
            Joint(
                name=names[pid],
                parent=index[h.parent[pid]] if h.parent[pid] else -1,
                translation=_p(t.m_LocalPosition),
                rotation=_q(t.m_LocalRotation),
                scale=(float(t.m_LocalScale.x), float(t.m_LocalScale.y), float(t.m_LocalScale.z)),
                inverse_bind=bind.get(pid),
                skinning=pid in used,
            )
        )
    palette = [index[pid] for pid in bones_used]
    return Skeleton(joints=joints, palette=palette), index


# ---------------------------------------------------------------- meshes
def _texture(ptr, cache: dict, textures: dict[str, TextureImage], slot: str) -> str | None:
    if ptr is None or not ptr.path_id:
        return None
    ck = (ptr.file_id, ptr.path_id)
    if ck in cache:
        return cache[ck]
    t = ptr.deref().read()
    key = t.m_Name
    rgba = np.array(t.image.convert("RGBA"), np.uint8)
    textures[key] = TextureImage(key=key, rgba=rgba, srgb=slot != "nor" and slot != "msk" and slot != "uti")
    cache[ck] = key
    return key


def _resize_channel(ch: np.ndarray, w: int, h: int) -> np.ndarray:
    if ch.shape == (h, w):
        return ch.astype(np.float32) / 255.0
    from PIL import Image

    return np.asarray(Image.fromarray(ch).resize((w, h), Image.BILINEAR), np.float32) / 255.0


def bake_tint(albedo: np.ndarray, mask: np.ndarray, primary, secondary, skin, mode: str = "zones") -> np.ndarray:
    """The game's character shader colours a (near-greyscale) albedo by mask region, multiplied in the albedo's own space.

    zones: R = _SkinColor, G = _PrimaryColor, B = _SecondaryColor, each a lerp from the previous result; untouched where
           the mask is black (clothes, shoes, accessories).
    skin:  body/face: the R and B regions carry the pink albedo and take _SkinColor (their _PrimaryColor is a cool shadow
           tint; applying it turned Lucas's face blue).
    hair:  the hair shader (_LIGHTING_HAIR) colours the whole texture with _PrimaryColor (Cynthia blonde, Lucas navy).
    """
    h, w = albedo.shape[:2]
    if mode == "hair":
        tint = np.broadcast_to(np.asarray(primary[:3], np.float32), (h, w, 3))
    elif mode == "skin":
        m = np.maximum(_resize_channel(mask[..., 0], w, h), _resize_channel(mask[..., 2], w, h))[..., None]
        tint = (1.0 - m) + np.asarray(skin[:3], np.float32) * m
    else:
        tint = np.ones((h, w, 3), np.float32)
        for c, col in enumerate((skin, primary, secondary)):
            m = _resize_channel(mask[..., c], w, h)[..., None]
            tint = tint * (1.0 - m) + np.asarray(col[:3], np.float32) * m
    out = albedo.copy()
    out[..., :3] = np.clip(albedo[..., :3].astype(np.float32) * tint, 0, 255).astype(np.uint8)
    return out


def _material(m, cache, textures) -> Material:
    props = m.m_SavedProperties
    tex = {name: v for name, v in props.m_TexEnvs}
    mat = Material(name=m.m_Name, roughness_factor=0.8)
    keys = {}
    for unity_name, slot in TEX_SLOTS.items():
        if unity_name in tex:
            keys[slot] = _texture(tex[unity_name].m_Texture, cache, textures, slot)
    mat.base_color = keys.get("col")
    mat.normal = keys.get("nor")
    kw = " ".join(m.m_ShaderKeywords) if isinstance(m.m_ShaderKeywords, list) else (m.m_ShaderKeywords or "")
    mat.extras["unity"] = {
        "keywords": kw,
        "masks": {s: k for s, k in keys.items() if s in ("msk", "uti") and k},
        "colors": {n: [round(float(c.r), 4), round(float(c.g), 4), round(float(c.b), 4), round(float(c.a), 4)] for n, c in props.m_Colors},
    }
    cols = mat.extras["unity"]["colors"]
    msk = keys.get("msk")
    if mat.base_color and msk and all(k in cols for k in ("_PrimaryColor", "_SecondaryColor", "_SkinColor")):
        mask = textures[msk].rgba
        # A tiny solid-green mask is the shader's "whole texture is _PrimaryColor" placeholder (every hair; Cynthia's black fur,
        # whose _SecondaryColor is gold but shows black in game), so it colours like hair instead of selecting the G zone.
        solid_g = max(mask.shape[:2]) <= 16 and mask[..., 1].min() >= 250 and mask[..., 0].max() <= 5 and mask[..., 2].max() <= 5
        mode = "hair" if "_LIGHTING_HAIR" in kw or solid_g else "zones"  # body/face too: R skin, G primary (the eyebrows), B secondary
        # Materials can share albedo + mask but differ in colours (Scientist's coat vs glasses, Twins' hair vs ribbon, Roughneck's
        # jeans vs metal): the baked texture is keyed by the colours and mode too, or the first material baked wins for all.
        tag = hashlib.md5(repr((mode, msk, [round(c, 3) for n in ("_PrimaryColor", "_SecondaryColor", "_SkinColor") for c in cols[n][:3]])).encode()).hexdigest()[:8]
        key = f"{mat.base_color}_tint_{tag}"
        if key not in textures:
            base = textures[mat.base_color]
            textures[key] = TextureImage(key=key, rgba=bake_tint(base.rgba, textures[msk].rgba, cols["_PrimaryColor"], cols["_SecondaryColor"], cols["_SkinColor"], mode), srgb=True)
        mat.extras["raw_base_color"] = mat.base_color
        mat.base_color = key
    if mat.base_color and "_ALPHABLEND_ON" in kw and textures[mat.base_color].rgba[..., 3].min() < 250:
        mat.alpha_mode = "BLEND"
    return mat



def _soften_lenses(meshes: list[Mesh], materials: dict[str, Material], textures: dict[str, TextureImage]) -> None:
    """Separate lens meshes (`*glassSkin`, `*gogglelensSkin`) whose lens area has a high albedo alpha (Scientist, Collector: ~0.9) are
    clear glass in game, not milky white; give them a copy of the material with 25% of that opacity. Lenses whose alpha is already ~0
    (School Kid) are left alone."""
    import dataclasses

    for mesh in meshes:
        if not re.search(r"(?i)(glass|lens)Skin$", mesh.name):
            continue
        for prim in mesh.primitives:
            mat = materials.get(prim.material)
            if mat is None or not mat.base_color or not prim.uvs or mat.base_color not in textures:
                continue
            tex = textures[mat.base_color].rgba
            hh, ww = tex.shape[:2]
            uv = np.asarray(prim.uvs[0], np.float64) % 1.0
            xs, ys = np.clip((uv[:, 0] * ww).astype(int), 0, ww - 1), np.clip((uv[:, 1] * hh).astype(int), 0, hh - 1)
            if tex[ys, xs, 3].mean() / 255.0 <= 0.6:
                continue
            key = f"{mat.base_color}_lens"
            if key not in textures:
                t = tex.copy()
                t[..., 3] = (t[..., 3].astype(np.float32) * 0.25).astype(np.uint8)
                textures[key] = TextureImage(key=key, rgba=t, srgb=True)
            new = dataclasses.replace(mat, name=f"{mat.name}_lens", base_color=key, alpha_mode="BLEND", extras=dict(mat.extras))
            materials[new.name] = new
            prim.material = new.name


def _mesh_primitives(smr, get_material, palette_of: dict[int, int]) -> list[Primitive]:
    from UnityPy.helpers.MeshHelper import MeshHandler

    mesh = smr.m_Mesh.deref().read()
    mh = MeshHandler(mesh)
    mh.process()
    pos = np.asarray(mh.m_Vertices, np.float32).reshape(-1, 3)
    nrm = np.asarray(mh.m_Normals, np.float32)[:, :3] if mh.m_Normals is not None else None
    tan = np.asarray(mh.m_Tangents, np.float32) if mh.m_Tangents is not None else None
    uv = np.asarray(mh.m_UV0, np.float32) if mh.m_UV0 is not None else None
    bi = np.asarray(mh.m_BoneIndices, np.int64) if mh.m_BoneIndices is not None else None
    bw = np.asarray(mh.m_BoneWeights, np.float32) if mh.m_BoneWeights is not None else None
    if bi is not None and bw is None:  # one bone per vertex: Unity stores indices without weights
        bw = np.ones(bi.shape, np.float32)
    if bi is not None and bw is not None and bi.shape[1] < 4:  # 1/2 bones-per-vertex meshes
        pad = 4 - bi.shape[1]
        bi = np.pad(bi, ((0, 0), (0, pad)))
        bw = np.pad(bw[:, : bi.shape[1] - pad], ((0, 0), (0, pad)))
    if bi is None and len(smr.m_Bones) > 0:
        # Rigid single-bone parts (goggle lens, glasses): a bind pose but no per-vertex weights; they follow bone 0 entirely.
        bi = np.zeros((len(pos), 4), np.int64)
        bw = np.tile(np.array([1, 0, 0, 0], np.float32), (len(pos), 1))
    mirror = np.array([-1.0, 1.0, 1.0], np.float32)
    pos = pos * mirror
    if nrm is not None:
        nrm = nrm * mirror
    if tan is not None:
        tan = tan * np.array([-1.0, 1.0, 1.0, -1.0], np.float32)
    if uv is not None:
        uv = uv * np.array([1.0, -1.0], np.float32) + np.array([0.0, 1.0], np.float32)
    remap = None
    if bi is not None:
        local = [b.path_id for b in smr.m_Bones]
        remap = np.array([palette_of.get(pid, 0) for pid in local], np.uint16)  # null bone slots -> palette 0
    prims = []
    mat_ptrs = list(smr.m_Materials)
    for si, tris in enumerate(mh.get_triangles()):
        t = np.asarray(tris, np.int64).reshape(-1, 3)
        if not len(t):
            continue
        used, inv = np.unique(t, return_inverse=True)
        idx = inv.reshape(-1, 3)[:, ::-1].reshape(-1).astype(np.uint32)  # mirrored: reverse winding
        mat_name = get_material(mat_ptrs[min(si, len(mat_ptrs) - 1)])
        prim = Primitive(material=mat_name, indices=idx, positions=pos[used])
        if nrm is not None:
            prim.normals = nrm[used]
        if tan is not None:
            prim.tangents = tan[used]
        if uv is not None:
            prim.uvs = [uv[used]]
        if bi is not None and remap is not None and bw is not None:
            prim.joints = remap[bi[used]]
            w = bw[used]
            s = w.sum(axis=1, keepdims=True)
            prim.weights = np.where(s > 0, w / np.maximum(s, 1e-8), np.array([1, 0, 0, 0], np.float32))
        prims.append(prim)
    return prims


# ---------------------------------------------------------------- animation
def _streamed(clip) -> dict[int, tuple[list[float], list[float]]]:
    """Streamed curve keys: frame = (time f32, key count u32, count x (curve index u32, 4 coeff f32 [slope.. , value]))."""
    data = np.asarray(clip.m_MuscleClip.m_Clip.data.m_StreamedClip.data, np.uint32)
    f = data.view(np.float32)
    curves: dict[int, tuple[list[float], list[float]]] = {}
    i, n = 0, len(data)
    while i + 1 < n:
        t, cnt = float(f[i]), int(data[i + 1])
        i += 2
        for _ in range(cnt):
            idx, val = int(data[i]), float(f[i + 4])
            i += 5
            if np.isfinite(t):
                ts, vs = curves.setdefault(idx, ([], []))
                t = max(t, 0.0)  # the first frame is stamped -FLT_MAX: it holds the initial value
                if ts and t <= ts[-1]:
                    vs[-1] = val
                else:
                    ts.append(t)
                    vs.append(val)
    return curves


def _clip_curves(clip) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    mc = clip.m_MuscleClip.m_Clip.data
    streamed_n = int(mc.m_StreamedClip.curveCount)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for idx, (ts, vs) in _streamed(clip).items():
        out[idx] = (np.asarray(ts, np.float32), np.asarray(vs, np.float32))
    dense = mc.m_DenseClip
    if dense.m_CurveCount and dense.m_FrameCount:
        arr = np.asarray(dense.m_SampleArray, np.float32).reshape(dense.m_FrameCount, dense.m_CurveCount)
        ts = (float(dense.m_BeginTime) + np.arange(dense.m_FrameCount) / float(dense.m_SampleRate)).astype(np.float32)
        for c in range(dense.m_CurveCount):
            out[streamed_n + c] = (ts, arr[:, c].copy())
    base = streamed_n + int(dense.m_CurveCount)
    for c, v in enumerate(mc.m_ConstantClip.data):
        out[base + c] = (np.zeros(1, np.float32), np.asarray([v], np.float32))
    return out


def _quat_signs(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] *= -1.0
    return out


def _clip_tracks(clip, h: Hierarchy, index: dict[int, int], skeleton: Skeleton) -> list[Track]:
    curves = _clip_curves(clip)
    tracks: list[Track] = []
    slot = 0
    for g in clip.m_ClipBindingConstant.genericBindings:
        if g.typeID != 4:
            slot += 1
            continue
        dim = 4 if g.attribute == ATTR_ROT else 3
        comps = [curves.get(slot + k) for k in range(dim)]
        slot += dim
        pid = h.hash.get(int(g.path))
        if pid is None or any(c is None for c in comps) or g.attribute not in (ATTR_POS, ATTR_ROT, ATTR_SCALE):
            continue
        times = np.unique(np.concatenate([c[0] for c in comps]))
        vals = np.stack([np.interp(times, c[0], c[1]) if len(c[0]) > 1 else np.full(len(times), c[1][0]) for c in comps], axis=1).astype(np.float32)
        joint = skeleton.joints[index[pid]]
        if g.attribute == ATTR_POS:
            kind, rest = "translation", np.asarray(joint.translation, np.float32)
            vals = vals * np.array([-1.0, 1.0, 1.0], np.float32)
        elif g.attribute == ATTR_SCALE:
            kind, rest = "scale", np.asarray(joint.scale, np.float32)
        else:
            kind, rest = "rotation", np.asarray(joint.rotation, np.float32)
            vals = vals * np.array([1.0, -1.0, -1.0, 1.0], np.float32)
            vals = _quat_signs(vals / np.maximum(np.linalg.norm(vals, axis=1, keepdims=True), 1e-8))
        same = np.abs(vals - rest).max() < REST_TOL or (kind == "rotation" and np.abs(vals + rest).max() < REST_TOL)
        if same:
            continue
        if np.abs(vals - vals[0]).max() < REST_TOL:
            tracks.append(Track(joint.name, kind, times[:1].copy(), vals[:1].copy()))
            continue
        t2, v2 = decimate(times, vals, CURVE_TOL)
        tracks.append(Track(joint.name, kind, t2.astype(np.float32), v2.astype(np.float32)))
    return tracks


# Eye gaze: each trainer bundle has a `UVConnection` MonoBehaviour whose Property00/01 point at the Leye/Reye materials (by material index into
# a SkinnedMeshRenderer); the clips animate `PropertyNN.offsetUV.x/.y` as typeID 114 bindings (attribute = crc32 of that path). The pupils sit in
# a 2x4 atlas, so the offset moves them (verified against the Bulbapedia VS images; the eyeball bones carry no gaze).
UV_PROPS = {zlib.crc32(f"Property0{n}.offsetUV.{a}".encode()): (n, a) for n in range(4) for a in "xy"}


def _uv_targets(objs, smr_by_id: dict, get_material) -> dict[int, str]:
    """UVConnection property index -> exported material name."""
    out: dict[int, str] = {}
    for o in objs:
        if o.type.name != "MonoBehaviour":
            continue
        try:
            t = o.read_typetree()
        except Exception:
            continue
        if not isinstance(t, dict) or "Property00" not in t:
            continue
        for n in range(4):
            try:
                p = t.get(f"Property0{n}")
                if not p or p.get("propertyName") != "_UVOffset":
                    continue
                smr = smr_by_id.get(p.get("renderer", {}).get("m_PathID"))
                mats = list(smr.m_Materials) if smr is not None else []
                if smr is not None and 0 <= int(p.get("materialIndex", -1)) < len(mats):
                    out[n] = get_material(mats[int(p["materialIndex"])])
            except (AttributeError, TypeError, ValueError):  # a different layout (player bundles): no gaze data
                continue
    return out


def _uv_offset_tracks(clip, targets: dict[int, str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not targets:
        return {}
    curves = _clip_curves(clip)
    slot = 0
    found: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for g in clip.m_ClipBindingConstant.genericBindings:
        if g.typeID == 4:
            slot += 4 if g.attribute == ATTR_ROT else 3
            continue
        key = UV_PROPS.get(int(g.attribute))
        c = curves.get(slot)
        slot += 1
        if key is not None and key[0] in targets and c is not None:
            found[key] = c
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for n, name in targets.items():
        cx, cy = found.get((n, "x")), found.get((n, "y"))
        if cx is None and cy is None:
            continue
        times = np.unique(np.concatenate([c[0] for c in (cx, cy) if c is not None]))

        def at(c):
            if c is None:
                return np.zeros(len(times), np.float32)
            return np.interp(times, c[0], c[1]).astype(np.float32) if len(c[0]) > 1 else np.full(len(times), c[1][0], np.float32)

        vals = np.stack([at(cx), at(cy)], axis=1)
        if np.abs(vals).max() < 1e-4:
            continue
        if np.abs(vals - vals[0]).max() < 1e-4:
            times, vals = times[:1], vals[:1]
        out[name] = (times.astype(np.float32), vals.astype(np.float32))
    return out


def _clips(objs, h: Hierarchy, index: dict[int, int], skeleton: Skeleton, uv_targets: dict[int, str] | None = None) -> list[Clip]:
    clips = []
    for o in objs:
        if o.type.name != "AnimationClip":
            continue
        c = o.read()
        rate = float(c.m_SampleRate) or 30.0
        stop = float(c.m_MuscleClip.m_StopTime)
        clips.append(
            Clip(
                name=c.m_Name,
                tracks=_clip_tracks(c, h, index, skeleton),
                uv_offset=_uv_offset_tracks(c, uv_targets or {}),
                frame_rate=rate,
                frame_count=int(round(stop * rate)) + 1,
                loop=bool(c.m_MuscleClip.m_LoopTime),
                category=categorize(c.m_Name),
                source=c.m_Name,
            )
        )
    clips.sort(key=lambda c: c.name)
    for i, c in enumerate(clips):
        c.index = i
    return clips


# ---------------------------------------------------------------- model
def import_bundle(path: str, model_id: str, display_name: str, kind: str = "trainer", game: str = "bdsp", with_anim: bool = True, deps: list[str] | None = None) -> Model:
    objs = _load(path, deps)
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

    textures: dict[str, TextureImage] = {}
    cache: dict = {}
    materials: dict[str, Material] = {}
    by_ptr: dict = {}

    def get_material(ptr) -> str:
        ck = (ptr.file_id, ptr.path_id)
        if ck not in by_ptr:
            m = ptr.deref().read()
            mat = _material(m, cache, textures)
            name = m.m_Name
            if name in materials:
                name = f"{name}.{len(materials)}"
            mat.name = name
            materials[name] = mat
            by_ptr[ck] = name
        return by_ptr[ck]

    meshes: list[Mesh] = []
    for smr in smrs:
        prims = _mesh_primitives(smr, get_material, palette_of)
        if prims:
            go = smr.m_GameObject.read()
            meshes.append(Mesh(name=go.m_Name, primitives=prims, visible=bool(go.m_IsActive) and bool(smr.m_Enabled)))

    _soften_lenses(meshes, materials, textures)

    smr_by_id = {o.path_id: o.read() for o in objs if o.type.name == "SkinnedMeshRenderer"}
    clips = _clips(objs, h, index, skeleton, _uv_targets(objs, smr_by_id, get_material)) if with_anim else []
    default = next((c.name for c in clips if re.search(r"wait|idle", c.name)), clips[0].name if clips else None)
    return Model(
        id=model_id,
        game=game,
        display_name=display_name,
        kind=kind,
        skeleton=skeleton,
        meshes=meshes,
        materials=materials,
        textures=textures,
        clips=clips,
        default_clip=default,
    )
