"""Reader for Let's Go Pikachu/Eevee `.gfbmdl` models (Game Freak's pre-Trinity FlatBuffers format).

Field indexes were verified on real LGPE files (see the format notes in the importer). Nothing is decoded
to the IR here; `parse_gfbmdl` returns plain numpy/dataclass data in the file's own units (centimetres).

Root table:  0 version, 1 bounds (6 f32), 2 texture names, 3 shader names, 6 materials, 7 groups,
             8 meshes, 9 bones.
Group:       0 bone index, 1 mesh index, 2 f32 (unexplained; ledger `gfbmdl:/7[]/2`), 3 u32 (sometimes).
Mesh:        0 polygons, 1 vertex attributes, 2 interleaved vertex buffer (bytes).
Polygon:     0 material index, 1 u16 indices (triangle list).
Attribute:   0 usage, 1 component type, 2 component count (packed in this order in the vertex buffer).
Bone:        0 name, 2 parent (-1 root), 5 scale, 6 euler XYZ radians (q = qZ*qY*qX), 7 translation.
Material:    0 name, 1 shader, 6 blend (field maps: 5 on water/glass), 7 cull (0 none = two-sided foliage, 1 back),
             11 textures {name, index into root[2], sampler {0 filter?, 1 wrap U, 2 wrap V, 3 wrap W, 4/5 f32}},
             12 switches {name, u8}, 13 floats {name, f32}, 14 colours {name, vec3},
             20 render states {0 [{name, u32}] toggles (1025 = on: DiscardEnable, CastShadow, FogEnable...),
             1 [{name, u32}] values (CullMode 0 back / 2 none, BlendMode, Tex0N_UV / UV_tex0N = UV set per texture...)}.
             Wrap enum as SwSh: 0 repeat, 1 clamp, 2 mirror, 3 border (clamp).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from ..za import fb

# vertex attribute usages
POSITION, NORMAL, TANGENT, UV0, UV1, UV2, COLOR0, COLOR1, BONE_INDEX, BONE_WEIGHT = 0, 1, 2, 3, 4, 5, 7, 8, 11, 12
_USAGE_FROM_RAW = {0: POSITION, 1: NORMAL, 2: TANGENT, 3: UV0, 4: UV1, 5: UV2, 7: COLOR0, 8: COLOR1, 11: BONE_INDEX, 12: BONE_WEIGHT}
WRAP = {0: "repeat", 1: "clamp", 2: "mirror", 3: "clamp"}
# component type -> (numpy dtype, bytes)
_TYPES = {0: ("<f4", 4), 1: ("<f2", 2), 3: ("u1", 1), 8: ("u1", 1)}


@dataclass
class GfBone:
    name: str
    parent: int
    scale: tuple[float, float, float]
    euler: tuple[float, float, float]
    translation: tuple[float, float, float]


@dataclass
class GfMaterial:
    name: str
    shader: str
    textures: dict[str, str]  # slot -> texture name (as listed in root[2])
    switches: dict[str, bool] = field(default_factory=dict)
    floats: dict[str, float] = field(default_factory=dict)
    colors: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    samplers: dict[str, tuple[str, str]] = field(default_factory=dict)  # slot -> (wrap U, wrap V)
    blend: int = 0  # field 6
    cull: int = 0  # field 7 (absent = 0)
    render: dict[str, int] = field(default_factory=dict)  # field 20: named render states (both lists merged)


@dataclass
class GfGroup:
    bone: int
    mesh: int
    sort: float  # field 2, meaning unknown (draw-order / distance key?)


@dataclass
class GfPolygon:
    material: int
    indices: np.ndarray  # (N,) uint16


@dataclass
class GfMesh:
    polygons: list[GfPolygon]
    positions: np.ndarray  # (V, 3) float32
    normals: np.ndarray | None
    uvs: list[np.ndarray]  # (V, 2) float32 each (UV0, UV1)
    colors: np.ndarray | None  # (V, 4) float32
    joints: np.ndarray | None  # (V, 4) uint16 (bone indices)
    weights: np.ndarray | None  # (V, 4) float32
    uv_sets: dict[int, np.ndarray] = field(default_factory=dict)  # set index (0 = UV0, 1 = UV1, 2 = UV2) -> (V, 2)


@dataclass
class Gfbmdl:
    version: int
    bounds: tuple[float, ...]
    textures: list[str]
    materials: list[GfMaterial]
    meshes: list[GfMesh]
    bones: list[GfBone]
    groups: list[GfGroup] = field(default_factory=list)


def _vec3(t: fb.Table, i: int) -> tuple[float, float, float]:
    f = t.floats(i, 3)
    return f if f else (0.0, 0.0, 0.0)


def _parse_material(m: fb.Table, textures: list[str]) -> GfMaterial:
    slots: dict[str, str] = {}
    samplers: dict[str, tuple[str, str]] = {}
    for t in m.tables(11):
        idx = t.u32(1)
        slots[t.string(0)] = textures[idx] if idx < len(textures) else ""
        st = t.table(2)
        samplers[t.string(0)] = (WRAP.get(st.u32(1), "repeat"), WRAP.get(st.u32(2), "repeat")) if st else ("repeat", "repeat")
    sw = {t.string(0): bool(t.u8(1)) for t in m.tables(12)}
    fl = {t.string(0): t.f32(1) for t in m.tables(13)}
    col = {t.string(0): _vec3(t, 1) for t in m.tables(14)}
    rs = m.table(20)
    render = {t.string(0): t.u32(1) for k in (0, 1) for t in (rs.tables(k) if rs else [])}
    return GfMaterial(m.string(0), m.string(1), slots, sw, fl, col, samplers, m.u32(6), m.u32(7), render)


def _parse_mesh(m: fb.Table) -> GfMesh:
    attrs = [(a.u32(0), a.u32(1), a.u32(2)) for a in m.tables(1)]
    for _, t, _ in attrs:
        if t not in _TYPES:
            raise ValueError(f"unknown gfbmdl vertex component type {t}")
    stride = sum(_TYPES[t][1] * c for _, t, c in attrs)
    vb = m.vector(2)
    n = vb.count // stride if vb and stride else 0
    raw = np.frombuffer(vb.buf, np.uint8, count=n * stride, offset=vb.pos).reshape(n, stride) if n else np.zeros((0, max(stride, 1)), np.uint8)
    arrays: dict[int, np.ndarray] = {}
    off = 0
    for usage, t, c in attrs:
        dt, size = _TYPES[t]
        chunk = np.ascontiguousarray(raw[:, off : off + size * c]).view(dt).reshape(n, c)
        off += size * c
        if t == 8:
            chunk = chunk.astype(np.float32) / 255.0
        arrays[_USAGE_FROM_RAW.get(usage, usage)] = chunk
    pos = arrays[POSITION].astype(np.float32)
    nrm = arrays.get(NORMAL)
    if nrm is not None:
        nrm = nrm[:, :3].astype(np.float32)
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = nrm / np.maximum(ln, 1e-8)
    uvs = [arrays[u][:, :2].astype(np.float32) for u in (UV0, UV1) if u in arrays]
    col = arrays.get(COLOR0)
    if col is not None:
        col = col.astype(np.float32) / 255.0
    polys = []
    for p in m.tables(0):
        v = p.vector(1)
        idx = v.scalars("<u2").copy() if v else np.zeros(0, np.uint16)
        polys.append(GfPolygon(p.u32(0), idx[: len(idx) // 3 * 3]))
    return GfMesh(
        polygons=polys,
        positions=pos,
        normals=nrm,
        uvs=uvs,
        colors=col,
        joints=arrays[BONE_INDEX].astype(np.uint16) if BONE_INDEX in arrays else None,
        weights=arrays[BONE_WEIGHT].astype(np.float32) if BONE_WEIGHT in arrays else None,
        uv_sets={k: arrays[u][:, :2].astype(np.float32) for k, u in enumerate((UV0, UV1, UV2)) if u in arrays},
    )


def parse_gfbmdl(data: bytes) -> Gfbmdl:
    r = fb.root(data)
    textures = r.strings(2)
    bones = []
    for b in r.tables(9):
        bones.append(
            GfBone(
                name=b.string(0),
                parent=b.i32(2, -1) if b.has(2) else 0,
                scale=b.floats(5, 3) or (1.0, 1.0, 1.0),
                euler=_vec3(b, 6),
                translation=_vec3(b, 7),
            )
        )
    return Gfbmdl(
        version=r.u32(0),
        bounds=r.floats(1, 6) or (0.0,) * 6,
        textures=textures,
        materials=[_parse_material(m, textures) for m in r.tables(6)],
        meshes=[_parse_mesh(m) for m in r.tables(8)],
        bones=bones,
        groups=[GfGroup(t.u32(0), t.u32(1), t.f32(2)) for t in r.tables(7)],
    )
