"""Readers for GameFreak Trinity model/animation FlatBuffers (Z-A revision).

Field indexes were verified against Pokémon Legends: Z-A files
(pm0001_00_00) and match the Scarlet/Violet layout documented by the
community (gftool / Titan). Only the fields the converter needs are read.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

from . import fb

# --------------------------------------------------------------------------
# TRMDL
# --------------------------------------------------------------------------


@dataclass
class Trmdl:
    meshes: list[str]  # .trmsh filenames (LOD0 first)
    skeleton: str  # .trskl filename
    materials: list[str]  # .trmtr filenames
    lod_mesh_indices: list[list[int]]  # per LOD, indices into `meshes`


def parse_trmdl(data: bytes) -> Trmdl:
    t = fb.root(data)
    meshes = [m.string(0) for m in t.tables(1)]
    skel_t = t.table(2)
    skeleton = skel_t.string(0) if skel_t else ""
    materials = t.strings(3)
    lods = []
    for lod in t.tables(4):
        lods.append([e.u32(0) for e in lod.tables(0)])
    return Trmdl(meshes, skeleton, materials, lods)


# --------------------------------------------------------------------------
# TRMSH / TRMBF
# --------------------------------------------------------------------------

# Vertex usage (TRVertexUsage)
USAGE_POSITION = 1
USAGE_NORMAL = 2
USAGE_TANGENT = 3
USAGE_BINORMAL = 4
USAGE_COLOR = 5
USAGE_TEXCOORD = 6
USAGE_BLEND_INDICES = 7
USAGE_BLEND_WEIGHTS = 8

# Vertex format (TRVertexFormat) -> byte size
FORMAT_BYTES = {20: 4, 22: 4, 36: 4, 37: 4, 39: 8, 43: 8, 48: 8, 51: 12, 52: 16, 54: 16}


@dataclass
class VertexElement:
    stream: int  # which vertex buffer / stride entry
    usage: int
    layer: int
    fmt: int
    offset: int


@dataclass
class VertexDeclaration:
    elements: list[VertexElement]
    strides: list[int]


@dataclass
class MeshPart:
    index_count: int
    index_offset: int
    material: str
    declaration: int


@dataclass
class BoneWeightSummary:
    rig_index: int
    weight: float


@dataclass
class MeshShape:
    name: str  # "<mesh>_shape"
    mesh_name: str  # "<mesh>"
    index_format: int  # 0=u8 1=u16 2=u32
    declarations: list[VertexDeclaration]
    parts: list[MeshPart]
    bone_weights: list[BoneWeightSummary]


@dataclass
class Trmsh:
    shapes: list[MeshShape]
    buffer_file: str


def parse_trmsh(data: bytes) -> Trmsh:
    t = fb.root(data)
    shapes = []
    for m in t.tables(1):
        decls = []
        for d in m.tables(3):
            elements = [
                VertexElement(e.i32(0), e.u32(1), e.i32(2), e.u32(3), e.i32(4)) for e in d.tables(0)
            ]
            strides = [s.i32(0) for s in d.tables(1)]
            decls.append(VertexDeclaration(elements, strides))
        parts = [MeshPart(p.i32(0), p.i32(1), p.string(3), p.i32(4)) for p in m.tables(4)]
        bw = [BoneWeightSummary(b.i32(0), b.f32(1)) for b in m.tables(10)]
        shapes.append(MeshShape(m.string(0), m.string(12), m.i32(2), decls, parts, bw))
    return Trmsh(shapes, t.string(2))


@dataclass
class MeshBuffers:
    index_buffers: list[memoryview]
    vertex_buffers: list[memoryview]


def parse_trmbf(data: bytes) -> list[MeshBuffers]:
    t = fb.root(data)
    out = []
    for b in t.tables(1):
        idx = [_bytes_vec(x, 0) for x in b.tables(0)]
        vtx = [_bytes_vec(x, 0) for x in b.tables(1)]
        out.append(MeshBuffers(idx, vtx))
    return out


def _bytes_vec(t: fb.Table, index: int) -> memoryview:
    v = t.vector(index)
    return v.raw(1) if v else memoryview(b"")


@dataclass
class DecodedVertices:
    count: int
    positions: np.ndarray
    normals: np.ndarray | None = None
    tangents: np.ndarray | None = None  # (V,4)
    binormals: np.ndarray | None = None
    uvs: list[np.ndarray] = field(default_factory=list)
    colors: np.ndarray | None = None
    joints: np.ndarray | None = None  # (V,4) uint16 (rig indices)
    weights: np.ndarray | None = None  # (V,4) float32


def _read_attribute(buf: memoryview, count: int, stride: int, offset: int, fmt: int, usage: int) -> np.ndarray:
    """Return (count, n) float32 (or uint for indices) array for one element."""
    signed = usage in (USAGE_NORMAL, USAGE_TANGENT, USAGE_BINORMAL)
    arr = np.frombuffer(buf, dtype=np.uint8, count=count * stride)
    arr = arr.reshape(count, stride)
    raw = arr[:, offset : offset + FORMAT_BYTES[fmt]]
    raw = np.ascontiguousarray(raw)
    if fmt == 51:  # RGB32F
        return raw.view("<f4").reshape(count, 3)
    if fmt == 48:  # RG32F
        return raw.view("<f4").reshape(count, 2)
    if fmt == 54:  # RGBA32F
        return raw.view("<f4").reshape(count, 4)
    if fmt == 43:  # RGBA16F
        return raw.view("<f2").reshape(count, 4).astype(np.float32)
    if fmt == 39:  # RGBA16 UNORM (or SNORM for normals/tangents)
        u = raw.view("<u2").reshape(count, 4)
        if signed:
            s = u.astype(np.int32)
            s[s >= 0x8000] -= 0x10000
            return np.maximum(-1.0, s.astype(np.float32) / 32767.0)
        return u.astype(np.float32) / 65535.0
    if fmt == 20:  # RGBA8 UNORM (SNORM for normals/tangents)
        u = raw.reshape(count, 4)
        if signed:
            s = u.astype(np.int16)
            s[s >= 128] -= 256
            return np.maximum(-1.0, s.astype(np.float32) / 127.0)
        return u.astype(np.float32) / 255.0
    if fmt == 22:  # RGBA8 UINT (blend indices)
        return raw.reshape(count, 4).astype(np.uint16)
    if fmt == 52:  # RGBA32 UINT
        return raw.view("<u4").reshape(count, 4)
    if fmt in (36, 37):  # R32 UINT / INT
        return raw.view("<u4" if fmt == 36 else "<i4").reshape(count, 1)
    raise ValueError(f"unsupported vertex format {fmt}")


def decode_vertices(decl: VertexDeclaration, buffers: MeshBuffers) -> DecodedVertices:
    pos_el = next((e for e in decl.elements if e.usage == USAGE_POSITION and e.layer == 0), None)
    if pos_el is None:
        raise ValueError("vertex declaration has no POSITION")
    pos_stride = decl.strides[pos_el.stream]
    pos_buf = buffers.vertex_buffers[pos_el.stream]
    count = len(pos_buf) // pos_stride
    out = DecodedVertices(count=count, positions=np.zeros((count, 3), np.float32))
    uvs: dict[int, np.ndarray] = {}
    for e in decl.elements:
        if e.fmt not in FORMAT_BYTES:
            continue
        stride = decl.strides[e.stream]
        buf = buffers.vertex_buffers[e.stream]
        if len(buf) < count * stride:
            raise ValueError("vertex buffer shorter than declared")
        vals = _read_attribute(buf, count, stride, e.offset, e.fmt, e.usage)
        if e.usage == USAGE_POSITION and e.layer == 0:
            out.positions = np.ascontiguousarray(vals[:, :3], np.float32)
        elif e.usage == USAGE_NORMAL and e.layer == 0:
            out.normals = np.ascontiguousarray(vals[:, :3], np.float32)
        elif e.usage == USAGE_TANGENT and e.layer == 0:
            t = np.ones((count, 4), np.float32)
            t[:, : min(4, vals.shape[1])] = vals[:, :4]
            out.tangents = t
        elif e.usage == USAGE_BINORMAL or (e.usage == USAGE_TANGENT and e.layer == 1):
            out.binormals = np.ascontiguousarray(vals[:, :3], np.float32)
        elif e.usage == USAGE_COLOR and e.layer == 0:
            c = np.ones((count, 4), np.float32)
            c[:, : vals.shape[1]] = vals
            out.colors = c
        elif e.usage == USAGE_TEXCOORD:
            uvs[e.layer] = np.ascontiguousarray(vals[:, :2], np.float32)
        elif e.usage == USAGE_BLEND_INDICES and e.layer == 0:
            out.joints = np.ascontiguousarray(vals[:, :4]).astype(np.uint16)
        elif e.usage == USAGE_BLEND_WEIGHTS and e.layer == 0:
            w = np.ascontiguousarray(vals[:, :4], np.float32)
            w = np.maximum(w, 0.0)
            s = w.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            out.weights = w / s
    out.uvs = [uvs[k] for k in sorted(uvs)]
    return out


def read_indices(shape: MeshShape, buffers: MeshBuffers, part: MeshPart) -> np.ndarray:
    width = 1 << shape.index_format
    dtype = {1: np.uint8, 2: np.uint16, 4: np.uint32}[width]
    buf = buffers.index_buffers[0]
    start = part.index_offset * width
    arr = np.frombuffer(buf, dtype=dtype, count=part.index_count, offset=start)
    return arr.astype(np.uint32)


# --------------------------------------------------------------------------
# TRSKL
# --------------------------------------------------------------------------


@dataclass
class TransformNode:
    name: str
    scale: tuple[float, float, float]
    rotation_euler: tuple[float, float, float]  # radians, XYZ extrinsic
    translation: tuple[float, float, float]
    parent: int
    joint_info: int  # index into Trskl.joints (rig/palette index), -1 if none
    scale_pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotate_pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class JointInfo:
    segment_scale_compensate: bool
    influence_skinning: bool
    inverse_bind: np.ndarray | None  # 4x4 row-major (row vectors: [axisX; axisY; axisZ; pos])


@dataclass
class Trskl:
    nodes: list[TransformNode]
    joints: list[JointInfo]
    palette_offset: int


def _vec3(t: fb.Table, index: int, default=(0.0, 0.0, 0.0)) -> tuple[float, float, float]:
    v = t.floats(index, 3)
    return tuple(v) if v else default


def parse_trskl(data: bytes) -> Trskl:
    t = fb.root(data)
    nodes = []
    for n in t.tables(1):
        srt = n.table(1)
        scale, rot, trans = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        if srt:
            scale = _vec3(srt, 0, scale)
            rot = _vec3(srt, 1, rot)
            trans = _vec3(srt, 2, trans)
        nodes.append(
            TransformNode(
                name=n.string(0),
                scale=scale,
                rotation_euler=rot,
                translation=trans,
                parent=n.i32(4, -1),
                joint_info=n.i32(5, -1),
                scale_pivot=_vec3(n, 2),
                rotate_pivot=_vec3(n, 3),
            )
        )
    joints = []
    for j in t.tables(2):
        m = j.table(2)
        inv = None
        if m:
            rows = [m.floats(i, 3) for i in range(4)]
            if all(r is not None for r in rows):
                inv = np.eye(4, dtype=np.float32)
                for i in range(4):
                    inv[i, :3] = rows[i]
        joints.append(JointInfo(j.bool(0), j.bool(1, True), inv))
    return Trskl(nodes, joints, t.i32(4, -1))


def euler_xyz_to_quat(rx: float, ry: float, rz: float) -> tuple[float, float, float, float]:
    """Extrinsic X->Y->Z rotation (q = qZ * qY * qX), returns (x, y, z, w)."""
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    qx = (sx, 0.0, 0.0, cx)
    qy = (0.0, sy, 0.0, cy)
    qz = (0.0, 0.0, sz, cz)
    return quat_mul(qz, quat_mul(qy, qx))


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


# --------------------------------------------------------------------------
# TRMTR / TRMMT
# --------------------------------------------------------------------------


@dataclass
class TextureBinding:
    slot: str  # e.g. BaseColorMap
    file: str  # e.g. pm0001_00_00_body_a_alb.bntx
    index: int


WRAP_NAMES = {0: "repeat", 1: "clamp", 6: "mirror", 7: "clamp"}  # 7 = mirror once: identity inside [0,1]


@dataclass
class TrMaterial:
    name: str
    shader: str
    shader_flags: dict[str, str]
    textures: list[TextureBinding]
    samplers: list[tuple[str, str]]  # (wrapU, wrapV) parallel to `textures`
    floats: dict[str, float]
    colors: dict[str, tuple[float, float, float, float]]
    ints: dict[str, int]
    alpha_type: str
    cull: int  # from "IntExtra" table field 1 (-1 unknown)


def parse_trmtr(data: bytes) -> list[TrMaterial]:
    t = fb.root(data)
    mats = []
    for m in t.tables(1):
        shaders = m.tables(1)
        shader = shaders[0].string(0) if shaders else ""
        flags = {}
        if shaders:
            for sv in shaders[0].tables(1):
                flags[sv.string(0)] = sv.string(1)
        textures = [TextureBinding(x.string(0), x.string(1), x.u32(2)) for x in m.tables(2)]
        samplers = [(WRAP_NAMES.get(s.u32(9), "repeat"), WRAP_NAMES.get(s.u32(10), "repeat")) for s in m.tables(3)]
        floats = {x.string(0): x.f32(1) for x in m.tables(4)}
        colors = {}
        for x in m.tables(7):
            v = x.floats(1, 4)
            colors[x.string(0)] = tuple(v) if v else (0.0, 0.0, 0.0, 0.0)
        ints = {x.string(0): x.i32(1) for x in m.tables(9)}
        extra = m.table(14)
        cull = extra.i32(1, -1) if extra else -1
        mats.append(TrMaterial(m.string(0), shader, flags, textures, samplers, floats, colors, ints, m.string(15), cull))
    return mats


@dataclass
class MaterialVariantSet:
    name: str  # "normal", "rare"
    material_files: list[str]
    shape_materials: dict[str, str]  # mesh shape name -> material name (if present)


def parse_trmmt(data: bytes) -> list[MaterialVariantSet]:
    t = fb.root(data)
    out = []
    for v in t.tables(2):
        files = v.strings(1)
        # Field 2 lists the mesh shapes this variant applies to; its second
        # field is not a string (packed flags), so only shape names are kept.
        mapping = {e.string(0): "" for e in v.tables(2)}
        out.append(MaterialVariantSet(v.string(0), files, mapping))
    return out


# --------------------------------------------------------------------------
# TRANM
# --------------------------------------------------------------------------

TRACK_FIXED = 1
TRACK_FRAMED = 2
TRACK_KEYED16 = 3
TRACK_KEYED8 = 4


@dataclass
class Channel:
    frames: np.ndarray  # (K,) int32
    values: np.ndarray  # (K, 3) or (K, 4) float32


@dataclass
class BoneTrack:
    bone: str
    scale: Channel | None
    rotation: Channel | None
    translation: Channel | None


@dataclass
class Tranm:
    looped: bool
    frame_count: int
    frame_rate: int
    tracks: list[BoneTrack]


def unpack_quaternions(packed: np.ndarray) -> np.ndarray:
    """packed: (K, 3) uint16 -> (K, 4) float32 quaternions (x, y, z, w)."""
    k = packed.shape[0]
    w = packed[:, 0].astype(np.uint64) | (packed[:, 1].astype(np.uint64) << 16) | (packed[:, 2].astype(np.uint64) << 32)
    scale = (math.pi / 2) / 32767.0
    q1 = ((w >> 3) & 0x7FFF).astype(np.float32) * scale - math.pi / 4
    q2 = ((w >> 18) & 0x7FFF).astype(np.float32) * scale - math.pi / 4
    q3 = ((w >> 33) & 0x7FFF).astype(np.float32) * scale - math.pi / 4
    missing_index = (w & 3).astype(np.int64)
    negative = (w & 4) != 0
    sq = q1 * q1 + q2 * q2 + q3 * q3
    missing = np.sqrt(np.maximum(0.0, 1.0 - sq)).astype(np.float32)
    out = np.empty((k, 4), np.float32)
    comps = np.stack([q1, q2, q3], axis=1)
    for i in range(k):
        mi = missing_index[i]
        row = list(comps[i])
        row.insert(mi, missing[i])
        out[i] = row
    out[negative] *= -1.0
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return out / norm


def _channel(track: fb.Table, type_index: int, rotation: bool) -> Channel | None:
    kind = track.u8(type_index)
    if not kind:
        return None
    content = track.table(type_index + 1)
    if content is None:
        return None
    width = 3 if not rotation else 4
    if rotation:
        elem = np.dtype([("x", "<u2"), ("y", "<u2"), ("z", "<u2")])
    else:
        elem = np.dtype("<f4")

    def values_from(vec: fb.Vector | None) -> np.ndarray:
        if vec is None or vec.count == 0:
            return np.zeros((0, width), np.float32)
        if rotation:
            s = vec.structs(elem)
            packed = np.stack([s["x"], s["y"], s["z"]], axis=1)
            return unpack_quaternions(packed)
        return np.frombuffer(vec.buf, dtype="<f4", count=vec.count * 3, offset=vec.pos).reshape(vec.count, 3).astype(np.float32)

    if kind == TRACK_FIXED:
        pos = content.struct_pos(0)
        if not pos:
            return None
        if rotation:
            packed = np.frombuffer(content.buf, dtype="<u2", count=3, offset=pos).reshape(1, 3)
            vals = unpack_quaternions(packed)
        else:
            vals = np.frombuffer(content.buf, dtype="<f4", count=3, offset=pos).reshape(1, 3).astype(np.float32)
        return Channel(np.zeros(1, np.int32), vals)
    if kind == TRACK_FRAMED:
        vals = values_from(content.vector(0))
        return Channel(np.arange(len(vals), dtype=np.int32), vals)
    if kind == TRACK_KEYED16:
        kv = content.vector(0)
        frames = kv.scalars("<u2").astype(np.int32) if kv else np.zeros(0, np.int32)
        vals = values_from(content.vector(1))
    elif kind == TRACK_KEYED8:
        kv = content.vector(0)
        frames = kv.scalars("u1").astype(np.int32) if kv else np.zeros(0, np.int32)
        vals = values_from(content.vector(1))
    else:
        raise ValueError(f"unknown track type {kind}")
    n = min(len(frames), len(vals))
    return Channel(frames[:n], vals[:n]) if n else None


def parse_tranm(data: bytes) -> Tranm:
    t = fb.root(data)
    info = t.table(0)
    skel = t.table(1)
    if info is None or skel is None:
        raise ValueError("TRANM missing info/skeletal tables")
    tracks = []
    for tr in skel.tables(0):
        tracks.append(
            BoneTrack(
                bone=tr.string(0),
                scale=_channel(tr, 1, rotation=False),
                rotation=_channel(tr, 3, rotation=True),
                translation=_channel(tr, 5, rotation=False),
            )
        )
    return Tranm(
        looped=info.u32(0) != 0,
        frame_count=info.u32(1),
        frame_rate=info.u32(2) or 30,
        tracks=tracks,
    )


# --------------------------------------------------------------------------
# TRACM (animation config): per-shape visibility tracks
# --------------------------------------------------------------------------


@dataclass
class VisibilityTrack:
    shape: str  # mesh shape name, e.g. cup_mesh_shape or cup_mesh_shape_lod1
    frames: np.ndarray  # (K,) int32
    values: np.ndarray  # (K,) bool


def parse_tracm_visibility(data: bytes) -> list[VisibilityTrack]:
    """Shape visibility channels: root[1] shapes, shape[5] -> [2] union {0: type, 1: data}.

    Types: 1 fixed (data[0] u8), 2 per-frame bits (data[0] byte vector, one
    bit per frame), 3/4 keyed (data[0] u16/u8 frames, data[1] bits).
    """
    t = fb.root(data)
    info = t.table(0)
    frame_count = info.u32(1) if info else 0
    out: list[VisibilityTrack] = []
    for shape in t.tables(1):
        vis = shape.table(5)
        union = vis.table(2) if vis else None
        if union is None:
            continue
        kind = union.u8(0)
        dat = union.table(1)
        if not kind or dat is None:
            continue
        if kind == TRACK_FIXED:
            frames = np.zeros(1, np.int32)
            values = np.array([dat.u8(0) != 0])
        elif kind in (TRACK_FRAMED, TRACK_KEYED16, TRACK_KEYED8):
            bits_vec = dat.vector(0 if kind == TRACK_FRAMED else 1)
            if bits_vec is None:
                continue
            bits = np.frombuffer(bits_vec.buf, dtype=np.uint8, count=bits_vec.count, offset=bits_vec.pos)
            if kind == TRACK_FRAMED:
                n = min(frame_count, bits_vec.count * 8) if frame_count else bits_vec.count * 8
                frames = np.arange(n, dtype=np.int32)
            else:
                kv = dat.vector(0)
                frames = kv.scalars("<u2" if kind == TRACK_KEYED16 else "u1").astype(np.int32) if kv else np.zeros(0, np.int32)
            idx = np.arange(len(frames))
            if len(bits):
                values = ((bits[np.minimum(idx >> 3, len(bits) - 1)] >> (idx & 7)) & 1) != 0
            else:
                values = np.zeros(len(frames), bool)
        else:
            continue
        if len(frames):
            out.append(VisibilityTrack(shape.string(0), frames, values.astype(bool)))
    return out
