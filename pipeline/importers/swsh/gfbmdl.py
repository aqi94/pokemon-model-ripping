"""Sword/Shield `.gfbmdl` (Game Freak Switch model) reader.

Field layout from the pkNX `GFBModel.fbs` schema (verified against real files): the file is a plain FlatBuffer without
identifier. `parse_gfbmdl` returns plain dataclasses with decoded vertex arrays; nothing here knows about glTF.
Units are centimetres, Y up, the model faces +Z.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..za import fb

# VertexAttributeType
V_POS, V_NRM, V_TAN, V_UV0, V_UV1, V_UV2, V_UV3, V_COL0, V_COL1, V_COL2, V_COL3 = range(11)
V_JOINTS, V_WEIGHTS = 11, 12
# DataType -> (numpy dtype, normalised)
_FORMATS = {
    0: ("<f4", False),
    1: ("<f2", False),
    2: ("i1", False),
    3: ("u1", False),
    4: ("<i2", False),
    5: ("<u2", False),
    6: ("<i4", False),
    7: ("<u4", False),
    8: ("u1", True),  # FixedPoint: unsigned byte / 255
}
WRAP = {0: "repeat", 1: "clamp", 2: "mirror", 3: "clamp"}
BLEND_NONE, BLEND_ALPHA, BLEND_ADD, BLEND_SUB, BLEND_MUL, BLEND_PREMUL = range(6)


@dataclass
class Bone:
    name: str
    type: int  # 0 transform, 1 joint, 2 locator, 3 transparency group
    parent: int
    visible: bool
    scale: tuple[float, float, float]
    rotation: tuple[float, float, float]  # euler XYZ radians
    translation: tuple[float, float, float]
    rigged: bool  # part of the skin palette (vertex joint ids index the rigged bones in order)


@dataclass
class TexSlot:
    sampler: str  # Col0Tex, NormalMapTex, ...
    file: str  # texture file stem (TextureFiles[index]); "" when the index is out of range
    wrap: tuple[str, str] = ("repeat", "repeat")


@dataclass
class Material:
    name: str
    shader: str
    blend: int
    cull: int
    alpha_test: bool
    textures: list[TexSlot]
    flags: dict[str, bool]
    values: dict[str, float]
    colors: dict[str, tuple[float, float, float]]

    def tex(self, sampler: str) -> TexSlot | None:
        return next((t for t in self.textures if t.sampler == sampler), None)


@dataclass
class Shape:
    attrs: dict[int, np.ndarray]  # vertex attribute type -> (V, count) float32 / uint arrays
    polygons: list[tuple[int, np.ndarray]]  # (material index, triangle indices (N,))


@dataclass
class MeshRef:
    bone: int
    shape: int


@dataclass
class GfbModel:
    texture_files: list[str]
    materials: list[Material]
    meshes: list[MeshRef]
    shapes: list[Shape]
    bones: list[Bone]
    bounds: tuple[float, ...] = ()
    extras: dict = field(default_factory=dict)


def _decode_shape(t: fb.Table) -> Shape:
    attrs = t.tables(1)
    spec = []
    stride = 0
    for a in attrs:
        fmt, cnt = a.u8(1), a.u32(2)
        dt, norm = _FORMATS[fmt]
        spec.append((a.u8(0), np.dtype(dt), norm, cnt, stride))
        stride += np.dtype(dt).itemsize * cnt
    vv = t.vector(2)
    raw = np.frombuffer(vv.buf, np.uint8, count=vv.count, offset=vv.pos) if vv else np.zeros(0, np.uint8)
    n = len(raw) // stride if stride else 0
    rows = raw[: n * stride].reshape(n, stride)
    out: dict[int, np.ndarray] = {}
    for vtype, dt, norm, cnt, off in spec:
        col = np.ascontiguousarray(rows[:, off : off + dt.itemsize * cnt]).view(dt).reshape(n, cnt)
        col = col.astype(np.float32) / 255.0 if norm else (col.astype(np.float32) if dt.kind == "f" else col)
        out[vtype] = col
    polys = []
    for p in t.tables(0):
        iv = p.vector(1)
        idx = iv.scalars("<u2").astype(np.uint32) if iv else np.zeros(0, np.uint32)
        polys.append((p.u32(0), idx))
    return Shape(out, polys)


def parse_gfbmdl(data: bytes) -> GfbModel:
    r = fb.root(data)
    materials = []
    tex_files = r.strings(2)
    for m in r.tables(6):
        slots = []
        for t in m.tables(11):
            idx = t.u32(1)
            st = t.table(2)
            wrap = (WRAP.get(st.u32(1), "repeat"), WRAP.get(st.u32(2), "repeat")) if st else ("repeat", "repeat")
            slots.append(TexSlot(t.string(0), tex_files[idx] if idx < len(tex_files) else "", wrap))
        materials.append(
            Material(
                name=m.string(0),
                shader=m.string(1),
                blend=m.u32(6),
                cull=m.u32(7, 0),
                alpha_test=bool(m.u8(18)),
                textures=slots,
                flags={f.string(0): f.bool(1) for f in m.tables(12)},
                values={f.string(0): f.f32(1) for f in m.tables(13)},
                colors={f.string(0): tuple(f.floats(1, 3) or (1.0, 1.0, 1.0)) for f in m.tables(14)},
            )
        )
    bones = []
    for b in r.tables(9):
        bones.append(
            Bone(
                name=b.string(0),
                type=b.u32(1),
                parent=b.i32(2),
                visible=b.bool(4, True),
                scale=b.floats(5, 3) or (1.0, 1.0, 1.0),
                rotation=b.floats(6, 3) or (0.0, 0.0, 0.0),
                translation=b.floats(7, 3) or (0.0, 0.0, 0.0),
                rigged=b.bool(10, True),
            )
        )
    return GfbModel(
        texture_files=tex_files,
        materials=materials,
        meshes=[MeshRef(m.u32(0), m.u32(1)) for m in r.tables(7)],
        shapes=[_decode_shape(s) for s in r.tables(8)],
        bones=bones,
        bounds=r.floats(1, 6) or (),
    )
