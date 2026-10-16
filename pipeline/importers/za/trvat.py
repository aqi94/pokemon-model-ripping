"""Z-A vertex animation textures (`.trvat`) for the crowd NPCs -> glTF morph targets.

`ik_chara/motion_uq/tr05xx_*/<model>_<idx>_<clip>.trvat` is a tiny FlatBuffer
(`root[1]` = shapes: [0] name "body", [1] position texture, [2] normal texture, [3] tangent
texture, [4] f32 half_range; `root[3]` = 1.0, `root[4]` = 1.618, meaning unknown) next to three BNTX
files `<clip>_body.bntx` (RGBA16F), `_body_normal.bntx` / `_body_tangent.bntx` (BC6H, not read).

Position texture (verified on tr0566..tr0581): width = number of unique vertex columns, height =
frames at 30 fps (the gameplay clips are 60 fps: 116 rows <-> 231 frames). Row f, column c holds
`(position_f[c] - rest[c]) / (2 * half_range) + 0.5` in R,G,B (A = 1), where `rest` is the mesh's
bind-pose vertex. The vertex -> column map is the mesh's second UV set (`TEXCOORD_1.x` = integer
column). Verified by edge-length preservation: only `k = 2 * half_range` (1.05 for tr0566 speak01)
keeps triangle edges at their rest length, and it brings the T-pose arms down to the sides.
"""

from __future__ import annotations

import struct

import numpy as np

from ..bntx.decoder import _deswizzle
from ...common.model import Clip
from . import fb
from .importer import ZipSource

VAT_FPS = 30.0


def read_rgba16f(data: bytes) -> np.ndarray:
    """First texture of a BNTX file whose format is R16G16B16A16 (any type) as (H, W, 4) float32."""
    if data[:4] != b"BNTX":
        raise ValueError("not a BNTX file")
    u32 = lambda o: struct.unpack_from("<I", data, o)[0]  # noqa: E731
    u64 = lambda o: struct.unpack_from("<Q", data, o)[0]  # noqa: E731
    at = u64(u64(0x28))
    fmt = u32(at + 0x1C)
    if fmt >> 8 != 0x15:
        raise NotImplementedError(f"BNTX format 0x{fmt:04X} is not RGBA16")
    w, h, size_range = u32(at + 0x24), u32(at + 0x28), u32(at + 0x34)
    off = u64(u64(at + 0x70))
    surf = np.frombuffer(data, np.uint8)[off:]
    lin = _deswizzle(surf, w * 8, h, 1 << size_range)
    return np.frombuffer(np.ascontiguousarray(lin).tobytes(), dtype="<f2").reshape(h, w, 4).astype(np.float32)


def parse_trvat(data: bytes) -> list[dict]:
    root = fb.root(data)
    return [
        {"name": t.string(0), "position": t.string(1), "normal": t.string(2), "tangent": t.string(3), "half_range": t.f32(4)}
        for t in root.tables(1)
    ]


def vat_deltas(tex: np.ndarray, half_range: float) -> np.ndarray:
    """(F, C, 4) texture -> (F, C, 3) metre offsets from the rest pose."""
    return (tex[..., :3] - 0.5) * (2.0 * half_range)


def attach_vat_clips(model, src: ZipSource) -> int:
    """Add every `.trvat` clip in `src` to `model` as morph-target animation (one target per frame).

    Returns the number of clips added. The mesh must carry the VAT column ids in its second UV
    set. All clips share one target list per mesh: target k belongs to exactly one clip frame.
    """
    names = sorted(n for n in src.list() if n.endswith(".trvat") and "/" not in n)
    if not names:
        return 0
    from .importer import clip_name_from_file  # local: avoid import cycle at module load

    added = 0
    per_mesh: dict[str, list[tuple[str, int, np.ndarray]]] = {}  # mesh -> [(clip name, index, deltas (F,C,3))]
    meta: dict[str, bool] = {}
    for n in names:
        parsed = clip_name_from_file(n[: -len(".trvat")] + ".tranm")
        if not parsed:
            continue
        for shape in parse_trvat(src.read(n)):
            mesh = next((m for m in model.meshes if m.name.startswith(shape["name"]) and not m.lod), None)
            if mesh is None or not mesh.primitives or len(mesh.primitives[0].uvs) < 2:
                continue
            tex = read_rgba16f(src.read(shape["position"].split("/")[-1]))
            per_mesh.setdefault(mesh.name, []).append((parsed[0], parsed[1], vat_deltas(tex, shape["half_range"])))
            meta[parsed[0]] = n.endswith("_loop.trvat") or "_loop" in n
    for mesh_name, items in per_mesh.items():
        mesh = next(m for m in model.meshes if m.name == mesh_name)
        prim = mesh.primitives[0]
        col = np.rint(prim.uvs[1][:, 0]).astype(np.int64)
        total = sum(d.shape[0] for _, _, d in items)
        prim.morph_targets = []
        base = 0
        spans = []
        for name, index, d in items:
            col_c = np.clip(col, 0, d.shape[1] - 1)
            for f in range(d.shape[0]):
                prim.morph_targets.append(np.ascontiguousarray(d[f][col_c], np.float32))
            spans.append((name, index, base, d.shape[0]))
            base += d.shape[0]
        for name, index, start, frames in spans:
            w = np.zeros((frames, total), np.float32)
            w[np.arange(frames), start + np.arange(frames)] = 1.0
            t = (np.arange(frames, dtype=np.float32) / VAT_FPS)
            model.clips.append(
                Clip(
                    name=name,
                    tracks=[],
                    frame_rate=VAT_FPS,
                    frame_count=frames,
                    loop=meta.get(name, False),
                    index=index,
                    category=_category(name),
                    source=f"{index:05d}_{name} (vat)",
                    morph={mesh_name: (t, w)},
                )
            )
            added += 1
    return added


def _category(name: str) -> str:
    from .importer import categorize

    return categorize(name)
