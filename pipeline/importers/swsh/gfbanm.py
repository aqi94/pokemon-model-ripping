"""Sword/Shield `.gfbanm` animation reader (schema: pkZukan/PokeDocs SWSH/Flatbuffers/Animation, verified on real files).

Each bone track holds scale / rotation / translation channels; a channel is a union: 1 fixed, 2 one key per frame,
3 keyed u16 frames, 4 keyed u8 frames. Rotations are the same 48-bit packed quaternions as Trinity (3 x u16).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from ..za import fb
from ..za.trinity import unpack_quaternions


@dataclass
class Channel:
    frames: np.ndarray  # (K,) int32
    values: np.ndarray  # (K, 3) float32 (vector) or (K, 4) float32 quaternion (x, y, z, w)


@dataclass
class BoneTrack:
    bone: str
    scale: Channel | None = None
    rotation: Channel | None = None
    translation: Channel | None = None


@dataclass
class Gfbanm:
    looped: bool
    frame_count: int
    frame_rate: int
    tracks: list[BoneTrack]
    visibility: dict[str, Channel] = field(default_factory=dict)  # mesh/shape name -> (K,1) 0/1 keys


def _channel(t: fb.Table, type_idx: int, rotation: bool) -> Channel | None:
    kind = t.u8(type_idx)
    c = t.table(type_idx + 1) if kind else None
    if c is None:
        return None
    if kind == 1:  # fixed
        pos = c.struct_pos(0)
        if not pos:
            return None
        if rotation:
            raw = np.array([struct.unpack_from("<3H", c.buf, pos)], np.uint16)
            vals = unpack_quaternions(raw)
        else:
            vals = np.array([struct.unpack_from("<3f", c.buf, pos)], np.float32)
        return Channel(np.zeros(1, np.int32), vals)
    if kind == 2:
        frames_v, co_i = None, 0
    else:
        frames_v, co_i = c.vector(0), 1
    cv = c.vector(co_i)
    if cv is None:
        return None
    if rotation:
        vals = unpack_quaternions(np.frombuffer(cv.buf, "<u2", count=cv.count * 3, offset=cv.pos).reshape(-1, 3))
    else:
        vals = np.frombuffer(cv.buf, "<f4", count=cv.count * 3, offset=cv.pos).reshape(-1, 3).astype(np.float32)
    if kind == 2:
        frames = np.arange(len(vals), dtype=np.int32)
    else:
        frames = frames_v.scalars("<u2" if kind == 3 else "u1").astype(np.int32) if frames_v else np.arange(len(vals), dtype=np.int32)
    n = min(len(frames), len(vals))
    return Channel(frames[:n], vals[:n])


def _byte_channel(t: fb.Table) -> Channel | None:
    kind = t.u8(1)
    c = t.table(2) if kind else None
    if c is None:
        return None
    if kind == 1:
        return Channel(np.zeros(1, np.int32), np.array([[c.u8(0)]], np.float32))
    fv = c.vector(0) if kind in (3, 4) else None
    bv = c.vector(1 if kind in (3, 4) else 0)
    if bv is None:
        return None
    vals = np.frombuffer(bv.buf, np.uint8, count=bv.count, offset=bv.pos).astype(np.float32).reshape(-1, 1)
    frames = fv.scalars("<u2" if kind == 3 else "u1").astype(np.int32) if fv else np.arange(len(vals), dtype=np.int32)
    n = min(len(frames), len(vals))
    return Channel(frames[:n], vals[:n])


def parse_gfbanm(data: bytes) -> Gfbanm:
    r = fb.root(data)
    info = r.table(0)
    if info is None:
        raise ValueError("gfbanm without info")
    tracks = []
    sk = r.table(1)
    for t in sk.tables(0) if sk else []:
        tracks.append(
            BoneTrack(
                t.string(0),
                scale=_channel(t, 1, False),
                rotation=_channel(t, 3, True),
                translation=_channel(t, 5, False),
            )
        )
    vis: dict[str, Channel] = {}
    vt = r.table(3)
    for t in vt.tables(0) if vt else []:
        # SkinTrack: name, values (ByteTrack union at fields 1/2)
        ch = _byte_channel(t)
        if ch is not None:
            vis[t.string(0)] = ch
    return Gfbanm(bool(info.u32(0)), int(info.u32(1)), int(info.u32(2)) or 30, tracks, vis)
