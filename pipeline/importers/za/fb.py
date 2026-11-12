"""Minimal, schema-free FlatBuffers table reader.

GameFreak's Trinity files (.trmdl, .trmsh, .trmbf, .trskl, .trmtr, .tranm, ...)
are plain FlatBuffers without a file identifier. We do not ship generated
schema classes; instead each format module declares which field index holds
what, and reads it through this helper. Field indexes follow the schema order
(index 0 == vtable offset 4).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator

import numpy as np


class FlatBufferError(ValueError):
    pass


@dataclass(frozen=True)
class Table:
    buf: memoryview
    pos: int  # absolute position of the table

    # -- vtable ----------------------------------------------------------
    def _field_offset(self, index: int) -> int:
        """Return absolute position of field `index`, or 0 if absent."""
        vt = self.pos - _i32(self.buf, self.pos)
        vt_size = _u16(self.buf, vt)
        slot = 4 + index * 2
        if slot + 2 > vt_size:
            return 0
        rel = _u16(self.buf, vt + slot)
        return self.pos + rel if rel else 0

    def has(self, index: int) -> bool:
        return self._field_offset(index) != 0

    def field_count(self) -> int:
        vt = self.pos - _i32(self.buf, self.pos)
        return (_u16(self.buf, vt) - 4) // 2

    # -- scalars ---------------------------------------------------------
    def u8(self, index: int, default: int = 0) -> int:
        o = self._field_offset(index)
        return self.buf[o] if o else default

    def i8(self, index: int, default: int = 0) -> int:
        o = self._field_offset(index)
        return struct.unpack_from("<b", self.buf, o)[0] if o else default

    def u16(self, index: int, default: int = 0) -> int:
        o = self._field_offset(index)
        return _u16(self.buf, o) if o else default

    def u32(self, index: int, default: int = 0) -> int:
        o = self._field_offset(index)
        return _u32(self.buf, o) if o else default

    def i32(self, index: int, default: int = 0) -> int:
        o = self._field_offset(index)
        return _i32(self.buf, o) if o else default

    def f32(self, index: int, default: float = 0.0) -> float:
        o = self._field_offset(index)
        return struct.unpack_from("<f", self.buf, o)[0] if o else default

    def bool(self, index: int, default: bool = False) -> bool:
        o = self._field_offset(index)
        return bool(self.buf[o]) if o else default

    # -- structs (inline) ------------------------------------------------
    def struct_pos(self, index: int) -> int:
        """Absolute position of an inline struct field, or 0."""
        return self._field_offset(index)

    def floats(self, index: int, count: int) -> tuple[float, ...] | None:
        o = self._field_offset(index)
        if not o:
            return None
        return struct.unpack_from(f"<{count}f", self.buf, o)

    # -- references ------------------------------------------------------
    def table(self, index: int) -> Table | None:
        o = self._field_offset(index)
        if not o:
            return None
        return Table(self.buf, o + _u32(self.buf, o))

    def string(self, index: int, default: str = "") -> str:
        o = self._field_offset(index)
        if not o:
            return default
        return _string_at(self.buf, o + _u32(self.buf, o))

    def vector(self, index: int) -> Vector | None:
        o = self._field_offset(index)
        if not o:
            return None
        start = o + _u32(self.buf, o)
        return Vector(self.buf, start + 4, _u32(self.buf, start))

    def tables(self, index: int) -> list[Table]:
        v = self.vector(index)
        return list(v.tables()) if v else []

    def strings(self, index: int) -> list[str]:
        v = self.vector(index)
        return list(v.strings()) if v else []


@dataclass(frozen=True)
class Vector:
    buf: memoryview
    pos: int  # position of first element (after the length prefix)
    count: int

    def tables(self) -> Iterator[Table]:
        for i in range(self.count):
            p = self.pos + i * 4
            yield Table(self.buf, p + _u32(self.buf, p))

    def strings(self) -> Iterator[str]:
        for i in range(self.count):
            p = self.pos + i * 4
            yield _string_at(self.buf, p + _u32(self.buf, p))

    def scalars(self, dtype: str) -> np.ndarray:
        """View the vector as a numpy array of `dtype` (e.g. '<u2', '<f4')."""
        dt = np.dtype(dtype)
        return np.frombuffer(self.buf, dtype=dt, count=self.count, offset=self.pos)

    def structs(self, dtype: np.dtype | str) -> np.ndarray:
        """View the vector as an array of inline structs described by `dtype`."""
        dt = np.dtype(dtype)
        return np.frombuffer(self.buf, dtype=dt, count=self.count, offset=self.pos)

    def raw(self, elem_size: int) -> memoryview:
        return self.buf[self.pos : self.pos + self.count * elem_size]


def root(data: bytes | bytearray | memoryview) -> Table:
    buf = memoryview(data).cast("B") if not isinstance(data, memoryview) else data.cast("B")
    if len(buf) < 8:
        raise FlatBufferError("buffer too small")
    off = _u32(buf, 0)
    if off + 4 > len(buf):
        raise FlatBufferError("root offset out of range")
    return Table(buf, off)


def _u16(buf: memoryview, pos: int) -> int:
    return struct.unpack_from("<H", buf, pos)[0]


def _u32(buf: memoryview, pos: int) -> int:
    return struct.unpack_from("<I", buf, pos)[0]


def _i32(buf: memoryview, pos: int) -> int:
    return struct.unpack_from("<i", buf, pos)[0]


def _string_at(buf: memoryview, pos: int) -> str:
    n = _u32(buf, pos)
    return bytes(buf[pos + 4 : pos + 4 + n]).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Debug helper: guess-dump a table so unknown files can be inspected quickly.
# ---------------------------------------------------------------------------

def dump(t: Table, depth: int = 0, max_depth: int = 4, max_items: int = 6) -> str:
    """Heuristic dump of a table's fields (for exploration only)."""
    lines = []
    ind = "  " * depth
    for i in range(t.field_count()):
        o = t._field_offset(i)
        if not o:
            continue
        desc = _guess(t, i, o, depth, max_depth, max_items)
        lines.append(f"{ind}[{i}] {desc}")
    return "\n".join(lines)


def _looks_like_table(buf: memoryview, pos: int) -> bool:
    try:
        if pos + 4 > len(buf):
            return False
        vt = pos - _i32(buf, pos)
        if vt < 0 or vt + 4 > len(buf):
            return False
        vt_size = _u16(buf, vt)
        tbl_size = _u16(buf, vt + 2)
        return 4 <= vt_size <= 4 + 2 * 200 and vt_size % 2 == 0 and tbl_size >= 4 and vt + vt_size <= len(buf)
    except Exception:
        return False


def _guess(t: Table, i: int, o: int, depth: int, max_depth: int, max_items: int) -> str:
    buf = t.buf
    raw = bytes(buf[o : o + 4])
    u = _u32(buf, o)
    f = struct.unpack_from("<f", buf, o)[0]
    # offset candidate
    target = o + u
    if 0 < u and target + 4 <= len(buf):
        # string?
        n = _u32(buf, target)
        if 0 < n < 512 and target + 4 + n < len(buf) and buf[target + 4 + n] == 0:
            s = bytes(buf[target + 4 : target + 4 + n])
            if all(32 <= c < 127 for c in s):
                return f'str "{s.decode()}"'
        # vector of tables?
        if 0 < n < 100000 and target + 4 + n * 4 <= len(buf):
            p0 = target + 4
            if n > 0 and _looks_like_table(buf, p0 + _u32(buf, p0)) and depth < max_depth:
                v = Vector(buf, p0, n)
                out = [f"vec<table>[{n}]"]
                for k, sub in enumerate(v.tables()):
                    if k >= max_items:
                        out.append("  " * (depth + 1) + "...")
                        break
                    out.append("  " * (depth + 1) + f"- {{")
                    out.append(dump(sub, depth + 2, max_depth, max_items))
                    out.append("  " * (depth + 1) + "}")
                return "\n".join(out)
            if n > 0:
                head = bytes(buf[p0 : p0 + min(16, n * 4)]).hex(" ")
                return f"vec?[{n}] first bytes {head}"
            return "vec[0]"
        if _looks_like_table(buf, target) and depth < max_depth:
            return "table {\n" + dump(Table(buf, target), depth + 1, max_depth, max_items) + "\n" + "  " * depth + "}"
    if -1e6 < f < 1e6 and abs(f) > 1e-6 and raw[3] not in (0, 0xFF):
        return f"f32 {f:.6g} (u32 {u})"
    return f"u32 {u} (bytes {raw.hex(' ')})"


# ---------------------------------------------------------------------------
# Coverage recorder (opt-in, for `pipeline audit-fields`).
#
#   with fb.recording() as rec:
#       parse_trmtr(data)            # every field accessor logs (table_path, field_idx)
#   rec.touched                      # {("trmtr:/1[]", 0), ...}
#   rec.tables                       # buffers + every table the parsers reached, by path
#
# A table path is the file type (the parser's name without `parse_`, e.g. `trmtr`, `gfbmdl`) plus the chain of
# field indexes from the root, vectors collapsed to `[]`: the material table of a .trmtr is `trmtr:/1[]`, its
# float params `trmtr:/1[]/4[]`. While recording is off nothing is patched, so the reader runs unchanged.
# ---------------------------------------------------------------------------

import contextlib as _contextlib
import sys as _sys
import zlib as _zlib


class _BufInfo:
    __slots__ = ("fmt", "buf", "key", "tables", "vectors")

    def __init__(self, fmt: str, buf: memoryview):
        self.fmt = fmt
        self.buf = buf  # held so id(buf) is not reused while recording
        self.key = (len(buf), _zlib.crc32(buf))
        self.tables: dict[int, str] = {}  # table pos -> path
        self.vectors: dict[int, str] = {}  # vector pos -> path ("…/i[]")


class Recorder:
    def __init__(self) -> None:
        self.touched: set[tuple[str, int]] = set()
        self.bufs: dict[int, _BufInfo] = {}
        self.formats: set[str] = set()

    # -- registration (called from the patched accessors) ----------------
    def _info(self, buf: memoryview) -> _BufInfo:
        info = self.bufs.get(id(buf))
        if info is None:  # a Table built by hand on a buffer root() never saw
            info = self.bufs[id(buf)] = _BufInfo("?", buf)
        return info

    def path_of(self, t: Table) -> str:
        info = self._info(t.buf)
        p = info.tables.get(t.pos)
        if p is None:
            p = info.tables[t.pos] = f"{info.fmt}:?"
        return p

    def _child(self, parent: Table, child_pos: int, suffix: str) -> None:
        info = self._info(parent.buf)
        info.tables.setdefault(child_pos, self.path_of(parent) + suffix)

    def take(self) -> list[_BufInfo]:
        """Return and forget the buffers seen so far (the touched set is kept)."""
        out = list(self.bufs.values())
        self.bufs = {}
        return out


_REC: Recorder | None = None
_ORIG = {
    "field_offset": Table._field_offset,
    "table": Table.table,
    "vector": Table.vector,
    "vtables": Vector.tables,
    "root": root,
}


def _fmt_from_caller(frame) -> str:
    name = frame.f_code.co_name
    if name.startswith("parse_"):
        return name[6:].split("_")[0]
    mod = frame.f_globals.get("__name__", "?").rsplit(".", 1)[-1]
    return f"{mod}.{name}"


def _rec_root(data):
    t = _ORIG["root"](data)
    r = _REC
    if r is not None:
        fmt = _fmt_from_caller(_sys._getframe(1))
        info = r.bufs[id(t.buf)] = _BufInfo(fmt, t.buf)
        info.tables[t.pos] = f"{fmt}:"
        r.formats.add(fmt)
    return t


def _rec_field_offset(self: Table, index: int) -> int:
    r = _REC
    if r is not None:
        r.touched.add((r.path_of(self), index))
    return _ORIG["field_offset"](self, index)


def _rec_table(self: Table, index: int):
    t = _ORIG["table"](self, index)
    r = _REC
    if r is not None and t is not None:
        r._child(self, t.pos, f"/{index}")
    return t


def _rec_vector(self: Table, index: int):
    v = _ORIG["vector"](self, index)
    r = _REC
    if r is not None and v is not None:
        r._info(self.buf).vectors.setdefault(v.pos, f"{r.path_of(self)}/{index}[]")
    return v


def _rec_vtables(self: Vector):
    r = _REC
    path = r._info(self.buf).vectors.get(self.pos) if r is not None else None
    for t in _ORIG["vtables"](self):
        if r is not None:
            r._info(self.buf).tables.setdefault(t.pos, path or f"{r._info(self.buf).fmt}:?[]")
        yield t


@_contextlib.contextmanager
def recording(rec: Recorder | None = None):
    """Log every field access of every table read inside the block (not re-entrant)."""
    global _REC, root
    if _REC is not None:
        raise RuntimeError("fb.recording() is already active")
    rec = rec or Recorder()
    _REC = rec
    Table._field_offset = _rec_field_offset
    Table.table = _rec_table
    Table.vector = _rec_vector
    Vector.tables = _rec_vtables
    root = _rec_root
    try:
        yield rec
    finally:
        Table._field_offset = _ORIG["field_offset"]
        Table.table = _ORIG["table"]
        Table.vector = _ORIG["vector"]
        Vector.tables = _ORIG["vtables"]
        root = _ORIG["root"]
        _REC = None


# -- layout helpers for coverage walks (never recorded) ----------------------

def field_layout(buf: memoryview, pos: int) -> list[tuple[int, int, int]]:
    """Present fields of the table at `pos` as (index, absolute offset, byte size incl. padding)."""
    vt = pos - _i32(buf, pos)
    vt_size = _u16(buf, vt)
    tbl_size = _u16(buf, vt + 2)
    rels = []
    for i in range((vt_size - 4) // 2):
        rel = _u16(buf, vt + 4 + 2 * i)
        if rel:
            rels.append((rel, i))
    rels.sort()
    out = []
    for k, (rel, i) in enumerate(rels):
        end = rels[k + 1][0] if k + 1 < len(rels) else max(tbl_size, rel + 1)
        out.append((i, pos + rel, max(1, end - rel)))
    out.sort()
    return out


def looks_like_table(buf: memoryview, pos: int) -> bool:
    if pos % 2 or not _looks_like_table(buf, pos):
        return False
    try:
        vt = pos - _i32(buf, pos)
        vt_size, tbl_size = _u16(buf, vt), _u16(buf, vt + 2)
        if pos + tbl_size > len(buf):
            return False
        return all(_u16(buf, vt + 4 + 2 * i) < tbl_size for i in range((vt_size - 4) // 2))
    except Exception:
        return False


def guess_field(buf: memoryview, off: int, size: int):
    """Best guess of a field's type and value: ("u8", 3) / ("f32", 1.5) / ("str", "x") / ("table", pos) /
    ("vec<table>", (n, first_pos)) / ("vec", (n, head_hex)) / ("struct", floats) / ("hex", ...)."""
    n_buf = len(buf)
    if size == 1 or off + 4 > n_buf:
        return ("u8", buf[off])
    if size == 2:
        return ("u16", _u16(buf, off))
    if size in (8, 12, 16) or (size > 4 and size % 4 == 0 and size <= 64):
        k = size // 4
        if off + 4 * k <= n_buf:
            f = struct.unpack_from(f"<{k}f", buf, off)
            if all(x == 0 or 1e-6 < abs(x) < 1e7 for x in f):
                return ("struct", tuple(round(x, 4) for x in f))
            return ("hex", bytes(buf[off : off + 4 * k]).hex())
    u = _u32(buf, off)
    target = off + u
    if 0 < u and target % 4 == 0 and target + 4 <= n_buf:
        n = _u32(buf, target)
        if n == 0:
            return ("empty", "empty vec/str")
        if n < 4096 and target + 4 + n < n_buf and buf[target + 4 + n] == 0:
            s = bytes(buf[target + 4 : target + 4 + n])
            if all(32 <= c < 127 for c in s):
                return ("str", s.decode())
        if 0 < n < 1_000_000 and target + 4 + n <= n_buf:
            p0 = target + 4
            if p0 + 4 <= n_buf and looks_like_table(buf, p0 + _u32(buf, p0)) and target + 4 + 4 * n <= n_buf:
                return ("vec<table>", (n, p0))
            return ("vec", (n, bytes(buf[p0 : p0 + min(8, n_buf - p0)]).hex()))
        if looks_like_table(buf, target):
            return ("table", target)
    f = struct.unpack_from("<f", buf, off)[0]
    raw3 = buf[off + 3]
    if 1e-6 < abs(f) < 1e7 and raw3 not in (0, 0xFF):
        return ("f32", round(f, 5))
    return ("i32", _i32(buf, off))
