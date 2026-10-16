"""Pure-numpy BC6H block decoder (DXGI_FORMAT_BC6H_UF16 / BC6H_SF16).

Transcribed from the Direct3D "BC6H Format" functional specification (the same
tables DirectXTex's BC6HBC7.cpp ``ms_aInfo``/``ms_aDesc`` encode):

* 14 modes selected by the first 2 bits (modes 1-2) or 5 bits (modes 3-14);
  the four 5-bit patterns 10011/10111/11011/11111 are reserved and decode to 0.
* Two-subset modes (1-10): 82 header bits (mode + endpoints + 5-bit partition
  ``d``) followed by 46 index bits (3 bits per texel, 2 for texel 0 and for the
  subset-1 fix-up texel).  One-subset modes (11-14): 65 header bits + 63 index
  bits (4 per texel, 3 for texel 0).
* Endpoint fields: ``w``/``x`` = endpoints 0/1 of subset 0, ``y``/``z`` =
  endpoints 0/1 of subset 1.  The field strings below list the bits in block
  order; ``f[a:b]`` means bit ``b`` is stored first (so ``rw[9:0]`` is LSB
  first while the high bits of modes 13/14, ``rw[10:11]``/``rw[10:15]``, are
  stored MSB first - a well-known quirk of the format).
* Delta ("transformed") modes store ``x``/``y``/``z`` as signed deltas from
  ``w`` with wrap-around modulo 2^EPB; signed blocks sign-extend ``w`` too.
* Unquantize to a 16-bit internal range (0..0xFFFF unsigned, -0x7FFF..0x7FFF
  signed), interpolate with the 64-based 3/4-bit weight tables, then scale by
  31/64 (unsigned) or 31/32 (signed): the result is the IEEE half-float bit
  pattern (sign bit set on the magnitude for negative signed results).

Reference: https://learn.microsoft.com/windows/win32/direct3d11/bc6h-format
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Field ids.  Endpoint e of subset s, channel c -> 3 * (2 * s + e) + c
# ---------------------------------------------------------------------------
_FIELD_NAMES = ["rw", "gw", "bw", "rx", "gx", "bx", "ry", "gy", "by", "rz", "gz", "bz", "d", "m"]
_FIELD = {n: i for i, n in enumerate(_FIELD_NAMES)}
_D, _M = _FIELD["d"], _FIELD["m"]
_NUM_FIELDS = len(_FIELD_NAMES)

# mode -> (endpoint precision, (delta bits r, g, b), transformed, subsets, field layout)
_MODE_TABLE: dict[int, tuple[int, tuple[int, int, int], bool, int, str]] = {
    1: (10, (5, 5, 5), True, 2,
        "m[1:0] gy[4] by[4] bz[4] rw[9:0] gw[9:0] bw[9:0] rx[4:0] gz[4] gy[3:0] gx[4:0] bz[0] "
        "gz[3:0] bx[4:0] bz[1] by[3:0] ry[4:0] bz[2] rz[4:0] bz[3] d[4:0]"),
    2: (7, (6, 6, 6), True, 2,
        "m[1:0] gy[5] gz[4] gz[5] rw[6:0] bz[0] bz[1] by[4] gw[6:0] by[5] bz[2] gy[4] bw[6:0] "
        "bz[3] bz[5] bz[4] rx[5:0] gy[3:0] gx[5:0] gz[3:0] bx[5:0] by[3:0] ry[5:0] rz[5:0] d[4:0]"),
    3: (11, (5, 4, 4), True, 2,
        "m[4:0] rw[9:0] gw[9:0] bw[9:0] rx[4:0] rw[10] gy[3:0] gx[3:0] gw[10] bz[0] gz[3:0] "
        "bx[3:0] bw[10] bz[1] by[3:0] ry[4:0] bz[2] rz[4:0] bz[3] d[4:0]"),
    4: (11, (4, 5, 4), True, 2,
        "m[4:0] rw[9:0] gw[9:0] bw[9:0] rx[3:0] rw[10] gz[4] gy[3:0] gx[4:0] gw[10] gz[3:0] "
        "bx[3:0] bw[10] bz[1] by[3:0] ry[3:0] bz[0] bz[2] rz[3:0] gy[4] bz[3] d[4:0]"),
    5: (11, (4, 4, 5), True, 2,
        "m[4:0] rw[9:0] gw[9:0] bw[9:0] rx[3:0] rw[10] by[4] gy[3:0] gx[3:0] gw[10] bz[0] "
        "gz[3:0] bx[4:0] bw[10] by[3:0] ry[3:0] bz[1] bz[2] rz[3:0] bz[4] bz[3] d[4:0]"),
    6: (9, (5, 5, 5), True, 2,
        "m[4:0] rw[8:0] by[4] gw[8:0] gy[4] bw[8:0] bz[4] rx[4:0] gz[4] gy[3:0] gx[4:0] bz[0] "
        "gz[3:0] bx[4:0] bz[1] by[3:0] ry[4:0] bz[2] rz[4:0] bz[3] d[4:0]"),
    7: (8, (6, 5, 5), True, 2,
        "m[4:0] rw[7:0] gz[4] by[4] gw[7:0] bz[2] gy[4] bw[7:0] bz[3] bz[4] rx[5:0] gy[3:0] "
        "gx[4:0] bz[0] gz[3:0] bx[4:0] bz[1] by[3:0] ry[5:0] rz[5:0] d[4:0]"),
    8: (8, (5, 6, 5), True, 2,
        "m[4:0] rw[7:0] bz[0] by[4] gw[7:0] gy[5] gy[4] bw[7:0] gz[5] bz[4] rx[4:0] gz[4] "
        "gy[3:0] gx[5:0] gz[3:0] bx[4:0] bz[1] by[3:0] ry[4:0] bz[2] rz[4:0] bz[3] d[4:0]"),
    9: (8, (5, 5, 6), True, 2,
        "m[4:0] rw[7:0] bz[1] by[4] gw[7:0] by[5] gy[4] bw[7:0] bz[5] bz[4] rx[4:0] gz[4] "
        "gy[3:0] gx[4:0] bz[0] gz[3:0] bx[5:0] by[3:0] ry[4:0] bz[2] rz[4:0] bz[3] d[4:0]"),
    10: (6, (6, 6, 6), False, 2,
         "m[4:0] rw[5:0] gz[4] bz[0] bz[1] by[4] gw[5:0] gy[5] by[5] bz[2] gy[4] bw[5:0] gz[5] "
         "bz[3] bz[5] bz[4] rx[5:0] gy[3:0] gx[5:0] gz[3:0] bx[5:0] by[3:0] ry[5:0] rz[5:0] d[4:0]"),
    11: (10, (10, 10, 10), False, 1,
         "m[4:0] rw[9:0] gw[9:0] bw[9:0] rx[9:0] gx[9:0] bx[9:0]"),
    12: (11, (9, 9, 9), True, 1,
         "m[4:0] rw[9:0] gw[9:0] bw[9:0] rx[8:0] rw[10] gx[8:0] gw[10] bx[8:0] bw[10]"),
    13: (12, (8, 8, 8), True, 1,
         "m[4:0] rw[9:0] gw[9:0] bw[9:0] rx[7:0] rw[10:11] gx[7:0] gw[10:11] bx[7:0] bw[10:11]"),
    14: (16, (4, 4, 4), True, 1,
         "m[4:0] rw[9:0] gw[9:0] bw[9:0] rx[3:0] rw[10:15] gx[3:0] gw[10:15] bx[3:0] bw[10:15]"),
}

# 5-bit mode patterns (modes 1 and 2 use the 2-bit patterns 00 / 01)
_MODE_BITS5 = {
    0b00010: 3, 0b00110: 4, 0b01010: 5, 0b01110: 6, 0b10010: 7, 0b10110: 8, 0b11010: 9,
    0b11110: 10, 0b00011: 11, 0b00111: 12, 0b01011: 13, 0b01111: 14,
}


def _parse_layout(layout: str) -> list[tuple[int, int]]:
    """Turn a layout string such as ``rw[9:0] bz[4] rw[10:11]`` into ``[(field, bit), ...]``
    in block-bit order (``f[a:b]``: bit ``b`` first, stepping toward ``a``)."""
    out: list[tuple[int, int]] = []
    for tok in layout.split():
        name, rng = tok[:-1].split("[")
        field = _FIELD[name]
        if ":" in rng:
            a, b = (int(v) for v in rng.split(":"))
            step = 1 if b <= a else -1
            out.extend((field, bit) for bit in range(b, a + step, step))
        else:
            out.append((field, int(rng)))
    return out


class _ModeInfo:
    __slots__ = ("mode", "epb", "delta", "transformed", "subsets", "desc", "header_bits", "index_bits")

    def __init__(self, mode: int) -> None:
        epb, delta, transformed, subsets, layout = _MODE_TABLE[mode]
        self.mode = mode
        self.epb = epb
        self.delta = delta
        self.transformed = transformed
        self.subsets = subsets
        self.desc = _parse_layout(layout)
        self.header_bits = len(self.desc)
        self.index_bits = 3 if subsets == 2 else 4
        expected = 82 if subsets == 2 else 65
        if self.header_bits != expected:
            raise AssertionError(f"mode {mode}: {self.header_bits} header bits, expected {expected}")
        # every field must receive exactly bits 0..n-1 once, with the spec's widths
        got: dict[int, list[int]] = {}
        for f, b in self.desc:
            got.setdefault(f, []).append(b)
        for f, bits in got.items():
            if sorted(bits) != list(range(len(bits))):
                raise AssertionError(f"mode {mode}: field {_FIELD_NAMES[f]} bits {sorted(bits)}")
        want = {_FIELD["rw"]: epb, _FIELD["gw"]: epb, _FIELD["bw"]: epb}
        for e in ("x", "y", "z")[: subsets * 2 - 1]:
            for c, n in zip("rgb", delta):
                want[_FIELD[c + e]] = n
        want[_M] = 2 if mode <= 2 else 5
        if subsets == 2:
            want[_D] = 5
        widths = {f: len(b) for f, b in got.items()}
        if widths != want:
            raise AssertionError(f"mode {mode}: field widths {widths} != {want}")


MODES: dict[int, _ModeInfo] = {m: _ModeInfo(m) for m in _MODE_TABLE}

# ---------------------------------------------------------------------------
# Two-subset partition table (shared with BC7) and the subset-1 fix-up texel
# ---------------------------------------------------------------------------
PARTITIONS = np.array([
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
], dtype=np.int64)

FIXUP = np.array([
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
    15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2,
    15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6,
    6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15,
], dtype=np.int64)

WEIGHTS3 = np.array([0, 9, 18, 27, 37, 46, 55, 64], dtype=np.int64)
WEIGHTS4 = np.array([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], dtype=np.int64)


# ---------------------------------------------------------------------------
# Spec helper functions (vectorised over int64 arrays)
# ---------------------------------------------------------------------------
def _sign_extend(v: np.ndarray, bits: int) -> np.ndarray:
    mask = np.int64(1) << (bits - 1)
    return (v ^ mask) - mask


def _unquantize(v: np.ndarray, bits: int, signed: bool) -> np.ndarray:
    v = v.astype(np.int64)
    if not signed:
        if bits >= 15:
            return v
        out = ((v << 16) + 0x8000) >> bits
        out = np.where(v == 0, 0, out)
        return np.where(v == (1 << bits) - 1, 0xFFFF, out)
    if bits >= 16:
        return v
    a = np.abs(v)
    out = ((a << 15) + 0x4000) >> (bits - 1)
    out = np.where(a == 0, 0, out)
    out = np.where(a >= (1 << (bits - 1)) - 1, 0x7FFF, out)
    return np.where(v < 0, -out, out)


def _finish_unquantize(v: np.ndarray, signed: bool) -> np.ndarray:
    if not signed:
        return (v * 31) >> 6
    m = (np.abs(v) * 31) >> 5
    return np.where(v < 0, -m, m)


def _to_half_bits(v: np.ndarray, signed: bool) -> np.ndarray:
    if signed:
        return np.where(v < 0, 0x8000 | (-v), v).astype(np.uint16)
    return np.clip(v, 0, 0x7BFF).astype(np.uint16)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
def _mode_of(blocks: np.ndarray) -> np.ndarray:
    """(N,16) uint8 -> (N,) mode number 1..14, or 0 for the reserved patterns."""
    b0 = blocks[:, 0].astype(np.int64)
    low2 = b0 & 3
    low5 = b0 & 31
    mode = np.zeros(len(blocks), dtype=np.int64)
    mode[low2 == 0] = 1
    mode[low2 == 1] = 2
    for pattern, m in _MODE_BITS5.items():
        mode[(low2 >= 2) & (low5 == pattern)] = m
    return mode


def _decode_mode(bits: np.ndarray, info: _ModeInfo, signed: bool) -> np.ndarray:
    """bits: (n,128) 0/1 for blocks of one mode -> (n,16,3) half-float bit patterns (uint16)."""
    n = len(bits)
    bits = bits.astype(np.int64)
    fields = np.zeros((n, _NUM_FIELDS), dtype=np.int64)
    for pos, (field, bit) in enumerate(info.desc):
        fields[:, field] |= bits[:, pos] << bit

    # endpoints (n, subsets, 2, 3): ep[:, s, 0] = first endpoint of subset s
    ep = fields[:, : info.subsets * 6].reshape(n, info.subsets, 2, 3).copy()
    epb = info.epb
    if signed:
        ep[:, 0, 0, :] = _sign_extend(ep[:, 0, 0, :], epb)
    if signed or info.transformed:
        for c, dbits in enumerate(info.delta):
            ep[:, 0, 1, c] = _sign_extend(ep[:, 0, 1, c], dbits)
            if info.subsets == 2:
                ep[:, 1, 0, c] = _sign_extend(ep[:, 1, 0, c], dbits)
                ep[:, 1, 1, c] = _sign_extend(ep[:, 1, 1, c], dbits)
    if info.transformed:
        base = ep[:, 0, 0, :].copy()
        wrap = (1 << epb) - 1
        for s in range(info.subsets):
            for e in range(2):
                if s == 0 and e == 0:
                    continue
                v = (ep[:, s, e, :] + base) & wrap
                ep[:, s, e, :] = _sign_extend(v, epb) if signed else v

    unq = _unquantize(ep, epb, signed)  # (n, subsets, 2, 3)

    # indices
    if info.subsets == 2:
        part = PARTITIONS[fields[:, _D]]  # (n,16)
        fix = FIXUP[fields[:, _D]]  # (n,)
        widths = np.full((n, 16), 3, dtype=np.int64)
        widths[:, 0] = 2
        widths[np.arange(n), fix] = 2
        weights = WEIGHTS3
    else:
        part = np.zeros((n, 16), dtype=np.int64)
        widths = np.full((n, 16), 4, dtype=np.int64)
        widths[:, 0] = 3
        weights = WEIGHTS4
    offsets = info.header_bits + np.cumsum(widths, axis=1) - widths
    idx = np.zeros((n, 16), dtype=np.int64)
    for k in range(int(widths.max())):
        pos = np.minimum(offsets + k, 127)
        bit = np.take_along_axis(bits, pos, axis=1)
        idx |= (bit & (k < widths)) << k

    rows = np.arange(n)[:, None]
    a = unq[rows, part, 0, :]  # (n,16,3)
    b = unq[rows, part, 1, :]
    w = weights[idx][..., None]
    fc = (a * (64 - w) + b * w + 32) >> 6
    fc = _finish_unquantize(fc, signed)
    return _to_half_bits(fc, signed)


def decode_bc6h_blocks(blocks: np.ndarray, signed: bool = True) -> np.ndarray:
    """(N,16) uint8 blocks -> (N,16,3) float32 texels (row-major within the 4x4 block)."""
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8).reshape(-1, 16)
    n = len(blocks)
    out = np.zeros((n, 16, 3), dtype=np.uint16)
    if n == 0:
        return out.view(np.float16).astype(np.float32)
    bits = np.unpackbits(blocks, axis=1, bitorder="little")  # (N,128): bit p of the 128-bit LE word
    modes = _mode_of(blocks)
    for m, info in MODES.items():
        sel = np.flatnonzero(modes == m)
        if sel.size:
            out[sel] = _decode_mode(bits[sel], info, signed)
    return out.view(np.float16).astype(np.float32)


def decode_bc6h(data: bytes, width: int, height: int, signed: bool = True) -> np.ndarray:
    """-> (height, width, 3) float32 linear RGB.

    Blocks are 4x4, row-major, 16 bytes each; width/height block-aligned (a
    non-aligned size is padded up and cropped).  ``signed`` selects BC6H_SF16
    (BNTX 0x1F05) versus BC6H_UF16 (0x1F0A).
    """
    bw, bh = (width + 3) // 4, (height + 3) // 4
    need = bw * bh * 16
    if len(data) < need:
        raise ValueError(f"BC6H: need {need} bytes for {width}x{height}, got {len(data)}")
    blocks = np.frombuffer(data, dtype=np.uint8, count=need).reshape(bw * bh, 16)
    tex = decode_bc6h_blocks(blocks, signed)  # (N,16,3)
    img = tex.reshape(bh, bw, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 3)
    return np.ascontiguousarray(img[:height, :width])


__all__ = ["decode_bc6h", "decode_bc6h_blocks", "PARTITIONS", "FIXUP", "WEIGHTS3", "WEIGHTS4", "MODES"]
