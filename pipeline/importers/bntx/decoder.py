"""Nintendo Switch BNTX texture container decoder (mip 0 only)."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import texture2ddecoder as t2d
from PIL import Image

_TYPES = {1: "UNORM", 2: "SNORM", 3: "UINT", 4: "SINT", 5: "FLOAT", 6: "SRGB", 10: "UFLOAT"}
_ASTC = [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8),
         (10, 5), (10, 6), (10, 8), (10, 10), (12, 10), (12, 12)]
# base id -> (name, block w, block h, bytes per block)
_FORMATS: dict[int, tuple[str, int, int, int]] = {
    0x02: ("R8", 1, 1, 1), 0x07: ("R5G6B5", 1, 1, 2), 0x09: ("R8G8", 1, 1, 2),
    0x0B: ("R8G8B8A8", 1, 1, 4), 0x0C: ("B8G8R8A8", 1, 1, 4),
    0x1A: ("BC1", 4, 4, 8), 0x1B: ("BC2", 4, 4, 16), 0x1C: ("BC3", 4, 4, 16),
    0x1D: ("BC4", 4, 4, 8), 0x1E: ("BC5", 4, 4, 16), 0x1F: ("BC6H", 4, 4, 16),
    0x20: ("BC7", 4, 4, 16),
}
for _i, (_bw, _bh) in enumerate(_ASTC):
    _FORMATS[0x2D + _i] = (f"ASTC_{_bw}x{_bh}", _bw, _bh, 16)


@dataclass
class BntxTexture:
    name: str
    width: int
    height: int
    format: int
    format_name: str
    srgb: bool
    rgba: np.ndarray  # (H, W, 4) uint8, top row first


def _deswizzle(data: np.ndarray, wb: int, h: int, block_height: int) -> np.ndarray:
    """Block-linear -> linear. wb = width in bytes, h = rows. Returns (h, wb) uint8."""
    gobs_per_row = (wb + 63) // 64
    x = np.arange(wb, dtype=np.int64)[None, :]
    y = np.arange(h, dtype=np.int64)[:, None]
    bh_rows = 8 * block_height
    addr = ((y // bh_rows) * 512 * block_height * gobs_per_row
            + (x // 64) * 512 * block_height
            + ((y % bh_rows) // 8) * 512
            + ((x % 64) // 32) * 256 + ((y % 8) // 2) * 64
            + ((x % 32) // 16) * 32 + (y % 2) * 16 + (x % 16))
    if addr.max() >= data.size:
        raise ValueError("BNTX swizzled surface is truncated")
    return data[addr]


def _bgra_to_rgba(raw: bytes, w: int, h: int) -> np.ndarray:
    a = np.frombuffer(raw, np.uint8).reshape(h, w, 4)
    return a[..., [2, 1, 0, 3]].copy()


def _decode_pixels(base: int, ftype: int, w: int, h: int, lin: bytes) -> np.ndarray:
    name, bw, bh, bpb = _FORMATS[base]
    if name.startswith("BC") or name.startswith("ASTC"):
        pw, ph = -(-w // bw) * bw, -(-h // bh) * bh  # decoders want block-aligned dims
        if name == "BC1":
            rgba = _bgra_to_rgba(t2d.decode_bc1(lin, pw, ph), pw, ph)
        elif name == "BC3":
            rgba = _bgra_to_rgba(t2d.decode_bc3(lin, pw, ph), pw, ph)
        elif name == "BC7":
            rgba = _bgra_to_rgba(t2d.decode_bc7(lin, pw, ph), pw, ph)
        elif name == "BC4":
            r = _bgra_to_rgba(t2d.decode_bc4(lin, pw, ph), pw, ph)[..., 0]
            rgba = np.stack([r, r, r, np.full_like(r, 255)], -1)
        elif name == "BC5":
            px = _bgra_to_rgba(t2d.decode_bc5(lin, pw, ph), pw, ph)
            xy = px[..., :2].astype(np.float32) / 255.0 * 2 - 1
            z = np.sqrt(np.clip(1 - xy[..., 0] ** 2 - xy[..., 1] ** 2, 0, 1))
            zb = np.round((z * 0.5 + 0.5) * 255).astype(np.uint8)
            rgba = np.dstack([px[..., 0], px[..., 1], zb, np.full_like(zb, 255)])
        elif name == "BC6H":
            rgba = _bgra_to_rgba(t2d.decode_bc6(lin, pw, ph, False), pw, ph)
        elif name.startswith("ASTC"):
            rgba = _bgra_to_rgba(t2d.decode_astc(lin, pw, ph, bw, bh), pw, ph)
        else:
            raise NotImplementedError(f"format {name}")
        return rgba[:h, :w]
    a = np.frombuffer(lin, np.uint8)
    if name == "R8G8B8A8":
        return a.reshape(h, w, 4).copy()
    if name == "B8G8R8A8":
        return a.reshape(h, w, 4)[..., [2, 1, 0, 3]].copy()
    if name == "R8":
        r = a.reshape(h, w)
        return np.stack([r, r, r, np.full_like(r, 255)], -1)
    if name == "R8G8":
        p = a.reshape(h, w, 2)
        return np.dstack([p, np.zeros((h, w), np.uint8), np.full((h, w), 255, np.uint8)])
    if name == "R5G6B5":
        v = a.view("<u2").reshape(h, w).astype(np.uint16)
        r = ((v >> 11) & 31) * 255 // 31
        g = ((v >> 5) & 63) * 255 // 63
        b = (v & 31) * 255 // 31
        return np.dstack([r, g, b, np.full_like(r, 255)]).astype(np.uint8)
    raise NotImplementedError(f"format {name}")


def parse_bntx(data: bytes) -> list[BntxTexture]:
    if data[:4] != b"BNTX":
        raise ValueError("not a BNTX file")
    if data[0x20:0x24] != b"NX  ":
        raise ValueError("unsupported BNTX platform")
    u16 = lambda o: struct.unpack_from("<H", data, o)[0]
    u32 = lambda o: struct.unpack_from("<I", data, o)[0]
    u64 = lambda o: struct.unpack_from("<Q", data, o)[0]
    count, table = u32(0x24), u64(0x28)
    buf = np.frombuffer(data, np.uint8)
    out = []
    for i in range(count):
        at = u64(table + 8 * i)
        if data[at:at + 4] != b"BRTI":
            raise ValueError("invalid BRTI entry")
        tile_mode = u16(at + 0x12)
        fmt = u32(at + 0x1C)
        w, h = u32(at + 0x24), u32(at + 0x28)
        size_range = u32(at + 0x34)
        name_ptr, mip_ptr = u64(at + 0x60), u64(at + 0x70)
        nlen = u16(name_ptr)
        name = data[name_ptr + 2:name_ptr + 2 + nlen].decode("utf-8", "replace")
        off = u64(mip_ptr)
        base, ftype = fmt >> 8, fmt & 0xFF
        if base not in _FORMATS:
            raise NotImplementedError(f"BNTX format 0x{fmt:04X} in {name}")
        fname, bw, bh, bpb = _FORMATS[base]
        wblk, hblk = -(-w // bw), -(-h // bh)
        wb = wblk * bpb
        end = min(len(data), u32(at + 0x50) + off) if u32(at + 0x50) else len(data)
        surf = buf[off:end]
        if tile_mode == 0:
            lin = _deswizzle(surf, wb, hblk, 1 << size_range)
        else:
            pitch = -(-wb // 64) * 64
            lin = surf[:pitch * hblk].reshape(hblk, pitch)[:, :wb]
        rgba = _decode_pixels(base, ftype, w, h, np.ascontiguousarray(lin).tobytes())
        suffix = _TYPES.get(ftype, f"T{ftype:02X}")
        out.append(BntxTexture(name, w, h, fmt, f"{fname}_{suffix}", ftype == 6, rgba))
    return out


def decode_bntx_file(path) -> list[BntxTexture]:
    return parse_bntx(Path(path).read_bytes())


def to_image(tex: BntxTexture) -> Image.Image:
    return Image.fromarray(tex.rgba, "RGBA")
