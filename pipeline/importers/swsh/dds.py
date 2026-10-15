"""Minimal DDS reader (mip 0 only) for the Sword/Shield trainer/object textures.

Pillow cannot open the sRGB DXGI variants (BC1/BC3_SRGB = 72/78) these files use, so decode the block formats directly.
"""

from __future__ import annotations

import struct

import numpy as np
import texture2ddecoder as t2d

# DXGI format -> kind
_DXGI = {
    28: "rgba", 29: "rgba", 87: "bgra", 91: "bgra", 88: "bgrx", 93: "bgrx",
    71: "bc1", 72: "bc1", 74: "bc2", 75: "bc2", 77: "bc3", 78: "bc3", 80: "bc4", 81: "bc4",
    83: "bc5", 84: "bc5", 98: "bc7", 99: "bc7",
}
_FOURCC = {b"DXT1": "bc1", b"DXT3": "bc2", b"DXT5": "bc3", b"ATI1": "bc4", b"BC4U": "bc4", b"ATI2": "bc5", b"BC5U": "bc5"}
SRGB_FORMATS = {72, 75, 78, 91, 93, 99}


def _bgra_to_rgba(raw: bytes, w: int, h: int) -> np.ndarray:
    return np.frombuffer(raw, np.uint8).reshape(h, w, 4)[..., [2, 1, 0, 3]].copy()


def parse_dds(data: bytes) -> tuple[np.ndarray, bool]:
    """-> (rgba (H, W, 4) uint8, srgb)."""
    if data[:4] != b"DDS " or len(data) < 128:
        raise ValueError("not a DDS file")
    h, w = struct.unpack_from("<II", data, 12)
    pf_flags, fourcc = struct.unpack_from("<I4s", data, 80)
    bits = struct.unpack_from("<I", data, 88)[0]
    off = 128
    dxgi = 0
    if fourcc == b"DX10":
        dxgi = struct.unpack_from("<I", data, 128)[0]
        kind = _DXGI.get(dxgi)
        off = 148
    elif fourcc in _FOURCC:
        kind = _FOURCC[fourcc]
    elif pf_flags & 0x40 and bits == 32:  # uncompressed RGB(A)
        rmask = struct.unpack_from("<I", data, 92)[0]
        kind = "bgra" if rmask == 0x00FF0000 else "rgba"
    else:
        kind = None
    if kind is None:
        raise NotImplementedError(f"DDS format fourcc={fourcc!r} dxgi={dxgi}")
    body = data[off:]
    pw, ph = -(-w // 4) * 4, -(-h // 4) * 4
    if kind == "rgba":
        rgba = np.frombuffer(body, np.uint8, count=w * h * 4).reshape(h, w, 4).copy()
    elif kind in ("bgra", "bgrx"):
        rgba = _bgra_to_rgba(body[: w * h * 4], w, h)
        if kind == "bgrx":
            rgba[..., 3] = 255
    elif kind == "bc1":
        rgba = _bgra_to_rgba(t2d.decode_bc1(body, pw, ph), pw, ph)
    elif kind == "bc2":
        rgba = _bgra_to_rgba(t2d.decode_bc2(body, pw, ph), pw, ph)
    elif kind == "bc3":
        rgba = _bgra_to_rgba(t2d.decode_bc3(body, pw, ph), pw, ph)
    elif kind == "bc4":
        r = _bgra_to_rgba(t2d.decode_bc4(body, pw, ph), pw, ph)[..., 0]
        rgba = np.stack([r, r, r, np.full_like(r, 255)], -1)
    elif kind == "bc5":
        px = _bgra_to_rgba(t2d.decode_bc5(body, pw, ph), pw, ph)
        xy = px[..., :2].astype(np.float32) / 255.0 * 2 - 1
        z = np.sqrt(np.clip(1 - xy[..., 0] ** 2 - xy[..., 1] ** 2, 0, 1))
        zb = np.round((z * 0.5 + 0.5) * 255).astype(np.uint8)
        rgba = np.dstack([px[..., 0], px[..., 1], zb, np.full_like(zb, 255)])
    else:  # bc7
        rgba = _bgra_to_rgba(t2d.decode_bc7(body, pw, ph), pw, ph)
    return rgba[:h, :w], dxgi in SRGB_FORMATS
