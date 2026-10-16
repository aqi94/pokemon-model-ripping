"""`probemain_diffuse/specular.bntx` light probes: 6-face cube maps, 64x64 BC6H signed float, block-linear.

The BNTX decoder in `importers/bntx` reads one 2D surface; cube arrays store the 6 faces one after another
(`imgsize / array_len` bytes each, mips included). Faces follow the usual +X -X +Y -Y +Z -Z order (assumed).
"""

from __future__ import annotations

import struct

import numpy as np

from ..importers.bntx import decoder as _dec


def read_cube(data: bytes) -> np.ndarray:
    """-> (6, H, W, 4) float32 RGBA, linear radiance (BC6H decodes to half floats, kept as float32)."""
    if data[:4] != b"BNTX":
        raise ValueError("not a BNTX file")
    u16 = lambda o: struct.unpack_from("<H", data, o)[0]
    u32 = lambda o: struct.unpack_from("<I", data, o)[0]
    u64 = lambda o: struct.unpack_from("<Q", data, o)[0]
    at = u64(u64(0x28))
    tile_mode, fmt = u16(at + 0x12), u32(at + 0x1C)
    w, h, arr = u32(at + 0x24), u32(at + 0x28), u32(at + 0x30)
    size_range, img_size = u32(at + 0x34), u32(at + 0x50)
    off = u64(u64(at + 0x70))
    base, ftype = fmt >> 8, fmt & 0xFF
    name, bw, bh, bpb = _dec._FORMATS[base]
    wblk, hblk = -(-w // bw), -(-h // bh)
    wb = wblk * bpb
    stride = img_size // max(1, arr)
    buf = np.frombuffer(data, np.uint8)
    faces = []
    for i in range(arr):
        surf = buf[off + i * stride: off + (i + 1) * stride]
        if tile_mode == 0:
            lin = _dec._deswizzle(surf, wb, hblk, 1 << size_range)
        else:
            pitch = -(-wb // 64) * 64
            lin = surf[:pitch * hblk].reshape(hblk, pitch)[:, :wb]
        faces.append(_decode_float(base, ftype, w, h, np.ascontiguousarray(lin).tobytes()))
    return np.stack(faces)


def _decode_float(base: int, ftype: int, w: int, h: int, lin: bytes) -> np.ndarray:
    name = _dec._FORMATS[base][0]
    if name == "BC6H":
        # BNTX type 5 = FLOAT (signed BC6H_SF16), 10 = UFLOAT; texture2ddecoder only handles the unsigned
        # variant, so use our own decoder (importers/bntx/bc6h.py). HDR values are kept as-is.
        from ..importers.bntx.bc6h import decode_bc6h
        rgb = decode_bc6h(lin, w, h, signed=(ftype != 10))
        return np.dstack([rgb, np.ones(rgb.shape[:2], np.float32)])
    return _dec._decode_pixels(base, ftype, w, h, lin).astype(np.float32) / 255.0


def hemisphere_means(cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sky rgb, ground rgb) = mean of the +Y face vs the -Y face plus the upper/lower halves of the side faces."""
    px, nx, py, ny, pz, nz = [cube[i, ..., :3] for i in range(6)]
    h = cube.shape[1] // 2
    sky = np.concatenate([py.reshape(-1, 3)] + [f[:h].reshape(-1, 3) for f in (px, nx, pz, nz)]).mean(0)
    ground = np.concatenate([ny.reshape(-1, 3)] + [f[h:].reshape(-1, 3) for f in (px, nx, pz, nz)]).mean(0)
    return sky, ground


def cube_to_cross(cube: np.ndarray) -> np.ndarray:
    """Horizontal cross layout (3 rows x 4 faces) as uint8 RGBA for a quick PNG."""
    n, h, w, _ = cube.shape
    out = np.zeros((3 * h, 4 * w, 4), np.uint8)
    img = (np.clip(cube, 0, 1) * 255).astype(np.uint8)
    pos = {0: (1, 2), 1: (1, 0), 2: (0, 1), 3: (2, 1), 4: (1, 1), 5: (1, 3)}
    for i in range(min(n, 6)):
        r, c = pos[i]
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = img[i]
    out[..., 3] = np.where(out[..., 3] == 0, 0, 255)
    return out
