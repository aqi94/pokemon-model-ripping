"""UNITE `LobbyPlayer` shaders (lobby NPCs, trainer avatar parts): colour maths read from the decompiled game shaders.

Source: the Marcy4000/Unite fan project's `Assets/Shaders/Compiled/*.txt` (GLSL ES dumps of the game's own shaders, with
the authors' slot comments in the Properties header). File and line references are of the ForwardLit gles3 fragment of the
first variant (`~/Documents/pokemon_raw_data/unite/marcy/Assets/Shaders/Compiled/`). Ledger: `docs/ledger/unite.json`
(`shader:lobbyPlayer*`).

Colour zones (`_MixMap` "Alpha(R) Mask(G) Emiss(B) Color(A)", read as raw bytes, no sRGB decode):
    w_k = clamp(1.031915 - |(M.a - k/3) * 3.191489|, 0, 1),  k = 0..3     (a tent around 0, 1/3, 2/3, 1)
    tint = sum w_k * _color{k+1}
Shaders (base colour in the shader's own space; textures are sRGB, so they are decoded first):
    HairColor_lv3  (HairColor_lv3.txt:536-560)   tint' = mix(tint, 1, M.g)
        a = mix(1, M.r, _Mask);  rgb = mix(_colorSkin, _MainTex.rgb * tint', a);  alpha = a, discard a < _Cutoff
        roughness = clamp(mix(dot(w, _roughv4) + _MainTex.a, _MainTex.a, M.g), 0.04, 1)
        emission  = M.b * _Emissive * _EmisssionScale * _MainTex.rgb (added after lighting)
    playerpbrSkin_lv3 / SkinNoShadow_lv3 (Skin_lv3.txt:442-464)
        rgb = _MainTex.rgb * mix(_colorSkin, 1, _BumpMapNr.a)      (the normal map's alpha is "Metalness")
        a = mix(1, M.r, _Mask) cut at _Cutoff;  roughness = clamp(_MainTex.a^2 + _RoughNessOffset, 0.04, 1)
        emission  = M.b * _Emissive * _EmisssionScale * _MainTex.rgb
    PlayerPBRColorEye (ColorEye.txt:468-490), unlit
        c = mix(1.2 * tint, 1, M.g);  out = c * _MainTex.rgb * (1 + M.b);  alpha = M.r
    playerpbrRampLight_lv3 (RampLight_lv3.txt:439-452, 630-660)
        base = _MainTex.rgb, roughness as Skin;  emission added after lighting:
        ramp(fract(_Time.x * _RampSpeed), 0.5) * mix(_Rampscale2, _Rampscale, _MixMap(uv1 + fract(_Time.y * _RampFlow)).r)
        only where the second UV's x <= 1 (vs_TEXCOORD9 = in_TEXCOORD1)
    Hidden/m_lob_PlayerPBRLengcy_lv3 (Lengcy_lv3.txt:746-749)
        base = _MainTex.rgb unchanged, roughness = clamp(_MainTex.a^2 + _RoughNessOffset, 0.04, 1)

The shaders' metallic (normal alpha / `_metalv4`), anisotropic hair highlights (`_HairMap`, `_specularColor1/2`), SH
ambient, rim and outline are not baked.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

PLAYER_KINDS = (
    (re.compile(r"LobbyPlayer/m_lob_PlayerPBRHairColor_lv3$", re.I), "hair"),
    (re.compile(r"LobbyPlayer/m_lob_playerpbrSkin(?:NoShadow)?_lv3$", re.I), "skin"),
    (re.compile(r"LobbyPlayer/m_lob_playerpbrRampLight_lv3$", re.I), "ramp"),
    (re.compile(r"LobbyPlayer/m_lob_PlayerPBRColorEye$", re.I), "eye"),
    (re.compile(r"^Hidden/m_lob_PlayerPBRLengcy_lv3$", re.I), "lengcy"),
)

# The shader's tent: 1 / (1/3) = 3, widened by 6.4% so neighbouring zones overlap slightly (read from the constants).
ZONE_SLOPE = 3.19148946
ZONE_PEAK = 1.03191495
ROUGH_MIN = 0.04  # 0.0399999991 in the shader


def player_kind(shader: str) -> str | None:
    return next((k for rx, k in PLAYER_KINDS if rx.search(shader or "")), None)


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def zone_weights(a: np.ndarray) -> np.ndarray:
    """(...,) colour-zone value (MixMap alpha, 0..1) -> (..., 4) weights of `_color1.._color4`."""
    k = np.array([0.0, 1 / 3, 2 / 3, 1.0], np.float32)
    return np.clip(ZONE_PEAK - np.abs((a[..., None] - k) * ZONE_SLOPE), 0.0, 1.0)


def zone_tint(mix: np.ndarray, colors: list) -> np.ndarray:
    """Sum of the four zone colours by the MixMap alpha: (H, W, 3)."""
    w = zone_weights(mix[..., 3])
    cols = np.asarray([c[:3] for c in colors], np.float32)  # (4, 3)
    return w @ cols


def fit(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resample (h0, w0, c) float data to (h, w, c) bilinearly (masks are often smaller than the albedo)."""
    if a.shape[:2] == (h, w):
        return a
    return np.stack([np.asarray(Image.fromarray(a[..., c].astype(np.float32), "F").resize((w, h), Image.BILINEAR))
                     for c in range(a.shape[2])], -1)


def _vec(colors: dict, key: str, default) -> list:
    v = colors.get(key)
    return list(v) if v is not None else list(default)


def _colors4(colors: dict) -> list:
    return [_vec(colors, f"_color{i}", (1, 1, 1, 1)) for i in range(1, 5)]


def alpha_of(mix: np.ndarray | None, fl: dict) -> np.ndarray | None:
    """Output alpha `mix(1, M.r, _Mask)` (None when it is 1 everywhere)."""
    mask = float(fl.get("_Mask", 0.0))
    if mix is None or mask <= 0.0:
        return None
    return 1.0 - mask * (1.0 - mix[..., 0])


def roughness_of(kind: str, main_a: np.ndarray, mix: np.ndarray | None, colors: dict, fl: dict) -> np.ndarray:
    if kind == "hair":
        w = zone_weights(mix[..., 3]) if mix is not None else np.zeros(main_a.shape + (4,), np.float32)
        rv = np.asarray(_vec(colors, "_roughv4", (1, 1, 1, 1)), np.float32)
        r = w @ rv + main_a
        g = mix[..., 1] if mix is not None else 0.0
        r = g * (main_a - r) + r
    else:
        r = main_a * main_a + float(fl.get("_RoughNessOffset", 0.0))
    return np.clip(r, ROUGH_MIN, 1.0)


def player_albedo(kind: str, main: np.ndarray, mix: np.ndarray | None, bump_a: np.ndarray | None,
                  colors: dict, fl: dict, linear_colors: bool = True) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """-> (rgb linear (H, W, 3), alpha or None, emission linear (H, W, 3) or None).

    `main` = linear `_MainTex.rgb`, `mix` = raw `_MixMap` (H, W, 4) in 0..1 (None = the shader default, black),
    `bump_a` = `_BumpMapNr` alpha. `linear_colors` False decodes the colour properties as sRGB first."""
    def col(c):
        c = np.asarray(c[:3], np.float32)
        return c if linear_colors else srgb_to_linear(c)

    h, w = main.shape[:2]
    if kind == "eye":
        m = mix if mix is not None else np.zeros((h, w, 4), np.float32)
        tint = zone_tint(m, [col(c) for c in _colors4(colors)])
        c = m[..., 1:2] * (1.0 - tint * 1.2) + tint * 1.2
        return c * main * (1.0 + m[..., 2:3]), m[..., 0], None
    if kind in ("ramp", "lengcy"):
        return main, None, None
    skin_c = col(_vec(colors, "_colorSkin", (1, 1, 1, 1)))
    alpha = alpha_of(mix, fl)
    if kind == "skin":
        metal = bump_a[..., None] if bump_a is not None else 0.0
        rgb = main * (skin_c * (1.0 - metal) + metal)
    else:  # hair
        m = mix if mix is not None else np.zeros((h, w, 4), np.float32)
        tint = zone_tint(m, [col(c) for c in _colors4(colors)])
        tint = tint + m[..., 1:2] * (1.0 - tint)
        a = alpha if alpha is not None else np.ones((h, w), np.float32)
        rgb = skin_c + a[..., None] * (main * tint - skin_c)
    emission = None
    amount = float(fl.get("_Emissive", 0.0)) * float(fl.get("_EmisssionScale", 0.0))
    if mix is not None and amount > 0.0 and float(mix[..., 2].max()) > 0.0:
        emission = main * (mix[..., 2:3] * amount)
    return rgb, alpha, emission


def ramp_emission(uv0: np.ndarray, uv1: np.ndarray, indices: np.ndarray, ramp_rgb: np.ndarray, flow_mask: np.ndarray,
                  fl: dict, size: tuple[int, int], time: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """RampLight emission baked into UV0 space. Per fragment: second UV interpolated, glow only where its x <= 1.

    `uv0` (V, 2) glTF convention (v down), `uv1` (V, 2) the shader's `in_TEXCOORD1` (Unity convention, v up, untouched),
    `ramp_rgb` (Hr, Wr, 3) linear, `flow_mask` (Hm, Wm) the `_MixMap` red channel. At `time` seconds:
    ramp u = fract(time/20 * _RampSpeed), mask offset = fract(time * _RampFlow) (Unity `_Time` = (t/20, t, 2t, 3t)).
    Returns (linear rgb (H, W, 3), covered (H, W) bool)."""
    h, w = size
    speed, flow = float(fl.get("_RampSpeed", 1.0)), float(fl.get("_RampFlow", 1.0))
    hi, lo = float(fl.get("_Rampscale", 1.0)), float(fl.get("_Rampscale2", 0.0))
    ru = (time / 20.0 * speed) % 1.0
    colour = ramp_rgb[min(int(0.5 * ramp_rgb.shape[0]), ramp_rgb.shape[0] - 1),
                      min(int(ru * ramp_rgb.shape[1]), ramp_rgb.shape[1] - 1)]
    off = (time * flow) % 1.0
    mh, mw = flow_mask.shape
    out = np.zeros((h, w, 3), np.float32)
    cover = np.zeros((h, w), bool)
    px = uv0[:, 0] * w
    py = uv0[:, 1] * h
    for tri in indices.reshape(-1, 3):
        if uv1[tri, 0].min() > 1.0:
            continue  # glow cut off everywhere
        x, y = px[tri], py[tri]
        x0, x1 = int(np.floor(x.min())), int(np.ceil(x.max()))
        y0, y1 = int(np.floor(y.min())), int(np.ceil(y.max()))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(d) < 1e-9:
            continue
        l0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d
        l1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
        if not inside.any():
            continue
        u = l0 * uv1[tri[0], 0] + l1 * uv1[tri[1], 0] + l2 * uv1[tri[2], 0]
        v = l0 * uv1[tri[0], 1] + l1 * uv1[tri[1], 1] + l2 * uv1[tri[2], 1]
        on = inside & (u <= 1.0)
        if not on.any():
            continue
        # bilinear, wrapping, texel centres (Unity v up = row 0 at the top of the file's bottom row: flip v)
        fx = ((u + off) % 1.0) * mw - 0.5
        fy = ((1.0 - ((v + off) % 1.0)) % 1.0) * mh - 0.5
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = fx - ix, fy - iy
        r = (flow_mask[iy % mh, ix % mw] * (1 - tx) * (1 - ty) + flow_mask[iy % mh, (ix + 1) % mw] * tx * (1 - ty)
             + flow_mask[(iy + 1) % mh, ix % mw] * (1 - tx) * ty + flow_mask[(iy + 1) % mh, (ix + 1) % mw] * tx * ty)
        e = (r * (hi - lo) + lo)[..., None] * colour[None, None, :]
        sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
        out[sl][on] = e[on]
        cover[sl] |= on
    return out, cover


def dilate(rgb: np.ndarray, cover: np.ndarray, steps: int = 2) -> np.ndarray:
    """Grow covered texels `steps` pixels outwards (so bilinear filtering at island edges does not blend black in)."""
    rgb = rgb.copy()
    cover = cover.copy()
    for _ in range(steps):
        grown = cover.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            src = np.roll(np.roll(cover, dy, 0), dx, 1)
            take = src & ~grown
            rgb[take] = np.roll(np.roll(rgb, dy, 0), dx, 1)[take]
            grown |= src
        cover = grown
    return rgb
