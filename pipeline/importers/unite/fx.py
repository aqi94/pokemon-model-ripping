"""WP-E: the UNITE `PGAME/Fx/*` effect shaders' formula, baked at t = 0 (docs/ledger/unite.json `fx:formula`).

Verified in `Assets/Shaders/Compiled/==PGAME==_Fx_m_1P_Base.txt` (fragment, `main`):
    rgb   = pow(Tex1.rgb, _Power1) * _Emission1 * _Color1.rgb * vertexColour.rgb
    alpha = Tex1.a * vertexColour.a ; output alpha = alpha * _Alpha1 (`_Color1.a` is never read)
    rgb  *= alpha when `_IF_ADD_MODE` (one-one blending then adds rgb * alpha)
    uv    = vertex uv * _tex1_st.xy + _tex1_st.zw + fract(time * _Tex1_Speed)   (vertex stage)
`m_2P_mask`: alpha *= Tex2.r (no `_Power1`); `m_2P_dissolve` / `m_3P_noise_dissolve`: alpha *= smoothstep of
Tex2.r through `_DissolveOffset1`/`_DissolveEdge1`; `m_3P_2Mask_Overlay`: colour texture `_Dtex`,
rgb = Dtex.rgb * tex1.rgb * Tex2.r, alpha = Dtex.a * tex1.a * Tex2.a.
Only `m_1P_Modulate`, `m_1P_fresnel_screen`, `m_2P_fresnel_noise` read `_Color1.a`.
"""
from __future__ import annotations

import hashlib
import numpy as np
from PIL import Image

COLOR_ALPHA_SHADERS = ("m_1P_Modulate", "m_1P_fresnel_screen", "m_2P_fresnel_noise")
FAMILIES = {
    "m_1P_Base": "base", "m_1P_Base_Half": "base",
    "m_2P_mask": "mask", "m_2P_mask_quickmask": "mask",
    "m_2P_dissolve": "dissolve", "m_3P_noise_dissolve": "dissolve",
    "m_3P_2Mask_Overlay": "overlay3",
}


def shader_tail(info: dict) -> str | None:
    s = info.get("shader") or ""
    return s.split("/")[-1] if "/Fx" in s else None


def uses_color_alpha(info: dict) -> bool:
    return shader_tail(info) in COLOR_ALPHA_SHADERS


def _v(info, k, default):
    v = info["colors"].get(k)
    return tuple(float(x) for x in v[:4]) if v is not None else default


def plan(info: dict) -> dict | None:
    """The per-family recipe for a JSON Fx material, or None (FBX-only / unknown shader: unchanged behaviour)."""
    tail = shader_tail(info)
    if tail is None or info.get("source") == "fbx":
        return None
    fam = FAMILIES.get(tail, "generic")
    fl, tx = info["floats"], info["textures"]
    p = {"family": fam, "shader": tail, "colorSlot": "_Dtex" if fam == "overlay3" else None,
         "power": float(fl.get("_Power1", 1.0)) if tail.startswith("m_1P_Base") else 1.0,
         "emission": float(fl.get("_Emission1", 1.0)),
         "st": {k: _v(info, k, (1, 1, 0, 0)) for k in ("_tex1_st", "_tex2_st", "_Dtex_st") if k in info["colors"]},
         "speeds": {k: _v(info, k, (0, 0, 0, 0))[:2] for k in ("_Tex1_Speed", "_Tex2_Speed", "_Tex3_Speed") if k in info["colors"]},
         "roles": {}}
    if fam == "mask" and not (tail.endswith("quickmask") and fl.get("_use_mask", 0.0)):
        p["roles"]["_Tex2"] = "mask"
    elif fam == "mask":
        p["roles"]["_Tex2"] = "mask"
    if fam == "dissolve":
        p["roles"]["_Tex2"] = "dissolve"
        p["dissolve"] = (float(fl.get("_DissolveOffset1", 0.5)), float(fl.get("_DissolveEdge1", 1.0)))
        if "_Tex3" in tx:
            p["roles"]["_Tex3"] = "noise"  # offsets the main uv by _noise1 (not baked)
    if fam == "overlay3":
        p["roles"].update({"_Dtex": "colour", "_tex1": "mask1", "_Tex2": "mask2"})
    return p


def _s2l(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _l2s(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _sample(img: np.ndarray, u: np.ndarray, v_unity: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.floor((u % 1.0) * w).astype(int) % w
    y = np.floor(((1.0 - v_unity) % 1.0) * h).astype(int) % h
    return img[y, x].astype(np.float32) / 255.0


def _rel(u, v, st_main, st_other):
    """Coordinates of the other texture for the grid point (u, v) given in the main texture's own frame."""
    return (((u - st_main[2]) / (st_main[0] or 1.0)) * st_other[0] + st_other[2],
            ((v - st_main[3]) / (st_main[1] or 1.0)) * st_other[1] + st_other[3])


def smoothstep_dissolve(t: np.ndarray, offset: float, edge: float) -> np.ndarray:
    """`m_2P_dissolve` fragment: x = (t + (2 off - 1) - (1 - edge)) / (2 edge - 1), smoothstep."""
    den = 2 * edge - 1
    x = (t + (2 * offset - 1) - (1 - edge)) / (den if abs(den) > 1e-6 else 1e-6)
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def needs_bake(p: dict) -> bool:
    return bool(p["roles"]) or abs(p["power"] - 1.0) > 1e-6


def bake(p: dict, main: np.ndarray, others: dict[str, np.ndarray]) -> np.ndarray:
    """RGBA uint8 in the main texture's frame with power, mask and dissolve baked at t = 0 (scroll offset 0)."""
    h, w = main.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w]
    u = (gx + 0.5) / w
    v = 1.0 - (gy + 0.5) / h
    st = p["st"]
    fam = p["family"]
    main_st = st.get("_Dtex_st" if fam == "overlay3" else "_tex1_st", (1, 1, 0, 0))
    m = main.astype(np.float32) / 255.0
    rgb, a = _s2l(m[..., :3]), m[..., 3].copy()
    if abs(p["power"] - 1.0) > 1e-6:
        rgb = np.maximum(rgb, 1e-6) ** p["power"]
    if fam in ("mask", "dissolve") and "_Tex2" in others:
        t = _sample(others["_Tex2"], *_rel(u, v, main_st, st.get("_tex2_st", (1, 1, 0, 0))))[..., 0]
        a = a * (t if fam == "mask" else smoothstep_dissolve(t, *p["dissolve"]))
    elif fam == "overlay3":
        if "_tex1" in others:
            s = _sample(others["_tex1"], *_rel(u, v, main_st, st.get("_tex1_st", (1, 1, 0, 0))))
            rgb, a = rgb * _s2l(s[..., :3]), a * s[..., 3]
        if "_Tex2" in others:
            s = _sample(others["_Tex2"], *_rel(u, v, main_st, st.get("_tex2_st", (1, 1, 0, 0))))
            rgb, a = rgb * s[..., :1], a * s[..., 3]
    out = np.concatenate([_l2s(rgb), a[..., None]], -1)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def key_suffix(p: dict, info: dict) -> str:
    sig = repr((p["family"], p["power"], p.get("dissolve"), sorted(p["st"].items()),
                sorted((k, v["name"]) for k, v in info["textures"].items())))
    return hashlib.md5(sig.encode()).hexdigest()[:8]


def extras(p: dict, info: dict) -> dict:
    return {"family": p["family"], "shader": p["shader"], "power": p["power"], "emission": p["emission"],
            "speeds": {k: list(v) for k, v in p["speeds"].items()},
            "st": {k: list(v) for k, v in p["st"].items()},
            "textures": {k: {"name": v["name"], "role": p["roles"].get(k, "colour" if k == (p["colorSlot"] or "_Tex1") else "other")}
                         for k, v in info["textures"].items()},
            "bakedAt": 0}
