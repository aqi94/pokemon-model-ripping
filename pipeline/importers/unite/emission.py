"""UNITE emission / AO from the game's own lit shaders (WP-C, 2026-11-09).

All three lit families read the packed `_MixMap` ("MROE": R metallic, G roughness, B occlusion, A emission) and add

    emission = albedo.rgb x MixMap.a x <scale> x <colour>                       (added to the lit result)

* `LobbyPokemon/m_lob_basepbr*`, `Lobby/*`, `Hidden/Lobby/*`, ghost*: `<scale>` = `_Emissive` x `_EmisssionScale`, no colour
  (Compiled/==PGAME==_LobbyPokemon_m_lob_basepbr_lv3.txt:479-480 scale = A * _EmisssionScale * _Emissive, :739 added to the
  lit colour x the albedo `u_xlat16_1`).
* `LobbyPokemon/m_lob_basepbrCloth_lv3` only: `<scale>` = `_Emiss` (a float toggle; the JSON's `_Emissive` is a stale
  leftover), colour = `_EmissColor.rgb` (HDR) (==PGAME==_LobbyPokemon_m_lob_basepbrCloth_lv3.txt:3035-3041; added at :3355).
* `Battle/m_bat_actorbase_lv3`: `_Emissive` x `_EmisssionScale` x `_EmisssionColor.rgb`, added AFTER the AO product
  (==PGAME==_Battle_m_bat_actorbase_lv3.txt:844-849: `(lit) * AO + albedo * A * scale * colour`).

Occlusion: lobby `basepbr` = `pow(B, _AOScale)` multiplying the ambient / environment terms only (basepbr_lv3.txt:469-471
and :582-583); Cloth uses B unpowered on its SH ambient (Cloth_lv3.txt:3287; `_AOScale` is declared but never read there);
the battle shader multiplies B into the whole result (actorbase_lv3.txt:849), so its scope is "total".

`_EMISSMASK` (Cloth): the Inspector text says the alpha then masks the Ramp/Matcap and cannot emit, but the compiled
code still adds the emission term in those variants, including the ones with `_IFRAMP_ON` (Cloth_lv3.txt:24231, :27578;
`_EMISSMASK_ON _IFRAMP_ON` variant at :27174, emission at +404..+409). So emission follows the code everywhere; with a ramp/matcap
the alpha is ALSO their mask (`mask_kind` "matcapMask"). Ledger `material:emissMask`.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image

CLOTH_SHADER = re.compile(r"LobbyPokemon/m_lob_basepbrCloth_lv\d$")
BATTLE_SHADER = re.compile(r"Battle/m_bat_actorbase_lv\d$")


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def mask_kind(info: dict) -> str:
    """What the mro(e) alpha is used for: "matcapMask" when the Cloth ramp/matcap consumes it, "emission" when emission
    is on, else "none"."""
    fl = info.get("floats", {})
    if fl.get("_EMISSMASK") and (fl.get("_IFRAMP") or fl.get("_IFMAT")):
        return "matcapMask"
    return "emission" if emission_params(info) else "none"


def emission_params(info: dict) -> dict | None:
    """{"form", "scale", "colour": [r, g, b]} when the material's shader emits, else None (the alpha may also be a ramp/matcap mask)."""
    shader = info.get("shader") or ""
    fl, col = info.get("floats", {}), info.get("colors", {})
    if not shader or "_MixMap" not in info.get("textures", {}):
        return None  # FBX-only: no shader data; every family default is "emission off" (`_Emissive` 0)
    if CLOTH_SHADER.search(shader):
        scale = float(fl.get("_Emiss", 0.0))
        colour = col.get("_EmissColor", (1.0, 1.0, 1.0))
        form = "cloth"
    elif BATTLE_SHADER.search(shader):
        scale = float(fl.get("_Emissive", 0.0)) * float(fl.get("_EmisssionScale", 0.0))
        colour = col.get("_EmisssionColor", (1.0, 1.0, 1.0))
        form = "battle"
    elif "_Emissive" in fl and "_EmisssionScale" in fl:
        scale = float(fl["_Emissive"]) * float(fl["_EmisssionScale"])
        colour = (1.0, 1.0, 1.0)
        form = "scale"
    else:
        return None
    if scale <= 0.0:
        return None
    return {"form": form, "scale": scale, "colour": [float(v) for v in list(colour)[:3]]}


def ao_params(info: dict) -> dict | None:
    """{"scope", "power"} for the occlusion channel (see the module docstring)."""
    shader = info.get("shader") or ""
    if not shader:
        return None
    fl = info.get("floats", {})
    if BATTLE_SHADER.search(shader):
        return {"scope": "total", "power": 1.0}
    if CLOTH_SHADER.search(shader):
        return {"scope": "indirect", "power": 1.0}
    return {"scope": "indirect", "power": float(fl.get("_AOScale", 1.0))}


def bake_emission(albedo_rgba: np.ndarray | None, mix_alpha: np.ndarray) -> np.ndarray:
    """sRGB RGBA texture of albedo.rgb x mix alpha (the scale/colour stay factors), at the albedo's resolution."""
    if albedo_rgba is None:
        albedo_rgba = np.full((*mix_alpha.shape[:2], 4), 255, np.uint8)
    h, w = albedo_rgba.shape[:2]
    a = mix_alpha
    if a.shape[:2] != (h, w):
        a = np.array(Image.fromarray(a).resize((w, h), Image.BILINEAR))
    lin = _srgb_to_linear(albedo_rgba[..., :3].astype(np.float32) / 255.0) * (a.astype(np.float32) / 255.0)[..., None]
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = np.round(_linear_to_srgb(lin) * 255.0)
    out[..., 3] = 255
    return out


def factors(params: dict) -> tuple[tuple[float, float, float], float | None]:
    """(emissive_factor <= 1, KHR strength or None) for scale x colour."""
    rgb = np.clip(np.array(params["colour"], np.float64), 0.0, None) * params["scale"]
    peak = float(rgb.max())
    if peak > 1.0:
        return tuple(float(v) for v in rgb / peak), round(peak, 4)
    return tuple(float(v) for v in rgb), None
