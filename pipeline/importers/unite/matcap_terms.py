"""The matcap family of UNITE's lit shaders, read from the decompiled shaders (WP-D of the 2026-11-09 shader survey).

`Assets/Shaders/Compiled/*.txt` (Marcy4000/Unite) shows that "matcap" is four unrelated terms, not one metal reflection:

* **matTex** (Cloth `_IFMAT_ON`, `_MatTex` x `_MatTexColor`, 239 of the lobby JSONs): an ADDITIVE highlight,
  `matcap(N_view.xy*0.5+0.5) * (0.25 + 0.75*N.L*shadow) * _MatTexColor`, added after the lit colour, independent of
  metal (`==PGAME==_LobbyPokemon_m_lob_basepbrCloth_lv3.txt` lines 9640-9653, the 0.25/0.75 at 9433-9437, the matcap UV
  basis at 9440-9467; the mask `u_xlat16_5` is the constant 1 there, line 9272).
* **rampLight** (Cloth `_IFRAMP_ON`, `_RampTex` x `_RampTexColor`, 106 JSONs): an additive fresnel ramp,
  `ramp(min(NoV^2, 0.96) * _RampTex_ST.x + _RampTex_ST.z, v = 0) * (same light term) * _RampTexColor` (same file,
  lines 5855-5876; NoV = line 5640 + 280-287 of that variant).
* **env** (Cloth `_MATCAPENV_ON` = JSON `_Matcap`, `_MatcapTex` x `_MatcapColor` 4.84..27.5): an environment term in
  place of the scene probe: `matcap(uv, lod = r*(1.7 - 0.7r)*6) * light * _MatcapColor * F0 * AO`, F0 =
  mix(0.04, albedo, metal) (Cloth, lines 49852-49881; NO Schlick factor there). Lengcy / MatcapMask (`_IF_MATREFL`,
  `_MatMap` x `_MatMapMul`) use the Unity indirect-specular form: `matcap(uv, lod) * _MatMapMul / (r^4 + 1) *
  mix(F0, grazing, (1 - NoV)^4)` (`Hidden_Lobby_m_lob_basepbrClothLengcy_lv3.txt` lines 735-760). The battle
  `actorMatcap` adds `matcap(uv, lod) * _SpecularColor` (`==PGAME==_Battle_m_bat_actorMatcap_lv3.txt` line 812-871).
  Without the keyword the shader samples the scene reflection probe (`unity_SpecCube0`), which is not in the rip: those
  materials get nothing (gap, ledger `material:matcapMetal`).
* **glass** (`ghostGlass` / battle `actorglass`): transparent face-on, opaque at grazing angles (see `glass_terms`).
* **bubble** (`basepbrBubble`, Buzzwole 0794 body): unlit `albedo + matcap(uv) + ramp*NoV + spec*0.1`
  (`==PGAME==_Lobby_m_lob_basepbrBubble_lv3.txt` lines 544-568).

The shader multiplies the finished colour by 0.45 and runs a fitted ACES curve (2.51/0.03/2.43/0.59/0.08, Cloth lines
9680-9702) in the fragment; the viewer has its own exposure and tone mapping, so these terms are added in linear light
before it, unscaled.

Everything here returns plain dicts for `extras.unite.{matTex,rampLight,env,glass}`; the viewer (`scene/matcap.ts`)
draws them. Texture references are `{"texture": key, "uv": 0}`, resolved by the glTF writer.
"""

from __future__ import annotations

import os
import re

import numpy as np
from PIL import Image

from ...common.model import TextureImage
from . import families

# Shader Properties defaults of the families a missing JSON stands for (decompiled `Properties` blocks):
CLOTH_MATTEX_COLOR = (2.0, 2.0, 2.0)  # basepbrCloth `_MatTexColor` default, line 33 of the dump
CLOTH_MATCAP_COLOR = (4.84, 4.84, 4.84)  # `_MatcapColor` default, line 40
MATMAP_MUL = 4.84  # Lengcy / MatcapMask `_MatMapMul` default (Lengcy dump line 20)
GLASS_DEFAULTS = {"offset": 0.0, "pow": 2.0, "scale": 1.0, "top": [2.5, 2.5, 2.5, 1.0], "bot": [2.5, 2.5, 2.5, 1.0],
                  "alpha": 1.0}  # ghostGlass Properties lines 11-18; actorglass lines 9-14 (same numbers)


def _c(info: dict, key: str, default) -> list[float]:
    c = info["colors"].get(key)
    return [float(x) for x in c] if c else [float(x) for x in default]


def _f(info: dict, key: str, default: float) -> float:
    v = info["floats"].get(key)
    return float(v) if v is not None else default


def _load(name: str, folder: str, index, textures: dict[str, TextureImage], png) -> str | None:
    path = png(folder, name, index)
    if not path:
        return None
    if name not in textures:
        textures[name] = TextureImage(key=name, rgba=np.array(Image.open(path).convert("RGBA"), np.uint8), srgb=True)
    return name


def family_of_info(info: dict, material: str = "") -> str:
    """Family token: WP-A's `info["family"]`, else from the JSON shader, else (FBX-only) inferred from the name."""
    fam = info.get("family")
    if fam:
        return fam
    if info.get("shader"):
        return families.family_of(info["shader"])
    kind = "battle" if re.search(r"_bat_", material or info.get("name", ""), re.I) else "lobby"
    return families.infer_family(material or info.get("name", ""), info.get("block") or [], kind)


def lit_terms(info: dict, folder: str, index, textures: dict[str, TextureImage], png, material: str = "") -> dict:
    """`extras.unite` sub-keys matTex / rampLight / env for a lit character material (empty dict = none apply)."""
    fam = family_of_info(info, material)
    tx = info["textures"]
    fbx = info.get("source") == "fbx"
    out: dict = {}

    def tref(slot: str) -> dict | None:
        t = tx.get(slot)
        key = _load(t["name"], folder, index, textures, png) if t else None
        return {"texture": key, "uv": 0} if key else None

    dflt = families.default_params(fam) if fbx else {}  # FBX-only: the shader's own Properties defaults are the data
    if fam == "cloth":
        if fbx:
            # keyword terms apply only when the shader default has them on (Cloth `_IFMAT` default 0): the keyword
            # statistics (219 of 357 Cloth JSONs naming a t_matcap_ have _IFMAT on with it in _MatTex) are not data for
            # one material (ledger `material:matcapMetal`)
            cap = tref("_MatCapTex")
            if cap and dflt.get("_IFMAT") == 1.0:
                out["matTex"] = {"texture": cap, "color": list(CLOTH_MATTEX_COLOR), "inferred": True}
            return out
        if _f(info, "_IFMAT", 0.0) == 1.0 and (r := tref("_MatTex")):
            out["matTex"] = {"texture": r, "color": _c(info, "_MatTexColor", CLOTH_MATTEX_COLOR + (0,))[:3]}
        if _f(info, "_IFRAMP", 0.0) == 1.0 and (r := tref("_RampTex")):
            t = tx["_RampTex"]
            out["rampLight"] = {"texture": r, "color": _c(info, "_RampTexColor", (1, 1, 1, 0))[:3],
                                "scale": float(t["scale"][0]), "offset": float(t["offset"][0])}
        if _f(info, "_Matcap", 0.0) == 1.0 and (r := tref("_MatcapTex")):
            out["env"] = {"kind": "cloth", "texture": r, "color": _c(info, "_MatcapColor", CLOTH_MATCAP_COLOR + (0,))[:3]}
    elif fam in ("lengcy", "matcapMask"):
        if fbx:
            cap = tref("_MatCapTex")  # `_MATREFL` default is 0 on Lengcy/MatcapMask: no term (17 of 23 JSON materials with a _MatMap have it on; not data)
            if cap and dflt.get("_MATREFL") == 1.0:
                out["env"] = {"kind": "pbr", "texture": cap, "color": [MATMAP_MUL] * 3, "inferred": True}
        elif _f(info, "_MATREFL", 0.0) == 1.0 and (r := tref("_MatMap")):
            out["env"] = {"kind": "pbr", "texture": r, "color": [_f(info, "_MatMapMul", MATMAP_MUL)] * 3}
    elif fam == "actorMatcap":
        slot = "_MatCapTex" if fbx else "_MatMap"
        if (r := tref(slot)):
            col = _c(info, "_SpecularColor", (1, 1, 1, 1))[:3]
            out["env"] = {"kind": "battle", "texture": r, "color": col, **({"inferred": True} if fbx else {})}
    elif fam == "bubble":
        if (r := tref("_MatMap")):
            out["env"] = {"kind": "bubble", "texture": r, "color": [1.0, 1.0, 1.0]}
    return out


def glass_terms(info: dict, matcap_key: str, material: str = "") -> dict:
    """`extras.unite.glass`: the ghostGlass / actorglass alpha and colour (constants from the JSON, else the shader
    defaults). ghostGlass (lobby; `ghostGlass_lv3.txt` lines 192-239)::

        NoV = min(|N.V| + 0.004, 1);  t = clamp((NoV - _ShadowOffset) * _ShadowPow, 0, 1);  k = clamp(1 - t + main.a, 0, 1)
        c = ramp(uv_matcap + k * _ShadowScale) * mix(_SHTopColor, _SHBotColor, k)        (rgba)
        alpha = clamp(_Cutoff * clamp(k * Bot.a + c.a, 0, 1) - NoV, 0, 1)                 (transparent face-on)
        rgb   = c.rgb + _Cutoff * (c.rgb * (L - 1) + main.rgb)

    actorglass (battle; `actorglass_lv3.txt` lines 199-217) differs in the alpha: ``clamp((k + c.a) * _CustomAlpha)`` with
    k = 1 - clamp((NoV - _RoughNessOffset) * _EmisssionScale), NoV = clamp(N.V) and the matcap offset `_MetallicOffset`.
    The decompiled property names differ, the defaults do not."""
    fbx = info.get("source") == "fbx"
    fl = info["floats"]
    actor = bool(re.search(r"actorglass", info.get("shader", ""), re.I)) or (
        fbx and re.search(r"_bat_", material, re.I) is not None)
    if actor:
        off, pw, sc = _f(info, "_RoughNessOffset", 0.0), _f(info, "_EmisssionScale", 2.0), _f(info, "_MetallicOffset", 1.0)
        alpha = _f(info, "_CustomAlpha", 1.0)
    else:
        d = GLASS_DEFAULTS
        off, pw, sc = _f(info, "_ShadowOffset", d["offset"]), _f(info, "_ShadowPow", d["pow"]), _f(info, "_ShadowScale", d["scale"])
        alpha = _f(info, "_Cutoff", d["alpha"])
    d = GLASS_DEFAULTS
    g = {"kind": "actor" if actor else "ghost", "texture": {"texture": matcap_key, "uv": 0}, "offset": off, "pow": pw,
         "scale": sc, "top": _c(info, "_SHTopColor", d["top"]), "bot": _c(info, "_SHBotColor", d["bot"]), "alpha": alpha}
    if fbx or not info.get("shader"):
        g["inferred"] = True  # no JSON: the shader's Properties defaults
    return g
