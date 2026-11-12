"""Shader family of a UNITE material (WP-A of the 2026-11-09 shader survey).

`family_of(shader_name)` maps a material JSON's `m_Shader.Name` to a family token; `infer_family(...)` guesses the
family of an FBX-only material (battle models, standalone lobby holowears: no `Materials/*.json`) from its name and
its texture block; `default_params(shader_or_family)` returns the Properties defaults of the decompiled shader
(`Assets/Shaders/Compiled`, written to `family_defaults.json` by `tools/gen_unite_family_defaults.py`), i.e. the values a
missing JSON would have held. Ledger `shader:familyInference`.
"""

from __future__ import annotations

import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))

# Ordered (regex on the shader name, token). First match wins.
_RULES: list[tuple[re.Pattern, str]] = [(re.compile(rx, re.I), tok) for rx, tok in [
    (r"ghostGlass|TranslateGlass|basepbrGlass", "glass"),
    (r"basepbrBubble", "bubble"),
    (r"PlayerPBR|m_lob_ghost(?:Cloth|eye|pbr)", "player"),
    (r"basepbrClothLengcy|PlayerPBRLengcy", "lengcy"),
    (r"bat_actorMatcap", "actorMatcap"),
    (r"MatcapMask", "matcapMask"),
    (r"Fur|basepbr5pass", "fur"),
    (r"basepbrmouth", "basepbrmouth"),
    (r"basepbreye", "basepbreye"),
    (r"basepbrOutlineAlpha|basepbrAnimation", "basepbr"),
    (r"basepbrCloth", "cloth"),
    (r"basepbr", "basepbr"),
    (r"bat_actorbaseeye", "actorbaseeye"),
    (r"bat_actorbasemouse", "actorbasemouth"),
    (r"bat_actorbase", "actorbase"),
    (r"/Fx2?/|UI/ui_3P|^Fx|m_banzi|ValentineCloud|water_reflection|RippleReflection", "fx"),
    (r"Fullroughness|NoHighLight|m_bat_Postboard|m_lob_Fullroughness", "fullroughness"),
    (r"MetalEmiss", "metalEmiss"),
    (r"Tree|Grass|Foliage|Snow|Decla|Terrian|terrianMask|BasicScene|cloudfront|SmallGrass", "scene"),
    (r"water|foam", "water"),
    (r"^Unlit/|LobbyScene|Lobby/|PlaneDotLight|FakeShadow|Battlefield|UI/|CustomTexture|^Standard|Universal Render", "unlit"),
]]


def family_of(shader_name: str) -> str:
    """Family token of a Unity shader name; an unknown or empty name returns the raw name ('' when empty)."""
    for rx, tok in _RULES:
        if rx.search(shader_name or ""):
            return tok
    return shader_name or ""


_defaults: dict | None = None


def _load_defaults() -> dict:
    global _defaults
    if _defaults is None:
        try:
            with open(os.path.join(_HERE, "family_defaults.json"), encoding="utf-8") as f:
                _defaults = json.load(f)
        except OSError:
            _defaults = {}
    return _defaults


def default_params(key: str) -> dict:
    """Properties defaults (floats, colours as 4-lists) of a shader name, or of a family token (the most common
    character shader of that family, `FAMILY_SHADER`). Unknown -> {}."""
    d = _load_defaults()
    name = key if key in d else FAMILY_SHADER.get(key, "")
    return dict(d.get(name, {}))


FAMILY_SHADER = {
    "basepbr": "==PGAME==/LobbyPokemon/m_lob_basepbr_lv3",
    "basepbreye": "==PGAME==/LobbyPokemon/m_lob_basepbreye_lv3",
    "basepbrmouth": "==PGAME==/LobbyPokemon/m_lob_basepbrmouth_lv3",
    "cloth": "==PGAME==/LobbyPokemon/m_lob_basepbrCloth_lv3",
    "lengcy": "Hidden/Lobby/m_lob_basepbrClothLengcy_lv3",
    "matcapMask": "==PGAME==/Lobby/m_lob_basepbrMatcapMask_lv3",
    "fur": "==PGAME==/Lobby/m_lob_basepbrClothFur5Pass_lv3",
    "actorbase": "==PGAME==/Battle/m_bat_actorbase_lv3",
    "actorMatcap": "==PGAME==/Battle/m_bat_actorMatcapMask_lv3",
    "fullroughness": "==PGAME==/BattleV3/m_bat_FullroughnessV3_lv0",
    "metalEmiss": "==PGAME==/BattleV3/m_bat_MetalEmissV3_lv3",
    "fx": "==PGAME==/Fx/m_1P_Base",
}


# ---- FBX-only materials ------------------------------------------------------------------------------------------
_FX_NAME = re.compile(r"(^|_)fx(_|$)", re.I)
_EYE = re.compile(r"eye", re.I)
_MOUTH = re.compile(r"mouth|mouse", re.I)
_BODY = re.compile(r"body", re.I)
_OUTFIT = re.compile(r"\d{6}[a-z]?(_|$)", re.I)


def _has(block: list[str], rx: str) -> bool:
    r = re.compile(rx, re.I)
    return any(r.match(os.path.basename(n)) for n in block)


def infer_family(material_name: str, block_names: list[str] | tuple[str, ...], kind: str) -> str:
    """Family of a material that has no JSON, from its name and texture block (`fbx.material_blocks`).

    kind = "battle" | "lobby" (scene materials count as lobby). Calibrated on the lobby scene materials that DO have
    JSONs (`tools/calibrate_unite_families.py`, ledger `shader:familyInference`). Evidence, strongest first:
      name `fx` token                         -> fx
      `t_nm_` normal map                      -> player (PlayerPBR / ghost shaders)
      eye / mouth in the name                 -> basepbreye / basepbrmouth (lobby), actorbaseeye / actorbasemouth (battle)
      `t_noise_` in the block                 -> fur (the fur shell's shape map)
      own `t_matcap_` + `t_mipmap03`          -> cloth (`_IFMAT` + generic env matcap)
      own `t_matcap_` on a body part          -> matcapMask when the map is a MROE (`t_mroe_`), else basepbr
      `t_ramp_` (no `t_m_`)                   -> cloth (`_IFRamp`)
      outfit with `t_fx_changeskin`, no mroe  -> lengcy
      other outfit                            -> cloth;  other body/short names -> basepbr
    Battle models use `actorbase` (`actorMatcap` when the block names its own matcap), `m_bat_actorbase*`.
    A matcap in the block means the albedo alpha is a mask, not opacity (survey)."""
    n = material_name or ""
    b = list(block_names or [])
    if _FX_NAME.search(n):
        return "fx"
    cap = _has(b, r"t_matcap")
    if kind == "battle":
        if _MOUTH.search(n):
            return "actorbasemouth"
        if _EYE.search(n):
            return "actorbaseeye"
        return "actorMatcap" if cap else "actorbase"
    if _has(b, r"t_nm_"):
        return "player"
    if _MOUTH.search(n):
        return "basepbrmouth"
    if _EYE.search(n):
        return "basepbreye"
    if _has(b, r"t_noise"):
        return "fur"
    env = _has(b, r"t_mipmap03")
    if cap and env:
        return "cloth"
    body = bool(_BODY.search(n))
    mroe = _has(b, r"t_mroe_")
    if cap and body:
        return "matcapMask" if mroe else "basepbr"
    if _has(b, r"t_ramp") and not _has(b, r"t_m_"):
        return "cloth"
    outfit = bool(_OUTFIT.search(n))
    if outfit:
        if _has(b, r"t_fx_changeskin") and not mroe and not cap:
            return "lengcy"
        return "cloth"
    return "basepbr"


# Outfit shaders that read the same inputs (cloth, Lengcy, ClothMatcapMask): a block cannot tell them apart (survey:
# 19 of 34 Lengcy and 12 of 42 MatcapMask materials look like plain cloth), so rules that need only the shared
# behaviour should test the group.
CLOTH_GROUP = frozenset({"cloth", "lengcy", "matcapMask"})
