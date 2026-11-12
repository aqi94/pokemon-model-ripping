"""UNITE alpha mode from shader data (WP-B, 2026-11-09). Ledger `material:transparency`.

Authority: decompiled shaders in marcy/Assets/Shaders/Compiled. The `Blend Zero Zero` line in a dump is a decompiler
placeholder; the real state is `Blend [_SrcBlendFactor] [_DstBlendFactor]`. Findings:

- basepbrCloth_lv3 / basepbrAnimation(Basic)_lv3: output alpha = lerp(1, albedo.a, saturate(_DstBlendFactor))
  (`==PGAME==_LobbyPokemon_m_lob_basepbrCloth_lv3.txt` ~l.1563-1570, `max(_DstBlendFactor, 0.0)`); alpha is the
  Ramp/Mat mask and the output stays 1 when `_ALPHAMASK` is on (properties l.9, 36). The `_ALPHATEST_ON` variant discards
  when albedo.a < _Cutoff (l.33134-33137); the material float for it is `_BlendMode` 1 (brief, survey).
- basepbr / basepbreye / basepbrmouth: `SV_Target0.w = _MainTex.w` (basepbr_lv3.txt l.824), blended by the factor props.
- basepbrBubble_lv3: `Blend SrcAlpha OneMinusSrcAlpha`, ZWrite Off hard-coded (l.46-48).
- Battle actorbase_lv3: `Blend SrcAlpha OneMinusSrcAlpha`, ZWrite Off (l.100-101), alpha = albedo.a * _CustomAlpha (l.363).
- OutlineAlpha: `_ZWrite 0` marks the blend (as before).
"""
from __future__ import annotations

import re

LERP_FAMILY = re.compile(r"m_lob_basepbr(Cloth|Animation|AnimationBasic)_lv3$", re.I)


def decide(shader: str | None, fl: dict) -> dict:
    """{mode: OPAQUE|BLEND|MASK, cutoff, source, alpha_factor} for a JSON material."""
    s = shader or ""
    cutoff = float(fl.get("_Cutoff", 0.5))
    out = {"mode": "OPAQUE", "cutoff": None, "source": "opaque", "alpha_factor": 1.0}
    if not s:
        return out
    alphamask = bool(fl.get("_ALPHAMASK"))
    if fl.get("_BlendMode") == 1.0 and not alphamask:
        return {**out, "mode": "MASK", "cutoff": cutoff, "source": "_BlendMode=1"}
    if re.search(r"basepbrBubble", s):
        return {**out, "mode": "BLEND", "source": "bubble (hard-coded SrcAlpha/OneMinusSrcAlpha)"}
    if re.search(r"m_bat_actorbase_lv3", s):
        return {**out, "mode": "BLEND", "source": "actorbase (hard-coded SrcAlpha/OneMinusSrcAlpha)",
                "alpha_factor": float(fl.get("_CustomAlpha", 1.0))}
    if fl.get("_ZWrite") == 0.0:
        return {**out, "mode": "BLEND", "source": "_ZWrite=0"}
    if LERP_FAMILY.search(s):
        if float(fl.get("_DstBlendFactor", 0.0)) > 0.0 and not alphamask:
            return {**out, "mode": "BLEND", "source": "lerp(1, a, saturate(_DstBlendFactor))"}
        return {**out, "source": "lerp alpha forced 1 (_DstBlendFactor 0 or _ALPHAMASK)"}
    if fl.get("_SrcBlendFactor") == 5.0 and fl.get("_DstBlendFactor") == 10.0 and re.search(r"basepbr", s):
        return {**out, "mode": "BLEND", "source": "_SrcBlendFactor 5 / _DstBlendFactor 10, alpha = albedo.a"}
    return out
