"""UNITE lighting constants per material (WP-F of the 2026-11-09 shader survey), written to `extras.unite.shading`.

Shaders: `Compiled/==PGAME==_LobbyPokemon_m_lob_basepbrCloth_lv3.txt` (lobby; fragment of the first variant, lines 330-700)
and `==PGAME==_Battle_m_bat_actorbase_lv3.txt` (battle; fragment lines 262-430, SH in the vertex stage lines 150-160).

Lit colour (linear, game units; the viewer's `scene/uniteShading.ts` evaluates it):

    d      = albedo (1 - metal)             F0 = mix(0.04, albedo, metal)        (Cloth :466-470, actorbase :317-322)
    spec   = min(D, 100) x (0.25 + 0.25 r) x F0 x NdL x _SpecularColor           (Cloth :523-546, actorbase :346-353)
             D = (a / ((a x)^2 + 1 - x^2))^2, a = r^2; x = N.V in the lobby shaders (their "N.H" slot is fed N.V), N.H in battle
    direct = lightColour x (NdL d + spec)                battle: lightColour x 1.2 clamped to 1.2 (actorbase :355-356)
    SH     = d x (A_top x _SHTopColor + A_bot x _SHBotColor) x _SHScale, A_top = max(0, .125 + .25 n + .125 n^2),
             A_bot = max(0, .125 - .25 n + .125 n^2), n = world normal y (Cloth :495-515; battle vertex stage)
    AO     = B of the mro map: lobby multiplies the indirect terms only, battle the whole result (WP-C `ao.scope`)
    out    = x (1.1295 x + 0.03) 0.45 / (x (1.0935 x + 0.59) 0.45 + 0.08)      (actorbase :414-423)  [tone curve, in the viewer]

Not material data: `_RimlightScale/Scale2/ShadowScale/Color`, `_RimLightDir`, `_MainLightColor` sit in the shader's GLOBAL
constant buffer (next to `_AddLightDir`, `_SunShift`...; Cloth :860-880), so the 91/405 Cloth and 169/291 basepbr JSONs that
list a `_RimlightColor` (up to (51, 13, 5)) hold stale values the shader never reads. Only the fur shaders
(`Hidden/Lobby/m_lob_basepbr5pass`, `ClothFur5Pass`...) declare the rim as Properties, so only they get `rim`.
`_ShadowOffset/_ShadowPow/_ShadowScale` are Properties of the player shaders only (`playerpbrSkin_lv3` :13-15, used at
:559-590): `shadow` is written for the `player` family. Ledger `shader:shading`.
"""
from __future__ import annotations

from . import families

LIT_FAMILIES = {"cloth", "basepbr", "basepbreye", "basepbrmouth", "lengcy", "matcapMask", "fur", "player",
                "actorbase", "actorbaseeye", "actorbasemouth", "actorMatcap"}
BATTLE_FAMILIES = {"actorbase", "actorbaseeye", "actorbasemouth", "actorMatcap"}
LIGHT_CLAMP = 1.2  # actorbase_lv3.txt:355-356


def _rgb(v, d=(1.0, 1.0, 1.0)) -> list[float]:
    v = list(v) if v is not None else list(d)
    return [round(float(x), 5) for x in v[:3]]


def shading_params(info: dict, family: str) -> dict | None:
    """`extras.unite.shading` for a lit character material, or None (fx, glass, unlit and scene shaders)."""
    if family not in LIT_FAMILIES:
        return None
    fbx = info.get("source") == "fbx" or not info.get("shader")
    base = families.default_params(info.get("shader") or family)
    fl = {**base, **(info.get("floats") or {})} if not fbx else dict(base)
    co = {**base, **(info.get("colors") or {})} if not fbx else dict(base)
    battle = family in BATTLE_FAMILIES
    out = {
        "form": "battle" if battle else ("cloth" if family == "cloth" else "pbr"),
        "specTerm": "ndh" if battle else "nov",
        "shTop": _rgb(co.get("_SHTopColor"), (2.5, 2.5, 2.5)),
        "shBot": _rgb(co.get("_SHBotColor"), (2.5, 2.5, 2.5)),
        "shScale": float(fl.get("_SHScale", 1.0)),
        "shShadow": float(fl.get("_SHshadow", 1.0)),
        "specColor": _rgb(co.get("_SpecularColor")),
        "roughOffset": float(fl.get("_RoughNessOffset", 0.0)),
        "metalOffset": float(fl.get("_MetallicOffset", 0.0)),
    }
    if battle:
        out["lightClamp"] = LIGHT_CLAMP
    if fbx:
        out["inferred"] = True  # the shader's own Properties defaults (no JSON in the rip)
    if family == "fur" and "_RimlightColor" in co and any(c > 0 for c in list(co["_RimlightColor"])[:3]):
        out["rim"] = {"color": [round(float(x), 5) for x in list(co["_RimlightColor"])[:4]], "scale": float(fl.get("_RimlightScale", 1.0)),
                      "shadowScale": float(fl.get("_RimlightShadowScale", 1.0))}
    if family == "player" and "_ShadowPow" in fl:
        out["shadow"] = {"offset": float(fl.get("_ShadowOffset", 0.0)), "pow": float(fl.get("_ShadowPow", 1.0)), "scale": float(fl.get("_ShadowScale", 0.0))}
    return out
