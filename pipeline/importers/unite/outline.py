"""Game outline data of a UNITE material (WP-G of the 2026-11-09 shader survey; ledger `shader:outline`).

The lit lobby shaders (`m_lob_basepbr*`, `basepbrCloth*`, ...) draw an inverted-hull pass `OUTLINE` (`Cull Front`):
  colour = mix(_OutlineColor.rgb, 1, clamp(dot(N_world, normalize(_lightDir.xyz)), 0, 1)) * (_VTEX ? _OutlineMap.rgb : COLOR_0.rgb)
          (`m_lob_basepbr_lv3.txt` 42185-42197, Cloth 606300-606330)
  width  = the vertex pushed along the projected normal by 1/216 NDC (0.0046296, :42220), times the clip w, so a
           constant screen width; plus a push away from the camera along the view ray of
           length(offset) * (_VTEX ? _OutlineMap.a : 1) * |_Offset| * 0.15 * 100 view units (:42198, :42232-42238).
Battle `m_bat_actorbase` has no such pass (only `UBuffOutLine`, the buff aura, width `_UBuffWidth`): no data, no outline.
"""

from __future__ import annotations

from . import families

# Shaders whose `_OutlineColor` is not an outline pass (Fx colour, unlit backdrops, the battle buff aura).
_NO_PASS = ("/Fx/", "Unlit", "Battlefield", "m_bat_", "NoOutline", "PlaneWater", "Valentine")
WIDTH_NDC = 1.0 / 216.0  # 0.00462962966 in the OUTLINE vertex programs


def _vec(v, n=3) -> list[float]:
    return [round(float(x), 5) for x in list(v)[:n]]


def outline_params(info: dict) -> dict | None:
    """`{color, lightDir, offset, width_ndc, vtex, vcolor2n, map?, source}` or None when the material has no outline pass.

    JSON materials: the shader must carry the OUTLINE pass (family check) and the JSON holds `_OutlineColor`.
    FBX-only materials: a `t_vc_` texture in the block marks a lit lobby shader; the colour/light/offset are the family
    defaults (`families.default_params`), the map is that texture."""
    fl, col, tx = info.get("floats") or {}, info.get("colors") or {}, info.get("textures") or {}
    shader = info.get("shader") or ""
    block = info.get("block") or []
    if shader:
        if "_OutlineColor" not in col or any(s in shader for s in _NO_PASS):
            return None
        src, mapname, dflt = "json", (tx.get("_OutlineMap") or {}).get("name"), {}
    else:
        mapname = next((n for n in block if n.lower().startswith("t_vc_")), None)
        fam = info.get("family") or ""
        if not mapname or fam.startswith("actor") or fam in ("fx", "unlit"):
            return None
        dflt = families.default_params(fam if fam in families.FAMILY_SHADER else "basepbr")
        src = "fbx"
    vtex = float(fl["_VTEX"]) if "_VTEX" in fl else (1.0 if mapname else float(dflt.get("_VTEX", 0.0)))  # FBX-only: a t_vc_ in the block = map in use
    out = {
        "color": _vec(col.get("_OutlineColor") or dflt.get("_OutlineColor") or (0.5, 0.5, 0.5)),
        "lightDir": _vec(col.get("_lightDir") or dflt.get("_lightDir") or (9.48, 3.68, 0.0)),
        "offset": float(fl.get("_Offset", dflt.get("_Offset", -5.0))),
        "width_ndc": WIDTH_NDC,
        "vtex": bool(vtex),
        "vcolor2n": bool(fl.get("_VCOLOR2N", 0.0)),
        "source": src,
    }
    if vtex and mapname:
        out["mapName"] = mapname
    return out


def add_outline(extras: dict, info: dict, load_map) -> None:
    """Write `extras.unite.outline`. `load_map(name) -> texture key | None` stores the map PNG (sRGB, alpha kept)."""
    o = outline_params(info)
    if not o:
        return
    name = o.pop("mapName", None)
    key = load_map(name) if name else None
    if key:
        o["map"] = {"texture": key, "uv": 0}
    extras.setdefault("unite", {})["outline"] = o
