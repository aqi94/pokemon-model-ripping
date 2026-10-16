"""Map a Trinity light rig onto the viewer's `Environment` (viewer/src/scene/environment.ts).

Mapping v1 (see docs/lighting_rigs.md for what is verified and what is inferred):
- effective rig = merge(master, area rig) then + weather deltas (`apply_delta`, the offset rigs are additive);
- a material category `c` (`.trmtr` CategoryLabel: SV trainer skin 5, Pokemon 6) selects `dir_c` (key light),
  `rim_c` (back/rim light) and `GILightCommon.Color{c}/Intensity{c}` (ambient);
- intensities are normalised so that `master` with category 5 lands on the viewer's Studio levels
  (key 1.2, ambient 1.9): the engine's units are unknown, the rigs' *ratios* are what we import;
- light colours are passed through as explicit linear colours (`keyColor`/`skyColor`/`backColor`), Kelvin fields
  are set to the closest blackbody temperature for the sliders;
- the key direction keeps the rig's elevation; the azimuth is world-relative in the game (characters face anywhere)
  and is left at the viewer's default, the rig value is recorded in the provenance;
- post-process: PfxTonemapMode 2 = Hable filmic (`hable` params), CbCrScale -> saturation, ContrastVol -> contrast.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

from ..importers.za.trlgt import LightRig

STUDIO_KEY = 1.2
STUDIO_AMBIENT = 1.9
STUDIO_AZIMUTH = -26.6

# viewer DEFAULT_ENV (keep in sync with environment.ts; tests check the key set)
DEFAULT_ENV: dict[str, object] = {
    "ambient": 1.9, "key": 1.2, "azimuth": -26.6, "elevation": 38.8, "keyTemp": 6500,
    "back": 0, "backAzimuth": 160, "backElevation": 25, "backTemp": 6500, "skyTemp": 6500, "ground": "#f6ecd8",
    "warmth": 0, "shadow": 0.65, "rim": 1, "detail": 0.5, "exposure": 1, "toneMapping": "linear", "contrast": 1,
    "saturation": 1, "gi": 1,
}

RGB = tuple[float, float, float]


@dataclass
class Provenance:
    rig: str
    category: int
    key_rig: tuple[float, float, float] | None = None  # colour x intensity before normalisation
    key_angles: tuple[float, float] | None = None
    notes: list[str] = field(default_factory=list)


def apply_delta(rig: LightRig, delta: LightRig) -> LightRig:
    """Weather offset rigs (`env/offset/weather_*.trlgt`) are deltas on the active rig: floats add
    (`dir_6 Intensity -2` on master's 4), colours are HSV offsets (hue, saturation, value, alpha): every sky
    gradient carries (0, -1, -0.05, 0) = fully desaturated and slightly darker, `GILightCommon Color0`
    (0, -0.71, 0.55) = greyer and brighter, `CloudDirectionalColor` (-0.012, -0.012, 0, 0). Inferred from the
    value pattern, not from code. Keys the base lacks are added as-is."""
    import colorsys

    out = deepcopy(rig)
    for dl in delta.lights:
        bl = out.get(dl.type)
        if bl is None:
            out.lights.append(deepcopy(dl))
            continue
        for k, v in dl.floats.items():
            bl.floats[k] = bl.floats.get(k, 0.0) + v
        for k, v in dl.colors.items():
            base = bl.colors.get(k)
            if base is None:
                bl.colors[k] = v
                continue
            h, s_, val = colorsys.rgb_to_hsv(max(0.0, base[0]), max(0.0, base[1]), max(0.0, base[2]))
            h = (h + v[0]) % 1.0
            s_ = min(1.0, max(0.0, s_ + v[1]))
            val = max(0.0, val + v[2])
            r, g, b = colorsys.hsv_to_rgb(h, s_, val)
            a = base[3] + v[3] if len(base) > 3 and len(v) > 3 else (base[3] if len(base) > 3 else 0.0)
            bl.colors[k] = (r, g, b, a)
    return out


def _rgb(c, default: RGB = (1.0, 1.0, 1.0)) -> RGB:
    if not c:
        return default
    return (max(0.0, c[0]), max(0.0, c[1]), max(0.0, c[2]))


def _hex(rgb: RGB, gamma: bool = True) -> str:
    def enc(v: float) -> int:
        v = min(1.0, max(0.0, v))
        if gamma:
            v = v * 12.92 if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055
        return round(v * 255)
    return "#%02x%02x%02x" % tuple(enc(v) for v in rgb)


def _norm(rgb: RGB) -> tuple[RGB, float]:
    """Split a linear colour into chromaticity (max channel 1) and its magnitude (max channel)."""
    m = max(rgb)
    return ((rgb[0] / m, rgb[1] / m, rgb[2] / m), m) if m > 0 else ((1.0, 1.0, 1.0), 0.0)


def kelvin_to_rgb(k: float) -> RGB:
    """Same fit as the viewer's `kelvinToRGB` (Tanner Helland, normalised so 6500 K is white)."""
    def raw(kk: float):
        t = min(40000.0, max(1000.0, kk)) / 100
        r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
        g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0 if t >= 66 else 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
        out = []
        for c in (r, g, b):
            s = min(255.0, max(0.0, c)) / 255
            out.append(s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4)
        return out
    c, w = raw(k), raw(6500)
    o = [a / b for a, b in zip(c, w)]
    m = max(o)
    return (o[0] / m, o[1] / m, o[2] / m)


def closest_kelvin(rgb: RGB) -> int:
    chroma, _ = _norm(rgb)
    best, best_d = 6500, 1e9
    for k in range(1500, 12001, 100):
        c = kelvin_to_rgb(k)
        d = sum((a - b) ** 2 for a, b in zip(c, chroma))
        if d < best_d:
            best, best_d = k, d
    return best


def key_magnitude(rig: LightRig, category: int) -> float:
    d = rig.get(f"dir_{category}") or rig.get("DirectionalMain")
    if d is None:
        return 0.0
    col = _rgb(d.colors.get("Color"))
    inten = d.floats.get("Intensity", 1.0)
    clamp = d.floats.get("ClampMax")
    if clamp is not None:
        inten = min(inten, clamp)
    return max(col) * max(0.0, inten)


def ambient_magnitude(rig: LightRig, category: int) -> float:
    gi = rig.get("GILightCommon")
    if gi is None:
        return 0.0
    col = _rgb(gi.colors.get(f"Color{category}", gi.colors.get("Color")))
    return max(col) * max(0.0, gi.floats.get("Intensity", 1.0)) * max(0.0, gi.floats.get(f"Intensity{category}", 1.0))


def rig_to_env(rig: LightRig, category: int, master: LightRig | None = None, rig_name: str = "?",
               probe_sky: RGB | None = None, probe_ground: RGB | None = None) -> tuple[dict[str, object], Provenance]:
    """`rig` must already be merged over master (and weather deltas applied). Returns (env, provenance)."""
    prov = Provenance(rig_name, category)
    env: dict[str, object] = dict(DEFAULT_ENV)
    ref = master or rig
    ref_key = key_magnitude(ref, category) or 1.0
    ref_amb = ambient_magnitude(ref, category) or 1.0

    # --- key light -------------------------------------------------------------------------
    d = rig.get(f"dir_{category}")
    sun = rig.get("DirectionalMain")
    src = d or sun
    if src is not None:
        col = _rgb(src.colors.get("Color"))
        chroma, mag = _norm(col)
        inten = src.floats.get("Intensity", 1.0)
        if "ClampMax" in src.floats:
            inten = min(inten, src.floats["ClampMax"])
        key_mag = mag * max(0.0, inten)
        env["key"] = round(STUDIO_KEY * key_mag / ref_key, 3)
        env["keyColor"] = _hex(chroma)
        env["keyTemp"] = closest_kelvin(chroma)
        prov.key_rig = tuple(round(c * inten, 3) for c in col)  # type: ignore[assignment]
    else:
        prov.notes.append("no dir_N / DirectionalMain light: key left at Studio")
    if sun is not None and sun.angles() is not None:
        az, el = sun.angles()
        env["elevation"] = round(max(0.0, min(90.0, el)), 1)
        env["azimuth"] = STUDIO_AZIMUTH
        prov.key_angles = (round(az, 1), round(el, 1))

    # --- ambient (GI colour x intensity, optionally tinted by the diffuse probe) ---------------
    gi = rig.get("GILightCommon")
    if gi is not None:
        col = _rgb(gi.colors.get(f"Color{category}", gi.colors.get("Color")))
        chroma, mag = _norm(col)
        amb_mag = mag * max(0.0, gi.floats.get("Intensity", 1.0)) * max(0.0, gi.floats.get(f"Intensity{category}", 1.0))
        env["ambient"] = round(STUDIO_AMBIENT * amb_mag / ref_amb, 3)
        sky: RGB = chroma
        ground: RGB = chroma
        ground_level = 0.85  # ground bounce relative to the sky when no probe is readable
        probe = rig.get("ProbeMain")
        if probe_sky is not None and probe_ground is not None and probe is not None:
            # The SV diffuse probes are a neutral top-to-bottom gradient (sky ~1.25, ground ~0.51, identical in
            # every rig), so they contribute the sky:ground ratio and a (usually white) tint, not a scene colour.
            pcol = _rgb(probe.colors.get("DiffuseProbeColor"))
            ps, s_mag = _norm((probe_sky[0] * pcol[0], probe_sky[1] * pcol[1], probe_sky[2] * pcol[2]))
            pg, g_mag = _norm((probe_ground[0] * pcol[0], probe_ground[1] * pcol[1], probe_ground[2] * pcol[2]))
            if s_mag > 0 and probe.floats.get("DiffuseProbeIntensity", 1.0) > 0:
                sky = (sky[0] * ps[0], sky[1] * ps[1], sky[2] * ps[2])
                ground = (ground[0] * pg[0], ground[1] * pg[1], ground[2] * pg[2])
                ground_level = min(1.0, max(0.1, g_mag / s_mag))
                prov.notes.append(f"diffuse probe sky:ground {s_mag:.2f}:{g_mag:.2f}")
        env["skyColor"] = _hex(_norm(sky)[0])
        env["skyTemp"] = closest_kelvin(sky)
        g0 = _norm(ground)[0]
        env["ground"] = _hex((ground_level * g0[0], ground_level * g0[1], ground_level * g0[2]))
    else:
        prov.notes.append("no GILightCommon: ambient left at Studio")

    # --- rim / back light ----------------------------------------------------------------------
    r = rig.get(f"rim_{category}")
    if r is not None and r.colors.get("Color"):
        col4 = r.colors["Color"]
        chroma, mag = _norm(_rgb(col4))
        strength = max(0.0, col4[3] if len(col4) > 3 else 0.0) * mag
        env["back"] = round(strength, 3)
        env["backColor"] = _hex(chroma)
        env["backTemp"] = closest_kelvin(chroma)
        env["backAzimuth"] = 160
        env["backElevation"] = 25
        env["rim"] = round(min(2.0, strength * (r.floats.get("Power", 3.0) / 3.0)), 3)
        prov.notes.append("rim_N alpha used as back-light strength (inferred)")

    # --- post-process --------------------------------------------------------------------------
    g = rig.get("Global")
    if g is not None:
        if g.floats.get("PfxTonemapMode", 0) == 2:
            env["toneMapping"] = "hable"
            env["hable"] = {
                "toeStrength": round(g.floats.get("PfxToeStrength", 0.0), 4),
                "toeLength": round(g.floats.get("PfxToeLength", 0.5), 4),
                "shoulderStrength": round(g.floats.get("PfxShoulderStrength", 1.0), 4),
                "shoulderLength": round(g.floats.get("PfxShoulderLength", 0.5), 4),
                "shoulderAngle": round(g.floats.get("PfxShoulderAngle", 0.0), 4),
                "gamma": round(g.floats.get("PfxTonemapGamma", 1.0), 4),
            }
        env["saturation"] = round(g.floats.get("CbCrScale", 1.0), 3)
        env["contrast"] = round(g.floats.get("ContrastVol", 1.0), 3)
    return env, prov
