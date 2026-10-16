"""Trinity scene light rigs (`.trlgt`): SV `romfs/env/**`, Z-A `light/**` (not in our Z-A zips).

Schema (PokeDocs `SV/Flatbuffers/field/trlgt.fbs`, verified on the SV dump): root {0 res_0 u32, 1 lights};
Light {0 light_type, 1 light_name, 2 transform, 3 float_params [{name, f32}], 4 vec4_params [{name, Vec4}],
5 string_params [{map_name, map_file}], 6 u32 (undocumented, 4 or 8)}. The schema calls field 2 a
"BoundMatrix" of three Vec4 rows, but the file stores three Vec3 fields laid out like a `.trskl` SRT:
field 0 scale, field 1 rotation (euler XYZ, radians), field 2 translation (verified: the master rig's sun
has rotation (-0.7, 3.0, 0) = 40 deg above the horizon; interiors (-1.02, 0, 0)).

A rig is a list of lights keyed by type: `Global` (post-process), `Sky`, `Fog`, `ProbeMain`,
`GILightCommon` (per-category ambient `Color0..7`), `DirectionalMain`, `Ambient`, `dir_0..7` ("Selectable"
per-category key lights), `rim_0..7`, `DirectionalForWater`, `RefSpecular`, `ShadowTransform`. Materials
pick their category through `.trmtr` int `CategoryLabel` (SV: trainer skin 5, Pokémon 6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import fb
from .trinity import euler_xyz_to_quat

Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]


@dataclass
class Light:
    type: str
    name: str
    scale: Vec3 | None
    rotation: Vec3 | None  # euler XYZ radians
    translation: Vec3 | None
    floats: dict[str, float] = field(default_factory=dict)
    colors: dict[str, Vec4] = field(default_factory=dict)
    textures: dict[str, str] = field(default_factory=dict)
    flags: int = 0

    @property
    def category(self) -> int | None:
        """0..7 for `dir_N` / `rim_N` lights (from `CategoryLabel`, falling back to the type suffix)."""
        if "CategoryLabel" in self.floats:
            return int(self.floats["CategoryLabel"])
        if "_" in self.type and self.type.rsplit("_", 1)[1].isdigit():
            return int(self.type.rsplit("_", 1)[1])
        return None

    def direction(self) -> Vec3 | None:
        """Unit vector the light travels along (Y-up), i.e. the rotated local -Z axis."""
        if self.rotation is None:
            return None
        x, y, z, w = euler_xyz_to_quat(*self.rotation)
        return _rotate((0.0, 0.0, -1.0), (x, y, z, w))

    def angles(self) -> tuple[float, float] | None:
        """(azimuth, elevation) in degrees of the point the light comes FROM, viewer convention
        (azimuth 0 = +Z in front of the model, positive toward +X; elevation above the horizon)."""
        d = self.direction()
        if d is None:
            return None
        fx, fy, fz = -d[0], -d[1], -d[2]
        az = math.degrees(math.atan2(fx, fz))
        el = math.degrees(math.atan2(fy, math.hypot(fx, fz)))
        return az, el


@dataclass
class LightRig:
    lights: list[Light]
    res0: int = 0

    def get(self, light_type: str) -> Light | None:
        for l in self.lights:
            if l.type == light_type:
                return l
        return None

    def types(self) -> list[str]:
        return [l.type for l in self.lights]

    def category(self, n: int) -> dict[str, object]:
        """Everything that applies to material category `n`: its key light, rim light and GI colour."""
        gi = self.get("GILightCommon")
        return {
            "dir": self.get(f"dir_{n}"),
            "rim": self.get(f"rim_{n}"),
            "gi_color": gi.colors.get(f"Color{n}") if gi else None,
            "gi_intensity": gi.floats.get(f"Intensity{n}") if gi else None,
        }


def parse_trlgt(data: bytes) -> LightRig:
    root = fb.root(data)
    lights = []
    for t in root.tables(1):
        srt = t.table(2)
        scale = rot = trans = None
        if srt is not None:
            scale, rot, trans = srt.floats(0, 3), srt.floats(1, 3), srt.floats(2, 3)
        floats = {p.string(0): p.f32(1) for p in t.tables(3)}
        colors = {p.string(0): (p.floats(1, 4) or (0.0, 0.0, 0.0, 0.0)) for p in t.tables(4)}
        textures = {p.string(0): p.string(1) for p in t.tables(5)}
        lights.append(Light(t.string(0), t.string(1), scale, rot, trans, floats, colors, textures, t.u32(6)))
    return LightRig(lights, root.u32(0))


def dump_rig(rig: LightRig) -> str:
    """Human-readable listing (one parameter per line), used by `pipeline lights dump`."""
    out = []
    for l in rig.lights:
        out.append(f"LIGHT type={l.type!r} name={l.name!r} flags={l.flags}")
        if l.rotation is not None:
            ang = l.angles()
            out.append(f"  srt scale={_fmt(l.scale)} rot={_fmt(l.rotation)} trans={_fmt(l.translation)}"
                       + (f"  -> from az {ang[0]:.1f} el {ang[1]:.1f} deg" if ang else ""))
        for k, v in l.floats.items():
            out.append(f"  f {k} = {v:g}")
        for k, v in l.colors.items():
            out.append(f"  v {k} = ({', '.join(f'{c:g}' for c in v)})")
        for k, v in l.textures.items():
            out.append(f"  s {k} = {v}")
    return "\n".join(out)


def _fmt(v: Vec3 | None) -> str:
    return "-" if v is None else "(" + ", ".join(f"{c:.3f}" for c in v) + ")"


def _rotate(v: Vec3, q: Vec4) -> Vec3:
    x, y, z, w = q
    vx, vy, vz = v
    # q * v * q^-1 via the cross-product form
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx))
