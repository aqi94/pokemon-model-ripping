"""Pokémon Legends: Z-A importer: Trinity files -> common `Model`."""

from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
import re
import zipfile
import dataclasses
from dataclasses import dataclass

import numpy as np
from PIL import Image

from ...common.model import (
    Clip,
    Joint,
    Material,
    Mesh,
    Model,
    Primitive,
    Skeleton,
    TextureImage,
    Track,
    Variant,
)
from ..bntx import parse_bntx
from . import trinity as T

# --------------------------------------------------------------------------
# Sources: a model lives in a folder, either on disk or inside a zip.
# --------------------------------------------------------------------------


class Source:
    def read(self, name: str) -> bytes:
        raise NotImplementedError

    def exists(self, name: str) -> bool:
        raise NotImplementedError

    def list(self) -> list[str]:
        raise NotImplementedError


class FolderSource(Source):
    def __init__(self, folder: str):
        self.folder = folder

    def read(self, name: str) -> bytes:
        with open(os.path.join(self.folder, name), "rb") as f:
            return f.read()

    def exists(self, name: str) -> bool:
        return os.path.isfile(os.path.join(self.folder, name))

    def list(self) -> list[str]:
        out = []
        for root, _, files in os.walk(self.folder):
            rel = os.path.relpath(root, self.folder).replace("\\", "/")
            for fn in files:
                out.append(fn if rel == "." else f"{rel}/{fn}")
        return out


class ZipSource(Source):
    """A folder inside one or more zips (later zips win on name clashes).

    Names are resolved relative to `prefix`, so a material's
    `../../share/common/x.bntx` reference reaches the shared-texture zips
    as long as they are part of `zips`.
    """

    def __init__(self, zips: list[zipfile.ZipFile], prefix: str):
        self.zips = zips
        self.prefix = prefix.rstrip("/") + "/"
        self._all: dict[str, zipfile.ZipFile] = {}
        for z in zips:
            for n in z.namelist():
                if not n.endswith("/"):
                    self._all[n] = z

    def _resolve(self, name: str) -> str:
        return posixpath.normpath(self.prefix + name.replace("\\", "/"))

    def read(self, name: str) -> bytes:
        full = self._resolve(name)
        return self._all[full].read(full)

    def exists(self, name: str) -> bool:
        return self._resolve(name) in self._all

    def list(self) -> list[str]:
        n = len(self.prefix)
        return sorted(k[n:] for k in self._all if k.startswith(self.prefix))


# --------------------------------------------------------------------------


@dataclass
class AnimationFile:
    name: str  # clip name e.g. walk01_loop
    index: int  # 5-digit index
    data: bytes
    config: bytes | None = None  # companion .tracm (visibility / material tracks)
    group: str | None = None  # alternate animation set label; None = derive from the index
    extra: bytes | None = None  # second .tranm whose bones this clip lacks (rival: body from motion_cc_base/rv, face/hair from motion_cc_ir)
    skip_bones: re.Pattern | None = None  # bones the game's layer mask keeps out of this clip (SV body sets: the eyes)
    rotation_only: frozenset[str] | None = None  # borrowed clips: bones whose bind offset differs keep only rotations


SKIN_MATERIALS = ("body_skin", "face_skin")
SKIN_TONES = 4  # zones 1..4 of the shared skin tables (deep .. pale)
SKIN_DEFAULT_TONE = 3
# FITTED: lipstick-layer size heuristic on the 14 cc faces with a lip layer, 2026-10-19 (engine-rules debt)
LIP_MAX_COVER = 0.012  # a face-skin mask layer this small, centred on the mouth, is lipstick (the others paint facial hair / makeup)

ANIM_RE = re.compile(r"^(?P<model>.+?)_(?P<idx>\d{5})_(?P<name>.+)\.tranm$")

CATEGORY_RULES = [
    ("idle", re.compile(r"wait|idle|eye|blink|sleep|breath")),
    ("move", re.compile(r"walk|run|dash|turn|jump|land|fly|swim|step|move|totter|spin|slide")),
    ("battle", re.compile(r"attack|damage|down|stun|charge|roar|range|guard|hit|dodge|faint|ko|battle")),
    ("emote", re.compile(r"happy|angry|sad|joy|emote|react|demo|appear|greet|cry|shake|nod|surprise")),
]


def categorize(name: str) -> str:
    n = name.lower()
    for cat, rx in CATEGORY_RULES:
        if rx.search(n):
            return cat
    return "other"


def clip_name_from_file(filename: str) -> tuple[str, int] | None:
    m = ANIM_RE.match(os.path.basename(filename))
    if not m:
        return None
    return m.group("name"), int(m.group("idx"))


def dedupe_clip_names(clips: list) -> None:
    """Clip names must be unique (glTF animation / Blender action names). One folder can reuse a name under different
    indexes: the Z-A player's `motion_pc/base` has `ballthrow01` as 00352 (field) and 01020 (battle), `damage01` as
    00500 and 07500. The lowest index keeps the bare name, later ones get ` [NNNNN]` (their 5-digit index)."""
    seen: set[str] = set()
    for c in sorted(clips, key=lambda c: (c.index, c.name)):
        if c.name in seen:
            c.name = f"{c.name} [{c.index:05d}]"
        seen.add(c.name)


# --------------------------------------------------------------------------


# Forms the game switches in code, not by data (Xerneas: Neutral Mode in the field, Active Mode in battle). The clips do toggle
# the meshes that only exist in one mode (field clips show `horn_a`, battle clips `horninc_a/b` = the rainbow tips and
# `backinc`), but the horn bodies are shared and their Active colour lives in emission layers authored at intensity 0
# (body_b_00: layer 4 cream on the horn bodies, layer 3 = the pastel crown leaves on horninc_b). `~active` copies of those
# materials switch the listed layers on (intensity 1), the active-only meshes use them, and the shared triangles are split
# into `<mesh>_neutral` / `<mesh>_active` meshes whose per-clip visibility follows the neutral / active meshes. Verified
# against the Z-A battle shot (samples/za_xerneas.png) and SwSh Active Mode (samples/xerneas_active_swsh.png).
# FITTED: per-species mode table, Xerneas pm0768 only; should become a clip-visibility rule shared with SPOT_GLOW_SPECIES, 2026-10-19 (engine-rules debt)
MODE_SWITCH = {
    ("za", "pm0768"): {"layers": (3, 4), "active": ("horninc_a_mesh", "horninc_b_mesh"), "neutral": "horn_a_mesh"},
}


def _mode_material(tm: T.TrMaterial, layers) -> T.TrMaterial | None:
    """Copy of `tm` with its intensity-0 chromatic emission layers in `layers` switched on, or None if it has none."""
    on = {}
    for i in layers:
        c = tm.colors.get(f"EmissionColorLayer{i}")
        # Chromatic only: white (1, 1, 1) is the authoring default (Xerneas' body_a carries it on its black layers 3/4).
        if c is not None and float(tm.floats.get(f"EmissionIntensityLayer{i}", 0.0)) == 0.0 and max(c[:3]) > 0.05 and max(c[:3]) - min(c[:3]) > 0.1:
            on[f"EmissionIntensityLayer{i}"] = 1.0
    return dataclasses.replace(tm, name=f"{tm.name}~active", floats={**tm.floats, **on}) if on else None


def _split_mode_meshes(model_id: str, mode: dict, meshes: list[Mesh], materials: dict, clips: list[Clip]) -> None:
    """See MODE_SWITCH. Mutates meshes (in place + appended) and the clips' visibility."""
    src = {n[: -len("~active")] for n in materials if n.endswith("~active")}
    if not src:
        return
    pre = f"{model_id}_"
    active_only = {pre + m for m in mode["active"]}
    neutral_mesh, active_mesh = pre + mode["neutral"], pre + mode["active"][0]
    rest = {m.name.split("_lod")[0]: m.visible for m in meshes if not m.lod}
    split: set[str] = set()
    for mesh in list(meshes):
        base = mesh.name.split("_lod")[0]
        if base in active_only:
            for p in mesh.primitives:
                if p.material in src:
                    p.material += "~active"
            continue
        moved = [p for p in mesh.primitives if p.material in src]
        if not moved or base == neutral_mesh:
            continue
        mesh.primitives = [p for p in mesh.primitives if p.material not in src]
        sfx = f"_lod{mesh.lod}" if mesh.lod else ""
        meshes.append(Mesh(name=f"{base}_neutral{sfx}", primitives=moved, lod=mesh.lod, visible=mesh.visible and rest.get(neutral_mesh, True)))
        meshes.append(Mesh(name=f"{base}_active{sfx}", primitives=[dataclasses.replace(p, material=p.material + "~active") for p in moved],
                           lod=mesh.lod, visible=mesh.visible and rest.get(active_mesh, False)))
        split.add(base)
    meshes[:] = [m for m in meshes if m.primitives]
    for c in clips:
        for base in split:
            if neutral_mesh in c.visibility:
                c.visibility[f"{base}_neutral"] = list(c.visibility[neutral_mesh])
            if active_mesh in c.visibility:
                c.visibility[f"{base}_active"] = list(c.visibility[active_mesh])


# ---- Holocast (hologram) material sets -----------------------------------------
# Trainer .trmmt files carry `holo` / `holo_red` sets (typo `horo`, numbered `holo00`): alternate
# .trmtr files with the same textures plus `EnableHolocastEffect`, `HolocastColor` (linear HDR) and
# procedural scanline params. The scanlines are drawn by the shader (no texture), so the viewer
# reproduces the effect; the importer only exports the parameters.

def is_holocast_set(name: str) -> bool:
    n = name.lower()
    return n.startswith("holo") or n.startswith("horo")


def holocast_params(tm: T.TrMaterial) -> dict | None:
    if tm.shader_flags.get("EnableHolocastEffect") != "True":
        return None
    c = tm.colors.get("HolocastColor", (0.223, 3.392, 3.392, 1.0))
    f = tm.floats
    return {
        "color": [round(float(x), 4) for x in c[:3]],
        "scanlineSpeed": round(float(f.get("ScanlineSpeed", 2.0)), 4),
        "scanlineDepth": round(float(f.get("ScanlineDepth", 0.7)), 4),
        "scanlineTiling": round(float(f.get("ScanlineTiling", 260.0)), 4),
        "scanGradationTiling": round(float(f.get("ScanGradationTiling", 1.0)), 4),
        "occlusionStrength": round(float(f.get("OcclusionStrength", 1.0)), 4),
        "shadingBias": round(float(f.get("ShadingBias", 1.0)), 4),
    }


def holocast_name(name: str) -> str:
    """Canonical set name: `horo*` typo -> `holo*`, `holo00` (first of a numbered pair) -> `holo`."""
    n = name.lower().replace("horo", "holo", 1)
    return "holo" if n == "holo00" else n


def holocast_label(name: str, color=None) -> str:
    n = holocast_name(name)
    rest = n[4:].strip("_")
    if not rest:
        # The plain set is usually cyan; name the hue when it is not (Collector red, Pa Boss magenta).
        if color is not None and max(color) > 0:
            r, g, b = (c / max(color) for c in color[:3])
            if r > 0.8 and b > 0.8 and g < 0.5:
                return "Hologram (magenta)"
            if r > 0.8 and g < 0.5 and b < 0.5:
                return "Hologram (red)"
        return "Hologram"
    if rest.isdigit():
        return f"Hologram {int(rest) + 1}"
    return f"Hologram ({rest.replace('_', ' ')})"


def dedupe_holocast(presets: list[dict]) -> list[dict]:
    """Drop presets that look the same as an earlier one (face parts' `holo01` = `holo` within 0.001)."""
    out: list[dict] = []
    for p in presets:
        if not any(all(abs(a - b) < 0.01 for a, b in zip(p["color"], q["color"]))
                   and all(p[k] == q[k] for k in ("scanlineSpeed", "scanlineDepth", "scanlineTiling", "scanGradationTiling")) for q in out):
            out.append(p)
    return out


def attach_holocast(materials: dict, sets: list, source, model_id: str, prefix: str = "") -> list[dict]:
    """Holo sets -> `Material.extras["holocast"][set] = params` on the normal materials (by name,
    optionally `prefix + name` for composite parts) and the model-level presets (each set's most
    common params) returned for `Model.extras["holocast"]`."""
    from collections import Counter
    presets = []
    for s in sets:
        if not is_holocast_set(s.name):
            continue
        files = s.material_files or [f"{model_id}_{s.name}.trmtr"]
        counts: Counter = Counter()
        by_key = {}
        for fn in files:
            if not source.exists(fn):
                continue
            try:
                tms = T.parse_trmtr(source.read(fn))
            except Exception:
                continue
            for tm in tms:
                p = holocast_params(tm)
                if p is None:
                    continue
                m = materials.get(prefix + tm.name)
                if m is not None:
                    m.extras.setdefault("holocast", {})[holocast_name(s.name)] = p
                key = (tuple(p["color"]), p["scanlineSpeed"], p["scanlineDepth"], p["scanlineTiling"], p["scanGradationTiling"])
                counts[key] += 1
                by_key[key] = p
        if counts:
            p = by_key[counts.most_common(1)[0][0]]
            presets.append({"name": holocast_name(s.name), "label": holocast_label(s.name, p["color"]), **{k: p[k] for k in ("color", "scanlineSpeed", "scanlineDepth", "scanlineTiling", "scanGradationTiling")}})
    return dedupe_holocast(presets)


def import_model(
    source: Source,
    model_id: str,
    display_name: str | None = None,
    kind: str = "pokemon",
    animations: list[AnimationFile] | None = None,
    game: str = "za",
    skeleton: Skeleton | None = None,
    lod0_only: bool = False,
    skin_tones: bool = False,
    body: dict | None = None,
) -> Model:
    """`body` = SV generic-NPC body type {"table": trbsm table, "weights": ccdata weights} (`trbsm.apply_body_type`);
    `skeleton` replaces the model's own .trskl (modular parts skin to a shared rig);
    `lod0_only` drops the _lodN mesh files; `skin_tones` bakes the shared skin palette
    zones of body_skin / face_skin materials into `extras["skin_tones"]`."""
    trmdl = T.parse_trmdl(source.read(f"{model_id}.trmdl"))

    # ---- skeleton -------------------------------------------------------
    if skeleton is None and trmdl.skeleton and source.exists(trmdl.skeleton):
        skeleton = _build_skeleton(T.parse_trskl(source.read(trmdl.skeleton)))

    # ---- materials & textures ---------------------------------------------
    textures: dict[str, TextureImage] = {}
    raw_tex: dict[str, np.ndarray] = {}  # bntx file -> rgba
    tex_meta: dict[str, bool] = {}  # bntx file -> srgb

    warnings: list[str] = []

    def load_tex(file: str) -> np.ndarray | None:
        if file in raw_tex:
            return raw_tex[file]
        if not source.exists(file):
            raw_tex[file] = None
            warnings.append(f"missing texture {file}")
            return None
        decoded = parse_bntx(source.read(file))
        if not decoded:
            raw_tex[file] = None
            return None
        raw_tex[file] = decoded[0].rgba
        tex_meta[file] = decoded[0].srgb
        return raw_tex[file]

    materials: dict[str, Material] = {}
    variants: list[Variant] = []

    # Material variant sets from .trmmt (e.g. normal / rare)
    sets = []
    if source.exists(f"{model_id}.trmmt"):
        sets = T.parse_trmmt(source.read(f"{model_id}.trmmt"))
    normal_files = trmdl.materials or [f"{model_id}.trmtr"]

    disp_maps: dict[str, tuple[np.ndarray, float, int, tuple]] = {}  # gas shells: material -> (density map, height, UV set, UVScaleOffset3)

    lid_unfit = _lid_atlas_unfit(animations or [])
    lid_parked = _lid_off_atlas(animations or [])
    lid_overlays: dict[str, list[Material]] = {}  # material -> overlay materials drawn on a copy of its triangles (sliding lids, layer 5)
    l5_moving = _layer5_moving(animations or [])
    mode = MODE_SWITCH.get((game, model_id[:6]))

    beard_mats = _beard_materials(source, trmdl) if skin_tones else set()

    def build_materials(files: list[str], tag: str, patch=None) -> dict[str, Material]:
        """`patch(tm) -> TrMaterial | None` (variation groups): convert only the materials it returns, as returned."""
        out = {}
        for fn in files:
            if not source.exists(fn):
                continue
            for tm in T.parse_trmtr(source.read(fn)):
                if patch is not None:
                    tm = patch(tm)
                    if tm is None:
                        continue
                ovs: list[Material] = []
                out[tm.name] = _convert_material(tm, tag, load_tex, textures, tex_meta, skin_tones, lid_atlas=tm.name not in lid_unfit, overlays=ovs,
                                                 lid_parked=tm.name in lid_parked, layer5_moves=tm.name in l5_moving, beard=tm.name in beard_mats)
                if tm.name in beard_mats:
                    out[tm.name].extras["beard"] = True
                for ov in ovs:
                    # Lids shade like the eye they cover (no per-texel specular mask of their own).
                    for k in ("shading", "pbr"):
                        if k in out[tm.name].extras:
                            ov.extras[k] = dict(out[tm.name].extras[k]) if isinstance(out[tm.name].extras[k], dict) else out[tm.name].extras[k]
                    ov.extras.get("shading", {}).pop("specularMaskChannels", None)
                    out[ov.name] = ov
                    if not tag:
                        lid_overlays.setdefault(tm.name, []).append(ov)
                act = _mode_material(tm, mode["layers"]) if mode else None
                if act is not None:
                    out[act.name] = _convert_material(act, tag, load_tex, textures, tex_meta, skin_tones, lid_atlas=tm.name not in lid_unfit,
                                                      lid_parked=tm.name in lid_parked, layer5_moves=tm.name in l5_moving)
                dfile = _tex_file(tm, "DisplacementMap")
                if dfile and _is_gas_shell(tm) and tm.name not in disp_maps and load_tex(dfile) is not None:
                    # The DisplacementMap is texture-coordinate "layer 3" (layer 1 = BaseColorMap/UVScaleOffset,
                    # layer 2 = NormalMap/UVScaleOffsetNormal): `UVIndexLayer3` picks its UV set (-1/0 = UV0,
                    # 1 = UV1) and `UVScaleOffset3` scales it. Every SV material with two UV sets and a
                    # displacement map has UVIndexLayer3 = 1; Gastly's UV1 is UV0 shrunk to [0.05, 0.95].
                    disp_maps[tm.name] = (
                        load_tex(dfile),
                        float(tm.floats.get("DisplacementHeight", 0.0)),
                        max(0, int(tm.ints.get("UVIndexLayer3", 0))),
                        tuple(float(x) for x in tm.colors.get("UVScaleOffset3", (1.0, 1.0, 0.0, 0.0))[:4]),
                    )
        return out

    materials = build_materials(normal_files, "")
    holo_presets = attach_holocast(materials, sets, source, model_id)
    for s in sets:
        if s.name == "normal" or is_holocast_set(s.name):
            continue
        files = s.material_files or [f"{model_id}_{s.name}.trmtr"]
        mats = build_materials(files, s.name)
        if mats:
            label = {"rare": "Shiny"}.get(s.name, s.name)
            variants.append(Variant(name=s.name, materials=mats, group="coloring", label=label))
    if variants:
        variants.insert(0, Variant(name="normal", materials={}, group="coloring", label="Normal", default=True))
    base_set = next((s for s in sets if s.name == "normal"), sets[0] if sets else None)
    if base_set is not None:
        variants += variation_variants(base_set, normal_files, source, build_materials)

    # ---- meshes ---------------------------------------------------------------
    meshes: list[Mesh] = []
    for mesh_file in trmdl.meshes:
        if not source.exists(mesh_file):
            continue
        lod = 0
        m = re.search(r"_lod(\d+)\.trmsh$", mesh_file)
        if m:
            lod = int(m.group(1))
        if lod and lod0_only:
            continue
        trmsh = T.parse_trmsh(source.read(mesh_file))
        buffers = T.parse_trmbf(source.read(trmsh.buffer_file))
        for si, shape in enumerate(trmsh.shapes):
            if si >= len(buffers):
                break
            used = {0} | {p.declaration for p in shape.parts}  # the others are blend-shape target layouts
            decoded = [T.decode_vertices(d, buffers[si]) if di in used else None for di, d in enumerate(shape.declarations)]
            prims = []
            for part in shape.parts:
                if part.index_count <= 0:
                    continue
                v = decoded[part.declaration] if part.declaration < len(decoded) and decoded[part.declaration] is not None else decoded[0]
                mat_name = part.material
                if mat_name not in materials:
                    materials[mat_name] = Material(name=mat_name)
                # Trinity UVs: bottom-left origin; each material's parts live
                # in their own integer V tile (body_a in [0,1], body_b in
                # [1,2], ...); the material's UVScaleOffset (su, sv, ou, ov)
                # is applied before sampling, usually (2,1,0,0) with a
                # mirrored U sampler for half-width symmetric textures.
                # Bake: uv_tex = uv * s + o, drop the V tile, flip V for glTF.
                # Mirrored samplers are baked into the textures (half-width
                # symmetric images are doubled), so `uv_scale_effective`
                # already includes the /2 and plain repeat wrapping suffices.
                su, sv, ou, ov = materials[mat_name].extras.get("uv_scale_effective", (1.0, 1.0, 0.0, 0.0))
                # UVCenter0: the scale pivots about this point (Falinks' eyes: (0.5, 0.5) with scale (2, 1), so both
                # eyes read the left column of the atlas and the right eye mirrors through 0). It only matters when
                # the scale is not 1 (Falinks, Budew).
                cu, cv = materials[mat_name].extras.get("uv_center", (0.0, 0.0))
                # The centre maps to itself in the game's (unbaked) texture, i.e. to centre/2 on a mirror-doubled
                # axis. Written without dividing by the scale: some materials have scale 0 (Magikarp, Drednaw).
                mu, mv = materials[mat_name].extras.get("mirror", (False, False))
                pu, pv = cu * (0.5 if mu else 1.0), cv * (0.5 if mv else 1.0)
                uvs = []
                for uv in v.uvs:
                    u_t = (uv[:, 0] - cu) * su + pu + ou
                    v_t = (uv[:, 1] - cv) * sv + pv + ov
                    v_t = v_t - np.floor(v_t)  # tile removal (V sampler is clamp)
                    uvs.append(np.stack([u_t, 1.0 - v_t], axis=1).astype(np.float32))
                positions = v.positions
                gas_weight = None
                if mat_name in disp_maps and v.normals is not None and v.uvs:
                    # Gas shell (Gastly's smoke): bake one frame of the game's drifting cloud, see gas_displacement.
                    dmap, dh, uv_idx, uvso3 = disp_maps[mat_name]
                    uv_idx = min(uv_idx, len(v.uvs) - 1)
                    uvd = v.uvs[uv_idx] * np.asarray(uvso3[:2], np.float32) + np.asarray(uvso3[2:], np.float32)
                    du = uvd[:, 0] - np.floor(uvd[:, 0])
                    dv = uvd[:, 1] - np.floor(uvd[:, 1])
                    dens = dmap[np.clip((1 - dv) * dmap.shape[0], 0, dmap.shape[0] - 1).astype(int), np.clip(du * dmap.shape[1], 0, dmap.shape[1] - 1).astype(int), 0] / 255.0
                    colour = smooth_over_mesh(v.colors[:, 0], T.read_indices(shape, buffers[si], part), GAS_WEIGHT_SMOOTH) if v.colors is not None else None
                    offset, gas_weight = gas_displacement(colour, dens.astype(np.float32), dh)
                    positions = (v.positions + v.normals * offset[:, None]).astype(np.float32)
                    materials[mat_name].extras.setdefault("gas", {})["texCoord"] = int(uv_idx)
                tangents = None
                if v.tangents is not None:
                    tangents = v.tangents.copy()
                    tangents[:, 3] = np.where(tangents[:, 3] < 0, 1.0, -1.0)
                prims.append(
                    Primitive(
                        material=mat_name,
                        indices=T.read_indices(shape, buffers[si], part),
                        positions=positions,
                        normals=v.normals,
                        tangents=tangents,
                        uvs=uvs,
                        colors=None,  # Trinity vertex colours are shader masks (EnableVertexSubLayer), not tint
                        joints=v.joints if skeleton else None,
                        weights=v.weights if skeleton else None,
                        gas_weight=gas_weight,
                    )
                )
                for ov in lid_overlays.get(mat_name, ()) if v.uvs else ():
                    base_prim = prims[-1]
                    lo = ov.extras.get("layer_overlay")
                    if lo is not None and lo["uv_set"] >= len(v.uvs):
                        continue  # the layer's UV set is missing on this mesh
                    prims.append(Primitive(
                        material=ov.name, indices=base_prim.indices, positions=base_prim.positions,
                        normals=base_prim.normals, tangents=base_prim.tangents,
                        # The tile comes from this part's vertices: SV shapes share one vertex buffer between the body
                        # (tile 1) and the eyes (tile 0), and the whole buffer's median put Corphish's lids a tile off.
                        uvs=[layer_overlay_uvs(v.uvs[lo["uv_set"]], lo, part_tile(v.uvs[lo["uv_set"]], base_prim.indices)) if lo is not None
                             else lid_overlay_uvs(v.uvs[0], ov.extras["lid_overlay"], part_tile(v.uvs[0], base_prim.indices))],
                        joints=base_prim.joints, weights=base_prim.weights,
                    ))
            name = shape.mesh_name or shape.name.replace("_shape", "")
            if lod and not name.endswith(f"_lod{lod}"):
                name = f"{name}_lod{lod}"
            mesh = Mesh(name=name, primitives=prims, lod=lod)
            if (shape.morph_regions or shape.morph_names) and prims and decoded[0] is not None:
                _attach_blend_shapes(mesh, shape, buffers[si], decoded[0].positions)
            meshes.append(mesh)

    # ---- animations -------------------------------------------------------
    morph_meshes = {m.name: m.extras["targetNames"] for m in meshes if m.extras.get("targetNames")}
    clips: list[Clip] = []
    rest_states: dict[str, bool] = {}
    for af in animations or []:
        if not af.data:
            # Config-only visibility state (visoff01 / vison01 ...): the "off"
            # state is the game's default; take its values as rest visibility.
            if af.config and "visoff" in af.name:
                for vt in T.parse_tracm_visibility(af.config):
                    shape = re.sub(r"_lod\d+$", "", vt.shape)
                    shape = shape[: -len("_shape")] if shape.endswith("_shape") else shape
                    rest_states.setdefault(shape, bool(vt.values[0]))
            continue
        try:
            tr = T.parse_tranm(af.data)
        except Exception as e:  # noqa: BLE001 - keep going on a bad clip
            warnings.append(f"skipped animation {af.name}: {e}")
            continue
        if af.skip_bones is not None:
            tr.tracks = [t for t in tr.tracks if not af.skip_bones.search(t.bone)]
        if af.rotation_only:
            # The donor keys translations at its own offsets; on a rig whose bind differs they would pull the
            # borrower's bones to the donor's proportions.
            for t in tr.tracks:
                if t.bone in af.rotation_only:
                    t.translation = None
        if af.extra:
            have = {t.bone for t in tr.tracks}
            tr.tracks += [t for t in T.parse_tranm(af.extra).tracks if t.bone not in have]
        clips.append(_convert_clip(tr, af))
        if af.config and morph_meshes:
            try:
                clips[-1].morph = blend_shape_weights(T.parse_tracm_blendshapes(af.config), morph_meshes, float(tr.frame_rate))
            except Exception as e:  # noqa: BLE001 - face tracks are optional
                warnings.append(f"{af.name}: blend-shape tracks unreadable ({e})")
        if af.config:
            try:
                uv = material_uv_offsets(T.parse_tracm_materials(af.config), materials, float(tr.frame_rate))
            except Exception as e:  # noqa: BLE001 - material tracks are optional
                warnings.append(f"{af.name}: material tracks unreadable ({e})")
                uv = {}
            if uv:
                clips[-1].uv_offset, clips[-1].uv_offset_mode = uv, "gltf_step"
    clips.sort(key=lambda c: (c.index, c.name))
    dedupe_clip_names(clips)
    for c in clips:
        # Clips without face tracks return the face to neutral (three.js would keep the previous clip's weights).
        for mname, names in morph_meshes.items():
            c.morph.setdefault(mname, (np.zeros(1, np.float32), np.zeros((1, len(names)), np.float32)))
    if skeleton is not None:
        split_scale_joints(skeleton, clips)
        if body:
            from .trbsm import apply_body_type

            apply_body_type(skeleton, clips, body["table"], body.get("weights"))
    # Rest visibility: a mesh that any clip toggles starts hidden, unless the
    # default clip's track shows it at frame 0 (Griselle's cup only appears in
    # sit/walk clips; Pokémon alt wing poses are mutually exclusive).
    toggled = {m for c in clips for m in c.visibility}
    if toggled:
        pref = next((c for c in clips if c.name in ("defaultwait01_loop", "battlewait01_loop", "defaultidle01")), clips[0] if clips else None)
        for mesh in meshes:
            base = mesh.name.split("_lod")[0]
            if base in toggled or mesh.name in toggled:
                keys = (pref.visibility.get(base) if pref else None) or []
                if keys:
                    mesh.visible = bool(keys[0][1])
                else:
                    # Majority of the clips' frame-0 values: props hidden in
                    # most clips (cup, tray, vines) start hidden; parts only
                    # ever marked visible stay visible.
                    shown = sum(1 for c in clips if c.visibility.get(base, [(0, False)])[0][1] and base in c.visibility)
                    hidden = sum(1 for c in clips if base in c.visibility and not c.visibility[base][0][1])
                    mesh.visible = shown >= hidden
    for mesh in meshes:
        base = mesh.name.split("_lod")[0]
        if base in rest_states:
            mesh.visible = rest_states[base]
    # Froakie line: the default clip shows `eye_b`, a dark overlay (a flat 32x32 bake, or Frogadier's mostly black
    # expression atlas), and hides `eye_a`, the yellow eye with its pupil. A dark eye mesh never is the open eye:
    # swap to its brighter sibling.
    # "Dark" = dark everywhere: Morpeko's normal eye (Z-A pm0941_11_00) is black with a white highlight disc, mean
    # 44.6, and a mean test swapped it for the red Hangry eye.
    def _blank(mesh: Mesh) -> bool:
        for p in mesh.primitives:
            ex = getattr(materials.get(p.material), "extras", {})
            if "lid_overlay" in ex or "layer_overlay" in ex:
                continue  # an overlay drawn over the eye (sliding lid), not the eye
            tex = textures.get(getattr(materials.get(p.material), "base_color", None) or "")
            if tex is None or float(np.percentile(tex.rgba[..., :3].max(axis=2), 99)) > 60.0:
                return False
        return bool(mesh.primitives)

    by_name = {m.name: m for m in meshes}
    swapped: set[str] = set()
    for mesh in meshes:
        mt = re.match(r"(.*eye_)([a-d])(_mesh.*)$", mesh.name)
        if not (mt and mesh.visible and _blank(mesh)):
            continue
        for letter in "abcd":
            sib = by_name.get(f"{mt.group(1)}{letter}{mt.group(3)}")
            if sib is not None and sib is not mesh and not sib.visible and not _blank(sib):
                mesh.visible, sib.visible = False, True
                swapped |= {mesh.name.split("_lod")[0], sib.name.split("_lod")[0]}
                break
    for c in clips:  # the pair's per-clip toggles would bring the blank mesh back
        for n in swapped:
            c.visibility.pop(n, None)
    if mode:
        _split_mode_meshes(model_id, mode, meshes, materials, clips)
    default_clip = None
    for pref in ("defaultwait01_loop", "battlewait01_loop", "defaultidle01"):
        if any(c.name == pref for c in clips):
            default_clip = pref
            break
    if default_clip is None and clips:
        default_clip = clips[0].name

    return Model(
        id=model_id,
        game=game,
        display_name=display_name or model_id,
        kind=kind,
        skeleton=skeleton,
        meshes=meshes,
        materials=materials,
        textures=textures,
        clips=clips,
        variants=variants,
        default_clip=default_clip,
        extras={**({"warnings": warnings} if warnings else {}), **({"holocast": holo_presets} if holo_presets else {})},
    )


# --------------------------------------------------------------------------


def _build_skeleton(sk: T.Trskl) -> Skeleton:
    joints = []
    for n in sk.nodes:
        inv = None
        skinning = n.joint_info >= 0
        if skinning and n.joint_info < len(sk.joints) and sk.joints[n.joint_info].inverse_bind is not None:
            # stored row-vector [axisX; axisY; axisZ; pos] -> column-vector 4x4
            inv = np.ascontiguousarray(sk.joints[n.joint_info].inverse_bind.T, np.float32)
        joints.append(
            Joint(
                name=n.name,
                parent=n.parent,
                translation=tuple(float(x) for x in n.translation),
                rotation=T.euler_xyz_to_quat(*n.rotation_euler),
                scale=tuple(float(x) for x in n.scale),
                inverse_bind=inv,
                skinning=skinning,
                # Only joints with a JointInfo carry the flag; it is written only when on, so a missing field = off
                # (SV trainers' `*_eyeball_segoff` under `eyescale` (1.3-1.6, 1, 1): their inverse binds include it).
                scale_compensate=(sk.joints[n.joint_info].segment_scale_compensate
                                  if 0 <= n.joint_info < len(sk.joints) else True),
            )
        )
    palette: list[int] = [-1] * len(sk.joints)
    for ni, n in enumerate(sk.nodes):
        if 0 <= n.joint_info < len(palette) and palette[n.joint_info] < 0:
            palette[n.joint_info] = ni
    # Fill holes (should not happen) with the root so indices stay valid.
    palette = [p if p >= 0 else 0 for p in palette]
    return Skeleton(joints=joints, palette=palette)


def _tex_file(tm: T.TrMaterial, slot: str) -> str | None:
    for b in tm.textures:
        if b.slot == slot:
            return b.file
    return None


MORPH_TARGET_RE = re.compile(r"^(?P<t>.+)_m(?P<k>\d)_(?P<mesh>.+)_shape$")


def _attach_blend_shapes(mesh: Mesh, shape: T.MeshShape, buffers: T.MeshBuffers, base: np.ndarray) -> None:
    """SV trainer face expressions: the shape's blend-shape targets as glTF morph targets on every primitive (glTF
    needs the same targets on all primitives of a mesh). Names become `<target>_m<region>` (`mouth_i_m1`), the key
    the clips' `blend_<deformer>:<target>` weights are matched on (`blend_shape_weights`)."""
    names, deltas = [], []
    for name, d in T.morph_targets(shape, buffers, base):
        m = MORPH_TARGET_RE.match(name)
        if m is None or not np.any(np.abs(d) > 1e-6):
            continue
        names.append(f"{m.group('t')}_m{m.group('k')}")
        deltas.append(d)
    if not names:
        return
    for p in mesh.primitives:
        if len(p.positions) == len(base):
            p.morph_targets = list(deltas)
    if any(len(p.morph_targets) != len(names) for p in mesh.primitives):
        for p in mesh.primitives:
            p.morph_targets = []
        return
    mesh.extras["targetNames"] = names


def blend_shape_weights(tracks: list, morph_meshes: dict[str, list[str]], fps: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Clip morph weights per mesh from `.tracm` blend-shape curves. A curve `blend_<deformer>:<target>` drives the
    targets named `<target>_m<k>` where k is the face region its deformer covers (from the region shapes it lists:
    `face_m1_mesh_shape` -> 1; mouth = m1, eyes m2/m3, brows m4). Weights are linear between keys."""
    by_key: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tr in tracks:
        target = tr.name.split(":", 1)[-1]
        k = next((m.group(1) for sname in tr.shapes for m in [re.search(r"(?:^|_)m(\d)(?:_|$)", sname)] if m), None)
        if k is None:
            continue
        f, idx = np.unique(tr.frames, return_index=True)
        by_key[f"{target}_m{k}"] = (f.astype(np.float64), tr.weights[idx].astype(np.float64))
    out = {}
    for mesh, names in morph_meshes.items():
        hits = {j: by_key[n] for j, n in enumerate(names) if n in by_key}
        if not hits:
            continue
        frames = np.unique(np.concatenate([f for f, _ in hits.values()]))
        w = np.zeros((len(frames), len(names)), np.float32)
        for j, (f, v) in hits.items():
            w[:, j] = np.interp(frames, f, v)
        out[mesh] = ((frames / fps).astype(np.float32), w)
    return out


VARIATION_COLOUR_GROUP = re.compile(r"^colou?r")  # per-NPC colour axes (`color_00`, `color01`, `color_skin`...)


def _patch_params(tm: T.TrMaterial, values: dict[str, tuple[float, ...]]) -> T.TrMaterial:
    colors, floats = dict(tm.colors), dict(tm.floats)
    for param, v in values.items():
        if param in floats or (param not in colors and len(v) == 1):
            floats[param] = float(v[0])
        else:
            colors[param] = tuple(float(x) for x in (tuple(v) + (1.0, 1.0, 1.0, 1.0))[:4])
    return dataclasses.replace(tm, colors=colors, floats=floats)


def _same_params(tm: T.TrMaterial, values: dict[str, tuple[float, ...]], tol: float = 2e-3) -> bool:
    for param, v in values.items():
        cur = tm.floats.get(param)
        cur = (cur,) if cur is not None else tm.colors.get(param)
        if cur is None or any(abs(float(a) - float(b)) > tol for a, b in zip(cur, v)):
            return False
    return True


def _variation_label(materials: set[str]) -> str:
    """Group name from the material names it recolours (the game's own names: `color_00`.. say nothing)."""
    kinds = [k for k, hit in (("Skin", lambda m: "skin" in m and "hair" not in m), ("Hair", lambda m: "hair" in m),
                              ("Eyes", lambda m: "eye" in m and "brow" not in m and "lash" not in m))
             if any(hit(m) for m in materials)]
    return " / ".join(kinds) if kinds else "Accessory colour"


def variation_variants(base_set: T.MaterialVariantSet, files: list[str], source, build_materials) -> list[Variant]:
    """SV per-NPC colour axes (`.trmmt` variation groups `color*`): one Variant per colour id of each group.

    The game picks an id per NPC (`.ccdata` race00..03; base-game generics: 4 skin/hair/eye tones 300..303) and sets
    the listed material params to the group's keyed values at that id. Each group becomes its own variant group
    (`tone:<group>`) whose options re-bake only that group's materials, so groups combine (the viewer only swaps
    meshes whose material differs inside the selected group). The option matching the authored .trmtr is default."""
    tms = {tm.name: tm for fn in files if source.exists(fn) for tm in T.parse_trmtr(source.read(fn))}
    out: list[Variant] = []
    for g in base_set.groups:
        if not VARIATION_COLOUR_GROUP.match(g.name) or not g.tracks:
            continue
        mats = sorted({t.material for t in g.tracks if t.material in tms})
        if not mats or g.last < g.first or g.last - g.first > 40:
            continue
        per_id = []
        for k in range(g.first, g.last + 1):
            vals: dict[str, dict[str, tuple[float, ...]]] = {}
            for t in g.tracks:
                if t.material in tms:
                    v = T.track_value(t, k)
                    if v is not None:
                        vals.setdefault(t.material, {}).setdefault(t.param, v)
            per_id.append((k, vals))
        # Ids that change nothing relative to the previous one are the same look: keep one option each.
        seen, opts = [], []
        for k, vals in per_id:
            key = repr(sorted((m, p, tuple(round(x, 4) for x in v)) for m, d in vals.items() for p, v in d.items()))
            if key not in seen:
                seen.append(key)
                opts.append((k, vals))
        if len(opts) < 2:
            continue
        group = f"tone:{g.name}"
        label = _variation_label(set(mats))
        defaults = [all(_same_params(tms[m], vals.get(m, {})) for m in mats) for _, vals in opts]
        for n, ((k, vals), dflt) in enumerate(zip(opts, defaults), 1):
            name = f"{g.name}_{k}"
            built = build_materials(files, name, patch=lambda tm, vals=vals: _patch_params(tm, vals[tm.name]) if tm.name in vals else None)
            out.append(Variant(name=name, materials=built, group=group, label=f"{label.split(' / ')[0]} {n}", group_label=label,
                               default=dflt and not any(defaults[: n - 1])))
    return out


def _finish_pbr_material(mat: Material, tm: T.TrMaterial, load_tex, register, stem,
                         lym: np.ndarray | None = None, lym_file: str | None = None) -> Material:
    mat.extras.pop("shading", None)
    mat.extras["pbr"] = True
    mat.metallic_factor = float(tm.floats.get("Metallic", 0.0))
    # Layer enable as for the colours and the per-layer specular (mask scale x layer strength).
    scales = [tm.floats.get(f"LayerMaskScale{i}", 1.0) * _layer_strength(tm.colors.get(f"BaseColorLayer{i}")) for i in range(1, 5)]
    if lym is not None and any(x > 0 for x in scales):
        mr = bake_layer_metal(tm.floats, lym, scales)
        if mr is not None:
            rgba, sig = mr
            mat.metallic_roughness = register(f"{stem(lym_file)}_mr_{sig}", rgba, False, mat.extras["mirror"])
            mat.wrap["metallic_roughness"] = ("repeat", "repeat")
            mat.metallic_factor = mat.roughness_factor = 1.0
    return mat


def bake_layer_metal(floats: dict, mask: np.ndarray, scales: list[float]) -> tuple[np.ndarray, str] | None:
    """Per-texel metallic/roughness from `MetallicLayerN` / `RoughnessLayerN` over the base `Metallic`/`Roughness`,
    lerped through the layer mask in order like the colours (SV/LA PBR shaders). Returns a glTF metallicRoughness
    texture (G roughness, B metallic) + a cache signature, or None when no layer differs from the base.

    SV eyes (`EyeClearCoat`) paint the iris as metallic layers (Vulpix: layers 1/3 brown, metallic 1, roughness 0.8):
    a metal has no diffuse term, so the iris reads dark brown in game. Baked as plain diffuse it was a lit pale disc.
    """
    base = (float(floats.get("Metallic", 0.0)), float(floats.get("Roughness", 0.8)))
    layers = []
    for i in range(1, 5):
        m, r = floats.get(f"MetallicLayer{i}"), floats.get(f"RoughnessLayer{i}")
        layers.append(None if m is None and r is None else
                      (base[0] if m is None else float(m), base[1] if r is None else float(r)))
    live = [l for i, l in enumerate(layers) if l is not None and scales[i] > 0]
    if not any(abs(l[0] - base[0]) > 0.02 or abs(l[1] - base[1]) > 0.02 for l in live):
        return None
    w = np.clip(mask.astype(np.float32) / 255.0 * np.asarray(scales, np.float32)[None, None, :], 0.0, 1.0)
    metal = np.full(mask.shape[:2], base[0], np.float32)
    rough = np.full(mask.shape[:2], base[1], np.float32)
    for i, l in enumerate(layers):
        if l is None or scales[i] <= 0:
            continue
        metal += (l[0] - metal) * w[..., i]
        rough += (l[1] - rough) * w[..., i]
    rgba = np.empty(mask.shape[:2] + (4,), np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = np.clip(rough * 255.0 + 0.5, 0, 255)
    rgba[..., 2] = np.clip(metal * 255.0 + 0.5, 0, 255)
    rgba[..., 3] = 255
    sig = hashlib.sha1(repr((base, layers, [float(x) for x in scales])).encode()).hexdigest()[:8]
    return rgba, sig


def _is_gas_shell(tm: T.TrMaterial) -> bool:
    """Gastly's smoke: a shell displaced by a cloud-density map by a large height (0.3; ZA `IkCharacter`,
    SV `NonDirectional`). Small heights (Palossand 0.1, Mega Raichu/Rotom sand 0.02) are ordinary surface
    relief, and the other large ones are effects with their own look: Blaziken's wrist flames (`Unlit`, 0.4,
    the density map is nearly black) and Blastoise's water jets (alpha-tested, 0.5), which must not turn
    into translucent gas."""
    return (
        any(t.slot == "DisplacementMap" for t in tm.textures)
        and tm.floats.get("DisplacementHeight", 0.0) >= 0.25
        and tm.shader != "Unlit"
        and tm.shader_flags.get("EnableAlphaTest") != "True"
    )


# Eyelid layers of the IkCharacter / SV eye shaders: (texture slot, UV param, colour layer, shadow layer, overlay key,
# EyelidType values that enable it). Upper lid = UpperEyelidColorMap x layer 7 (shadow tint 9) through UVScaleOffset3;
# lower lid = LowerEyelidColorMap x layer 8 (shadow 10) through UVScaleOffset4. UVScaleOffset2 never moves on its own
# (ZA/SV/LA survey: equal to 3 in every clip that keys both), while 3 moves alone where it is the visible lid: SV
# Sinistcha (pm1134) steps its expression atlas from the spiral row to the closed row (blink) through 3 only, ZA
# pm0860_51 blinks its (0, 1)-scale lid edge through 3 with 2 parked at (1, 1, 0, 0).
LID_LAYERS = (
    ("UpperEyelidColorMap", "UVScaleOffset3", 7, 9, "lidU", ("All", "Upper")),
    ("LowerEyelidColorMap", "UVScaleOffset4", 8, 10, "lidL", ("All", "Lower")),
)


def _lid_overlay(tm: T.TrMaterial, spec, tag: str, load_tex, register, wrap_of) -> Material | None:
    """One sliding eyelid as its own blended material, drawn over a copy of the eye triangles.

    The game samples the lid with `uv' = (uv + o - centre) * scale + centre` (UVCenter0; offsets before scaling, like
    the Falinks atlas) through a clamp sampler, and clips animate `o`: Greninja's lids have scale (0, 1), i.e. a
    horizontal edge (the centre column of the lid texture) that slides down over the eye (rest V -0.5 = shut, battle
    clips 0.206 = the half-lidded glare, lower lid parked out of view at -0.5). Lid colour = its RGB x the layer
    colour (ZA lids are white; SV lids carry the colour with a white layer), alpha = the lid. The primitive UVs get
    the rest transform (`extras.lid_overlay`, applied in import_model) and clips move it by UVScaleOffsetN."""
    slot, param, cl, sl, key, _ = spec
    f = _tex_file(tm, slot)
    img = load_tex(f) if f else None
    if img is None:
        return None
    a = img[..., 3]
    cover = float((a >= 128).mean())
    if not 0.001 < cover < 0.999:
        return None  # no lid shape (fully clear or a solid placeholder)
    col = tm.colors.get(f"BaseColorLayer{cl}") or (1.0, 1.0, 1.0, 1.0)
    shd = tm.colors.get(f"BaseColorLayer{sl}")
    lin = srgb_to_linear(img[..., :3].astype(np.float32) / 255.0)
    lit = lin * np.asarray(col[:3], np.float32)
    dark = lit * (np.asarray(shd[:3], np.float32) if shd else 0.6)
    enc = lambda x: np.clip(linear_to_srgb(x) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba = np.concatenate([enc(lit), a[..., None]], axis=2)
    srgba = np.concatenate([enc(dark), a[..., None]], axis=2)
    sig = hashlib.sha1(repr((tuple(col[:3]), tuple(shd[:3]) if shd else None)).encode()).hexdigest()[:8]
    base = os.path.splitext(os.path.basename(f))[0]
    so = tuple(float(x) for x in tm.colors.get(param, (1.0, 1.0, 0.0, 0.0))[:4])
    cen = tm.colors.get("UVCenter0", (0.0, 0.0, 0.0, 0.0))
    w = wrap_of(slot)
    wrap = tuple("clamp" if m not in ("repeat", "mirror") else m for m in w)
    ov = Material(name=f"{tm.name}~{key}")
    ov.base_color = register(f"{base}_{sig}{'_' + tag if tag else ''}_lid", rgba, True)
    ov.wrap["base_color"] = wrap
    ov.alpha_mode = "BLEND"
    ov.double_sided = tm.cull == 2
    ov.roughness_factor, ov.metallic_factor = 0.75, 0.0
    ov.extras.update({
        "shadow_color": register(f"{base}_{sig}{'_' + tag if tag else ''}_lidshadow", srgba, True),
        "lid_overlay": {"scale": so[:2], "offset": so[2:], "center": (float(cen[0]), float(cen[1]))},
        "uv_scale_offset": so, "uv_track_param": param, "uv_track_src": tm.name, "mirror": (False, False),
        "shader": tm.shader,
    })
    return ov


def part_tile(uv: np.ndarray, indices: np.ndarray) -> float:
    """Integer V tile of the vertices a part's triangles use (median, so a stray vertex does not move it)."""
    used = np.unique(np.asarray(indices).reshape(-1))
    return float(np.floor(np.median(uv[used, 1]))) if used.size else float(np.floor(np.median(uv[:, 1])))


def lid_overlay_uvs(uv: np.ndarray, lo: dict, tile: float | None = None) -> np.ndarray:
    """Trinity UVs (bottom-left origin, material V tile included) -> glTF UVs of a lid overlay at its rest offset.

    In glTF/image space (+V down) a larger game offset samples further down the lid texture, the direction the
    viewer applies clip offsets in (verified on Falinks' blink rows), so the rest transform is
    `v' = (v + o - c) * s + c` there too. No tile wrap after the transform: the lid relies on its clamp sampler.
    `tile` = the part's V tile (`part_tile`); default: the median of all of `uv`."""
    if tile is None:
        tile = np.floor(np.median(uv[:, 1]))
    u = uv[:, 0]
    v = 1.0 - (uv[:, 1] - tile)
    (su, sv), (ou, ov), (cu, cv) = lo["scale"], lo["offset"], lo["center"]
    cv = 1.0 - cv
    return np.stack([(u + ou - cu) * su + cu, (v + ov - cv) * sv + cv], axis=1).astype(np.float32)


def _layer5(tm: T.TrMaterial, load_tex, wrap_of) -> dict | None:
    """Material colour layer 5 (`NumMaterialLayer` 6): a fifth colour layer whose mask is its own texture.

    mask = `OpacityMap1` R channel, sampled on UV set `UVIndexLayer5` through `UVScaleOffset5`; colour `BaseColorLayer5`
    (shadow `ShadowingColorLayer5`, glow `EmissionColorLayer5` x `EmissionIntensityLayer5`), strength `LayerMaskScale5`.
    It covers layers 1-4. Yveltal (Z-A pm0769) paints its black wings/back/veins this way (UV0, static); Xerneas
    (pm0768) uses it for the horn glow its battle clips sweep along UV1. Unlike the main `UVScaleOffset` the layer samples
    `uv * scale - offset` (Trinity space): with +offset the clip tracks of Crabrawler/Crabominable (U -> -0.75), Xerneas
    (U 0.5 -> 0) and SV Black/White Kyurem (rest V -0.6 / -1.0) would never show the layer, or show it fully at rest."""
    f = _tex_file(tm, "OpacityMap1")
    c = tm.colors.get("BaseColorLayer5")
    if not f or c is None:
        return None
    try:
        if int(tm.shader_flags.get("NumMaterialLayer", "0")) < 6:
            return None
    except ValueError:
        return None
    strength = float(tm.floats.get("LayerMaskScale5", 1.0)) * _layer_strength(c)
    img = load_tex(f) if strength > 0 else None
    if img is None:
        return None
    sc = tm.colors.get("ShadowingColorLayer5")
    ei = float(tm.floats.get("EmissionIntensityLayer5", 0.0))
    ec = tm.colors.get("EmissionColorLayer5", (1.0, 1.0, 1.0, 1.0))
    return {
        "file": f,
        "mask": np.ascontiguousarray(img[..., 0]),
        "uv_set": max(0, int(tm.ints.get("UVIndexLayer5", 0))),
        "so": tuple(float(x) for x in tm.colors.get("UVScaleOffset5", (1.0, 1.0, 0.0, 0.0))[:4]),
        "colour": tuple(float(x) for x in c[:3]),
        "strength": strength,
        "shadow": tuple(float(x) for x in sc[:3]) if sc else None,
        "emission": tuple(float(x) * ei for x in ec[:3]) if ei > 0 and max(ec[:3]) > 0.05 else None,
        "wrap": wrap_of("OpacityMap1"),
    }


def _layer5_moving(animations: list[AnimationFile]) -> set[str]:
    """Materials whose layer-5 mask a clip moves (`.tracm` UVScaleOffset5): they need the overlay, not a bake."""
    out: set[str] = set()
    for af in animations:
        if not af.config:
            continue
        try:
            tracks = T.parse_tracm_materials(af.config)
        except Exception:  # noqa: BLE001 - material tracks are optional
            continue
        out |= {t.material for t in tracks if t.param == "UVScaleOffset5"}
    return out


def _full_width(mask: np.ndarray, alb: np.ndarray) -> bool:
    """A mask that is twice as wide (relative to its height) as a half-width mirrored albedo."""
    r = mask.shape[1] / mask.shape[0]
    return abs(r - 2.0 * alb.shape[1] / alb.shape[0]) < 0.05 * r


def bake_extra_layer(img: np.ndarray, albedo: np.ndarray, mask: np.ndarray, colour, strength: float, multiply: bool = True) -> np.ndarray:
    """One more layer over an already-baked image, like one step of bake_layers: lerp (linear light) toward
    albedo x colour by mask x strength. `mask` is a single channel; all three images are resampled to the largest."""
    h = max(img.shape[0], albedo.shape[0], mask.shape[0])
    w = max(img.shape[1], albedo.shape[1], mask.shape[1])
    if img.shape[:2] != (h, w):
        img = resize_rgba(img, w, h)
    if albedo.shape[:2] != (h, w):
        albedo = resize_rgba(albedo, w, h)
    if mask.shape[:2] != (h, w):
        mask = np.asarray(Image.fromarray(mask, "L").resize((w, h), Image.BILINEAR))
    a = srgb_to_linear(albedo[..., :3].astype(np.float32) / 255.0)
    out = srgb_to_linear(img[..., :3].astype(np.float32) / 255.0)
    wt = np.clip(mask.astype(np.float32) / 255.0 * float(strength), 0.0, 1.0)[..., None]
    col = np.asarray(colour, np.float32)[None, None, :]
    tinted = a * col if multiply else np.broadcast_to(col, a.shape)
    out = out + (tinted - out) * wt
    rgba = np.empty_like(img)
    rgba[..., :3] = np.clip(linear_to_srgb(out) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba[..., 3] = img[..., 3]
    return rgba


def _layer5_overlay(tm: T.TrMaterial, l5: dict, tag: str, register) -> Material | None:
    """Layer 5 on another UV set or moved by clips (Xerneas' horn glow): its own blended material on a copy of the
    triangles, like the sliding eyelids. Colour = BaseColorLayer5 (not x albedo: the albedo lives in UV0 space),
    alpha = mask x strength, glow = emissive factor. Clips drive it through UVScaleOffset5 (`uv_track_gain`: the layer
    subtracts its offset, so the glTF U delta is negated; V is flipped twice and keeps its sign)."""
    m = l5["mask"].astype(np.float32) / 255.0 * min(1.0, l5["strength"])
    if float(m.max()) < 0.02:
        return None
    c = np.asarray(l5["colour"], np.float32)
    shd = c * (np.asarray(l5["shadow"], np.float32) if l5["shadow"] else np.float32(0.6))
    alpha = np.clip(m * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def flat(col: np.ndarray) -> np.ndarray:
        rgba = np.empty(m.shape + (4,), np.uint8)
        rgba[..., :3] = np.clip(linear_to_srgb(col) * 255.0 + 0.5, 0, 255).astype(np.uint8)[None, None, :]
        rgba[..., 3] = alpha
        return rgba

    sig = hashlib.sha1(repr((l5["colour"], l5["shadow"], l5["strength"])).encode()).hexdigest()[:8]
    base = os.path.splitext(os.path.basename(l5["file"]))[0]
    so = l5["so"]
    ov = Material(name=f"{tm.name}~l5")
    ov.base_color = register(f"{base}_{sig}{'_' + tag if tag else ''}_l5", flat(c), True)
    ov.wrap["base_color"] = tuple("clamp" if w not in ("repeat", "mirror") else w for w in l5["wrap"])
    ov.alpha_mode = "BLEND"
    ov.double_sided = tm.cull == 2
    ov.roughness_factor, ov.metallic_factor = 0.75, 0.0
    if l5["emission"] is not None:
        ov.emissive_factor = tuple(min(1.0, float(x)) for x in l5["emission"])
    ov.extras.update({
        "shadow_color": register(f"{base}_{sig}{'_' + tag if tag else ''}_l5shadow", flat(shd), True),
        "layer_overlay": {"uv_set": l5["uv_set"], "scale": so[:2], "offset": so[2:]},
        "uv_scale_offset": so, "uv_track_param": "UVScaleOffset5", "uv_track_src": tm.name, "uv_track_gain": (-1.0, 1.0),
        "mirror": (False, False), "shader": tm.shader,
    })
    return ov


def layer_overlay_uvs(uv: np.ndarray, lo: dict, tile: float | None = None) -> np.ndarray:
    """Trinity UVs of the layer's UV set -> glTF UVs of a layer-5 overlay at its rest offset: `uv * scale - offset` with
    the V tile dropped (see _layer5), then V flipped. No wrap: the overlay relies on its sampler (clamp)."""
    if tile is None:
        tile = np.floor(np.median(uv[:, 1]))
    (su, sv), (ou, ov) = lo["scale"], lo["offset"]
    u = uv[:, 0] * su - ou
    v = (uv[:, 1] - tile) * sv - ov
    return np.stack([u, 1.0 - v], axis=1).astype(np.float32)


def _lid_off_atlas(animations: list[AnimationFile]) -> set[str]:
    """Materials some clip moves the lower-lid atlas (UVScaleOffset4) outside [0, 1) on: off the clamp-sampled atlas,
    i.e. no lid at all (Froakie parks it at V 2.0)."""
    out: set[str] = set()
    for af in animations:
        if not af.config:
            continue
        try:
            tracks = T.parse_tracm_materials(af.config)
        except Exception:  # noqa: BLE001
            continue
        for t in tracks:
            if t.param == "UVScaleOffset4" and len(t.channels) >= 4:
                vals = np.concatenate([t.channels[2][1], t.channels[3][1]])
                if len(vals) and (vals.min() < -1e-3 or vals.max() >= 1.0 - 1e-3):
                    out.add(t.material)
    return out


def _lid_atlas_unfit(animations: list[AnimationFile]) -> set[str]:
    """Materials whose lower-eyelid atlas cannot be baked together with the eye (see the eyelid block in
    _convert_material): some clip moves the eye (UVScaleOffset) and the lid (UVScaleOffset4) differently (Froakie parks
    only its lid at V 2.0, off the clamp-sampled atlas; Ash-Greninja and Munkidori slide the lid in over a still eye).
    Those need the lid as a separate layer, which the viewer does not have yet."""
    bad: set[str] = set()
    for af in animations:
        if not af.config:
            continue
        try:
            tracks = T.parse_tracm_materials(af.config)
        except Exception:  # noqa: BLE001
            continue
        per: dict[str, dict[str, T.MaterialParamTrack]] = {}
        for t in tracks:
            if t.param in ("UVScaleOffset", "UVScaleOffset4") and len(t.channels) >= 4:
                per.setdefault(t.material, {}).setdefault(t.param, t)
        for name, ps in per.items():
            eye, lid = ps.get("UVScaleOffset"), ps.get("UVScaleOffset4")
            frames = np.unique(np.concatenate([t.channels[k][0] for t in ps.values() for k in (2, 3)] + [np.zeros(1, np.float32)]))
            for k in (2, 3):
                e = _held(*eye.channels[k], frames, 0.0) if eye else None
                l_ = _held(*lid.channels[k], frames, 0.0) if lid else None
                if e is not None and l_ is not None and np.abs(e - l_).max() > 1e-3:
                    bad.add(name)
                if (e is None) != (l_ is None) and np.abs(e if e is not None else l_).max() > 1e-3:
                    bad.add(name)
    return bad


def _convert_material(tm: T.TrMaterial, tag: str, load_tex, textures: dict[str, TextureImage], tex_meta, skin_tones: bool = False, lid_atlas: bool = True,
                      overlays: list[Material] | None = None, lid_parked: bool = False, layer5_moves: bool = False, beard: bool = False) -> Material:
    def stem(f: str) -> str:
        return os.path.splitext(os.path.basename(f))[0]

    def register(key: str, rgba: np.ndarray, srgb: bool, mirror: tuple[bool, bool] = (False, False)) -> str:
        if mirror[0]:
            rgba = np.concatenate([rgba, rgba[:, ::-1]], axis=1)
            key += "_mu"
        if mirror[1]:
            # V is flipped after scaling, so the mirrored copy goes first.
            rgba = np.concatenate([rgba[::-1], rgba], axis=0)
            key += "_mv"
        if key not in textures:
            textures[key] = TextureImage(key=key, rgba=np.ascontiguousarray(rgba, np.uint8), srgb=srgb)
        return key

    mat = Material(name=tm.name)
    flags = tm.shader_flags
    uvso = tm.colors.get("UVScaleOffset", (1.0, 1.0, 0.0, 0.0))
    uvso_n = tm.colors.get("UVScaleOffsetNormal", uvso) if flags.get("EnableUVScaleOffsetNormal", "True") != "False" else uvso
    # `UVTransformMode T` (SV/LA eyes; the flag's only value) = translate only: the scale is not applied. Every such
    # eye with a scale != 1 (Magnemite/Lumineon (2, 4), Magikarp/Cutiefly (1, 0), LA Carnivine (1, 2)) has UVs that
    # already cover exactly one cell of a 2x2 expression atlas (U < 0.5, V > 0.5) and clips step the offset between
    # cells (Magnemite: U 0 -> 0.5 = pupil -> closed); applied, the scale smeared the atlas or collapsed it to one row.
    translate_only = flags.get("UVTransformMode") == "T"
    if translate_only:
        uvso = (1.0, 1.0, float(uvso[2]), float(uvso[3]))
        uvso_n = (1.0, 1.0, float(uvso_n[2]), float(uvso_n[3]))
        mat.extras["uv_translate_only"] = True
    mat.extras["uv_scale_offset"] = tuple(float(x) for x in uvso[:4])

    def wrap_of(slot: str) -> tuple[str, str]:
        for i, b in enumerate(tm.textures):
            if b.slot == slot and i < len(tm.samplers):
                return tm.samplers[i]
        return ("repeat", "repeat")

    # Mirrored samplers: the game stores half of a symmetric texture and
    # samples it with UVScaleOffset (2,1) + mirror. Downstream tools (Blender
    # FBX/OBJ, Unreal) handle plain repeat far more reliably, so mirrored
    # textures are doubled (mirror-concatenated) and the UV scale halved.
    # Only double when the scaled UV range actually spans the mirror, i.e.
    # scale >= 2; a mirror sampler with scale 1 never leaves [0,1].
    base_wrap = wrap_of("BaseColorMap")
    centre = tm.colors.get("UVCenter0", (0.0, 0.0, 0.0, 0.0))
    centred = any(abs(float(centre[k])) > 1e-6 for k in range(2)) and any(abs(float(uvso[k]) - 1.0) > 1e-6 for k in range(2))
    if centred:
        mat.extras["uv_center"] = (float(centre[0]), float(centre[1]))
        if wrap_of("LayerMaskMap")[0] == "mirror":
            # Centred eyes (Falinks) put a placeholder albedo on a clamp sampler, but the layer mask that carries the
            # eye shapes is mirror-sampled and the scaled UVs go negative on one eye: bake the mirror like the albedo.
            base_wrap = ("mirror", base_wrap[1])
    mirror_u = base_wrap[0] == "mirror" and float(uvso[0]) >= 2.0
    mirror_v = base_wrap[1] == "mirror" and float(uvso[1]) >= 2.0
    su_eff = float(uvso[0]) / (2.0 if mirror_u else 1.0)
    sv_eff = float(uvso[1]) / (2.0 if mirror_v else 1.0)
    mat.extras["uv_scale_effective"] = (su_eff, sv_eff, float(uvso[2]), float(uvso[3]))
    mat.extras["mirror"] = (mirror_u, mirror_v)
    for slot in ("base_color", "normal", "occlusion", "emissive"):
        mat.wrap[slot] = ("repeat", "repeat")
    # The normal map may use a different UV scale than the base colour
    # (typically (1,1) on a full-width map vs (2,1) on the half-width albedo).
    n_wrap = wrap_of("NormalMap")
    n_mirror_u = n_wrap[0] == "mirror" and float(uvso_n[0]) >= 2.0
    n_mirror_v = n_wrap[1] == "mirror" and float(uvso_n[1]) >= 2.0
    sun_eff = float(uvso_n[0]) / (2.0 if n_mirror_u else 1.0)
    svn_eff = float(uvso_n[1]) / (2.0 if n_mirror_v else 1.0)
    if (sun_eff, svn_eff) != (su_eff, sv_eff) and su_eff and sv_eff:
        mat.uv_transform["normal"] = (sun_eff / su_eff, svn_eff / sv_eff, 0.0, 0.0)
    mat.extras["mirror_normal"] = (n_mirror_u, n_mirror_v)

    # Base colour with baked layer colours
    alb_file = _tex_file(tm, "BaseColorMap")
    alb = load_tex(alb_file) if alb_file and flags.get("EnableBaseColorMap", "True") != "False" else None
    alb = clear_near_opaque_lens(alb, tm.alpha_type)
    lym_file = _tex_file(tm, "LayerMaskMap")
    lym = load_tex(lym_file) if lym_file else None
    mask_is_full_width = (
        lym is not None and alb is not None
        and abs(lym.shape[1] / lym.shape[0] - 2.0 * alb.shape[1] / alb.shape[0]) < 0.05 * lym.shape[1] / lym.shape[0]
    )
    if lym is not None and mirror_u and (wrap_of("LayerMaskMap")[0] != "mirror" or mask_is_full_width):
        # The albedo stores half of a symmetric layout (mirror sampler) but this mask is a
        # full-width clamp-sampled texture (Croconaw body_b: 1024x1024 vs a 512x256 albedo).
        # Baking it as-is and then mirror-concatenating repeats the whole layout twice across
        # U. The layout is symmetric, so its left half is the mirrored half the albedo matches.
        lym = lym[:, : lym.shape[1] // 2]
    layer_colors = []
    layer_enable = []
    for i in range(1, 5):
        c = tm.colors.get(f"BaseColorLayer{i}")
        layer_colors.append(c[:3] if c else None)
        # The colour's alpha is the layer's enable/strength: Pokemon use 1.0,
        # trainer clothing/skin use 0.0 with a placeholder black mask.
        layer_enable.append(_layer_strength(c))
    base_factor = tm.colors.get("BaseColor", (1.0, 1.0, 1.0, 1.0))
    base_factor = fresnel_facing_color(tm, base_factor)
    l5_bake = None  # layer 5 baked into the albedo (see _layer5)
    if alb is not None:
        # ColorTableMap (trainer faces): a small palette (ColorTableDivideNumber
        # cells per axis) selected per texel by the layer mask (row from R,
        # column from G) and multiplied into the albedo; it holds the skin tone.
        ct_file = _tex_file(tm, "ColorTableMap")
        table_shadow = None
        ct_sig = None  # colour-table identity for the bake key (same albedo + mask, different zones per face/outfit)
        if ct_file and flags.get("EnableColorTableMap") == "True" and lym is not None:
            ct = load_tex(ct_file)
            if ct is not None:
                div = int(tm.ints.get("ColorTableDivideNumber", 2))
                idxs = [int(tm.ints.get(f"BaseColorIndex{i}", 1)) for i in range(1, 5)]
                eye = flags.get("EnableEyeOptions") == "True"
                modular = skin_tones  # modular NPC part (composite importer)
                skin = modular and tm.name in SKIN_MATERIALS
                # Eyes and modular parts honour LayerMaskScaleN (the mask alpha is often fully on while layer 4 is unused).
                # Modular clothes/hair/eyes only tint where a mask channel covers them (the sclera and
                # unpainted cloth stay untinted); skin keeps layer 1 as the base under everything.
                masked = eye or (modular and not skin)
                scl = [tm.floats.get(f"LayerMaskScale{i}", 1.0) for i in range(1, 5)] if (eye or modular) else None
                skin_layers: list[int] = []
                if skin:
                    idxs[0] = SKIN_DEFAULT_TONE
                    skin_layers = skin_zone_layers(lym, scl, idxs)
                    for sl in skin_layers:
                        idxs[sl - 1] = SKIN_DEFAULT_TONE
                raw_alb = alb
                alb, table_shadow = apply_color_table(raw_alb, lym, ct, div, idxs, masked_only=masked, scales=scl, deeper=skin_layers)
                ct_sig = (stem(ct_file), div, tuple(idxs), tuple(float(s) for s in scl) if scl else None, masked)
                if modular and not skin:
                    mask_r = lym if lym.shape[:2] == raw_alb.shape[:2] else resize_rgba(lym, raw_alb.shape[1], raw_alb.shape[0])
                    cover = [float(((mask_r[..., c] / 255.0 * (scl[c] if scl else 1.0)) > 0.02).mean()) for c in range(4)]
                    layers = [c + 1 for c in range(4) if cover[c] > 0.01]
                    if layers:
                        # NPC hair colour is runtime data; the part's table holds only a few authoring swatches (Sleeves
                        # F: grey/rust/cream/navy, default cream), so every hair also offers the player's 32 hair colours.
                        extra = _hair_colors() if tm.name.startswith("hairstyle") and layers == [1] else None
                        layer_palette(mat, raw_alb, mask_r, ct, div, idxs, scl, layers, alb_file, lym_file, register, extra=extra)
                if skin:
                    # Body and head share one skin palette: zone N of both tables is the same tone, so
                    # tone variants keep head and body matched. Layer 4 (lips) keeps its own zone.
                    # A face whose mask has a lipstick layer also gets a runtime lip-colour row (custom_lip_palette).
                    tones = []
                    # Keyed by mask + table too: outfits share sh_body_skin_01_alb but paint different layers (Artist's hands).
                    skin_key = f"{stem(alb_file)}_{stem(lym_file)}_{hashlib.md5(repr((stem(ct_file), div, tuple(idxs[1:]), tuple(scl), tuple(skin_layers))).encode()).hexdigest()[:8]}"
                    for z in range(1, min(div, SKIN_TONES) + 1):
                        lit, shd = apply_color_table(raw_alb, lym, ct, div, [z] + [z if c + 1 in skin_layers else idxs[c] for c in range(1, 4)], scales=scl, deeper=skin_layers)
                        entry = (z, register(f"{skin_key}_skin{z}", lit, True), register(f"{skin_key}_skin{z}_shadow", shd, True))
                        tones.append(entry)
                    mat.extras["skin_tones"] = tones
                    if tm.name == "body_skin":
                        # Painted non-skin layers on the body (Occult F's leggings = layer 2): a runtime palette whose layer 1 follows the skin tone.
                        mask_r = lym if lym.shape[:2] == raw_alb.shape[:2] else resize_rgba(lym, raw_alb.shape[1], raw_alb.shape[0])
                        body_layers = [c + 1 for c in range(1, 4)
                                       if scl[c] > 0 and idxs[c] != idxs[0] and float(((mask_r[..., c] / 255.0 * scl[c]) > 0.02).mean()) > 0.01]
                        if body_layers:
                            layer_palette(mat, raw_alb, mask_r, ct, div, idxs, scl, body_layers, alb_file, lym_file, register, skin_link=True, skin_layers=skin_layers)
                    lip = tm.name == "face_skin" and lip_layer(lym, scl, idxs)
                    beard = 0 if lip or tm.name != "face_skin" else beard_paint_layer(lym, scl, idxs)
                    if lip:
                        custom_lip_palette(mat, raw_alb, lym, ct, div, idxs, scl, lip, alb_file, lym_file, register, skin_layers=skin_layers)
                    elif beard:
                        # A beard painted on the skin mask (Middle A M, Jogger M): "No beard" + the player's hair colours.
                        custom_lip_palette(mat, raw_alb, lym, ct, div, idxs, scl, beard, alb_file, lym_file, register, colors=_hair_colors(), none_label="No beard", skin_layers=skin_layers)
                        mat.extras["beard"] = True
        beard_neutral = None
        if (skin_tones and flags.get("EnableEyeOptions") == "True" and lym is not None and layer_enable[1] == 0
                and layer_colors[1] is not None and tm.floats.get("LayerMaskScale2", 1.0) > 0):
            # Common A/B eyes ship placeholder swatches for the iris layers (the game colours them per NPC): give them a
            # default iris so the bake is not left unpainted and custom_eye_palette can offer the player's contacts.
            default_iris = _default_iris_colors()
            layer_colors[1], layer_colors[2] = default_iris
            layer_enable[1] = layer_enable[2] = 1.0
        if skin_tones and "palette" not in mat.extras and flags.get("EnableEyeOptions") == "True":
            custom_eye_palette(mat, tm, lym, alb, alb_file, lym_file, layer_colors, layer_enable, register)
        elif skin_tones and "palette" not in mat.extras and tm.name == "eyebrow" and lym is not None:
            custom_brow_palette(mat, tm, lym, alb, alb_file, lym_file, layer_colors, layer_enable, register)
        elif skin_tones and "palette" not in mat.extras and beard and alb is not None:
            # Beard mesh without a colour table (Common B): the game tints it per NPC; offer the player's hair colours.
            beard_neutral = custom_hair_palette(mat, alb, alb_file, register, default_zone=BEARD_DEFAULT_ZONE)
        elif skin_tones and "palette" not in mat.extras and tm.name == "hairstyle" and not any(tm.floats.get(f"LayerMaskScale{i}", 1.0) > 0 for i in (1, 2, 3, 4)):
            custom_hair_palette(mat, alb, alb_file, register)
        baked = alb
        scales = [tm.floats.get(f"LayerMaskScale{i}", 1.0) * layer_enable[i - 1] for i in range(1, 5)]
        multiply = flags.get("BaseColorMultiply", "True") != "False"
        bc = tuple(float(base_factor[k]) for k in range(3))
        # SV materials often define only layers 1..3 (Geeta's face, Ryme's skin): missing layers are disabled, not "no bake".
        layered = lym is not None and any(e > 0 for e in layer_enable)
        gas = _is_gas_shell(tm)
        lym_bake = lym
        if layered:
            layer_colors = [c if c is not None else (1.0, 1.0, 1.0) for c in layer_colors]
            # BaseColor goes into the texture (see bake_layers), the glTF factor becomes 1.
            if gas:
                # Gas shell masks partition the surface (R+G+B+A = 1) instead of stacking, and the
                # sequential lerp would leave the white albedo showing through (pale grey Gastly).
                # Re-weight each channel by its share of the running total so the layers average.
                m = lym.astype(np.float32) / 255.0 * np.asarray(scales, np.float32)[None, None, :]
                run = np.cumsum(m, axis=2)
                lym_bake = np.clip(m / np.maximum(run, 1e-6) * 255.0 + 0.5, 0, 255).astype(np.uint8)
                lym_bake[..., 3] = np.where(run[..., 3] > 0, lym_bake[..., 3], 0)
            baked = bake_layers(alb, lym_bake, layer_colors, [1.0 if gas else s_ for s_ in scales], multiply, bc if multiply else (1.0, 1.0, 1.0))
        if beard_neutral is not None:
            # The default beard colour is baked (the palette only replaces it): luminance albedo x the default hair zone.
            tint = srgb_to_linear(np.asarray(mat.extras["palette"]["lit"][BEARD_DEFAULT_ZONE - 1], np.float32) / 255.0)
            lin = srgb_to_linear(beard_neutral[..., :3].astype(np.float32) / 255.0) * tint
            baked = beard_neutral.copy()
            baked[..., :3] = np.clip(linear_to_srgb(lin) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        # Layer 5 (OpacityMap1, see _layer5): baked over layers 1-4 when it maps like the albedo (UV0, same scale, no
        # offsets, no clip moves it: Yveltal's black); otherwise an overlay material (Xerneas' horn glow on UV1).
        l5 = _layer5(tm, load_tex, wrap_of)
        if l5 is not None:
            so5 = l5["so"]
            same_uv = (l5["uv_set"] == 0 and not layer5_moves
                       and all(abs(so5[k] - float(uvso[k])) < 1e-6 for k in (0, 1))
                       and all(abs(so5[k]) < 1e-6 and abs(float(uvso[k])) < 1e-6 for k in (2, 3)))
            if same_uv:
                m5 = l5["mask"]
                if mirror_u and (l5["wrap"][0] != "mirror" or _full_width(m5, alb)):
                    m5 = m5[:, : m5.shape[1] // 2]  # full-width mask over a half-width mirrored albedo (see lym above)
                l5_bake = dict(l5, mask=m5)
                baked = bake_extra_layer(baked, alb, m5, l5["colour"], l5["strength"], multiply)
            elif overlays is not None:
                ov = _layer5_overlay(tm, l5, tag, register)
                if ov is not None:
                    overlays.append(ov)
        # Lower-eyelid expression atlas (EyelidType "Lower": Falinks, Dartrix line, Carbink, Yamask (ZA); Pawmot,
        # Munkidori, Terapagos ... (SV)). The LowerEyelidColorMap is drawn OVER the eye: its alpha is the lid (opaque) with
        # holes in the eye shapes, an atlas of expression cells (Falinks: a 4x4 grid of right/left pairs: open, half,
        # closed, "> <", spirals, the Mega leader's side dots) addressed at UV scale 1 through UVScaleOffset4. Lid colour =
        # its RGB x BaseColorLayer8 (ZA lids are white and layer 8 carries the colour: Falinks' visor = the body's layer 4;
        # SV lids carry the colour with layer 8 white), ZA shadow tint BaseColorLayer10. Through the holes the usual layer
        # bake shows (Falinks: R = layer 1 blue everywhere, so a thin ring is left around the G white).
        # The game offsets before scaling, uv' = (uv + o - centre) * scale + centre, and clips move the eye and lid offsets
        # in lockstep (Falinks: UVScaleOffset == UVScaleOffset4 in every clip of both forms), so a lid texel q = p + o maps
        # to the eye texture at (q - centre) * scale + centre whatever the offset, and the eye is baked once in lid space.
        # (Offsetting after the scale put the white of Falinks' offset-0.5 expressions outside their holes.) The material
        # then uses plain (1, 1) UVs driven by the lid track; an eye track that moves on its own (gaze) is not followed.
        c8 = tm.colors.get("BaseColorLayer8")
        lid_file = _tex_file(tm, "LowerEyelidColorMap")
        # A one-colour eye looks the same under any offset, so the lid may move on its own (Greninja's field eye_b:
        # a black eye under a white lid whose dash/"> <" holes the sleep/glad clips switch).
        # Not when a clip parks the lid off the atlas (Froakie: V 2.0 on a clamp sampler = no lid): the baked eye-in-lid
        # texture repeats, so the parked lid would come back.
        eye_flat = baked is not None and int(np.ptp(baked[..., :3].reshape(-1, 3), axis=0).max()) <= 3 and not lid_parked
        lid = load_tex(lid_file) if lid_file and flags.get("EyelidType") == "Lower" and (lid_atlas or eye_flat) else None
        s4 = tm.colors.get("UVScaleOffset4", (1.0, 1.0, 0.0, 0.0))
        eye_fix = False
        eye_shadow = None
        if (layered and baked is not alb and c8 and lid is not None and abs(float(s4[0]) - 1) < 1e-6 and abs(float(s4[1]) - 1) < 1e-6
                and 0.001 < float((lid[..., 3] < 128).mean()) < 0.999):
            to_lin = lambda img: srgb_to_linear(img[..., :3].astype(np.float32) / 255.0)
            # Shadow side of the eye: each layer x its ShadowingColorLayerN, like the body's shadow bake.
            shd_pal = []
            for i in range(4):
                sc = tm.colors.get(f"ShadowingColorLayer{i + 1}")
                shd_pal.append(tuple(float(layer_colors[i][k]) * (float(sc[k]) if sc else 0.6) for k in range(3)))
            eye_shd = bake_layers(alb, lym_bake, shd_pal, scales, multiply, bc if multiply else (1.0, 1.0, 1.0))
            hh, ww = lid.shape[:2]
            cen = tm.colors.get("UVCenter0", (0.0, 0.0, 0.0, 0.0))
            cu, cv = float(cen[0]), 1.0 - float(cen[1])  # V centre in top-left (image) space
            du_, dv_ = float(uvso[2]) - float(s4[2]), float(uvso[3]) - float(s4[3])  # rest offsets of eye vs lid
            su_ = float(uvso[0]) * ((np.arange(ww, dtype=np.float32) + 0.5) / ww + du_ - cu) + cu
            sv_ = float(uvso[1]) * ((np.arange(hh, dtype=np.float32) + 0.5) / hh + dv_ - cv) + cv

            def wrap(x, mode):
                if mode == "mirror":
                    x = np.abs(x) % 2.0
                    return np.where(x > 1.0, 2.0 - x, x)
                if mode == "repeat":
                    return x - np.floor(x)
                return np.clip(x, 0.0, 1.0)

            wm = wrap_of("LayerMaskMap") if lym is not None else wrap_of("BaseColorMap")
            su_, sv_ = wrap(su_, wm[0]), wrap(sv_, wm[1])
            ex = np.clip((su_ * baked.shape[1]).astype(int), 0, baked.shape[1] - 1)
            ey = np.clip((sv_ * baked.shape[0]).astype(int), 0, baked.shape[0] - 1)
            lid_rgb = to_lin(lid) * np.asarray(c8[:3], np.float32)
            c10 = tm.colors.get("BaseColorLayer10")
            lid_shd = lid_rgb * (np.asarray(c10[:3], np.float32) if c10 else 0.6)
            a = lid[..., 3:4].astype(np.float32) / 255.0  # 1 = lid, 0 = eye hole
            lit = to_lin(baked[ey][:, ex]) * (1 - a) + lid_rgb * a
            shd = to_lin(eye_shd[ey][:, ex]) * (1 - a) + lid_shd * a
            baked = np.empty((hh, ww, 4), np.uint8)
            baked[..., :3] = np.clip(linear_to_srgb(lit) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            baked[..., 3] = 255
            eye_shadow = baked.copy()
            eye_shadow[..., :3] = np.clip(linear_to_srgb(shd) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            # The mesh now samples the lid atlas directly: plain (1, 1) UVs, no centre, no mirror; clips drive it through
            # UVScaleOffset4. The eye normal map / layer emission use the old layout and are dropped.
            mat.extras.update({"uv_scale_effective": (1.0, 1.0, 0.0, 0.0), "uv_scale_offset": (1.0, 1.0, float(s4[2]), float(s4[3])), "mirror": (False, False), "uv_track_param": "UVScaleOffset4", "eyelid_atlas": True})
            mat.extras.pop("uv_center", None)
            mat.uv_transform.pop("normal", None)
            eye_fix = True
        # Sliding eyelids (EyelidType All/Upper, and Lower lids that move apart from a patterned eye): separate layers
        # the game draws over the eye with their own UVScaleOffsetN, which clips animate (Greninja's battle glare,
        # Feebas/Metagross blinks). They become overlay materials on a copy of the eye triangles (see _lid_overlay).
        made = set()
        # (No EnableEyeOptions gate: only Z-A writes that flag. SV and LA lids have EyelidType + the lid textures alone.)
        if overlays is not None and not eye_fix:
            et = flags.get("EyelidType", "None")
            for spec in LID_LAYERS:
                if et in spec[5]:
                    ov = _lid_overlay(tm, spec, tag, load_tex, register, wrap_of)
                    if ov is not None:
                        overlays.append(ov)
                        made.add(spec[4])
        # Upper eyelid (Malamar's drooping lid): EyelidShadowMaskMap is white where the lid covers the eye (a slanted
        # edge across the disc), filled with the lid colour (layer 7). Only without an upper-lid overlay, and only for a
        # mask with an edge: a flat white one (Sinistcha's 32x32 placeholder) painted the whole eye in the lid colour.
        lid_file = _tex_file(tm, "EyelidShadowMaskMap")
        c7 = tm.colors.get("BaseColorLayer7")
        if layered and baked is not alb and lid_file and c7 and flags.get("EyelidType") in ("All", "Upper") and "lidU" not in made:
            lid = load_tex(lid_file)
            if lid is not None and not _is_flat(lid):
                a = np.asarray(Image.fromarray(lid[..., 0]).resize((baked.shape[1], baked.shape[0]), Image.BILINEAR), np.float32)[..., None] / 255.0
                col = np.clip(linear_to_srgb(np.asarray(c7[:3], np.float32)) * 255.0 + 0.5, 0, 255)
                baked = baked.copy()
                baked[..., :3] = (baked[..., :3] * (1 - a) + col * a + 0.5).astype(np.uint8)
        # Key on albedo + mask so materials sharing a placeholder albedo
        # (eyes and flames both use white_msk) do not collide.
        # Key on albedo + mask + the layer colours/scales: materials can share
        # textures but differ in colours (Lucario torso vs legs, Persian body
        # vs ear tips), and identical colour sets (left/right eye) still dedupe.

        # When the layers are not baked the glTF factor keeps BaseColor, and the shadow-side
        # texture must carry that same tint: the viewer's Game shader uses the shadow texture
        # *instead of* diffuseColor (factor x map), so an untinted shadow texture renders the
        # raw near-white albedo on the shadow/ambient side (Rika's teal hair went grey-sage,
        # ZA bare skin went pale). Colour-table skin tones replace the factor, so no tint there.
        baked_bc = layered and multiply
        shadow_bc = (1.0, 1.0, 1.0) if (baked_bc or "skin_tones" in mat.extras) else bc
        l5_sig = (stem(l5_bake["file"]), l5_bake["colour"], l5_bake["strength"], l5_bake["shadow"]) if l5_bake else None
        sig = hashlib.sha1(repr((layer_colors, scales if lym is not None else None, multiply if lym is not None else None, bc if layered else None, shadow_bc)
                                + ((l5_sig,) if l5_sig else ()) + (("ct", ct_sig) if ct_sig else ())).encode()).hexdigest()[:8]
        key = f"{stem(alb_file)}_{stem(lym_file) if lym_file else 'l5'}_{sig}{'_' + tag if tag else ''}_baked" if (baked is not alb or ct_sig) else stem(alb_file)
        # Gas shells (Gastly "smoke"): an OPAQUE shell whose DisplacementMap is a cloud density map the game
        # pans with time. In-game footage (SV, ZA) shows solid, hard-edged violet lumps, and the face shows
        # through a hole where the displaced shell sinks into the black body (see gas_displacement), so the
        # shell stays opaque. Ship the density map (grey; the map tiles seamlessly, so the writer wraps it
        # with repeat) plus the panner parameters so a viewer can animate it (viewer/src/scene/gas.ts).
        disp_file = _tex_file(tm, "DisplacementMap")
        if disp_file and _is_gas_shell(tm) and layered:
            disp = load_tex(disp_file)
            if disp is not None:
                dens = disp[..., :1].repeat(3, axis=2)
                dens = np.concatenate([dens, np.full_like(dens[..., :1], 255)], axis=2)
                mat.extras["gas_density"] = register(stem(disp_file) + "_gas", dens, False)
                mat.extras["gas"] = {
                    "height": float(tm.floats.get("DisplacementHeight", 0.0)),
                    "texCoord": max(0, int(tm.ints.get("UVIndexLayer3", 0))),  # import_model clamps it to the UV sets present
                    "speed": list(SMOKE_DRIFT),
                }
        mat.base_color = register(key, baked, True, mat.extras["mirror"])
        if baked is not alb:
            # Keep the un-layered albedo too so viewers can show the difference.
            mat.extras["raw_base_color"] = register(stem(alb_file), alb, True, mat.extras["mirror"])
        # Shadow-side colour: the game shades each layer with its own
        # ShadowingColorLayerN (dark, saturated counterparts), blended through
        # the same mask, with ShadowStrength as the amount. Bake it as a second
        # texture so viewers can cross-fade base -> shadow by light.
        strength = float(tm.floats.get("ShadowStrength", 0.7))
        sh_base = tm.colors.get("ShadowingColor", (1.0, 1.0, 1.0, 0.0))[:3]
        shadow_palette = []
        shadow_scales = []
        for i in range(4):
            shl = tm.colors.get(f"ShadowingColorLayer{i + 1}")
            # Trainers' clothing: base layers are disabled (white, alpha 0) and
            # the albedo is the lit colour, but the shadow layers still carry
            # the garment hues (Griselle: blue-grey jacket, black bow). So the
            # shadow bake follows the mask wherever a shadow layer exists.
            enabled = layer_enable[i] > 0
            base_i = layer_colors[i] if (enabled and layer_colors[i] is not None) else (bc if layered else (1.0, 1.0, 1.0))
            # Shadow colour = layer colour x ShadowingColorLayerN (undiluted):
            # measured against in-game shots (Griselle's bow albedo x layer 4 =
            # the game's near-black navy). ShadowStrength is how far the shader
            # blends toward it, which the viewer applies at runtime.
            tint = tuple(float(shl[k]) for k in range(3)) if shl else tuple(1.0 + (float(sh_base[k]) - 1.0) * strength for k in range(3))
            if gas:
                # Same hue as the lit layer, darker: ShadowingColorLayerN are magenta-leaning and turned the
                # underside of the gas maroon, while the game's gas fades to dark indigo.
                tint = (0.2, 0.2, 0.28)
            shadow_palette.append(tuple(float(base_i[k]) * tint[k] for k in range(3)))
            shadow_scales.append(tm.floats.get(f"LayerMaskScale{i + 1}", 1.0) * (1.0 if (shl is not None or enabled) else 0.0))
        if lym is not None and any(sc > 0 for sc in shadow_scales):
            # Gas shells: same partition-mask normalisation as the lit bake, or the white albedo survives the
            # sequential lerp and the shadow side comes out light grey instead of the layers' dark indigo.
            shadow = bake_layers(alb, lym_bake, shadow_palette, [1.0 if gas else s_ for s_ in shadow_scales], multiply, bc if baked_bc else (1.0, 1.0, 1.0))
            if shadow_bc != (1.0, 1.0, 1.0):
                shadow = tint_linear(shadow, shadow_bc)
        else:
            tint = np.asarray([(1.0 + (float(sh_base[k]) - 1.0) * strength) * float(shadow_bc[k]) for k in range(3)], np.float32)
            lin = srgb_to_linear(alb[..., :3].astype(np.float32) / 255.0) * tint[None, None, :]
            shadow = np.empty_like(alb)
            shadow[..., :3] = np.clip(linear_to_srgb(lin) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            shadow[..., 3] = alb[..., 3]
        if eye_fix:
            shadow = eye_shadow  # same masks as the lit bake (the plain layer-1 shadow would paint the visor blue)
        elif table_shadow is not None:
            shadow = table_shadow  # skin: the colour table's bottom half is the shadow tone
        elif not any(tm.colors.get(f"ShadowingColorLayer{i + 1}") for i in range(4)):
            shadow = darken_saturate(shadow, 0.78, 1.15)  # no per-layer shadow data: generic darker shade
        if l5_bake is not None and not eye_fix and table_shadow is None:
            # Layer 5 on the shadow side: its colour x ShadowingColorLayer5, like layers 1-4.
            st5 = l5_bake["shadow"] or tuple(1.0 + (float(sh_base[k]) - 1.0) * strength for k in range(3))
            shadow = bake_extra_layer(shadow, alb, l5_bake["mask"], tuple(l5_bake["colour"][k] * st5[k] for k in range(3)), l5_bake["strength"], multiply)
        mat.extras["shadow_color"] = register(f"{stem(alb_file)}_{sig}{'_' + tag if tag else ''}_shadow", shadow, True, mat.extras["mirror"])
    baked_bc = alb is not None and layered and multiply
    # (shadow texture: see shadow_bc above; it carries the factor whenever this keeps BaseColor)
    # Skin tone comes from the colour table; the body material's authored BaseColor tint would
    # make it redder than the face, so it is dropped for modular skin.
    mat.base_color_factor = (1.0, 1.0, 1.0, 1.0) if (baked_bc or "skin_tones" in mat.extras) else (
        float(base_factor[0]),
        float(base_factor[1]),
        float(base_factor[2]),
        # Texture-less Blend materials (df_face0016 lens, shader "Transparent", BaseColor alpha 0.15)
        # take their opacity from the BaseColor alpha; textured ones from the albedo alpha.
        float(base_factor[3]) if alb is None and tm.alpha_type and tm.alpha_type.lower() not in ("opaque", "") else 1.0,
    )

    nrm_file = _tex_file(tm, "NormalMap")
    if nrm_file and flags.get("EnableNormalMap", "True") != "False" and not mat.extras.get("eyelid_atlas"):
        nrm = load_tex(nrm_file)
        if nrm is not None:
            mat.normal = register(stem(nrm_file), nrm, False, mat.extras["mirror_normal"])

    ao_file = _tex_file(tm, "OcclusionMap") or _tex_file(tm, "AOMap")  # SV names the slot AOMap
    if ao_file and flags.get("EnableAOMap", "True") != "False":
        ao = load_tex(ao_file)
        if ao is not None and not _is_flat(ao):
            mat.occlusion = register(stem(ao_file), ao, False, mat.extras["mirror"])

    # Emission (rarely used on Pokémon bodies; eyes/markings sometimes)
    emi = tm.floats.get("EmissionIntensity", 0.0)
    emi_file = _tex_file(tm, "EmissionColorMap")
    if emi > 0 and emi_file:
        e = load_tex(emi_file)
        if e is not None:
            mat.emissive = register(stem(emi_file), e, True)
            c = tm.colors.get("EmissionColor", (1.0, 1.0, 1.0, 1.0))
            mat.emissive_factor = tuple(min(1.0, float(x) * emi) for x in c[:3])

    # Eye highlights: layer 5 emission masked by HighlightMaskMap. Baked as
    # an emissive texture so the white glint shows regardless of lighting.
    hl_file = _tex_file(tm, "HighlightMaskMap")
    hl_int = float(tm.floats.get("EmissionIntensityLayer5", 0.0))
    if hl_file and flags.get("EnableHighlight") == "True" and hl_int > 0 and mat.emissive is None:
        hl = load_tex(hl_file)
        if hl is not None and not _is_flat(hl):
            c = tm.colors.get("EmissionColorLayer5", (1.0, 1.0, 1.0, 1.0))
            rgb = linear_to_srgb(np.asarray(c[:3], np.float32))
            e = np.empty_like(hl)
            for i in range(3):
                e[..., i] = np.clip(hl[..., 0].astype(np.float32) * rgb[i] + 0.5, 0, 255).astype(np.uint8)
            e[..., 3] = 255
            mat.emissive = register(stem(hl_file) + "_hl", e, True, mat.extras["mirror"])
            mat.emissive_factor = (min(1.0, hl_int),) * 3

    # Per-layer emission (Golurk's glowing seals and eyes): EmissionColorLayerN x EmissionIntensityLayerN,
    # routed through the _lym channels like the base colours (each layer covers the previous).
    lym_file = _tex_file(tm, "LayerMaskMap")
    if mat.emissive is None and lym_file and flags.get("EnableEmissionLayer", "True") != "False" and not mat.extras.get("eyelid_atlas"):
        emis = []
        for i in range(1, 5):
            c = tm.colors.get(f"EmissionColorLayer{i}", (0.0, 0.0, 0.0, 1.0))
            k = float(tm.floats.get(f"EmissionIntensityLayer{i}", 0.0))
            if (k == 0 and "eye" not in tm.name and os.path.basename(lym_file).startswith(SPOT_GLOW_SPECIES)
                    and max(c[:3]) - min(c[:3]) > 0.25):
                # Inkay/Malamar's yellow spots: a chromatic emission colour at intensity 0 that is lit in game.
                # Not general: at intensity 0 Xerneas/Yveltal's chromatic layers are their off (Neutral) mode.
                k = 1.0
            emis.append(np.asarray(c[:3], np.float32) * k if k > 0 and max(c[:3]) > 0.05 else None)
        lm = load_tex(lym_file)
        e5 = l5_bake["emission"] if l5_bake is not None else None
        if lm is not None and (any(e is not None for e in emis) or e5 is not None):
            if mirror_u and (wrap_of("LayerMaskMap")[0] != "mirror" or mask_is_full_width):
                lm = lm[:, : lm.shape[1] // 2]  # full-width mask over a half-width mirrored layout (see above)
            w = lm.astype(np.float32) / 255.0 * np.asarray([tm.floats.get(f"LayerMaskScale{i}", 1.0) for i in range(1, 5)], np.float32)
            em = np.zeros(lm.shape[:2] + (3,), np.float32)
            for i, e in enumerate(emis):
                em += ((e if e is not None else 0.0) - em) * np.clip(w[..., i : i + 1], 0.0, 1.0)
            if l5_bake is not None:
                # Layer 5 covers the glow of layers 1-4 too (with its own glow, or none).
                m5 = np.asarray(Image.fromarray(l5_bake["mask"], "L").resize((lm.shape[1], lm.shape[0]), Image.BILINEAR), np.float32)
                w5 = np.clip(m5 / 255.0 * l5_bake["strength"], 0.0, 1.0)[..., None]
                em += ((np.asarray(e5, np.float32) if e5 is not None else 0.0) - em) * w5
            if em.max() > 0.01:
                rgba = np.empty(lm.shape[:2] + (4,), np.uint8)
                rgba[..., :3] = np.clip(linear_to_srgb(np.clip(em, 0.0, 1.0)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
                rgba[..., 3] = 255
                sig_e = hashlib.sha1(repr([None if e is None else tuple(float(x) for x in e) for e in emis]
                                          + ([(stem(l5_bake["file"]), e5)] if l5_bake is not None else [])).encode()).hexdigest()[:8]
                mat.emissive = register(f"{stem(lym_file)}_emi_{sig_e}", rgba, True, mat.extras["mirror"])
                mat.emissive_factor = (1.0, 1.0, 1.0)

    # Blend type wins over alpha test: glasses (IkCharacterTransparent, "Blend")
    # also set EnableAlphaTest with DiscardValue 0, and exporting MASK made their
    # lenses (albedo alpha 0.3-0.5) fully opaque. extras.blend keeps the game's
    # mode ("Blend" / "BlendPreMultiAlpha" / "Add"), which glTF cannot express;
    # the viewer maps it to three.js blending.
    if tm.alpha_type and tm.alpha_type.lower() not in ("opaque", ""):
        mat.alpha_mode = "BLEND"
        mat.extras["blend"] = tm.alpha_type
    elif flags.get("EnableAlphaTest") == "True":
        mat.alpha_mode = "MASK"
        mat.alpha_cutoff = float(tm.floats.get("DiscardValue", 0.5))
    mat.double_sided = tm.cull == 2
    mat.roughness_factor = 0.75
    mat.metallic_factor = 0.0
    mat.extras["shader"] = tm.shader  # keep uv_scale_effective / mirror flags set above
    # Stylised-shading parameters of the game's IkCharacter shader, for the
    # viewer's "Game" shading mode. Colours are linear floats.
    f, c = tm.floats, tm.colors
    # The scene light rig lights each CategoryLabel with its own dir_N / rim_N / GILightCommon.ColorN set
    # (Z-A: 4 = the character's own skin, hair, eyes; 5 = clothing and accessories; 6 = Pokémon; 0 = props).
    if "CategoryLabel" in tm.ints:
        mat.extras["category"] = int(tm.ints["CategoryLabel"])
    # SV/LA catchlights: `EnableHighlight` eyes without a HighlightMaskMap (~1430 SV materials, Pikachu included) carry
    # a highlight lobe of their own (RoughnessHighlight / MetallicHighlight), a glow EmissionIntensityLayer5 x
    # EmissionColorLayer5 and usually a per-eye PointLightIndex (Pikachu: left 1, right 2): the game reflects a
    # dedicated eye light off the clear coat (the rig's `EyeHighlightIntensity` / `EyeHighlightMaxLuminance` scale and
    # clamp it). The light's position is not in the files; the viewer draws it (viewer/src/scene/eyeHighlight.ts).
    if (tm.shader_flags.get("EnableHighlight") == "True" and "RoughnessHighlight" in f
            and not _tex_file(tm, "HighlightMaskMap") and float(f.get("EmissionIntensityLayer5", 0.0)) > 0):
        ec = c.get("EmissionColorLayer5", (1.0, 1.0, 1.0, 1.0))
        mat.extras["eyeHighlight"] = {
            "roughness": float(f["RoughnessHighlight"]),
            "metallic": float(f.get("MetallicHighlight", 0.0)),
            "intensity": float(f["EmissionIntensityLayer5"]),
            "color": [float(ec[0]), float(ec[1]), float(ec[2])],
            "light": int(tm.ints.get("PointLightIndex", 0)),
            "normalHeight": float(f.get("NormalHeight1", 1.0)),  # dome strength (Chansey 0.2: a flatter dome)
        }
        # The lobe forms on the clear-coat dome: NormalMap1 (`eye2_nrm`, convex), not NormalMap (`eye_nrm`, concave: the
        # iris' sunken look). On the near-flat eye disc the mesh normal alone lit the whole eye at once.
        n1 = _tex_file(tm, "NormalMap1")
        if n1 and tm.shader_flags.get("EnableNormalMap1", "True") != "False":
            img = load_tex(n1)
            if img is not None:
                mat.extras["highlight_normal"] = register(stem(n1), img, False, mat.extras.get("mirror", (False, False)))
    # SV/LA `EyeClearCoat`: a glossy dielectric coat over the (often black metal) iris, on the domed NormalMap1. Without it
    # a black eye (Chansey: every layer 0.013, iris metallic) rendered as a flat black hole: the coat's sheen and
    # reflections are what make it read as a wet eye in game. Written as KHR_materials_clearcoat.
    if tm.shader == "EyeClearCoat" and float(f.get("MetallicClearCoat", 0.0)) < 0.5:
        cc = {"factor": 1.0, "roughness": float(f.get("RoughnessClearCoat", 0.2))}
        n1 = _tex_file(tm, "NormalMap1")
        if n1 and tm.shader_flags.get("EnableNormalMap1", "True") != "False":
            img = load_tex(n1)
            if img is not None:
                cc["normal"] = register(stem(n1), img, False, mat.extras.get("mirror", (False, False)))
                cc["normalScale"] = float(f.get("NormalHeight1", 1.0))
        mat.extras["clearcoat"] = cc
    if not tm.shader.startswith("IkCharacter") and tm.shader not in ("NonDirectional", "Unlit", "FresnelEffect"):
        # Scarlet/Violet (SSS, EyeClearCoat, Standard, Hair...) is plain PBR: no IkCharacter toon params.
        return _finish_pbr_material(mat, tm, load_tex, register, stem, lym, lym_file)
    sc = c.get("ShadowingColor", (0.6, 0.6, 0.6, 0.0))
    mat.extras["shading"] = {
        "halfLambertBias": float(f.get("HalfLambertBias", 0.5)),
        "shadowStrength": float(f.get("ShadowStrength", 0.8)),
        "shadowColor": [float(sc[0]), float(sc[1]), float(sc[2])],
        "shadowShift": float(f.get("ShadowingShift", 0.0)),
        "rimIntensity": float(f.get("RimLightIntensity", 0.0)),
        "rimOffset": float(f.get("RimLightOffset", 0.0)),
        "rimContrast": float(f.get("RimLightContrast", 1.0)),
        "specularIntensity": float(f.get("SpecularIntensity", 0.04)),
        "occlusionStrength": float(f.get("OcclusionStrength", 1.0)),
        "normalHeight": float(f.get("NormalHeight", 1.0)),
        # SV `NonDirectional` (Gastly smoke, Blastoise jets) is lit by the scene with no direction: footage
        # shows the gas pale lavender by day and dusk-blue at night, i.e. albedo x ambient/key colour, never
        # the raw dark layer colour an emissive material would give. `Unlit` (fire) is truly emissive.
        "unlit": tm.shader == "Unlit",
        "nonDirectional": tm.shader == "NonDirectional",
        "specularOffset": float(f.get("SpecularOffset", 0.0)),
        "specularContrast": float(f.get("SpecularContrast", 1.0)),
        "backRimIntensity": float(f.get("BackRimLightIntensity", 0.0)),
        "hairSpecular": tm.shader_flags.get("EnableHairSpecular") == "True",
        # FresnelEffect shader (gems, Mega Stones): metallic-ish surface with a
        # white fresnel edge glow driven by EmissionIntensity.
        "fresnel": tm.shader == "FresnelEffect",
        "fresnelIntensity": float(f.get("EmissionIntensity", 0.0)) if tm.shader == "FresnelEffect" else 0.0,
        "roughness": float(f.get("Roughness", 0.8)),
        "metallic": float(f.get("Metallic", 0.0)),
        "giGain": float(f.get("ShadowingGIGain", 0.5)),
        # Hue-shift ramp terms (not drawn by the viewer yet; exported so a later shader needs no reconvert).
        "hueShift": {k: float(f[k]) for k in ("HueShiftBias", "HueShiftAreaValue", "MidAreaShift", "MidAreaContrast",
                                               "MidAreaHueOffset", "DarkAreaShift", "DarkAreaContrast", "DarkAreaHueOffset",
                                               "SaturationPower", "BaseColorDarkness") if k in f},
    }
    spec_mask = None
    # Specular / rim masks (single-channel). Shared white/black placeholders
    # are skipped; the viewer falls back to 1.0 (white) when absent.
    for slot, key in (("SpecularMaskMap", "specular_mask"), ("RimLightMaskMap", "rim_mask")):
        mf = _tex_file(tm, slot)
        if mf and not os.path.basename(mf).startswith("sh_") and "white_msk" not in mf:
            mt = load_tex(mf)
            # A flat white mask is a no-op, but a flat dark one is a real multiplier: Onix's spe_msk is all black,
            # so its layer-1 specular 1.0 is zeroed in game (matte rock).
            if mt is not None and not (_is_flat(mt) and int(mt[..., 0].min()) >= 250):
                mat.extras[key] = register(stem(mf), mt, False, mat.extras["mirror"])
                if key == "specular_mask":
                    spec_mask = mt
    # Per-layer specular: the game evaluates SpecularLayerN* through the layer mask like the colours
    # (Lebanne's skin is layer 1 = 0.0 while the base 0.74 only covers unpainted texels).
    if lym is not None:
        lscales = [tm.floats.get(f"LayerMaskScale{i}", 1.0) * _layer_strength(tm.colors.get(f"BaseColorLayer{i}")) for i in range(1, 5)]
        ls = bake_layer_specular(tm.floats, lym, spec_mask, lscales)
        if ls is not None:
            mask_rgba, inten, off, con = ls
            sh = mat.extras["shading"]
            sh["specularIntensity"], sh["specularOffset"], sh["specularContrast"] = inten, off, con
            if mask_rgba is None:
                mat.extras.pop("specular_mask", None)
            else:
                sh["specularMaskChannels"] = True  # mask G/B = per-texel offset/contrast
                sig = hashlib.sha1(repr((inten, off, con, lscales)).encode()).hexdigest()[:8]
                mat.extras["specular_mask"] = register(f"{stem(lym_file)}_{stem(_tex_file(tm, 'SpecularMaskMap') or '')}_{sig}_spec", mask_rgba, False, mat.extras["mirror"])
    return mat


def bake_layer_specular(floats: dict, mask: np.ndarray, spec_mask: np.ndarray | None, scales: list[float]):
    """Compose SpecularLayerN{Intensity,Offset,Contrast} over the base values with the layer mask.

    Layers lerp over what is below, in order, exactly like the colour layers. Returns None when no
    layer differs from the base (nothing to do), else (mask rgba or None, intensity, offset, contrast):
    the per-texel intensity is normalised into the specular mask R, offset is stored in G as
    (offset + 1) / 2 and contrast in B as contrast / 2 (the shader saturates contrast * 0.5). The
    returned offset/contrast are intensity-weighted means for consumers that only read R. One mean
    offset was not enough: the Courier outfit's base 0.4 offset widened the black trousers'
    highlight although their layer offset is 0.
    """
    base = (float(floats.get("SpecularIntensity", 0.04)), float(floats.get("SpecularOffset", 0.0)), float(floats.get("SpecularContrast", 1.0)))
    layers = []
    for i in range(1, 5):
        inten = floats.get(f"SpecularLayer{i}Intensity")
        layers.append(None if inten is None else (float(inten), float(floats.get(f"SpecularLayer{i}Offset", base[1])), float(floats.get(f"SpecularLayer{i}Contrast", base[2]))))
    if not any(l is not None and abs(l[0] - base[0]) > 0.02 for l in layers):
        return None
    h, w = mask.shape[:2]
    if spec_mask is not None:
        h, w = max(h, spec_mask.shape[0]), max(w, spec_mask.shape[1])
        if spec_mask.shape[:2] != (h, w):
            spec_mask = resize_rgba(spec_mask, w, h)
    if mask.shape[:2] != (h, w):
        mask = resize_rgba(mask, w, h)
    weights = np.clip(mask.astype(np.float32) / 255.0 * np.asarray(scales, np.float32)[None, None, :], 0.0, 1.0)
    eff = np.stack([np.full((h, w), v, np.float32) for v in base])  # intensity, offset, contrast
    for i, l in enumerate(layers):
        if l is None:
            continue
        wt = weights[..., i]
        for k in range(3):
            eff[k] += (l[k] - eff[k]) * wt
    inten = eff[0]
    if spec_mask is not None:
        inten = inten * (spec_mask[..., 0].astype(np.float32) / 255.0)
    peak = float(inten.max())
    if peak <= 1e-4:
        return None, 0.0, base[1], base[2]
    tot = float(eff[0].sum()) or 1.0
    off = float((eff[1] * eff[0]).sum() / tot)
    con = float((eff[2] * eff[0]).sum() / tot)
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., 0] = np.clip(inten / peak * 255.0 + 0.5, 0, 255)
    rgba[..., 1] = np.clip((eff[1] + 1.0) * 0.5 * 255.0 + 0.5, 0, 255)
    rgba[..., 2] = np.clip(eff[2] * 0.5 * 255.0 + 0.5, 0, 255)
    rgba[..., 3] = 255
    return rgba, peak, off, con


def _is_flat(rgba: np.ndarray) -> bool:
    return bool(rgba[..., :3].min() == rgba[..., :3].max())


# Default "unset" swatches the authoring tool leaves in unused layers (red,
# blue, green, cyan). Their alpha is 0, but alpha 0 is also used by real
# layers (Tarragon's black eyelashes / white brows), so detect the swatches.
# Gas shells: how far (in units of DisplacementHeight) the face band of the shell is pulled inside the body,
# fading to nothing by weight GAS_RETRACT_END. Measured on Gastly's rest pose (ray from the smoke centre
# through each shell vertex to the body surface): the shell sits 0.04-0.18 m outside the body over the face
# band (colour R 0.5/0.6, weight 0/0.2; the mouth bulge even pokes through) and 0.06-0.19 m at the ring and
# back (R >= 0.7), so no uniform sink opens the face without also exposing the sides. The face band needs
# >= 0.15 m to vanish, the R 0.7 ring must keep its sink <= 0.07 m to stay visible: 1.0 x 0.3 m at w = 0,
# 0.167 m at w = 0.2, 0.033 m at w = 0.4, 0 from w = 0.45. (A uniform 0.075 m buried the face; linear pulls of
# 0.5-0.7 and a quadratic 0.8 sank the ring too and left a hood of gas over an exposed lower body.)
# FITTED: gas-shell sink on Gastly pm0092 rest pose only (0 since gas draws under the body; comment above describes the old 1.0 pull), 2026-10-19 (engine-rules debt)
GAS_RETRACT = 0.0
GAS_RETRACT_END = 0.45
# Passes of neighbour averaging over the painted weights: six coarse levels on ~1,900 vertices give the face
# hole a stepped, jagged edge, while the game's is a clean circle.
# FITTED: edge smoothness on Gastly only, 2026-10-19 (engine-rules debt)
GAS_WEIGHT_SMOOTH = 3


def smooth_over_mesh(values: np.ndarray, indices: np.ndarray, passes: int) -> np.ndarray:
    """Average a per-vertex scalar with its triangle neighbours `passes` times (keeps the 0.5 floor intact)."""
    tri = indices.reshape(-1, 3)
    a = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2], tri[:, 1], tri[:, 2], tri[:, 0]])
    b = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0], tri[:, 0], tri[:, 1], tri[:, 2]])
    out = values.astype(np.float32)
    n = len(out)
    for _ in range(passes):
        acc = np.bincount(a, weights=out[b], minlength=n)
        cnt = np.bincount(a, minlength=n)
        out = np.where(cnt > 0, 0.5 * out + 0.5 * acc / np.maximum(cnt, 1), out).astype(np.float32)
    return out
# Drift of the cloud map in glTF UV units per second (u, v). The smoke sphere's UV is a front planar
# projection (+u = +x, Trinity +v = up at the front; V is flipped for glTF, so a positive V speed makes the
# pattern rise). Not in the data: the per-clip `.tracm` files carry no smoke tracks and the SV shader binary
# (`non_directional.bnsh`) is stripped Maxwell code, so this is estimated from 5 fps footage, where a lump
# crosses its own width (~0.2 UV) in 2-3 s.
# FITTED: cloud pan speed from 5 fps Gastly footage, 2026-10-19 (engine-rules debt)
SMOKE_DRIFT = (0.01, 0.07)


def gas_displacement(colour_r: np.ndarray | None, density: np.ndarray, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Gas-shell vertex offset along the normal for one frame, and the per-vertex weight shipped as `_GASWEIGHT`.

    The game displaces the sphere by DisplacementHeight x a cloud density it pans with time, weighted by the
    vertex colour R. The material is Opaque and the lumps render solid; the black body shows through a round
    hole at the front because the shell sits INSIDE the body there once it is sunk a little (the rest radius
    0.47 already touches the body's front at 0.48, and the eyes lie on the body surface), so the displacement
    is signed: weighted outward lumps minus a pull that only the face band feels:
        offset = H * (w * d - GAS_RETRACT * max(0, 1 - w / GAS_RETRACT_END)),  w = (R - 0.5) * 2
    Gastly paints six levels 0.5, 0.6 .. 1.0 (0.5 around the mouth and face, 1.0 over the top and back): w = 0
    sinks the shell 0.3 m with no lumps, w >= 0.45 is the plain outward cloud. The viewer adds
    H * w * (d(t) - d(0)) on top, which keeps this frame exact (the pull is constant in time). Colour ranges
    that never reach 0.5 (Koffing 0..0.8, Entei 0..0.15) are not this
    encoding and keep a plain weight-scaled outward push until checked against footage. The exported weight is
    kept >= 1e-3 because a disabled vertex attribute also reads 0 in the viewer shader (which then means "1").
    """
    n = len(density)
    if colour_r is None:
        w = np.ones(n, np.float32)
        offset = height * density
    elif float(colour_r.min()) >= 0.5 - 1e-3:
        w = np.clip((colour_r.astype(np.float32) - 0.5) * 2.0, 0.0, 1.0)
        offset = height * (w * density - GAS_RETRACT * np.clip(1.0 - w / GAS_RETRACT_END, 0.0, 1.0))
    else:
        w = np.clip(colour_r.astype(np.float32), 0.0, 1.0)
        offset = height * w * density
    return offset.astype(np.float32), np.ascontiguousarray(np.maximum(w, 1e-3), np.float32)

_PLACEHOLDER_LAYER_COLOURS = [(1.0, 0.0, 0.0), (0.014, 0.177, 0.981), (0.102, 1.0, 0.0), (0.023, 0.972, 0.875)]


def _layer_strength(c) -> float:
    if not c:
        return 0.0
    if len(c) > 3 and float(c[3]) > 0:
        return float(c[3])
    rgb = [float(x) for x in c[:3]]
    for ph in _PLACEHOLDER_LAYER_COLOURS:
        if max(abs(a - b) for a, b in zip(rgb, ph)) < 0.02:
            return 0.0
    return 1.0


def resize_rgba(rgba: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize each channel independently.

    Pillow resamples RGBA with premultiplied alpha, which zeroes the colour
    channels wherever alpha is 0. Layer masks use alpha as an ordinary data
    channel, so they must be resampled per channel.
    """
    out = np.empty((h, w, 4), np.uint8)
    for c in range(4):
        out[..., c] = np.asarray(Image.fromarray(rgba[..., c], "L").resize((w, h), Image.BILINEAR))
    return out


# FITTED: lens clearing on one face (df_face0000 glasses vs samples/nu_glasses.png); MIN/MAX/CLEAR_ALPHA, 2026-10-19 (engine-rules debt)
LENS_NEAR_OPAQUE_MIN = 0.85  # a Blend albedo whose lowest alpha is at least this has lenses authored ~0.9 (df_face0000 glasses)
LENS_NEAR_OPAQUE_MAX = 0.97  # texels below this are lens; frames are 1.0
LENS_CLEAR_ALPHA = 0.18      # in-game look (samples/nu_glasses.png): nearly clear, like the 0.1 eyewear atlas


def clear_near_opaque_lens(alb: np.ndarray | None, alpha_type: str | None) -> np.ndarray | None:
    """Glasses in the default Adult F face ship lens alpha 0.91-0.97 but render nearly clear in game.

    Lenses of every other Blend material sit at 0.1-0.5 (Mable 0.30), so only albedos whose minimum
    alpha is >= LENS_NEAR_OPAQUE_MIN are touched. Returns a copy; the decoded texture cache is shared.
    """
    if alb is None or not alpha_type or alpha_type.lower() != "blend":
        return alb
    a = alb[..., 3]
    if a.min() < LENS_NEAR_OPAQUE_MIN * 255:
        return alb
    lens = a < LENS_NEAR_OPAQUE_MAX * 255
    if not lens.any():
        return alb
    out = alb.copy()
    out[..., 3][lens] = round(LENS_CLEAR_ALPHA * 255)
    return out


def bake_layers(
    albedo: np.ndarray,
    mask: np.ndarray,
    layer_colors: list,
    scales: list[float],
    multiply: bool = True,
    base: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Blend up to four layer colours over the albedo using the RGBA mask.

    Mirrors the in-game shader: the mask channels partition a palette; where
    the mask is fully covered the albedo is tinted by the palette colour.
    Mask is resampled to the albedo size. Albedo and mask are treated as the
    same colour space (sRGB-encoded 8-bit), which is how the game stores them.
    """
    h = max(albedo.shape[0], mask.shape[0])
    w = max(albedo.shape[1], mask.shape[1])
    if albedo.shape[:2] != (h, w):
        albedo = resize_rgba(albedo, w, h)
    if mask.shape[:2] != (h, w):
        mask = resize_rgba(mask, w, h)
    # The shader works in linear light: the albedo is sRGB-encoded, the
    # layer colours are linear floats. Convert, blend, re-encode.
    a = srgb_to_linear(albedo[..., :3].astype(np.float32) / 255.0)
    m = mask.astype(np.float32) / 255.0
    weights = np.clip(m * np.asarray(scales, np.float32)[None, None, :], 0.0, 1.0)
    # Layers are applied in order, each covering what is below it (Pikachu's
    # cheek mask has layer 1 and layer 2 both fully on; the cheek is layer 2).
    # Unmasked texels keep the material's BaseColor tint; each layer lerps from
    # there to albedo x its own colour (it replaces the base tint, it does not
    # stack on it: Grisham's hands carry the skin tone in BaseColor AND layers
    # 1-2, and multiplying both made them redder than his face in-game).
    out = a * np.asarray(base, np.float32)[None, None, :] if multiply else a.copy()
    for i in range(4):
        colour = np.asarray(layer_colors[i], np.float32)[None, None, :]
        tinted = a * colour if multiply else np.broadcast_to(colour, a.shape)
        out = out + (tinted - out) * weights[..., i : i + 1]
    rgba = np.empty_like(albedo)
    rgba[..., :3] = np.clip(linear_to_srgb(out) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    rgba[..., 3] = albedo[..., 3]
    return rgba


# FITTED: colour-table shadow-cell repair on Schoolkids F bu_face0003 brows; also the 0.78 literals, 2026-10-19 (engine-rules debt)
TABLE_SHADOW_MAX = 0.95  # shadow cell luminance above this x the lit cell's = not a real shadow tone
TABLE_SHADOW_DARKEN = 0.78  # ... rescaled to this x the lit luminance (same factor as darken_saturate)


def table_shadow_cell(lit: np.ndarray, shadow: np.ndarray) -> np.ndarray:
    """Colour-table shadow cell (linear RGB) made no brighter than its lit cell.

    Modular NPC tables put vivid colours in the shadow row under near-black lit
    cells (Schoolkids F: green (80,138,90) under #252626 brows; ~11% of all cells),
    which cannot be absolute shadow colours: the brows are dark in game. Their hue
    is not trusted either (keeping it left the brows dark green), so such cells
    become TABLE_SHADOW_DARKEN x the lit colour (black lit -> black shadow).
    Normal, darker shadow cells pass unchanged.
    """
    w = np.asarray([0.2126, 0.7152, 0.0722], np.float32)
    ll, ls = float(lit @ w), float(shadow @ w)
    if ls <= ll * TABLE_SHADOW_MAX:
        return shadow
    return (lit * TABLE_SHADOW_DARKEN).astype(np.float32)


def table_shadow_rgb(lit: list[int], shadow: list[int]) -> list[int]:
    """table_shadow_cell on sRGB 0..255 triples (manifest/extras palette entries)."""
    lin = lambda c: srgb_to_linear(np.asarray(c[:3], np.float32) / 255.0)
    out = linear_to_srgb(table_shadow_cell(lin(lit), lin(shadow)))
    return [int(v) for v in np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)]


def apply_color_table(albedo: np.ndarray, mask: np.ndarray, table: np.ndarray, divide: int, indices: list[int] | None = None, masked_only: bool = False, scales: list[float] | None = None,
                      deeper: tuple[int, ...] | list[int] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Skin tone from a face ColorTableMap. Returns (lit, shadow) RGBA images.

    The table is `divide` zones wide; the top half holds each zone's lit tint
    and the bottom half its shadow tint. The layer mask's R,G,B,A channels are
    layers 1..4 and `BaseColorIndexN` (1-based) picks the zone layer N uses;
    layers are applied in order, each covering the previous. Every trainer
    face has BaseColorIndex1 = 1 (R covers ~87% of the face = skin) and A on
    the mouth/lips. Picking the zone from R's *value* (the old rule) chose the
    last zone for all skin: near-white on 2-zone faces (too pale) and black on
    Mable's 4-zone face. Multiplied into the albedo in linear light.

    `masked_only` (eyes): texels outside every layer stay untinted, so the sclera
    stays white and only the masked iris takes its zone (layer 1 = mask R too).

    `deeper` (1-based layers, `skin_zone_layers`): wrinkle strokes drawn one skin shade deeper than their zone t, i.e. zone t-1,
    and for t = 1 (the deepest) zone 1 x zone 1 / zone 2 (`deeper_skin`).
    """
    h, w = albedo.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = resize_rgba(mask, w, h)
    th, tw = table.shape[:2]
    d = max(1, divide)
    idx = list(indices or [1, 1, 1, 1]) + [1] * 4
    m = mask.astype(np.float32) / 255.0
    if scales is not None:
        m = m * np.asarray(list(scales) + [1.0] * 4, np.float32)[None, None, :4]
    a_lin = srgb_to_linear(albedo[..., :3].astype(np.float32) / 255.0)
    out = []
    def raw_cell(ty: int, zone_1based: int) -> np.ndarray:
        z = min(max(zone_1based, 1), d) - 1
        tx = min(((z * 2 + 1) * tw) // (2 * d), tw - 1)
        return srgb_to_linear(table[min(ty, th - 1), tx, :3].astype(np.float32) / 255.0)

    for ty in (th // 4, (3 * th) // 4):
        def cell(zone_1based: int) -> np.ndarray:
            if ty == th // 4:
                return raw_cell(ty, zone_1based)
            return table_shadow_cell(raw_cell(th // 4, zone_1based), raw_cell(ty, zone_1based))
        # Unmasked texels take layer 1's zone (the base skin), so there are no untinted seams.
        if masked_only:
            tint = np.ones_like(a_lin)
            first = 0
        else:
            tint = np.broadcast_to(cell(idx[0]), a_lin.shape).copy()
            first = 1
        for c in range(first, 4):
            col = deeper_skin(cell, idx[c]) if (c + 1) in deeper else cell(idx[c])
            tint += (col[None, None, :] - tint) * m[..., c : c + 1]
        img = np.empty_like(albedo)
        img[..., :3] = np.clip(linear_to_srgb(a_lin * tint) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        img[..., 3] = albedo[..., 3]
        out.append(img)
    return out[0], out[1]


# FITTED(za): wrinkle strokes drawn one skin zone deeper than the selected tone (the colour-table blend makes a same-zone
# layer invisible; in-game Rust Grunt M furrow is darker than his skin) on tr0030/tr0031, Jub F, Ga M B, 2026-10-23
def deeper_skin(cell, zone: int) -> np.ndarray:
    """Linear colour one skin shade deeper than `zone` (zones 1..SKIN_TONES run deep..pale): zone - 1, and below zone 1 the
    same step again (zone 1 x zone 1 / zone 2). Wrinkle strokes authored in the skin's own zone (Rust Grunt M/F brow furrow,
    Jub F, Ga M B) show darker than the skin in game (`samples/za_rust_syndicate_grunt_vs.png`); with the layer covering the
    skin they would otherwise be invisible. User decision 2026-10-23: one shade deeper than the selected tone, at every tone."""
    z = max(zone - 1, 1)
    return cell(z) * cell(z) / np.maximum(cell(z + 1), 1e-4) if zone <= 1 else cell(z)


def tint_linear(rgba: np.ndarray, tint: tuple[float, float, float]) -> np.ndarray:
    """Multiply an sRGB-encoded image by a linear-light colour factor (what the glTF factor does)."""
    lin = srgb_to_linear(rgba[..., :3].astype(np.float32) / 255.0) * np.asarray(tint, np.float32)[None, None, :]
    out = np.empty_like(rgba)
    out[..., :3] = np.clip(linear_to_srgb(lin) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = rgba[..., 3]
    return out


_EYE_COLORS: list[dict] | None = None
_BROW_COLORS: list[dict] | None = None


def paint_eye_regions(alb: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None:
    """Split a painted eye (no layer mask: Nu F, Jub F, Pclady F) into pupil / iris / inner iris regions.

    The eye shape = everything the flood fill from the corner does not reach; its outline is eroded away, the pupil is the
    dark interior blob and the inner iris the clearly lighter part of the rest. Returns (neutral albedo, mask, own colours):
    inside each region the albedo becomes its own luminance (normalised per region) so `albedo x zone colour` reproduces the
    shading, the mask is a layer mask (R pupil, G iris, B inner iris; alpha 0) in the sequential-lerp form `bakePalette`
    expects, and `own` are linear zone colours that give the painted look back."""
    from PIL import ImageDraw, ImageFilter

    h, w = alb.shape[:2]
    flood = Image.fromarray(np.ascontiguousarray(alb[..., :3]))
    ImageDraw.floodfill(flood, (0, 0), (255, 0, 255), thresh=45)  # thresh sums the channel differences
    shape = (np.asarray(flood) == (255, 0, 255)).all(axis=2) == False  # noqa: E712
    if not 0.05 < shape.mean() < 0.8:
        return None
    interior = np.asarray(Image.fromarray(shape.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(7))) > 127
    lin = srgb_to_linear(alb[..., :3].astype(np.float32) / 255.0)
    y = lin @ np.asarray([0.2126, 0.7152, 0.0722], np.float32)
    if interior.sum() < 40:
        return None
    pupil = interior & (y < 0.45 * np.median(y[interior]))
    iris = interior & ~pupil
    if iris.sum() < 20:
        return None
    patch = iris & (y > 1.5 * np.median(y[iris]))
    iris &= ~patch
    cores = [pupil, iris, patch]
    soft = [np.asarray(Image.fromarray(c.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255.0 for c in cores]
    tot = np.maximum(sum(soft), 1.0)
    wr, wg, wb = (x / tot for x in soft)
    cover = wr + wg + wb
    grey = np.zeros_like(y)
    own: list[np.ndarray] = []
    for core, wt in zip(cores, (wr, wg, wb)):
        if core.sum() < 5:
            own.append(own[-1] if own else np.ones(3, np.float32))
            continue
        g = np.clip(y / max(float(np.percentile(y[core], 95)), 1e-4), 0.0, 1.0)
        grey += wt * g
        own.append(np.clip(lin[core].mean(axis=0) / max(float(g[core].mean()), 1e-4), 0.0, 1.0))
    neutral = lin * (1.0 - cover)[..., None] + (grey)[..., None]
    out = np.empty_like(alb)
    out[..., :3] = np.clip(linear_to_srgb(neutral) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = alb[..., 3]
    # Sequential lerp (layer 1 first) that equals sum(w_i z_i) + (1 - sum(w)) over the partition.
    eps = 1e-6
    mb = wb
    mg = np.clip(wg / np.maximum(1.0 - wb, eps), 0, 1)
    mr = np.clip(wr / np.maximum(1.0 - wb - wg, eps), 0, 1)
    mask = np.zeros_like(alb)
    mask[..., 0], mask[..., 1], mask[..., 2] = (np.clip(m * 255.0 + 0.5, 0, 255).astype(np.uint8) for m in (mr, mg, mb))
    return out, mask, own


def _default_iris_colors(label: str = "brown") -> tuple[tuple, tuple]:
    """Linear (layer 2, layer 3) colours of the player's eye contact whose label contains `label`."""
    global _EYE_COLORS
    if _EYE_COLORS is None:
        with open(os.path.join(os.path.dirname(__file__), "eye_colors.json"), encoding="utf-8") as f:
            _EYE_COLORS = json.load(f)
    entry = next((e for e in _EYE_COLORS if label in e["label"]), _EYE_COLORS[0])
    return tuple(tuple(srgb_to_linear(np.asarray(c, np.float32) / 255.0).tolist()) for c in entry["lit"][:2])


def _palette_planes(alb: np.ndarray, lym: np.ndarray, alb_key: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Albedo + layer mask at a common size for a runtime palette, the larger of the two like `bake_layers`. Shrinking
    the mask to a placeholder albedo (the Common faces' brows: 32x32 white vs a 32x64 brow/lash mask) blurred the
    brow/lash boundary, so a recolour turned the left lash grey. Returns (albedo, mask, albedo texture key)."""
    h, w = max(alb.shape[0], lym.shape[0]), max(alb.shape[1], lym.shape[1])
    if alb.shape[:2] != (h, w):
        alb = resize_rgba(alb, w, h)
        alb_key = f"{alb_key}{w}x{h}"
    return alb, (lym if lym.shape[:2] == (h, w) else resize_rgba(lym, w, h)), alb_key


def _single_layer_eye_palette(mat: Material, tm: T.TrMaterial, own: list, mask: np.ndarray, alb: np.ndarray, alb_file: str,
                              lym_file: str, scl: list[float], register, alb_key: str | None = None) -> None:
    """Adult M Common A-F (`dm_face1000..1005`): the mask's R channel is the whole iris (the pupil is darker in the albedo)
    and layer 2 (white, a tiny G highlight) is not an iris tint. One swatch row recolours layer 1 with each player contact's
    iris colour. Zones: 1 = layer 2 as authored (not offered), 2 = the authored iris, then one zone per contact."""
    s8 = lambda c: [int(v) for v in np.clip(linear_to_srgb(np.asarray(c[:3], np.float32)) * 255.0 + 0.5, 0, 255).astype(np.uint8)]
    shd_of = lambda i: tm.colors.get(f"ShadowingColorLayer{i + 1}", own[i])
    lit = [s8(own[1]), s8(own[0])]
    shadow = [table_shadow_rgb(lit[1 - j], s8(shd_of(1 - j))) for j in range(2)]
    for e in _EYE_COLORS:
        lit.append(e["lit"][0])
        shadow.append(e["shadow"][0])
    stem = lambda f: os.path.splitext(os.path.basename(f))[0]
    h, w = alb.shape[:2]
    mir = mat.extras["mirror"]
    mask_rgb = mask.copy()
    mask_rgb[..., 3] = 255
    mask_alpha = np.repeat(mask[..., 3:4], 4, axis=2).copy()
    mask_alpha[..., 3] = 255
    key = stem(lym_file)
    mat.extras["palette"] = {
        "albedo": register(alb_key or f"{stem(alb_file)}_raw", alb, True, mir),
        "mask": register(f"{key}_mask{w}x{h}", mask_rgb, False, mir),
        "maskA": register(f"{key}_maskA{w}x{h}", mask_alpha, False, mir),
        "divide": len(lit),
        "indices": [2, 1, 1, 1],
        "scales": [scl[0], scl[1] if scl[1] > 0 else 0.0, 0.0, 0.0],
        "maskedOnly": True,
        "layers": [1],
        "lit": lit,
        "shadow": shadow,
        "custom": True,
        "zoneLabels": ["Original base", "Original"] + [e["label"] for e in _EYE_COLORS],
    }


def custom_eye_palette(mat: Material, tm: T.TrMaterial, lym: np.ndarray | None, alb: np.ndarray, alb_file: str, lym_file: str,
                       layer_colors: list, layer_enable: list[float], register) -> None:
    """Runtime-recolourable iris for modular eyes without a ColorTableMap (Common faces `<b>_face1000..1009`, painted eyes).

    Common eyes have fixed layer colours on the material but the game recolours them per NPC (data not in the rip), so the
    viewer gets the player's eye contacts (tools/gen_eye_colors.py) as `extras.palette` with `custom`
    set: zone 1 = the pupil (layer 1) as authored, zones 2-3 = the material's own iris pair, then one (layer 2, layer 3)
    pair per shipped colour (pair j = zones 2j+2, 2j+3; pair 0 is the original). `pairs` makes the viewer show a single
    swatch row that sets layers 2 and 3 together. Eyes whose layer mask is off (Nu F: the iris is painted into the albedo)
    get a layer mask and neutral albedo derived from the painting (`paint_eye_regions`)."""
    global _EYE_COLORS
    if _EYE_COLORS is None:
        with open(os.path.join(os.path.dirname(__file__), "eye_colors.json"), encoding="utf-8") as f:
            _EYE_COLORS = json.load(f)
    s8 = lambda c: [int(v) for v in np.clip(linear_to_srgb(np.asarray(c[:3], np.float32)) * 255.0 + 0.5, 0, 255).astype(np.uint8)]
    scl = [tm.floats.get(f"LayerMaskScale{i}", 1.0) * (1.0 if i <= 3 and layer_enable[i - 1] > 0 else 0.0) for i in range(1, 5)]
    h, w = alb.shape[:2]
    stem = lambda f: os.path.splitext(os.path.basename(f))[0]
    mir = mat.extras["mirror"]
    if scl[1] <= 0:
        painted = paint_eye_regions(alb) if not any(float(tm.floats.get(f"LayerMaskScale{i}", 1.0)) > 0 for i in (1, 2, 3)) else None
        if painted is None:
            return
        alb, mask, own = painted
        lit = [s8(c) for c in own]
        shadow = [list(c) for c in lit]
        key = f"{stem(alb_file)}_eyepaint"
        scl = [1.0, 1.0, 1.0, 0.0]
        albedo_key = register(f"{key}_neutral", alb, True, mir)
    else:
        own = [layer_colors[i] for i in range(3)]
        # Layer 2 must be an iris tint: a white layer 2 (Common A's) is a different mask layout, left alone.
        if own[0] is not None and own[1] is not None and min(own[1]) >= 0.8 and scl[0] > 0 and scl[2] <= 0:
            alb_p, mask, alb_key = _palette_planes(alb, lym, f"{stem(alb_file)}_raw")
            _single_layer_eye_palette(mat, tm, own, mask, alb_p, alb_file, lym_file, scl, register, alb_key)
            return
        if own[0] is None or own[1] is None or own[2] is None or min(own[1]) >= 0.8:
            return
        shd_of = lambda i: tm.colors.get(f"ShadowingColorLayer{i + 1}", own[i])
        lit = [s8(own[0]), s8(own[1]), s8(own[2])]
        shadow = [table_shadow_rgb(lit[i], s8(shd_of(i))) for i in range(3)]
        alb, mask, alb_key = _palette_planes(alb, lym, f"{stem(alb_file)}_raw")
        h, w = alb.shape[:2]
        key = stem(lym_file)
        albedo_key = register(alb_key, alb, True, mir)
    # Layer 3 is the lighter inner iris on most faces; where it is darker (a pupil-like ring) it keeps its colour.
    inner_is_iris = sum(lit[2]) >= 0.9 * sum(lit[1])
    for e in _EYE_COLORS:
        lit += e["lit"] if inner_is_iris else [e["lit"][0], lit[2]]
        shadow += e["shadow"] if inner_is_iris else [e["shadow"][0], shadow[2]]
    mask_rgb = mask.copy()
    mask_rgb[..., 3] = 255
    mask_alpha = np.repeat(mask[..., 3:4], 4, axis=2).copy()
    mask_alpha[..., 3] = 255
    mat.extras["palette"] = {
        "albedo": albedo_key,
        "mask": register(f"{key}_mask{w}x{h}", mask_rgb, False, mir),
        "maskA": register(f"{key}_maskA{w}x{h}", mask_alpha, False, mir),
        "divide": len(lit),
        "indices": [1, 2, 3, 1],
        "scales": scl,
        "maskedOnly": True,
        "layers": [2],
        "lit": lit,
        "shadow": shadow,
        "custom": True,
        "pairs": True,
        "pairLabels": ["Original"] + [e["label"] for e in _EYE_COLORS],
    }


def custom_brow_palette(mat: Material, tm: T.TrMaterial, lym: np.ndarray, alb: np.ndarray, alb_file: str, lym_file: str,
                        layer_colors: list, layer_enable: list[float], register) -> None:
    """Runtime-recolourable eyebrows for Common faces (no ColorTableMap, fixed layer colours; the game recolours them per
    NPC, data not in the rip). Mask layer 1 is the dark base and layer 2 the brow colour, so the palette is zone 1 = the
    authored base, zone 2 = the authored brow colour, then one layer 2 colour per entry of `eyebrow_colors.json` (the player's 32 brow
    colours, tools/gen_eye_colors.py). Flagged `custom` (no `pairs`): one swatch row for layer 2."""
    global _BROW_COLORS
    if _BROW_COLORS is None:
        with open(os.path.join(os.path.dirname(__file__), "eyebrow_colors.json"), encoding="utf-8") as f:
            _BROW_COLORS = json.load(f)
    own = layer_colors[:2]
    scl = [tm.floats.get(f"LayerMaskScale{i}", 1.0) * (1.0 if i <= 2 and layer_enable[i - 1] > 0 else 0.0) for i in range(1, 5)]
    if own[0] is None or own[1] is None or scl[1] <= 0:
        return
    s8 = lambda c: [int(v) for v in np.clip(linear_to_srgb(np.asarray(c[:3], np.float32)) * 255.0 + 0.5, 0, 255).astype(np.uint8)]
    lit = [s8(c) for c in own]
    shadow = [table_shadow_rgb(lit[i], s8(tm.colors.get(f"ShadowingColorLayer{i + 1}", own[i]))) for i in range(2)]
    for e in _BROW_COLORS:
        lit += e["lit"]
        shadow += e["shadow"]
    stem = lambda f: os.path.splitext(os.path.basename(f))[0]
    alb, mask, alb_key = _palette_planes(alb, lym, f"{stem(alb_file)}_raw")
    h, w = alb.shape[:2]
    mask_rgb = mask.copy()
    mask_rgb[..., 3] = 255
    mask_alpha = np.repeat(mask[..., 3:4], 4, axis=2).copy()
    mask_alpha[..., 3] = 255
    mir = mat.extras["mirror"]
    mat.extras["palette"] = {
        "albedo": register(alb_key, alb, True, mir),
        "mask": register(f"{stem(lym_file)}_mask{w}x{h}", mask_rgb, False, mir),
        "maskA": register(f"{stem(lym_file)}_maskA{w}x{h}", mask_alpha, False, mir),
        "divide": len(lit),
        "indices": [1, 2, 1, 1],
        "scales": scl,
        "maskedOnly": True,
        "layers": [2],
        "lit": lit,
        "shadow": shadow,
        "custom": True,
        "zoneLabels": ["Original base", "Original"] + [e["label"] for e in _BROW_COLORS],
    }


def custom_hair_palette(mat: Material, alb: np.ndarray, alb_file: str, register, default_zone: int = 2) -> np.ndarray | None:
    """Runtime-recolourable hair for painted hair (Pclady F `df_hrs0004_00`: layer mask off, the colour is in the albedo).
    The albedo is flattened to luminance (its 85th percentile = 1) and tinted by one mask layer covering everything, with
    the player's hair colours (`eyebrow_colors.json` = `p0_hairstyle_pallete01`, the table eyebrows, lashes and hair share).
    Zone 2 = the painting's mean colour (an approximation of the original: the paint's own shading is flattened)."""
    global _BROW_COLORS
    if _BROW_COLORS is None:
        with open(os.path.join(os.path.dirname(__file__), "eyebrow_colors.json"), encoding="utf-8") as f:
            _BROW_COLORS = json.load(f)
    opaque = alb[..., 3] > 128
    if int(opaque.sum()) < 64:
        return None
    rgb = alb[..., :3].astype(np.float32)
    lin = srgb_to_linear(rgb / 255.0)
    lum = (lin * np.asarray([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=2)
    top = max(float(np.percentile(lum[opaque], 85)), 1e-4)
    neutral = alb.copy()
    neutral[..., :3] = np.clip(linear_to_srgb(np.repeat((np.clip(lum / top, 0, 1))[..., None], 3, axis=2)) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    mean = [int(v) for v in np.clip(linear_to_srgb(lin[opaque].mean(axis=0) / np.maximum(float(np.clip(lum[opaque] / top, 0, 1).mean()), 1e-4)) * 255.0 + 0.5, 0, 255)]
    lit, shadow = [mean, mean], [[round(v * 0.78) for v in mean]] * 2
    for e in _BROW_COLORS:
        lit += e["lit"]
        shadow += e["shadow"]
    h, w = alb.shape[:2]
    mask = np.zeros((h, w, 4), np.uint8)
    mask[..., 0] = 255
    mask[..., 3] = 255
    mask_alpha = np.full((h, w, 4), 255, np.uint8)
    stem = os.path.splitext(os.path.basename(alb_file))[0]
    mir = mat.extras["mirror"]
    mat.extras["palette"] = {
        "albedo": register(f"{stem}_hairneutral", neutral, True, mir),
        "mask": register(f"hair_full_mask{w}x{h}", mask, False, mir),
        "maskA": register(f"hair_full_maskA{w}x{h}", mask_alpha, False, mir),
        "divide": len(lit),
        "indices": [default_zone, 1, 1, 1],
        "scales": [1.0, 0.0, 0.0, 0.0],
        "maskedOnly": True,
        "layers": [1],
        "lit": lit,
        "shadow": shadow,
        "custom": True,
        "zoneLabels": ["Own", "Original (flat)"] + [e["label"] for e in _BROW_COLORS],
    }
    return neutral


# FITTED: stand-in beard colour (per-NPC runtime data not in the rip), 2026-10-19 (engine-rules debt)
BEARD_DEFAULT_ZONE = 6  # "Player 04 - dark brown" of the hair table: the stand-in colour for beards the game tints at runtime
BEARD_MESH = ("beard", "face_hair")


def _beard_materials(source: Source, trmdl) -> set[str]:
    """Materials used by the facial-hair meshes (`beard_mesh`, `face_hair_mesh`) of a modular face part."""
    out: set[str] = set()
    for mf in trmdl.meshes[:1]:
        for sh in T.parse_trmsh(source.read(mf)).shapes:
            if any(k in sh.name for k in BEARD_MESH):
                out.update(p.material for p in sh.parts)
    return out


def lip_layer(lym: np.ndarray, scl: list[float], idxs: list[int]) -> int:
    """1-based mask layer (2..4) that paints lipstick on a face-skin mask, else 0: a small compact blob centred on the mouth
    (x ~0.5, y 0.62-0.85 of the texture) in a zone other than skin. Most faces paint their lips into the albedo and have none."""
    for c in (1, 2, 3):
        if scl[c] <= 0 or idxs[c] == idxs[0]:
            continue
        m = lym[..., c] > 128
        cover = float(m.mean())
        if not 0.0005 < cover < LIP_MAX_COVER:
            continue
        ys, xs = np.nonzero(m)
        h, w = m.shape
        if 0.4 < xs.mean() / w < 0.6 and 0.62 < ys.mean() / h < 0.85 and (xs.max() - xs.min()) / w < 0.25:
            return c + 1
    return 0


def skin_zone_layers(lym: np.ndarray, scl: list[float], idxs: list[int]) -> list[int]:
    """1-based mask layers (2..4) of a skin material that point at one of the skin zones (1..SKIN_TONES) of the shared table.
    These are wrinkle strokes authored in the skin's own zone (Rust Grunt M/F brow furrow, Jub F nose bridge and mouth
    corners, Ga M B). They follow the selected tone like layer 1, drawn one shade deeper (`deeper_skin`): the game shows the
    Grunt's furrow darker than his skin; left at the authored zone they became pale lines at deeper tones. No coverage
    floor: Grunt F's strokes cover 0.03% of the texture, and a layer that paints nothing looks the same whatever zone it follows.
    The lip and beard layers keep their own zones (their palette rows offer the choice)."""
    own = {lip_layer(lym, scl, idxs), beard_paint_layer(lym, scl, idxs)}
    return [c + 1 for c in range(1, 4)
            if (c + 1) not in own and scl[c] > 0 and 1 <= idxs[c] <= SKIN_TONES and bool((lym[..., c] > 0).any())]


_LIP_COLORS: list[dict] | None = None


def beard_paint_layer(lym: np.ndarray, scl: list[float], idxs: list[int]) -> int:
    """1-based mask layer that paints a beard on a face-skin mask, else 0: layer 2, a big mass (> 5% of the texture, > 60% of
    its width) in a zone other than skin. The other layer-2/3 paint is makeup, earrings or neck jewellery (checked on every face)."""
    if scl[1] <= 0 or idxs[1] == idxs[0]:
        return 0
    m = lym[..., 1] > 128
    if float(m.mean()) < 0.05:
        return 0
    _, xs = np.nonzero(m)
    return 2 if (xs.max() - xs.min()) / m.shape[1] > 0.6 else 0


def _hair_colors() -> list[dict]:
    """The player's 32 hair colours (`eyebrow_colors.json`, zones of p0_hairstyle_pallete01)."""
    global _BROW_COLORS
    if _BROW_COLORS is None:
        with open(os.path.join(os.path.dirname(__file__), "eyebrow_colors.json"), encoding="utf-8") as f:
            _BROW_COLORS = json.load(f)
    return _BROW_COLORS


def layer_palette(mat: Material, raw_alb: np.ndarray, mask_r: np.ndarray, ct: np.ndarray, div: int, idxs: list[int], scl: list[float],
                  layers: list[int], alb_file: str, lym_file: str, register, skin_link: bool = False, extra: list[dict] | None = None,
                  skin_layers: list[int] | None = None) -> None:
    """`extras.palette` for a colour-table material: raw albedo + layer mask (the viewer re-bakes it per zone choice) and the table's lit/shadow zones.
    `layers` = the 1-based mask layers offered as swatch rows; `skin_link` = layer 1 is the skin tone (driven by the skin variant group);
    `skin_layers` = further layers that point at a skin zone and follow the tone too (`skin_zone_layers`).
    `extra` = colours appended after the table's own zones (`own` = how many zones the table has; labels in `zoneLabels`)."""
    stem = lambda f: os.path.splitext(os.path.basename(f))[0]
    th, tw = ct.shape[:2]

    def zone_rgb(row: int, z: int) -> list[int]:
        tx = min(((z * 2 + 1) * tw) // (2 * max(1, div)), tw - 1)
        return [int(v) for v in ct[min(row, th - 1), tx, :3]]

    h, w = raw_alb.shape[:2]
    mask_rgb = mask_r.copy()
    mask_rgb[..., 3] = 255
    mask_alpha = np.repeat(mask_r[..., 3:4], 4, axis=2).copy()
    mask_alpha[..., 3] = 255
    lit = [zone_rgb(th // 4, z) for z in range(div)]
    shadow = [table_shadow_rgb(zone_rgb(th // 4, z), zone_rgb((3 * th) // 4, z)) for z in range(div)]
    labels = None
    if extra:
        labels = [f"Palette {z + 1}" for z in range(div)] + [e["label"] for e in extra]
        lit += [e["lit"][0] for e in extra]
        shadow += [e["shadow"][0] for e in extra]
    mat.extras["palette"] = {
        "albedo": register(f"{stem(alb_file)}_raw", raw_alb, True, mat.extras["mirror"]),
        # RGB and alpha travel as two opaque images: browsers premultiply alpha on canvas reads and would zero RGB where A=0.
        "mask": register(f"{stem(lym_file)}_mask{w}x{h}", mask_rgb, False, mat.extras["mirror"]),
        "maskA": register(f"{stem(lym_file)}_maskA{w}x{h}", mask_alpha, False, mat.extras["mirror"]),
        "divide": len(lit),
        "indices": list(idxs),
        "scales": [float(s) for s in scl],
        "maskedOnly": True,
        "layers": layers,
        "lit": lit,
        "shadow": shadow,
        **({"skinLink": True} if skin_link else {}),
        **({"skinLayers": list(skin_layers)} if skin_link and skin_layers else {}),
        **({"own": div, "zoneLabels": labels} if labels else {}),
    }


def custom_lip_palette(mat: Material, raw_alb: np.ndarray, lym: np.ndarray, ct: np.ndarray, div: int, idxs: list[int], scl: list[float],
                       layer: int, alb_file: str, lym_file: str, register, colors: list[dict] | None = None, none_label: str = "No lipstick",
                       skin_layers: list[int] | None = None) -> None:
    """Runtime lip colour for a face whose skin mask paints lipstick on `layer` (the player's lip colours,
    `lip_colors.json` = zones 5-13 of p0_face_pallete01). The palette bakes the whole face (skin layer 1 + the other layers
    with their authored zones), so the viewer drives zone `indices[0]` from the selected skin tone (`skinLink`).
    Offered swatches (`offer`): zone 0 = "No lipstick" (the lip layer in the skin zone: the albedo's own painted lips stay),
    the authored zone, then one zone per player colour appended after the table's own zones."""
    global _LIP_COLORS
    if colors is None:
        if _LIP_COLORS is None:
            with open(os.path.join(os.path.dirname(__file__), "lip_colors.json"), encoding="utf-8") as f:
                _LIP_COLORS = json.load(f)
        colors = _LIP_COLORS
    th, tw = ct.shape[:2]
    cell = lambda row, z: [int(v) for v in ct[min(row, th - 1), min(((z * 2 + 1) * tw) // (2 * max(1, div)), tw - 1), :3]]
    lit = [cell(th // 4, z) for z in range(div)]
    shadow = [table_shadow_rgb(lit[z], cell((3 * th) // 4, z)) for z in range(div)]
    own = idxs[layer - 1]
    for e in colors:
        lit += e["lit"]
        shadow += e["shadow"]
    stem = lambda f: os.path.splitext(os.path.basename(f))[0]
    h, w = raw_alb.shape[:2]
    mask = lym if lym.shape[:2] == (h, w) else resize_rgba(lym, w, h)
    mask_rgb = mask.copy()
    mask_rgb[..., 3] = 255
    mask_alpha = np.repeat(mask[..., 3:4], 4, axis=2).copy()
    mask_alpha[..., 3] = 255
    mir = mat.extras["mirror"]
    mat.extras["palette"] = {
        "albedo": register(f"{stem(alb_file)}_raw", raw_alb, True, mir),
        "mask": register(f"{stem(lym_file)}_mask{w}x{h}", mask_rgb, False, mir),
        "maskA": register(f"{stem(lym_file)}_maskA{w}x{h}", mask_alpha, False, mir),
        "divide": len(lit),
        "indices": list(idxs),
        "scales": [float(x) for x in scl],
        "maskedOnly": True,
        "layers": [layer],
        "lit": lit,
        "shadow": shadow,
        "custom": True,
        "skinLink": True,
        **({"skinLayers": list(skin_layers)} if skin_layers else {}),
        "offer": [0, own] + list(range(div + 1, div + 1 + len(colors))),
        "noneLabel": none_label,
        "zoneLabels": [f"Original (palette {z})" if z == own else f"Palette {z}" for z in range(1, div + 1)] + [e["label"] for e in colors],
    }


def darken_saturate(rgba: np.ndarray, value: float, saturation: float) -> np.ndarray:
    """Scale brightness by `value` and saturation by `saturation` in linear light."""
    lin = srgb_to_linear(rgba[..., :3].astype(np.float32) / 255.0) * value
    luma = (lin * np.asarray([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=2, keepdims=True)
    lin = luma + (lin - luma) * saturation
    out = np.empty_like(rgba)
    out[..., :3] = np.clip(linear_to_srgb(lin) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = rgba[..., 3]
    return out


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055).astype(np.float32)


def _strict(ch: T.Channel) -> tuple[np.ndarray, np.ndarray]:
    """Return (frames, values) with strictly increasing frames (first wins)."""
    f, v = ch.frames, ch.values
    if len(f) <= 1:
        return f, v
    keep = np.ones(len(f), bool)
    keep[1:] = f[1:] > np.maximum.accumulate(f)[:-1]
    return f[keep], v[keep]


def decimate(frames: np.ndarray, values: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Drop keys that linear interpolation of their neighbours reproduces within `tol`.

    Greedy single pass: a key is kept if interpolating between the last kept
    key and the next key misses it by more than tol. First/last always kept.
    """
    n = len(frames)
    if n <= 2 or tol <= 0:
        return frames, values
    keep = [0]
    last = 0
    for i in range(1, n - 1):
        f0, f2 = frames[last], frames[i + 1]
        t = (frames[i] - f0) / max(f2 - f0, 1e-9)
        interp = values[last] + (values[i + 1] - values[last]) * t
        if np.abs(interp - values[i]).max() > tol:
            keep.append(i)
            last = i
    keep.append(n - 1)
    k = np.asarray(keep)
    return frames[k], values[k]


# FITTED: per-species emission rule, Inkay/Malamar only; opposite reading of intensity-0 emission vs MODE_SWITCH, 2026-10-19 (engine-rules debt)
SPOT_GLOW_SPECIES = ("pm0726_", "pm0727_")


def fresnel_facing_color(tm: T.TrMaterial, base: tuple) -> tuple:
    """Diffuse colour of an unmasked `FresnelEffect` material as seen face-on.

    The shader blends `BaseColorLayer1` (the fresnel layer, `BaseColorMap1` = a white fresnel texture) over
    `BaseColor` with a weight going from `FresnelAlphaMin` (facing) to `FresnelAlphaMax` (grazing). Mega Tatsugiri's
    roe (`body_f`) is BaseColor black + layer 1 orange at Min 1 / Max 0: reading BaseColor alone rendered black
    balls; the game shows orange roe with dark edges. Only applied when BaseColor is black (the base then adds
    nothing): applying it to every FresnelEffect material shifts 37 gem/crystal materials (Meloetta, Staryu, Mega
    Rayquaza...) by unverified amounts. Materials with a LayerMaskMap bake their layers instead."""
    layer = tm.colors.get("BaseColorLayer1")
    if (
        tm.shader != "FresnelEffect"
        or layer is None
        or max(float(x) for x in base[:3]) > 0.02
        or tm.shader_flags.get("EnableFresnelTexture") != "True"
        or _tex_file(tm, "LayerMaskMap")
    ):
        return base
    w = min(1.0, max(0.0, float(tm.floats.get("FresnelAlphaMin", 0.0))))
    return tuple(float(base[k]) + (float(layer[k]) - float(base[k])) * w for k in range(3)) + (float(base[3]) if len(base) > 3 else 1.0,)


def _held(frames: np.ndarray, values: np.ndarray, at: np.ndarray, default: float) -> np.ndarray:
    """Value of the last key at or before each frame in `at` (keys are held, not interpolated)."""
    if not len(frames):
        return np.full(len(at), default, np.float32)
    order = np.argsort(frames, kind="stable")
    f, v = frames[order], values[order]
    i = np.clip(np.searchsorted(f, at, side="right") - 1, 0, len(f) - 1)
    return v[i].astype(np.float32)


def material_uv_offsets(tracks: list[T.MaterialParamTrack], materials: dict, fps: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """`.tracm` UVScaleOffset tracks -> per-material texture offsets relative to the baked rest UVs.

    The importer bakes the material's static UVScaleOffset into the vertex UVs, so a clip only needs the change of
    the offset (glTF space: +x right, +y down the image; Falinks' blink steps the V offset 0 -> 0.25 -> 0.5 down the
    rows open / half / closed), scaled by the material's UV scale (the game offsets before scaling) and halved on
    mirror-doubled axes. Tracks that change the scale
    are skipped (not seen on eyes so far). Shiny copies (`name@rare`) get the same track.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    # A material follows the track of `uv_track_src` (default: itself) / `uv_track_param`: lid overlays
    # (`r_eye_a~lidU`) follow their eye's UVScaleOffset3/4.
    by_src: dict[tuple[str, str], list[str]] = {}
    for k, m in materials.items():
        src = m.extras.get("uv_track_src", k.split("@")[0])
        by_src.setdefault((src, m.extras.get("uv_track_param", "UVScaleOffset")), []).append(k)
    for mt in tracks:
        names = by_src.get((mt.material, mt.param), [])
        if len(mt.channels) < 4 or not names:
            continue
        for name in names:
            _uv_track_to(out, mt, name, [k for k in materials if k == name or k.startswith(name + "@")], materials[name], fps)
    return out


def _uv_track_to(out: dict, mt: T.MaterialParamTrack, name: str, targets: list[str], m: Material, fps: float) -> None:
    if any(t in out for t in targets):
        return  # LOD copies and shared materials carry the same track
    rest = [float(x) for x in m.extras.get("uv_scale_offset", (1.0, 1.0, 0.0, 0.0))]
    mir = m.extras.get("mirror", (False, False))
    if not m.extras.get("uv_translate_only") and any(len(mt.channels[k][1]) and float(np.abs(mt.channels[k][1] - rest[k]).max()) > 1e-4 for k in (0, 1)):
        return  # (translate-only materials ignore the scale channels, see UVTransformMode in _convert_material)
    frames = np.unique(np.concatenate([mt.channels[2][0], mt.channels[3][0], np.zeros(1, np.float32)]))
    du = _held(*mt.channels[2], frames, rest[2]) - rest[2]
    dv = _held(*mt.channels[3], frames, rest[3]) - rest[3]
    if not (np.abs(du).max() > 1e-6 or np.abs(dv).max() > 1e-6):
        return
    # The game offsets before scaling ((uv + o - centre) * scale + centre, see Falinks' eye_fix), so a change of o
    # moves the texture by o * scale; mirror-doubled axes halve it again.
    k_u = rest[0] / (2.0 if mir[0] else 1.0)
    k_v = rest[1] / (2.0 if mir[1] else 1.0)
    if "uv_track_gain" in m.extras:
        k_u, k_v = m.extras["uv_track_gain"]  # layer-5 overlays: offset after the scale, U subtracted (see _layer5)
    xy = np.stack([du * k_u, dv * k_v], axis=1).astype(np.float32)
    keep = np.ones(len(frames), bool)
    keep[1:] = np.any(np.abs(np.diff(xy, axis=0)) > 1e-6, axis=1)  # drop repeated holds
    times = (frames[keep] / fps).astype(np.float32)
    for t in targets:
        out[t] = (times, xy[keep])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (q[:, i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def _sample(track: Track | None, rest, times: np.ndarray) -> np.ndarray:
    rest = np.asarray(rest, np.float64)
    if track is None:
        return np.tile(rest, (len(times), 1))
    v = np.stack([np.interp(times, track.times, track.values[:, k]) for k in range(track.values.shape[1])], 1)
    if v.shape[1] == 4:
        v /= np.linalg.norm(v, axis=1, keepdims=True)  # nlerp is enough between dense keys
    return v


def _non_uniform(v: np.ndarray) -> bool:
    v = np.asarray(v, np.float64).reshape(-1, 3)
    return bool(np.ptp(v, axis=1).max() > 0.05 * max(float(np.abs(v).max()), 1e-6))


def split_scale_joints(skeleton: Skeleton, clips: list[Clip]) -> list[str]:
    """Trinity joints are Maya-style (`segment_scale_compensate`): a parent's scale moves its children (their offsets
    scale) but is not inherited by them. glTF always inherits it, so a non-uniform scale on a bone with children
    stretches and shears everything below. Seviper's tongue and Florges' vines are retracted by (0.15, 1, 1) /
    (0.05, 1, 1) scale tracks along whole chains; dropping those tracks (the old anti-shear rule) left them sticking
    out, and compensating each child's scale cannot undo the shear once a child is rotated (Binacle's bent arm).

    Exact instead: each such bone gets a leaf joint `<bone>_scale` (identity local, same inverse bind) that takes
    over its skin weights and its scale; the bone itself stays unscaled, and its children's translations are
    multiplied by the game's scale (T(ps * t), the compensate rule). Clips that only scale the bone uniformly
    (hide at 0, grow) keep that scale on the bone, so glTF inheritance still hides the chain as before.
    Children with segment scale compensate OFF (`Joint.scale_compensate`, Trinity `*_segoff`) do inherit the scale:
    they are re-parented under the leaf with their raw offsets, so glTF inheritance scales both their offset and their
    mesh as the game does (treated as compensating, SV trainers' irises/pupils were squashed 20-33% under `eyescale`).
    Returns the split bone names."""
    joints = skeleton.joints
    kids: dict[int, list[int]] = {}
    inherit: dict[int, list[int]] = {}  # scale_compensate off: follow the leaf
    for i, j in enumerate(joints):
        if j.parent >= 0:
            kids.setdefault(j.parent, []).append(i)
    idx = {j.name: i for i, j in enumerate(joints)}
    split = set()
    for c in clips:
        for t in c.tracks:
            if t.kind == "scale" and idx.get(t.joint) in kids and _non_uniform(t.values):
                split.add(idx[t.joint])
    split |= {i for i in kids if _non_uniform(joints[i].scale)}
    if not split:
        return []
    for i in split:
        inherit[i] = [k for k in kids[i] if not joints[k].scale_compensate]
        kids[i] = [k for k in kids[i] if joints[k].scale_compensate]
    rest_scale = {i: np.asarray(joints[i].scale, np.float64) for i in split}
    rest_trans = {k: np.asarray(joints[k].translation, np.float64) for i in split for k in kids[i]}
    proxy: dict[int, int] = {}
    for i in sorted(split):
        b = joints[i]
        joints.append(Joint(name=f"{b.name}_scale", parent=i, scale=tuple(float(x) for x in rest_scale[i]),
                            inverse_bind=b.inverse_bind, skinning=b.skinning))
        proxy[i] = len(joints) - 1
        b.scale = (1.0, 1.0, 1.0)
        for k in inherit[i]:
            joints[k].parent = proxy[i]
        for k in kids[i]:
            joints[k].translation = tuple(float(x) for x in rest_trans[k] * rest_scale[i])
    skeleton.palette = [proxy.get(p, p) for p in skeleton.palette]
    for c in clips:
        tr = {(t.joint, t.kind): t for t in c.tracks}
        out: list[Track] = [t for t in c.tracks if not ((t.kind == "scale" and idx.get(t.joint) in split)
                                                       or (t.kind == "translation" and idx.get(t.joint) in rest_trans))]
        for i in sorted(split):
            b = joints[i]
            st = tr.get((b.name, "scale"))
            t0 = np.zeros(1, np.float32)
            if st is not None and not _non_uniform(st.values):
                # Uniform this clip: the old inherited behaviour (hide/show the whole chain).
                out.append(st)
                out.append(Track(f"{b.name}_scale", "scale", t0, rest_scale[i][None].astype(np.float32)))
                ps_tr = None
            else:
                # Every clip keys both, so switching clips never leaves a stale scale behind.
                out.append(Track(b.name, "scale", t0, np.ones((1, 3), np.float32)))
                if st is None:
                    # No scale key this clip: the rest scale holds, and the children's offsets must still be scaled by
                    # it (writing the raw offsets pulled SV trainers' eyeballs together under `eyescale` (1.6, 1, 1)).
                    st = Track(b.name, "scale", t0, rest_scale[i][None].astype(np.float32))
                out.append(Track(f"{b.name}_scale", "scale", st.times, st.values))
                ps_tr = st
            for k in kids[i]:
                ct = tr.get((joints[k].name, "translation"))
                if ps_tr is None:
                    if ct is not None:  # uniform clip: children inherit; their offsets stay as authored
                        out.append(ct)
                    else:
                        out.append(Track(joints[k].name, "translation", t0, rest_trans[k][None].astype(np.float32)))
                    continue
                times = np.unique(np.concatenate([ps_tr.times] + ([ct.times] if ct is not None else []))).astype(np.float64)
                ps = _sample(ps_tr, rest_scale[i], times)
                tv = _sample(ct, rest_trans[k], times)
                out.append(Track(joints[k].name, "translation", times.astype(np.float32), (ps * tv).astype(np.float32)))
        c.tracks = out
    return [joints[i].name for i in sorted(split)]


def _convert_clip(tr: T.Tranm, af: AnimationFile, tol: float = 1e-3, shear_bones: frozenset[str] | None = None) -> Clip:
    fps = float(tr.frame_rate)
    tracks: list[Track] = []
    seen: set[str] = set()
    for bt in tr.tracks:
        # Some source clips list the whole track table twice (SV `tr0510_00_01320_ballthrow01`: 59 identical
        # repeats; LA pm0493): glTF forbids two channels on one target, keep the first.
        if bt.bone in seen:
            continue
        seen.add(bt.bone)
        if bt.translation is not None:
            f, v = decimate(*_strict(bt.translation), tol)
            tracks.append(Track(bt.bone, "translation", f.astype(np.float32) / fps, v))
        if bt.rotation is not None:
            f, v = _strict(bt.rotation)
            f, v = decimate(f, _fix_quat_signs(v), tol)
            tracks.append(Track(bt.bone, "rotation", f.astype(np.float32) / fps, v))
        if bt.scale is not None:
            f, v = decimate(*_strict(bt.scale), tol)
            # Non-uniform scale on a bone with children is Maya-style (children do not inherit it): kept, and the
            # children compensated in import_model (compensate_segment_scale).
            tracks.append(Track(bt.bone, "scale", f.astype(np.float32) / fps, v))
    # Indexes 10000, 20000, ... are whole alternate animation sets that reuse
    # the main set's clip names; keep names unique and record the set.
    anim_set = af.index // 10000
    group = af.group if af.group is not None else (f"set{anim_set}" if anim_set else "")
    name = f"{af.name} [{group}]" if group else af.name
    visibility: dict[str, list[tuple[int, bool]]] = {}
    if af.config:
        try:
            for vt in T.parse_tracm_visibility(af.config):
                shape = re.sub(r"_lod\d+$", "", vt.shape)
                if shape.endswith("_shape"):
                    shape = shape[: -len("_shape")]
                if shape in visibility:
                    continue  # LOD copies carry the same track
                keys = [(int(f), bool(v)) for f, v in zip(vt.frames.tolist(), vt.values.tolist())]
                compact = [k for i, k in enumerate(keys) if i == 0 or k[1] != keys[i - 1][1]]
                visibility[shape] = compact
        except Exception:  # noqa: BLE001 - visibility is optional
            visibility = {}
    return Clip(
        name=name,
        tracks=tracks,
        frame_rate=fps,
        frame_count=tr.frame_count,
        loop=tr.looped or af.name.endswith("_loop"),
        index=af.index,
        category=categorize(af.name),
        group=group,
        source=f"{af.index:05d}_{af.name}",
        visibility=visibility,
    )


def _fix_quat_signs(q: np.ndarray) -> np.ndarray:
    """Make consecutive quaternions take the short arc (q and -q are equal)."""
    out = q.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] *= -1.0
    return out


# --------------------------------------------------------------------------


def collect_animations(source: Source, model_id: str, subdir: str = "") -> list[AnimationFile]:
    """Find `<model_id>_#####_<name>.tranm` files in the source (optionally a subdir)."""
    out = []
    for n in source.list():
        if subdir and not n.startswith(subdir.rstrip("/") + "/"):
            continue
        base = os.path.basename(n)
        if not base.startswith(model_id + "_") or not base.endswith(".tranm"):
            continue
        parsed = clip_name_from_file(base)
        if not parsed:
            continue
        name, idx = parsed
        cfg = n[: -len(".tranm")] + ".tracm"
        out.append(AnimationFile(name=name, index=idx, data=source.read(n), config=source.read(cfg) if source.exists(cfg) else None))
    return out
