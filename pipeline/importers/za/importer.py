"""Pokémon Legends: Z-A importer: Trinity files -> common `Model`."""

from __future__ import annotations

import math
import os
import posixpath
import re
import zipfile
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


SKIN_MATERIALS = ("body_skin", "face_skin")
SKIN_TONES = 4  # zones 1..4 of the shared skin tables (deep .. pale)
SKIN_DEFAULT_TONE = 3

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


# --------------------------------------------------------------------------


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
) -> Model:
    """`skeleton` replaces the model's own .trskl (modular parts skin to a shared rig);
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

    def build_materials(files: list[str], tag: str) -> dict[str, Material]:
        out = {}
        for fn in files:
            if not source.exists(fn):
                continue
            for tm in T.parse_trmtr(source.read(fn)):
                out[tm.name] = _convert_material(tm, tag, load_tex, textures, tex_meta, skin_tones)
        return out

    materials = build_materials(normal_files, "")
    for s in sets:
        if s.name == "normal":
            continue
        files = s.material_files or [f"{model_id}_{s.name}.trmtr"]
        mats = build_materials(files, s.name)
        if mats:
            label = {"rare": "Shiny"}.get(s.name, s.name)
            variants.append(Variant(name=s.name, materials=mats, group="coloring", label=label))
    if variants:
        variants.insert(0, Variant(name="normal", materials={}, group="coloring", label="Normal", default=True))

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
            decoded = [T.decode_vertices(d, buffers[si]) for d in shape.declarations]
            prims = []
            for part in shape.parts:
                if part.index_count <= 0:
                    continue
                v = decoded[part.declaration] if part.declaration < len(decoded) else decoded[0]
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
                uvs = []
                for uv in v.uvs:
                    u_t = uv[:, 0] * su + ou
                    v_t = uv[:, 1] * sv + ov
                    v_t = v_t - np.floor(v_t)  # tile removal (V sampler is clamp)
                    uvs.append(np.stack([u_t, 1.0 - v_t], axis=1).astype(np.float32))
                tangents = None
                if v.tangents is not None:
                    tangents = v.tangents.copy()
                    tangents[:, 3] = np.where(tangents[:, 3] < 0, 1.0, -1.0)
                prims.append(
                    Primitive(
                        material=mat_name,
                        indices=T.read_indices(shape, buffers[si], part),
                        positions=v.positions,
                        normals=v.normals,
                        tangents=tangents,
                        uvs=uvs,
                        colors=None,  # Trinity vertex colours are shader masks (EnableVertexSubLayer), not tint
                        joints=v.joints if skeleton else None,
                        weights=v.weights if skeleton else None,
                    )
                )
            name = shape.mesh_name or shape.name.replace("_shape", "")
            if lod and not name.endswith(f"_lod{lod}"):
                name = f"{name}_lod{lod}"
            meshes.append(Mesh(name=name, primitives=prims, lod=lod))

    # ---- animations -------------------------------------------------------
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
        clips.append(_convert_clip(tr, af))
    clips.sort(key=lambda c: (c.index, c.name))
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
        extras={"warnings": warnings} if warnings else {},
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


def _finish_pbr_material(mat: Material, tm: T.TrMaterial, load_tex, register, stem) -> Material:
    mat.extras.pop("shading", None)
    mat.extras["pbr"] = True
    mat.metallic_factor = float(tm.floats.get("Metallic", 0.0))
    return mat


def _convert_material(tm: T.TrMaterial, tag: str, load_tex, textures: dict[str, TextureImage], tex_meta, skin_tones: bool = False) -> Material:
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
    lym_file = _tex_file(tm, "LayerMaskMap")
    lym = load_tex(lym_file) if lym_file else None
    layer_colors = []
    layer_enable = []
    for i in range(1, 5):
        c = tm.colors.get(f"BaseColorLayer{i}")
        layer_colors.append(c[:3] if c else None)
        # The colour's alpha is the layer's enable/strength: Pokemon use 1.0,
        # trainer clothing/skin use 0.0 with a placeholder black mask.
        layer_enable.append(_layer_strength(c))
    base_factor = tm.colors.get("BaseColor", (1.0, 1.0, 1.0, 1.0))
    if alb is not None:
        # ColorTableMap (trainer faces): a small palette (ColorTableDivideNumber
        # cells per axis) selected per texel by the layer mask (row from R,
        # column from G) and multiplied into the albedo; it holds the skin tone.
        ct_file = _tex_file(tm, "ColorTableMap")
        table_shadow = None
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
                if skin:
                    idxs[0] = SKIN_DEFAULT_TONE
                raw_alb = alb
                alb, table_shadow = apply_color_table(raw_alb, lym, ct, div, idxs, masked_only=masked, scales=scl)
                if modular and not skin:
                    mask_r = lym if lym.shape[:2] == raw_alb.shape[:2] else resize_rgba(lym, raw_alb.shape[1], raw_alb.shape[0])
                    cover = [float(((mask_r[..., c] / 255.0 * (scl[c] if scl else 1.0)) > 0.02).mean()) for c in range(4)]
                    layers = [c + 1 for c in range(4) if cover[c] > 0.01]
                    if layers:
                        th, tw = ct.shape[:2]

                        def zone_rgb(row: int, z: int) -> list[int]:
                            tx = min(((z * 2 + 1) * tw) // (2 * max(1, div)), tw - 1)
                            return [int(v) for v in ct[min(row, th - 1), tx, :3]]

                        h, w = raw_alb.shape[:2]
                        mask_rgb = mask_r.copy()
                        mask_rgb[..., 3] = 255
                        mask_alpha = np.repeat(mask_r[..., 3:4], 4, axis=2).copy()
                        mask_alpha[..., 3] = 255
                        mat.extras["palette"] = {
                            "albedo": register(f"{stem(alb_file)}_raw", raw_alb, True, mat.extras["mirror"]),
                            # RGB and alpha travel as two opaque images: browsers premultiply alpha on canvas reads and would zero RGB where A=0.
                            "mask": register(f"{stem(lym_file)}_mask{w}x{h}", mask_rgb, False, mat.extras["mirror"]),
                            "maskA": register(f"{stem(lym_file)}_maskA{w}x{h}", mask_alpha, False, mat.extras["mirror"]),
                            "divide": div,
                            "indices": idxs,
                            "scales": [float(s) for s in scl],
                            "maskedOnly": True,
                            "layers": layers,
                            "lit": [zone_rgb(th // 4, z) for z in range(div)],
                            "shadow": [zone_rgb((3 * th) // 4, z) for z in range(div)],
                        }
                if skin:
                    # Body and head share one skin palette: zone N of both tables is the same tone, so
                    # tone variants keep head and body matched. Layer 4 (lips) keeps its own zone.
                    tones = []
                    for z in range(1, min(div, SKIN_TONES) + 1):
                        lit, shd = apply_color_table(raw_alb, lym, ct, div, [z] + idxs[1:], scales=scl)
                        tones.append((z, register(f"{stem(alb_file)}_skin{z}", lit, True), register(f"{stem(alb_file)}_skin{z}_shadow", shd, True)))
                    mat.extras["skin_tones"] = tones
        baked = alb
        scales = [tm.floats.get(f"LayerMaskScale{i}", 1.0) * layer_enable[i - 1] for i in range(1, 5)]
        multiply = flags.get("BaseColorMultiply", "True") != "False"
        bc = tuple(float(base_factor[k]) for k in range(3))
        layered = lym is not None and all(c is not None for c in layer_colors) and any(e > 0 for e in layer_enable)
        if layered:
            # BaseColor goes into the texture (see bake_layers), the glTF factor becomes 1.
            baked = bake_layers(alb, lym, layer_colors, scales, multiply, bc if multiply else (1.0, 1.0, 1.0))
        # Key on albedo + mask so materials sharing a placeholder albedo
        # (eyes and flames both use white_msk) do not collide.
        # Key on albedo + mask + the layer colours/scales: materials can share
        # textures but differ in colours (Lucario torso vs legs, Persian body
        # vs ear tips), and identical colour sets (left/right eye) still dedupe.
        import hashlib

        sig = hashlib.sha1(repr((layer_colors, scales if lym is not None else None, multiply if lym is not None else None, bc if layered else None)).encode()).hexdigest()[:8]
        key = f"{stem(alb_file)}_{stem(lym_file)}_{sig}{'_' + tag if tag else ''}_baked" if baked is not alb else stem(alb_file)
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
            shadow_palette.append(tuple(float(base_i[k]) * tint[k] for k in range(3)))
            shadow_scales.append(tm.floats.get(f"LayerMaskScale{i + 1}", 1.0) * (1.0 if (shl is not None or enabled) else 0.0))
        if lym is not None and any(sc > 0 for sc in shadow_scales):
            shadow = bake_layers(alb, lym, shadow_palette, shadow_scales, multiply, bc if (layered and multiply) else (1.0, 1.0, 1.0))
        else:
            tint = np.asarray([1.0 + (float(sh_base[k]) - 1.0) * strength for k in range(3)], np.float32)
            lin = srgb_to_linear(alb[..., :3].astype(np.float32) / 255.0) * tint[None, None, :]
            shadow = np.empty_like(alb)
            shadow[..., :3] = np.clip(linear_to_srgb(lin) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            shadow[..., 3] = alb[..., 3]
        if table_shadow is not None:
            shadow = table_shadow  # skin: the colour table's bottom half is the shadow tone
        elif not any(tm.colors.get(f"ShadowingColorLayer{i + 1}") for i in range(4)):
            shadow = darken_saturate(shadow, 0.78, 1.15)  # no per-layer shadow data: generic darker shade
        mat.extras["shadow_color"] = register(f"{stem(alb_file)}_{sig}{'_' + tag if tag else ''}_shadow", shadow, True, mat.extras["mirror"])
    baked_bc = alb is not None and layered and multiply
    # Skin tone comes from the colour table; the body material's authored BaseColor tint would
    # make it redder than the face, so it is dropped for modular skin.
    mat.base_color_factor = (1.0, 1.0, 1.0, 1.0) if (baked_bc or "skin_tones" in mat.extras) else (
        float(base_factor[0]),
        float(base_factor[1]),
        float(base_factor[2]),
        1.0,
    )

    nrm_file = _tex_file(tm, "NormalMap")
    if nrm_file and flags.get("EnableNormalMap", "True") != "False":
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
    if not tm.shader.startswith("IkCharacter") and tm.shader not in ("NonDirectional", "Unlit", "FresnelEffect"):
        # Scarlet/Violet (SSS, EyeClearCoat, Standard, Hair...) is plain PBR: no IkCharacter toon params.
        return _finish_pbr_material(mat, tm, load_tex, register, stem)
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
        "unlit": tm.shader in ("NonDirectional", "Unlit"),
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
    }
    # Specular / rim masks (single-channel). Shared white/black placeholders
    # are skipped; the viewer falls back to 1.0 (white) when absent.
    for slot, key in (("SpecularMaskMap", "specular_mask"), ("RimLightMaskMap", "rim_mask")):
        mf = _tex_file(tm, slot)
        if mf and not os.path.basename(mf).startswith("sh_") and "white_msk" not in mf and "black_msk" not in mf:
            mt = load_tex(mf)
            if mt is not None and not _is_flat(mt):
                mat.extras[key] = register(stem(mf), mt, False, mat.extras["mirror"])
    return mat


def _is_flat(rgba: np.ndarray) -> bool:
    return bool(rgba[..., :3].min() == rgba[..., :3].max())


# Default "unset" swatches the authoring tool leaves in unused layers (red,
# blue, green, cyan). Their alpha is 0, but alpha 0 is also used by real
# layers (Tarragon's black eyelashes / white brows), so detect the swatches.
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


def apply_color_table(albedo: np.ndarray, mask: np.ndarray, table: np.ndarray, divide: int, indices: list[int] | None = None, masked_only: bool = False, scales: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
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
    for ty in (th // 4, (3 * th) // 4):
        def cell(zone_1based: int) -> np.ndarray:
            z = min(max(zone_1based, 1), d) - 1
            tx = min(((z * 2 + 1) * tw) // (2 * d), tw - 1)
            return srgb_to_linear(table[min(ty, th - 1), tx, :3].astype(np.float32) / 255.0)
        # Unmasked texels take layer 1's zone (the base skin), so there are no untinted seams.
        if masked_only:
            tint = np.ones_like(a_lin)
            first = 0
        else:
            tint = np.broadcast_to(cell(idx[0]), a_lin.shape).copy()
            first = 1
        for c in range(first, 4):
            tint += (cell(idx[c])[None, None, :] - tint) * m[..., c : c + 1]
        img = np.empty_like(albedo)
        img[..., :3] = np.clip(linear_to_srgb(a_lin * tint) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        img[..., 3] = albedo[..., 3]
        out.append(img)
    return out[0], out[1]


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


def _convert_clip(tr: T.Tranm, af: AnimationFile, tol: float = 1e-3) -> Clip:
    fps = float(tr.frame_rate)
    tracks: list[Track] = []
    for bt in tr.tracks:
        if bt.translation is not None:
            f, v = decimate(*_strict(bt.translation), tol)
            tracks.append(Track(bt.bone, "translation", f.astype(np.float32) / fps, v))
        if bt.rotation is not None:
            f, v = _strict(bt.rotation)
            f, v = decimate(f, _fix_quat_signs(v), tol)
            tracks.append(Track(bt.bone, "rotation", f.astype(np.float32) / fps, v))
        if bt.scale is not None:
            f, v = decimate(*_strict(bt.scale), tol)
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
