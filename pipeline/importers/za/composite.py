"""Modular generic NPCs (`model_cc_*`): one base rig plus swappable face / hair / dress parts.

`model_cc_base/<b>/<b>_base.trmdl` is a skeleton only. Each part under
`model_cc_vr/{face,hrs,drs}/<b>_<kind>NNNN_VV_<name>/` is a mesh set that skins
to the shared rig. Vertex joint indices address one combined palette: the base
rig at 0.., then each part's own bones (facial bones, skirt chains) at that
part's `palette_offset`. Parts that ship bones list them as root nodes whose
local transform is relative to a base bone (face -> head, skirt -> waist).

The composite keeps every selectable part as separate meshes tagged with
`extras = {group, option}`; the manifest's `partGroups` lets the viewer show
exactly one option per group.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

import numpy as np

from ...common.model import Joint, Mesh, Model, Skeleton, Variant
from . import trinity as T
from .importer import SKIN_DEFAULT_TONE, AnimationFile, Source, _build_skeleton, import_model

GROUP_LABELS = {"face": "Face", "hrs": "Hair", "drs": "Outfit"}
GROUP_ORDER = ["face", "hrs", "drs"]


@dataclass
class CcPart:
    group: str  # face | hrs | drs
    id: str  # model id inside its folder, e.g. dm_drs0000_00
    label: str
    folder: str  # zip-internal folder prefix
    default: bool = False


def part_label(name: str, variant: int) -> str:
    label = name.replace("_", " ").title()
    return f"{label} {variant + 1}" if variant else label


PALETTE_LABELS = {
    "tops": "Top", "bottoms": "Bottom", "body": "Body", "obj": "Accessory", "obj_drs": "Accessory", "obj_drs_01": "Accessory",
    "hairstyle": "Hair", "hairstyle_obj": "Hair accessory", "obj_hrs": "Hair accessory", "obj_hrs00": "Hair accessory",
    "eyebrow": "Eyebrows", "l_eye": "Eyes", "r_eye": "Eyes", "eyewear": "Eyewear",
    "obj_face": "Face accessory", "obj_face00": "Face accessory", "obj_face01": "Face accessory", "obj00_face": "Face accessory",
}


def _palette_entries(mats) -> list[dict]:
    """Manifest descriptors of a part's recolourable materials (left/right eyes share one entry)."""
    out: dict[tuple, dict] = {}
    for m in mats:
        pal = m.extras.get("palette")
        if not pal:
            continue
        label = PALETTE_LABELS.get(m.name.split("__", 1)[-1], m.name.split("__", 1)[-1].replace("_", " ").title())
        key = (label, tuple(pal["indices"]), tuple(pal["layers"]))
        e = out.setdefault(
            key,
            {
                "label": label,
                "materials": [],
                "divide": pal["divide"],
                "indices": pal["indices"],
                "layers": pal["layers"],
                "zones": ["#%02x%02x%02x" % tuple(c) for c in pal["lit"]],
            },
        )
        e["materials"].append(m.name)
    return list(out.values())


def _local_matrix(j: Joint) -> np.ndarray:
    x, y, z, w = j.rotation
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    m = np.eye(4)
    m[:3, :3] = r * np.array(j.scale)
    m[:3, 3] = j.translation
    return m


def _world_matrices(joints: list[Joint]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}

    def world(i: int) -> np.ndarray:
        if i not in out:
            j = joints[i]
            m = _local_matrix(j)
            out[i] = world(j.parent) @ m if j.parent >= 0 else m
        return out[i]

    for i in range(len(joints)):
        world(i)
    return out


@dataclass
class _PartRig:
    lut: dict[int, int]  # part-global rig index -> palette position


def _add_part_bones(
    sk: T.Trskl,
    part: CcPart,
    joints: list[Joint],
    palette: list[int],
    pos_of: dict[int, int],
    known: dict[tuple, int],
    base_world: dict[int, np.ndarray],
    warnings: list[str],
) -> _PartRig:
    """Append (or reuse) a part's bones on the shared rig, return its palette lookup."""
    part_sk = _build_skeleton(sk)
    mapping: dict[int, int] = {}
    for ni, pj in enumerate(part_sk.joints):
        ibm = None if pj.inverse_bind is None else np.round(pj.inverse_bind, 4).tobytes()
        key = (pj.name, ibm)
        if key in known:
            mapping[ni] = known[key]
            continue
        name = pj.name
        if any(j.name == name for j in joints):
            name = f"{name}@{part.id}"
        mapping[ni] = len(joints)
        known[key] = len(joints)
        joints.append(Joint(name=name, parent=-1, translation=pj.translation, rotation=pj.rotation, scale=pj.scale, inverse_bind=pj.inverse_bind, skinning=pj.skinning))
    for ni, pj in enumerate(part_sk.joints):
        idx = mapping[ni]
        if joints[idx].parent != -1 or idx < len(base_world):
            continue
        if pj.parent >= 0:
            joints[idx].parent = mapping[pj.parent]
            continue
        # Root bone of the part: its local transform is relative to the base
        # bone whose bind pose reproduces the bone's own bind position.
        if pj.inverse_bind is None:
            joints[idx].parent = 0
            continue
        target = np.linalg.inv(pj.inverse_bind)[:3, 3]
        local = np.append(pj.translation, 1.0)
        best_err, best = min((float(np.linalg.norm((m @ local)[:3] - target)), bi) for bi, m in base_world.items())
        if best_err > 1e-3:
            warnings.append(f"{part.id}: bone {pj.name} attaches to {joints[best].name} with error {best_err:.4f}")
        joints[idx].parent = best
    lut: dict[int, int] = {}
    if sk.palette_offset >= 0:
        for ni, node in enumerate(sk.nodes):
            if node.joint_info < 0:
                continue
            ji = mapping[ni]
            pos = pos_of.get(ji)
            if pos is None:
                pos = pos_of[ji] = len(palette)
                palette.append(ji)
            lut[sk.palette_offset + node.joint_info] = pos
    return _PartRig(lut)


def _remap_joints(prim, rig: _PartRig, base_count: int) -> int:
    """Rewrite a primitive's joint indices into the combined palette; returns #unmapped weighted verts."""
    if prim.joints is None:
        return 0
    top = int(prim.joints.max()) + 1
    table = np.zeros(max(top, base_count), np.uint16)
    table[:base_count] = np.arange(base_count)
    mapped = np.zeros(len(table), bool)
    mapped[:base_count] = True
    for g, pos in rig.lut.items():
        if g < len(table):
            table[g] = pos
            mapped[g] = True
    bad = int(((~mapped[prim.joints]) & (prim.weights > 0)).sum()) if prim.weights is not None else 0
    prim.joints = table[prim.joints].astype(np.uint16)
    return bad


def import_cc_model(
    base_source: Source,
    base_id: str,
    parts: list[CcPart],
    part_source,
    model_id: str,
    display_name: str,
    animations: list[AnimationFile] | None = None,
    game: str = "za",
) -> Model:
    """`part_source(part) -> Source` opens a part's folder."""
    warnings: list[str] = []
    base_trmdl = T.parse_trmdl(base_source.read(f"{base_id}.trmdl"))
    base_sk = _build_skeleton(T.parse_trskl(base_source.read(base_trmdl.skeleton)))
    joints = list(base_sk.joints)
    palette = list(base_sk.palette)
    base_count = len(palette)
    pos_of = {ji: i for i, ji in enumerate(palette)}
    base_world = _world_matrices(joints)
    known: dict[tuple, int] = {}

    # pass 1: bones of every part
    sources: dict[str, Source] = {}
    rigs: dict[str, _PartRig] = {}
    for part in parts:
        src = sources[part.id] = part_source(part)
        trmdl = T.parse_trmdl(src.read(f"{part.id}.trmdl"))
        if trmdl.skeleton and src.exists(trmdl.skeleton):
            rigs[part.id] = _add_part_bones(T.parse_trskl(src.read(trmdl.skeleton)), part, joints, palette, pos_of, known, base_world, warnings)
        else:
            rigs[part.id] = _PartRig({})
    skeleton = Skeleton(joints=joints, palette=palette)

    # base model: clips only (skeleton passed in)
    base_model = import_model(base_source, base_id, display_name, kind="trainer", animations=animations, game=game, skeleton=skeleton)

    # pass 2: meshes / materials of every part
    meshes: list[Mesh] = []
    materials = {}
    textures = dict(base_model.textures)
    unmapped = 0
    skin_mats: dict[str, list] = {}
    part_palettes: dict[str, list] = {}
    for part in sorted(parts, key=lambda p: (GROUP_ORDER.index(p.group), p.id)):
        pm = import_model(sources[part.id], part.id, kind="trainer", game=game, skeleton=skeleton, lod0_only=True, skin_tones=True)
        warnings += [f"{part.id}: {w}" for w in pm.extras.get("warnings", [])]
        rename = {n: f"{part.id}__{n}" for n in pm.materials}
        for n, mat in pm.materials.items():
            mat.name = rename[n]
            materials[mat.name] = mat
            tones = mat.extras.pop("skin_tones", None)
            if tones:
                skin_mats[mat.name] = tones
        for key, tex in pm.textures.items():
            textures.setdefault(key, tex)
        for mesh in pm.meshes:
            mesh.name = f"{part.id}__{mesh.name}"
            accessory = all(rename.get(p.material, p.material).split("__", 1)[-1].startswith("obj") for p in mesh.primitives)
            mesh.visible = part.default and not accessory
            mesh.extras = {"group": part.group, "option": part.id, **({"accessory": True} if accessory else {})}
            for prim in mesh.primitives:
                prim.material = rename.get(prim.material, prim.material)
                unmapped += _remap_joints(prim, rigs[part.id], base_count)
            meshes.append(mesh)
        used = {p.material for m in pm.meshes for p in m.primitives}
        part_palettes[part.id] = _palette_entries([materials[n] for n in rename.values() if n in used])
    if unmapped:
        warnings.append(f"{unmapped} weighted vertices reference joints missing from the rig")

    # Skin tone: one variant per shared palette zone, overriding every body_skin / face_skin material at once
    # (body and head stay matched).
    variants: list[Variant] = []
    tone_ids = sorted({t[0] for tones in skin_mats.values() for t in tones})
    for z in tone_ids:
        over = {}
        for name, tones in skin_mats.items():
            t = next((t for t in tones if t[0] == z), None)
            if t:
                m = materials[name]
                over[name] = replace(m, base_color=t[1], extras={**m.extras, "shadow_color": t[2]})
        variants.append(Variant(name=f"tone{z}", materials=over, group="skin", label=f"Tone {z}", default=z == SKIN_DEFAULT_TONE))

    groups = []
    for g in GROUP_ORDER:
        opts = [p for p in parts if p.group == g]
        if opts:
            groups.append(
                {
                    "name": g,
                    "label": GROUP_LABELS[g],
                    "options": [
                        {"id": p.id, "label": p.label, "default": p.default, **({"palettes": part_palettes[p.id]} if part_palettes.get(p.id) else {})}
                        for p in sorted(opts, key=lambda p: p.id)
                    ],
                }
            )
    extras = {"part_groups": groups}
    if warnings:
        extras["warnings"] = warnings
    return Model(
        id=model_id,
        game=game,
        display_name=display_name,
        kind="trainer",
        skeleton=skeleton,
        meshes=meshes,
        materials=materials,
        textures=textures,
        clips=base_model.clips,
        variants=variants,
        default_clip=base_model.default_clip,
        extras=extras,
    )
