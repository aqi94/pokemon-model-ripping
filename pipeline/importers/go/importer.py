"""Pokémon GO importer: binary FBX + PNG textures -> common Model.

The FBX is read by headless Blender (`blender/dump_fbx.py`), which writes a neutral dump already in glTF space
(Y up, metres). This module turns that dump into the IR; `model_from_dump` is separate from `dump_fbx` so tests can
feed a synthetic dump without Blender.
"""

from __future__ import annotations

import os
import re
import tempfile

import numpy as np
from PIL import Image

from ...common.model import Clip, Joint, Material, Mesh, Model, Primitive, Skeleton, TextureImage, Track
from ..za.importer import categorize, decimate

REST_TOL = 1e-4
# Translation tracks are kept only for the root chain (rig / Root / Center ...): GO bakes one shared clip set into every
# character's FBX with the translations of the skeleton it was authored on, so on a rig with other proportions they push
# face/hip/shoulder bones off the body (Jessie's eyes stretch 7 cm out of her face). Rotations retarget cleanly.
TRANSLATION_DEPTH = 2


def dump_fbx(fbx: str, out_dir: str | None = None) -> tuple[dict, dict]:
    """Run Blender on `fbx`; returns (metadata, arrays)."""
    from ...blender_runner import BLENDER_DIR, run_script

    out_dir = out_dir or tempfile.mkdtemp(prefix="gofbx_")
    meta = run_script(os.path.join(BLENDER_DIR, "dump_fbx.py"), [os.path.abspath(fbx), os.path.abspath(out_dir)], os.path.join(out_dir, "result.json"))
    if not meta.get("ok"):
        raise RuntimeError(f"Blender FBX dump failed for {fbx}: {meta.get('error', '')[-800:]}")
    with np.load(os.path.join(out_dir, "dump.npz")) as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


def _find_texture(folder: str, name: str | None) -> str | None:
    if not name:
        return None
    want = name.lower()
    for f in os.listdir(folder):
        if f.lower() == want:
            return os.path.join(folder, f)
    return None


SKIN_PARTS = {"nude", "arms", "chest", "face", "feet", "hands", "legs", "scalp", "waist"}


def _mesh_texture(folder: str, mesh: str) -> str | None:
    """Modular avatar parts share one untextured material; pick `<m|f>_<hair|skin>_default.png` by part name."""
    toks = [t for t in re.split(r"[^a-z0-9]+", mesh.lower()) if t]
    if not toks:
        return None
    cat = "hair" if "hair" in toks else "skin" if SKIN_PARTS & set(toks) else None
    return _find_texture(folder, f"{toks[0]}_{cat}_default.png") if cat else None


def _quat_xyzw(wxyz) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in wxyz)
    return (x, y, z, w)


def _fix_quat_signs(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0:
            out[i] *= -1.0
    return out


def _depth(joints: list[dict], i: int) -> int:
    d = 0
    while joints[i]["parent"] >= 0:
        i = joints[i]["parent"]
        d += 1
    return d


def _clip_tracks(joints: list[dict], arrays: dict, key: str, frames: int, fps: float, tol: float) -> list[Track]:
    loc, rot, scl = arrays[key + "loc"], arrays[key + "rot"], arrays[key + "scl"]
    times = np.arange(frames, dtype=np.float32) / fps
    tracks: list[Track] = []
    for i, j in enumerate(joints):
        rest = {
            "translation": np.asarray(j["t"], np.float32),
            "rotation": np.asarray([j["r"][1], j["r"][2], j["r"][3], j["r"][0]], np.float32),
            "scale": np.asarray(j["s"], np.float32),
        }
        chans = {
            "translation": loc[:, i],
            "rotation": np.concatenate([rot[:, i, 1:], rot[:, i, :1]], axis=1),  # wxyz -> xyzw
            "scale": scl[:, i],
        }
        for kind, vals in chans.items():
            if kind == "translation" and _depth(joints, i) > TRANSLATION_DEPTH:
                continue
            if kind == "rotation":
                vals = _fix_quat_signs(vals)
                same_as_rest = np.abs(vals - rest[kind]).max() < REST_TOL or np.abs(vals + rest[kind]).max() < REST_TOL
            else:
                same_as_rest = np.abs(vals - rest[kind]).max() < REST_TOL
            if same_as_rest:
                continue
            if np.abs(vals - vals[0]).max() < REST_TOL:
                tracks.append(Track(j["name"], kind, times[:1].copy(), vals[:1].astype(np.float32)))
                continue
            f, v = decimate(times, vals.astype(np.float32), tol)
            tracks.append(Track(j["name"], kind, f.astype(np.float32), v.astype(np.float32)))
    return tracks


def model_from_dump(meta: dict, arrays: dict, folder: str, model_id: str, display_name: str, kind: str = "trainer", game: str = "go") -> Model:
    warnings: list[str] = []
    skeleton = None
    if meta.get("joints"):
        joints = []
        for i, j in enumerate(meta["joints"]):
            joints.append(
                Joint(
                    name=j["name"],
                    parent=j["parent"],
                    translation=tuple(float(x) for x in j["t"]),
                    rotation=_quat_xyzw(j["r"]),
                    scale=tuple(float(x) for x in j["s"]),
                    inverse_bind=np.ascontiguousarray(arrays["ibm"][i], np.float32),
                    skinning=True,
                )
            )
        skeleton = Skeleton(joints=joints, palette=list(range(len(joints))))

    textures: dict[str, TextureImage] = {}
    materials: dict[str, Material] = {}

    def add_material(name: str, tex_path: str | None) -> None:
        mat = Material(name=name, roughness_factor=0.8)
        if tex_path:
            key = os.path.splitext(os.path.basename(tex_path))[0]
            if key not in textures:
                rgba = np.array(Image.open(tex_path).convert("RGBA"), np.uint8)
                rgba[..., 3] = 255
                textures[key] = TextureImage(key=key, rgba=rgba, srgb=True)
            mat.base_color = key
        materials[name] = mat

    unresolved: set[str] = set()
    for m in meta["materials"]:
        tex_path = _find_texture(folder, m.get("base")) or next((p for p in (_find_texture(folder, i) for i in m.get("images", [])) if p), None)
        if tex_path is None:
            unresolved.add(m["name"])
        else:
            add_material(m["name"], tex_path)

    meshes: list[Mesh] = []
    flip_v = np.array([1.0, -1.0], np.float32), np.array([0.0, 1.0], np.float32)  # Blender bottom-left -> glTF top-left
    for md in meta["meshes"]:
        prims = []
        for s in md["slots"]:
            pre = s["key"]
            mat_name = s["material"]
            if mat_name in unresolved:
                # Untextured material: per-mesh material, texture by part name (avatars), else a plain colour.
                mat_name = f"{md['name']}_{mat_name}"
                if mat_name not in materials:
                    tex = _mesh_texture(folder, md["name"])
                    if tex is None:
                        warnings.append(f"mesh {md['name']}: no texture for material {s['material']}")
                    add_material(mat_name, tex)
            prim = Primitive(
                material=mat_name,
                indices=arrays[pre + "idx"].astype(np.uint32),
                positions=arrays[pre + "pos"].astype(np.float32),
                normals=arrays[pre + "nrm"].astype(np.float32),
                uvs=[arrays[pre + "uv"].astype(np.float32) * flip_v[0] + flip_v[1]],
            )
            if s.get("skinned") and (pre + "joints") in arrays:
                prim.joints = arrays[pre + "joints"].astype(np.uint16)
                prim.weights = arrays[pre + "weights"].astype(np.float32)
            prims.append(prim)
        if prims:
            meshes.append(Mesh(name=md["name"], primitives=prims))

    fps = float(meta.get("fps") or 30.0)
    clips: list[Clip] = []
    for ci, c in enumerate(meta.get("clips", [])):
        tracks = _clip_tracks(meta["joints"], arrays, c["key"], c["frames"], fps, 1e-3)
        name = c["name"]
        clips.append(
            Clip(
                name=name,
                tracks=tracks,
                frame_rate=fps,
                frame_count=int(c["frames"]),
                loop=bool(re.search(r"idle|loop", name.lower())),
                index=ci,
                category=categorize(name),
                source=name,
            )
        )
    clips.sort(key=lambda c: c.name)
    default_clip = next((c.name for c in clips if c.name.lower().endswith("idle")), clips[0].name if clips else None)
    return Model(
        id=model_id,
        game=game,
        display_name=display_name,
        kind=kind,
        skeleton=skeleton,
        meshes=meshes,
        materials=materials,
        textures=textures,
        clips=clips,
        default_clip=default_clip,
        extras={"warnings": warnings} if warnings else {},
    )


def import_model(fbx: str, model_id: str, display_name: str, kind: str = "trainer", game: str = "go", dump_dir: str | None = None) -> Model:
    meta, arrays = dump_fbx(fbx, dump_dir)
    return model_from_dump(meta, arrays, os.path.dirname(os.path.abspath(fbx)), model_id, display_name, kind, game)
