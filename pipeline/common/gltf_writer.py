"""Write a `Model` (IR) to a self-contained .glb.

Produces glTF 2.0 with: one scene, joint hierarchy, one skin, PBR materials
with embedded PNG textures, KHR_materials_variants for alternate colourings,
and one animation per clip. No external files.
"""

from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .model import Clip, Material, Mesh, Model, Primitive, Skeleton, TextureImage

GLB_MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

COMPONENT = {
    np.dtype("float32"): 5126,
    np.dtype("uint32"): 5125,
    np.dtype("uint16"): 5123,
    np.dtype("uint8"): 5121,
}
TYPES = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}


@dataclass
class _Builder:
    json: dict
    bin: bytearray = field(default_factory=bytearray)
    accessor_cache: dict = field(default_factory=dict)
    image_cache: dict = field(default_factory=dict)

    def add_bytes(self, data: bytes, target: int | None = None, stride: int | None = None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin += data
        view = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            view["target"] = target
        if stride:
            view["byteStride"] = stride
        self.json["bufferViews"].append(view)
        return len(self.json["bufferViews"]) - 1

    def add_accessor(
        self,
        arr: np.ndarray,
        target: int | None = None,
        normalized: bool = False,
        minmax: bool = False,
        dtype=None,
    ) -> int:
        # Cache on the caller's array object. Keep a reference so its id()
        # cannot be recycled for a different temporary array.
        key = (id(arr), target, normalized, str(dtype))
        if key in self.accessor_cache:
            return self.accessor_cache[key][0]
        source = arr
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 1:
            n_comp = 1
        else:
            n_comp = arr.shape[1]
        view = self.add_bytes(arr.tobytes(), target)
        acc = {
            "bufferView": view,
            "componentType": COMPONENT[arr.dtype],
            "count": int(arr.shape[0]),
            "type": TYPES[n_comp],
        }
        if normalized:
            acc["normalized"] = True
        if minmax:
            a2 = arr.reshape(arr.shape[0], -1)
            acc["min"] = [float(x) for x in a2.min(axis=0)]
            acc["max"] = [float(x) for x in a2.max(axis=0)]
        self.json["accessors"].append(acc)
        idx = len(self.json["accessors"]) - 1
        self.accessor_cache[key] = (idx, source)
        return idx

    def add_image(self, tex: TextureImage) -> int:
        if tex.key in self.image_cache:
            return self.image_cache[tex.key]
        buf = io.BytesIO()
        Image.fromarray(tex.rgba, "RGBA").save(buf, format="PNG", optimize=False, compress_level=6)
        view = self.add_bytes(buf.getvalue())
        self.json["images"].append({"bufferView": view, "mimeType": "image/png", "name": tex.key})
        idx = len(self.json["images"]) - 1
        self.image_cache[tex.key] = idx
        return idx


def write_glb(model: Model, path: str) -> dict:
    """Write the model and return a small summary dict."""
    g = {
        "asset": {"version": "2.0", "generator": "pokemon-model-pipeline"},
        "scene": 0,
        "scenes": [{"nodes": [], "name": model.id}],
        "nodes": [],
        "meshes": [],
        "materials": [],
        "textures": [],
        "images": [],
        "samplers": [],
        "accessors": [],
        "bufferViews": [],
        "buffers": [],
        "animations": [],
        "skins": [],
        "extensionsUsed": [],
    }
    b = _Builder(g)

    # ---- textures / materials -------------------------------------------
    texture_index: dict[tuple, int] = {}
    sampler_index: dict[tuple[str, str], int] = {}
    WRAP = {"repeat": 10497, "mirror": 33648, "clamp": 33071}

    def sampler(wrap: tuple[str, str]) -> int:
        if wrap not in sampler_index:
            g["samplers"].append({"magFilter": 9729, "minFilter": 9987, "wrapS": WRAP.get(wrap[0], 10497), "wrapT": WRAP.get(wrap[1], 10497)})
            sampler_index[wrap] = len(g["samplers"]) - 1
        return sampler_index[wrap]

    def tex(key: str | None, wrap: tuple[str, str] = ("repeat", "repeat")) -> int | None:
        if key is None or key not in model.textures:
            return None
        k = (key, wrap)
        if k not in texture_index:
            img = b.add_image(model.textures[key])
            g["textures"].append({"sampler": sampler(wrap), "source": img, "name": key})
            texture_index[k] = len(g["textures"]) - 1
        return texture_index[k]

    def tex_ref(m: Material, slot: str, key: str | None) -> dict | None:
        t = tex(key, m.wrap.get(slot, ("repeat", "repeat")))
        if t is None:
            return None
        ref: dict = {"index": t, "texCoord": m.uv_sets.get(slot, 0)}
        tr = m.uv_transform.get(slot)
        if tr and tuple(tr) != (1.0, 1.0, 0.0, 0.0):
            ref["extensions"] = {"KHR_texture_transform": {"scale": [float(tr[0]), float(tr[1])], "offset": [float(tr[2]), float(tr[3])]}}
            if "KHR_texture_transform" not in g["extensionsUsed"]:
                g["extensionsUsed"].append("KHR_texture_transform")
        return ref

    material_index: dict[tuple[str, str | None], int] = {}

    def material(m: Material, variant: str | None) -> int:
        k = (m.name, variant)
        if k in material_index:
            return material_index[k]
        pbr: dict = {
            "baseColorFactor": [min(1.0, max(0.0, float(x))) for x in m.base_color_factor],
            "metallicFactor": m.metallic_factor,
            "roughnessFactor": m.roughness_factor,
        }
        gm: dict = {"name": m.name if variant is None else f"{m.name}@{variant}", "pbrMetallicRoughness": pbr}
        r = tex_ref(m, "base_color", m.base_color)
        if r is not None:
            pbr["baseColorTexture"] = r
        r = tex_ref(m, "metallic_roughness", m.metallic_roughness)
        if r is not None:
            pbr["metallicRoughnessTexture"] = r
        r = tex_ref(m, "normal", m.normal)
        if r is not None:
            gm["normalTexture"] = r
        r = tex_ref(m, "occlusion", m.occlusion)
        if r is not None:
            gm["occlusionTexture"] = r
        r = tex_ref(m, "emissive", m.emissive)
        if r is not None:
            gm["emissiveTexture"] = r
        if any(m.emissive_factor):
            gm["emissiveFactor"] = list(m.emissive_factor)
        if m.alpha_mode != "OPAQUE":
            gm["alphaMode"] = m.alpha_mode
            if m.alpha_mode == "MASK":
                gm["alphaCutoff"] = m.alpha_cutoff
        if m.double_sided:
            gm["doubleSided"] = True
        extras = dict(m.extras) if m.extras else {}
        raw_key = extras.pop("raw_base_color", None)
        if raw_key:
            t = tex(raw_key, m.wrap.get("base_color", ("repeat", "repeat")))
            if t is not None:
                extras["rawBaseColorTexture"] = t
        shadow_key = extras.pop("shadow_color", None)
        if shadow_key:
            t = tex(shadow_key, m.wrap.get("base_color", ("repeat", "repeat")))
            if t is not None:
                extras["shadowColorTexture"] = t
        pal = extras.pop("palette", None)
        if pal:
            # runtime recolour data (modular NPCs): raw albedo + layer mask textures, palette zones
            pal = dict(pal)
            ta = tex(pal["albedo"], ("repeat", "repeat"))
            tm_ = tex(pal["mask"], ("repeat", "repeat"))
            tma = tex(pal["maskA"], ("repeat", "repeat"))
            if ta is not None and tm_ is not None and tma is not None:
                pal["albedo"], pal["mask"], pal["maskA"] = ta, tm_, tma
                extras["palette"] = pal
        for src_key, dst_key in (("specular_mask", "specularMaskTexture"), ("rim_mask", "rimMaskTexture")):
            mask_key = extras.pop(src_key, None)
            if mask_key:
                t = tex(mask_key, m.wrap.get("base_color", ("repeat", "repeat")))
                if t is not None:
                    extras[dst_key] = t
        for internal in ("uv_scale_offset", "uv_scale_effective", "mirror", "mirror_normal"):
            extras.pop(internal, None)
        if extras:
            gm["extras"] = extras
        g["materials"].append(gm)
        material_index[k] = len(g["materials"]) - 1
        return material_index[k]

    # ---- skeleton --------------------------------------------------------
    joint_node: dict[int, int] = {}  # joint index -> node index
    root_nodes: list[int] = []
    skin_index = None
    if model.skeleton:
        sk = model.skeleton
        for ji, j in enumerate(sk.joints):
            node = {
                "name": j.name,
                "translation": list(map(float, j.translation)),
                "rotation": list(map(float, j.rotation)),
                "scale": list(map(float, j.scale)),
            }
            g["nodes"].append(node)
            joint_node[ji] = len(g["nodes"]) - 1
        for ji, j in enumerate(sk.joints):
            if j.parent >= 0:
                g["nodes"][joint_node[j.parent]].setdefault("children", []).append(joint_node[ji])
            else:
                root_nodes.append(joint_node[ji])
        if sk.palette:
            ibm = np.zeros((len(sk.palette), 16), np.float32)
            for pi, ji in enumerate(sk.palette):
                m = sk.joints[ji].inverse_bind
                if m is None:
                    m = np.eye(4, dtype=np.float32)
                ibm[pi] = np.asarray(m, np.float32).T.reshape(16)  # column-major
            skin = {
                "joints": [joint_node[ji] for ji in sk.palette],
                "inverseBindMatrices": b.add_accessor(ibm),
                "name": "skin",
            }
            if root_nodes:
                skin["skeleton"] = root_nodes[0]
            g["skins"].append(skin)
            skin_index = 0

    # ---- variants ----------------------------------------------------------
    variant_names = [v.name for v in model.variants]
    if variant_names:
        g["extensionsUsed"].append("KHR_materials_variants")
        g["extensions"] = {"KHR_materials_variants": {"variants": [{"name": n} for n in variant_names]}}

    # ---- meshes ------------------------------------------------------------
    mesh_nodes: list[int] = []
    for mesh in model.meshes:
        prims = []
        for p in mesh.primitives:
            attrs = {"POSITION": b.add_accessor(p.positions, 34962, minmax=True, dtype=np.float32)}
            if p.normals is not None:
                attrs["NORMAL"] = b.add_accessor(p.normals, 34962, dtype=np.float32)
            if p.tangents is not None:
                attrs["TANGENT"] = b.add_accessor(p.tangents, 34962, dtype=np.float32)
            for i, uv in enumerate(p.uvs):
                attrs[f"TEXCOORD_{i}"] = b.add_accessor(uv, 34962, dtype=np.float32)
            if p.colors is not None:
                attrs["COLOR_0"] = b.add_accessor(p.colors, 34962, dtype=np.float32)
            skinned = skin_index is not None and p.joints is not None and p.weights is not None
            if skinned:
                attrs["JOINTS_0"] = b.add_accessor(p.joints, 34962, dtype=np.uint16)
                attrs["WEIGHTS_0"] = b.add_accessor(p.weights, 34962, dtype=np.float32)
            idx_dtype = np.uint16 if p.positions.shape[0] <= 65535 else np.uint32
            targets = [{"POSITION": b.add_accessor(d, 34962, minmax=True, dtype=np.float32)} for d in p.morph_targets]
            prim = {
                "attributes": attrs,
                "indices": b.add_accessor(p.indices, 34963, dtype=idx_dtype),
                "mode": 4,
                "material": material(model.materials[p.material], None),
            }
            if targets:
                prim["targets"] = targets
            if variant_names:
                mappings = []
                for vi, v in enumerate(model.variants):
                    mat = v.materials.get(p.material, model.materials[p.material])
                    mappings.append({"material": material(mat, v.name if p.material in v.materials else None), "variants": [vi]})
                prim["extensions"] = {"KHR_materials_variants": {"mappings": mappings}}
            prims.append(prim)
        g["meshes"].append({"name": mesh.name, "primitives": prims})
        if mesh.primitives and mesh.primitives[0].morph_targets:
            g["meshes"][-1]["weights"] = [0.0] * len(mesh.primitives[0].morph_targets)
        node = {"name": mesh.name, "mesh": len(g["meshes"]) - 1}
        extras = {"lod": mesh.lod, "visible": mesh.visible}
        extras.update(mesh.extras)
        node["extras"] = extras
        if skin_index is not None and any(p.joints is not None for p in mesh.primitives):
            node["skin"] = skin_index
        g["nodes"].append(node)
        mesh_nodes.append(len(g["nodes"]) - 1)

    g["scenes"][0]["nodes"] = root_nodes + mesh_nodes

    # ---- animations ----------------------------------------------------------
    for clip in model.clips:
        samplers, channels = [], []
        for tr in clip.tracks:
            ji = model.skeleton.index_of(tr.joint) if model.skeleton else -1
            if ji < 0 or len(tr.times) == 0:
                continue
            times = tr.times
            samplers.append(
                {
                    "input": b.add_accessor(tr.times, minmax=True, dtype=np.float32),
                    "output": b.add_accessor(tr.values, dtype=np.float32),
                    "interpolation": "LINEAR" if len(times) > 1 else "STEP",
                }
            )
            channels.append({"sampler": len(samplers) - 1, "target": {"node": joint_node[ji], "path": tr.kind}})
        for mesh_name, (mtimes, mweights) in clip.morph.items():
            mi = next((i for i, m in enumerate(model.meshes) if m.name == mesh_name), -1)
            if mi < 0 or len(mtimes) == 0:
                continue
            samplers.append(
                {
                    "input": b.add_accessor(mtimes, minmax=True, dtype=np.float32),
                    "output": b.add_accessor(np.ascontiguousarray(mweights).reshape(-1), dtype=np.float32),
                    "interpolation": "LINEAR",
                }
            )
            channels.append({"sampler": len(samplers) - 1, "target": {"node": mesh_nodes[mi], "path": "weights"}})
        if not channels:
            continue
        g["animations"].append(
            {
                "name": clip.name,
                "samplers": samplers,
                "channels": channels,
                "extras": {
                    "index": clip.index,
                    "loop": clip.loop,
                    "category": clip.category,
                    "frameRate": clip.frame_rate,
                    "frameCount": clip.frame_count,
                    "default": clip.name == model.default_clip,
                },
            }
        )

    # ---- finish --------------------------------------------------------------
    g["buffers"].append({"byteLength": len(b.bin)})
    for k in ("animations", "skins", "extensionsUsed", "images", "textures"):
        if not g[k]:
            del g[k]
    g["extras"] = {
        "game": model.game,
        "id": model.id,
        "displayName": model.display_name,
        "kind": model.kind,
        "variantGroups": _variant_groups(model),
    }
    json_bytes = json.dumps(g, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4:
        json_bytes += b" "
    bin_bytes = bytes(b.bin)
    while len(bin_bytes) % 4:
        bin_bytes += b"\0"
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", GLB_MAGIC, 2, total))
        f.write(struct.pack("<II", len(json_bytes), CHUNK_JSON))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_bytes), CHUNK_BIN))
        f.write(bin_bytes)
    return {
        "bytes": total,
        "meshes": len(g["meshes"]),
        "primitives": sum(len(m["primitives"]) for m in g["meshes"]),
        "nodes": len(g["nodes"]),
        "joints": len(joint_node),
        "materials": len(g["materials"]),
        "images": len(g.get("images", [])),
        "animations": len(g.get("animations", [])),
    }


def _variant_groups(model: Model) -> list[dict]:
    groups: dict[str, dict] = {}
    for vi, v in enumerate(model.variants):
        grp = groups.setdefault(v.group, {"name": v.group, "options": []})
        grp["options"].append({"variant": vi, "name": v.name, "label": v.label or v.name, "default": v.default})
    return list(groups.values())
