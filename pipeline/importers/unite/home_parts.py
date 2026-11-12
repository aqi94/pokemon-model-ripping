"""Colour for UNITE parts that have no texture anywhere in the rip (`missingTexture`: Fuecoco's head tuft, Skeledirge's
lobby crown...) taken from the same species in Pokémon HOME (user decision 2026-11-09: HOME is the fallback source for
UNITE effect colours; nothing is invented).

Rule (ledger `fx:homeColouredParts`): both models are normalised by their body bounding box at rest (origin = the model origin, unit = larger horizontal extent). The texture-less part is matched to the HOME mesh whose vertices lie nearest to it
(mean nearest distance); each part vertex then takes the HOME texel (x the material's base colour) at its nearest
vertex of that mesh, written as COLOR_0 on an ordinary lit material. Quality gate: mean match distance <= `MAX_MEAN`
and >= `MIN_COVER` of the vertices within `NEAR` of the HOME mesh; the HOME mesh must be opaque and no more than
`MAX_BODY_OVERLAP` of it may lie on the UNITE body (a part with a counterpart, not a shell on the body skin). Parts that fail stay hidden `~notex` parts.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.common.model import Material, Model, TextureImage


REPO = Path(__file__).resolve().parents[3]
HOME_OUT = Path(os.environ.get("POKEMODELS_HOME_OUT", REPO / "out" / "home"))
HOME_FX = REPO / "pipeline" / "catalog" / "unite_home_fx.json"
MAX_SEAM_SHARE = 0.02  # UV transfer is used only when <= 2% of triangles still cross a texture seam after the splits


def home_as_fire(dex: int) -> dict[str, dict]:
    """HOME `_AsFire` materials of a species (`unite_home_fx.json` `asFire`): material name -> record (motion, scale)."""
    try:
        with open(HOME_FX, encoding="utf-8") as f:
            sp = json.load(f)["species"].get(str(int(dex))) or {}
    except OSError:
        return {}
    return {e["material"]: e for e in sp.get("asFire", [])}

# FITTED: match gate (units of the body footprint) on Fuecoco/Crocalor/Skeledirge/Hoopa tufts and the 83 texture-less materials, 2026-11-09
MAX_MEAN = 0.06
NEAR = 0.10
MIN_COVER = 0.90
BACK_NEAR = 0.05
MIN_BACK_COVER = 0.7  # FITTED: share of the HOME mesh lying on the UNITE part, 2026-11-09 (Fuecoco tuft 1.0, Skeledirge flames 1.0)
SANE_EXTENT = 1.5  # raw part extent / body extent above which positions are carried to the rest pose
BODY_NEAR = 0.02
MAX_BODY_OVERLAP = 0.7  # FITTED: share of the HOME mesh within BODY_NEAR of the UNITE body (body skins 0.8-1.0, tufts/crowns 0.15-0.6) on Fuecoco, Skeledirge, Charizard, 2026-11-09


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _accessor(g, blob: bytes, idx: int) -> np.ndarray:
    a = g.accessors[idx]
    bv = g.bufferViews[a.bufferView]
    ncomp = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a.type]
    dt = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[a.componentType]
    itemsize = np.dtype(dt).itemsize * ncomp
    stride = bv.byteStride or itemsize
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    raw = np.frombuffer(blob, np.uint8, count=stride * a.count, offset=off).reshape(a.count, stride)[:, :itemsize]
    out = np.ascontiguousarray(raw).view(dt).reshape(a.count, ncomp).astype(np.float32)
    if a.normalized:
        out /= float(np.iinfo(dt).max)
    return out


def load_home_meshes(glb: Path) -> list[dict]:
    """[{name, pos (V,3), uv (V,2), factor (4,), image (H,W,4) uint8 | None}] per HOME primitive (variant 0)."""
    from pygltflib import GLTF2
    g = GLTF2().load(str(glb))
    blob = g.binary_blob()
    out = []
    for m in g.meshes:
        for p in m.primitives:
            if p.attributes.POSITION is None:
                continue
            mat = g.materials[p.material] if p.material is not None else None
            pbr = mat.pbrMetallicRoughness if mat is not None else None
            factor = np.array(pbr.baseColorFactor if pbr is not None and pbr.baseColorFactor else [1, 1, 1, 1], np.float32)
            img = None
            if pbr is not None and pbr.baseColorTexture is not None:
                src = g.textures[pbr.baseColorTexture.index].source
                bv = g.bufferViews[g.images[src].bufferView]
                img = np.asarray(Image.open(io.BytesIO(blob[(bv.byteOffset or 0):(bv.byteOffset or 0) + bv.byteLength])).convert("RGBA"))
            uv = _accessor(g, blob, p.attributes.TEXCOORD_0) if p.attributes.TEXCOORD_0 is not None else None
            hx = ((mat.extras or {}).get("home") or {}) if mat is not None else {}
            out.append({"name": m.name, "blend": (mat.alphaMode if mat is not None else "OPAQUE") == "BLEND", "pos": _accessor(g, blob, p.attributes.POSITION), "uv": uv, "factor": factor, "image": img,
                        "material": mat.name if mat is not None else None, "asFire": bool((hx.get("floats") or {}).get("_AsFire"))})
    return out


def _norm_frame(points: np.ndarray) -> tuple[np.ndarray, float]:
    lo, hi = points.min(0), points.max(0)
    # unit = the larger horizontal extent: a head tuft/crest adds height in one game's body but is a separate part in the
    # other (Fuecoco's tuft is 33% of HOME's height, none of UNITE's body bounds), while the footprint is the same
    # origin = the model's own origin (feet at y 0, centred, in both games: a bbox bottom/centre is thrown off by stray
    # meshes such as Skeledirge's HOME bird B, which hangs 1.2 m below the floor)
    return np.zeros(3), max(float(hi[0] - lo[0]), float(hi[2] - lo[2]), 1e-6)


def _nearest(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index into b and distance of the nearest b point for every a point (chunked brute force)."""
    idx = np.empty(len(a), np.int64)
    dist = np.empty(len(a), np.float64)
    for s in range(0, len(a), 512):
        d = ((a[s:s + 512, None, :] - b[None, :, :]) ** 2).sum(-1)
        k = d.argmin(1)
        idx[s:s + 512] = k
        dist[s:s + 512] = np.sqrt(d[np.arange(len(k)), k])
    return idx, dist


def _joint_world(sk) -> list[np.ndarray]:
    out: list[np.ndarray | None] = [None] * len(sk.joints)

    def world(i: int) -> np.ndarray:
        if out[i] is None:
            j = sk.joints[i]
            x, y, z, w = j.rotation
            r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
            m = np.eye(4)
            m[:3, :3] = r * np.asarray(j.scale)[None, :]
            m[:3, 3] = j.translation
            out[i] = (world(j.parent) @ m) if j.parent >= 0 else m
        return out[i]

    return [world(i) for i in range(len(sk.joints))]


def rest_positions(model: Model, mesh, prim, world: list[np.ndarray] | None = None, raw_extent: float | None = None) -> np.ndarray:
    """Vertex positions for matching with HOME (whose raw positions are bind-pose body space). The raw positions are
    used while they lie inside the body's extent; a skinned part whose raw (mesh-space) positions are far larger (a
    mesh/bone scale: Fuecoco skin 01's tuft is 38x too large) is carried to the rest pose by its joints instead."""
    pos = np.asarray(prim.positions, np.float64)
    sk = model.skeleton
    if sk is None or prim.joints is None or prim.weights is None:
        return pos
    if raw_extent is not None and float(np.ptp(pos, axis=0).max()) <= SANE_EXTENT * raw_extent:
        return pos
    world = world or _joint_world(sk)
    cache: dict[int, np.ndarray] = {}

    def mat(k: int) -> np.ndarray:
        if k not in cache:
            ib = mesh.inverse_binds.get(k) if mesh.inverse_binds else None
            if ib is None:
                ib = sk.joints[sk.palette[k]].inverse_bind
            cache[k] = np.eye(4) if ib is None else world[sk.palette[k]] @ np.asarray(ib, np.float64)
        return cache[k]

    homo = np.concatenate([pos, np.ones((len(pos), 1))], 1)
    out = np.zeros_like(pos)
    for c in range(prim.joints.shape[1]):
        w = prim.weights[:, c].astype(np.float64)
        if not (w > 0).any():
            continue
        for k in np.unique(prim.joints[w > 0, c]):
            sel = (prim.joints[:, c] == k) & (w > 0)
            out[sel] += w[sel, None] * (homo[sel] @ mat(int(k)).T)[:, :3]
    tot = prim.weights.sum(1)
    out[tot <= 0] = pos[tot <= 0]
    return out


def _candidates(q: np.ndarray, body_pos: np.ndarray, uo, us, ho, hs, home: list[dict]) -> list[dict]:
    """HOME meshes that may be this part's counterpart: textured albedo meshes off the UNITE body surface, most of whose
    vertices lie on the part."""
    out = []
    bq = (body_pos - uo) / us
    for h in home:
        if h["uv"] is None or h["image"] is None or h.get("blend"):  # baked flame/smoke cards are not albedo
            continue
        hp = (h["pos"] - ho) / hs
        # the counterpart must be a part of HOME's model the UNITE body does not have: a mesh lying on the UNITE
        # body surface (the body skin itself) is no counterpart for a shell/glow card hugging it
        _, back = _nearest(hp[:: max(1, len(hp) // 400)], bq[:: max(1, len(bq) // 1500)])
        if float((back <= BODY_NEAR).mean()) > MAX_BODY_OVERLAP:
            continue
        # mutual: the HOME mesh must be the same part, so most of it must lie on the UNITE part too (a glow shell
        # covers only a patch of a big HOME mesh; a tuft or crown is covered whole)
        _, back_part = _nearest(hp[:: max(1, len(hp) // 400)], q[:: max(1, len(q) // 1500)])
        if float((back_part <= BACK_NEAR).mean()) < MIN_BACK_COVER:
            continue
        out.append({"mesh": h, "hp": hp})
    return out


def match_part(part_pos: np.ndarray, body_pos: np.ndarray, home: list[dict]) -> dict | None:
    """Match one texture-less part (rest positions) to HOME; returns the per-vertex linear colours, per-vertex HOME
    mesh/vertex indices and the gate values, or None when HOME has no usable mesh. `body_pos` = the UNITE body's rest
    vertices (for normalisation).

    One UNITE mesh may hold several HOME meshes (Skeledirge's lobby `fx_lv004` = HOME's three birds `BirdA/B/CSkin`, of
    which the idle clip shows one): every candidate counterpart (each lying on the part) is tried alone and all of them
    together (each vertex takes its nearest vertex of any of them); the lowest mean distance wins."""
    home_all = np.concatenate([h["pos"] for h in home])
    uo, us = _norm_frame(body_pos)
    ho, hs = _norm_frame(home_all)
    q = (part_pos - uo) / us
    cands = _candidates(q, body_pos, uo, us, ho, hs, home)
    if not cands:
        return None
    sets = [[c] for c in cands] + ([cands] if len(cands) > 1 else [])
    best = None
    for group in sets:
        pts = np.concatenate([c["hp"] for c in group])
        owner = np.concatenate([np.full(len(c["hp"]), i) for i, c in enumerate(group)])
        local = np.concatenate([np.arange(len(c["hp"])) for c in group])
        idx, dist = _nearest(q, pts)
        rec = {"group": group, "mesh_of": owner[idx], "idx": local[idx], "dist": dist,
               "mean": float(dist.mean()), "cover": float((dist <= NEAR).mean())}
        if best is None or rec["mean"] < best["mean"] - 1e-9:
            best = rec
    col = np.ones((len(q), 4), np.float32)
    for i, c in enumerate(best["group"]):
        sel = best["mesh_of"] == i
        if not sel.any():
            continue
        h = c["mesh"]
        img = h["image"]
        uv = h["uv"][best["idx"][sel]] % 1.0
        x = np.clip((uv[:, 0] * img.shape[1]).astype(int), 0, img.shape[1] - 1)
        y = np.clip((uv[:, 1] * img.shape[0]).astype(int), 0, img.shape[0] - 1)
        rgb = _srgb_to_linear(img[y, x, :3].astype(np.float32) / 255.0) * h["factor"][:3]
        col[sel, :3] = np.clip(rgb, 0, 1)
    meshes = [c["mesh"] for c in best["group"]]
    return {"colors": col, "homeMesh": "+".join(h["name"] for h in meshes), "homeMeshes": meshes,
            "meshOf": best["mesh_of"], "homeIndex": best["idx"], "dist": best["dist"],
            "matchDistance": best["mean"], "cover": best["cover"],
            "ok": best["mean"] <= MAX_MEAN and best["cover"] >= MIN_COVER}


SEAM_SPAN = 0.5  # a triangle whose transferred UVs span more than this crosses a texture seam
SEAM_K = 12  # nearby HOME vertices a seam corner may take its UV from


def transfer_uvs(part_pos: np.ndarray, indices: np.ndarray, hmesh: dict, idx: np.ndarray, dist: np.ndarray,
                 us: float, hs: float, uo=None, ho=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """UV0 for a part that IS a HOME mesh (same surface, possibly a coarser tessellation): each vertex takes the UV of
    a nearby HOME vertex. HOME splits vertices at UV seams, so a spot can carry several UVs; the copy agreeing with the
    vertex's triangle neighbours is chosen (relaxation passes). A triangle still spanning a seam gets its own copies of
    the odd vertices, each taking the UV of whichever of its `SEAM_K` nearest HOME vertices lies closest (in UV) to the
    triangle's other corners. Returns (uv, new indices, source vertex of every output vertex, share of triangles still
    crossing a seam)."""
    huv = np.asarray(hmesh["uv"], np.float64)
    hp = (np.asarray(hmesh["pos"], np.float64) - (0 if ho is None else ho)) / hs
    q = (np.asarray(part_pos, np.float64) - (0 if uo is None else uo)) / us
    uv = huv[idx].copy()
    near: list[np.ndarray] = []  # SEAM_K nearest HOME vertices per part vertex, nearest first
    same: list[np.ndarray] = []  # those as near as the nearest
    for i in range(len(q)):
        d = np.sqrt(((hp - q[i]) ** 2).sum(-1))
        k = np.argsort(d)[:SEAM_K]
        near.append(k)
        same.append(k[d[k] <= dist[i] + 1e-4])
    tri = np.asarray(indices, np.int64).reshape(-1, 3).copy()
    nbr: list[set] = [set() for _ in range(len(q))]
    for a, b, c in tri:
        nbr[a] |= {b, c}; nbr[b] |= {a, c}; nbr[c] |= {a, b}
    multi = [i for i in range(len(q)) if len(same[i]) > 1]
    for _ in range(4):
        for i in multi:
            if not nbr[i]:
                continue
            nb = uv[list(nbr[i])]
            opts = huv[same[i]]
            cost = np.abs(opts[:, None, :] - nb[None, :, :]).sum(-1).sum(-1)
            uv[i] = opts[int(cost.argmin())]
    src = list(range(len(q)))
    uvs = [u for u in uv]
    for t in np.nonzero(np.ptp(uv[tri], axis=1).max(-1) > SEAM_SPAN)[0]:
        corners = tri[t]
        # keep the corner pair that agrees best; re-pick the odd corner from its nearby HOME vertices
        pairs = [(0, 1, 2), (1, 2, 0), (0, 2, 1)]
        a, b, odd = min(pairs, key=lambda p: np.abs(uv[corners[p[0]]] - uv[corners[p[1]]]).sum())
        ref = (uv[corners[a]] + uv[corners[b]]) / 2
        v = corners[odd]
        opts = huv[near[v]]
        best = opts[int(np.abs(opts - ref).sum(-1).argmin())]
        if np.abs(best - uv[v]).sum() > 1e-6:
            src.append(int(v))
            uvs.append(best)
            tri[t, odd] = len(src) - 1
    uv_out = np.asarray(uvs, np.float32)
    span = np.ptp(uv_out[tri], axis=1).max(-1)
    return uv_out, tri.reshape(-1).astype(np.uint32), np.asarray(src, np.int64), float((span > SEAM_SPAN).mean())


def home_glb(dex: int, ensure: bool = True) -> Path | None:
    """HOME's form-00 model for a species (converted on demand into out/home)."""
    mid = f"pm{int(dex):04d}_00_00"
    glb = HOME_OUT / mid / "model.glb"
    if not glb.exists() and ensure:
        try:
            subprocess.run([sys.executable, "-m", "pipeline", "--game", "home", "convert", "--id", mid],
                           cwd=REPO, check=False, capture_output=True, timeout=600)
        except Exception:  # noqa: BLE001
            pass
    return glb if glb.exists() else None


def colour_from_home(model: Model, dex: int | None) -> dict:
    """Give every texture-less primitive that passes the gate HOME colours as COLOR_0 on an ordinary lit material
    (no `missingTexture`, so `split_missing_texture_parts` leaves it visible). Records `model.extras['homeColouredParts']`
    and removes the parts from the 'imported hidden' warning. Returns {"done": [...], "failed": [...], "reason": str|None}."""
    # texture-less parts, and parts whose own texture is a sparse overlay (`sparseTexture`, importer.flag_sparse_effects)
    todo = [(m, p) for m in model.meshes for p in m.primitives
            if model.materials[p.material].extras.get("missingTexture") or "sparseTexture" in model.materials[p.material].extras]
    res: dict = {"done": [], "failed": [], "reason": None}
    if not todo:
        return res
    if not dex:
        res["reason"] = "no dex number"
        return res
    glb = home_glb(dex)
    if glb is None:
        res["reason"] = f"HOME pm{int(dex):04d}_00_00 not available"
        return res
    home = load_home_meshes(glb)
    miss_ids = {id(p) for _, p in todo}
    world = _joint_world(model.skeleton) if model.skeleton is not None else None
    body = [p.positions for m in model.meshes for p in m.primitives
            if id(p) not in miss_ids and m.visible and not model.materials[p.material].extras.get("effect")]
    if not body:
        res["reason"] = "no body mesh"
        return res
    body_pos = np.concatenate(body)
    extent = float(np.ptp(body_pos, axis=0).max())
    home_id = glb.parent.name
    as_fire = home_as_fire(dex)
    made: dict[tuple[str, str], str] = {}
    for mesh, p in todo:
        old = model.materials[p.material]
        rest = rest_positions(model, mesh, p, world, extent)
        r = match_part(rest, body_pos, home)
        label = f"{mesh.name}/{old.name}"
        if r is None or not r["ok"]:
            if not old.extras.get("missingTexture"):
                continue  # a sparse-texture part HOME cannot match keeps its own texture
            res["failed"].append({"part": label, "matchDistance": None if r is None else round(r["matchDistance"], 4),
                                  "cover": None if r is None else round(r["cover"], 3)})
            continue
        fire_uv = None
        hm = r["homeMeshes"][0]
        if len(r["homeMeshes"]) == 1 and hm.get("asFire") and hm["material"] in as_fire:
            # painted HOME flames (`_AsFire`: Skeledirge's `FlameSkin`): the part IS that mesh, so it takes HOME's own UVs
            # and texture, and the viewer scrolls it at the loop clip's speed (ledger `fx:homeMotion`)
            _, us = _norm_frame(body_pos)
            _, hs = _norm_frame(np.concatenate([h["pos"] for h in home]))
            uv, new_idx, src, seam = transfer_uvs(rest, p.indices, hm, r["homeIndex"], r["dist"], us, hs)
            if seam <= MAX_SEAM_SHARE:
                fire_uv = (uv, new_idx, src)
            else:
                model.extras.setdefault("warnings", []).append(f"{mesh.name}: HOME UV transfer crosses seams ({seam:.1%}), colours only")
        key = (old.name, r["homeMesh"])
        name = made.get(key)
        if name is None:
            name = f"{old.name}.home"
            while name in model.materials:
                name += "_"
            rec = {"source": "home", "homeModel": home_id, "homeMesh": r["homeMesh"], "matchDistance": round(r["matchDistance"], 4)}
            ex = {k: v for k, v in old.extras.items() if k not in ("effect", "solidEffect", "missingTexture", "blend", "vertexColour")}
            if "sparseTexture" in old.extras:
                ex["replacedTexture"] = old.emissive or old.base_color  # the UNITE overlay mask HOME's colour replaces
            ex.setdefault("unite", {})["colourFrom"] = rec
            mat = Material(name=name, base_color_factor=(1.0, 1.0, 1.0, 1.0), roughness_factor=1.0,
                           double_sided=True, extras=ex)
            if fire_uv is not None:
                rec_f = as_fire[hm["material"]]
                tkey = f"home:{home_id}:{hm['material']}"
                if tkey not in model.textures:
                    model.textures[tkey] = TextureImage(key=tkey, rgba=np.ascontiguousarray(hm["image"]), srgb=True)
                # INFERRED: `_AsFire` = self-lit flame paint (base black, colour as emission); HOME's shader is not in the rip
                mat.base_color = tkey
                mat.emissive = tkey
                mat.base_color_factor = (0.0, 0.0, 0.0, 1.0)
                mat.emissive_factor = (1.0, 1.0, 1.0)
                mot = rec_f.get("motion") or {}
                cu, cv = (mot.get("ColorUVTranslate") or [0.0, 0.0])[:2]
                # glTF UV offset per second, as the eye helpers (home importer `_uv_helper_offsets`: U / the baked tiling)
                ex["homeFire"] = {"material": hm["material"], "scroll": [round(cu / max(1.0, rec_f["colScale"][0]), 5), round(cv, 5)],
                                  "loop": mot.get("loop"),
                                  # not drawn: the displacement panner's operator (`_DISPOPE_SIN`) is not in the rip
                                  "displaceGap": {"height": rec_f.get("displaceHeight"), "speed": mot.get("DisplaceUVTranslate")}}
            model.materials[name] = mat
            made[key] = name
        p.material = name
        if fire_uv is not None and model.materials[name].extras.get("homeFire"):
            uv, new_idx, src = fire_uv
            # seam corners were duplicated: every per-vertex attribute follows its source vertex
            for attr in ("positions", "normals", "joints", "weights", "flame_t", "gas_weight"):
                a = getattr(p, attr)
                if a is not None:
                    setattr(p, attr, np.ascontiguousarray(np.asarray(a)[src]))
            p.morph_targets = [np.ascontiguousarray(np.asarray(m)[src]) for m in p.morph_targets]
            p.indices = new_idx
            p.uvs = [uv]
            p.colors = None
            p.tangents = None
        else:
            p.colors = r["colors"]
        res["done"].append({"part": label, "homeModel": home_id, "homeMesh": r["homeMesh"],
                            "matchDistance": round(r["matchDistance"], 4), "cover": round(r["cover"], 3)})
    used = {p.material for m in model.meshes for p in m.primitives}
    model.materials = {k: v for k, v in model.materials.items() if k in used}
    if res["done"]:
        model.extras["homeColouredParts"] = res["done"]
        done = {d["part"] for d in res["done"]}
        ws = model.extras.get("warnings", [])
        pre = "effect parts with no texture in the rip, imported hidden: "
        out = []
        for w in ws:
            if w.startswith(pre):
                left = [x for x in w[len(pre):].split("; ") if x not in done]
                if left:
                    out.append(pre + "; ".join(left))
            else:
                out.append(w)
        out.append("texture-less parts coloured from Pokémon HOME (home_parts): " + "; ".join(
            f"{d['part']} <- {d['homeModel']}/{d['homeMesh']} (d={d['matchDistance']})" for d in res["done"]))
        model.extras["warnings"] = out
    if res["failed"]:
        model.extras.setdefault("warnings", []).append(
            "texture-less parts HOME could not colour (gate): " + "; ".join(
                f"{f['part']} (d={f['matchDistance']}, cover={f['cover']})" for f in res["failed"]))
    return res


if __name__ == "__main__":  # survey: python -m pipeline.importers.unite.home_parts  (reads out/unite manifests)
    print(json.dumps({"home_out": str(HOME_OUT)}))
