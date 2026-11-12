"""Data scan of converted GLBs for two render faults that show as black/white parts and z-fighting.

    ./.venv/Scripts/python.exe tools/unite_material_scan.py [--game unite] [--id ...] [--out out/unite_holowear_audit/scan.json]

Per visible primitive (node extras.visible, LOD 0, default material) the albedo is sampled at the primitive's own UVs
(KHR_texture_transform applied, repeat wrap) x baseColorFactor, giving the colour the surface can show at best:
  black      mean sRGB luminance < 0.06 (non-effect)           -> a broken or unread colour source is likely
  white      mean > 0.93 and chroma < 0.03 (non-effect)
  flat       no baseColorTexture on a non-effect material       -> one colour over the whole part
  offAtlas   >= 30% of the UV samples hit alpha < 0.05 texels of an OPAQUE material (UVs on empty atlas space)
  metalDark  mean metallic >= 0.5 and no unite env/matcap term   -> a metal with nothing to reflect renders dark
Z-fighting: triangle centroids of every visible opaque/mask primitive are tested against the planes of nearby
triangles (5 mm cells, samples every <= 2.5 mm) of every OTHER visible primitive: plane distance < 0.4 mm x
max(1, height) and |n.n'| > 0.97 counts as coincident. Reported per primitive pair as the coincident area fraction.
Thresholds only pick candidates for review; they are not import rules.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys

import numpy as np
from PIL import Image
from pygltflib import GLTF2

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def accessor(g: GLTF2, blob: bytes, idx: int) -> np.ndarray:
    a = g.accessors[idx]
    bv = g.bufferViews[a.bufferView]
    dt = np.dtype(COMP[a.componentType])
    n = NCOMP[a.type]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    stride = bv.byteStride or dt.itemsize * n
    if stride == dt.itemsize * n:
        arr = np.frombuffer(blob, dt, a.count * n, off).reshape(a.count, n)
    else:
        raw = np.frombuffer(blob, np.uint8, stride * (a.count - 1) + dt.itemsize * n, off)
        arr = np.lib.stride_tricks.as_strided(raw, (a.count, dt.itemsize * n), (stride, 1)).copy().view(dt).reshape(a.count, n)
    arr = arr.astype(np.float32)
    if a.normalized:
        arr /= float(np.iinfo(dt).max)
    return arr


def node_world(g: GLTF2) -> dict[int, np.ndarray]:
    parent = {c: i for i, n in enumerate(g.nodes) for c in (n.children or [])}
    cache: dict[int, np.ndarray] = {}

    def local(n) -> np.ndarray:
        if n.matrix:
            return np.array(n.matrix, np.float64).reshape(4, 4).T
        t = np.array(n.translation or [0, 0, 0], np.float64)
        x, y, z, w = n.rotation or [0, 0, 0, 1]
        s = np.array(n.scale or [1, 1, 1], np.float64)
        r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        m = np.eye(4)
        m[:3, :3] = r * s
        m[:3, 3] = t
        return m

    def world(i: int) -> np.ndarray:
        if i not in cache:
            m = local(g.nodes[i])
            cache[i] = world(parent[i]) @ m if i in parent else m
        return cache[i]

    return {i: world(i) for i in range(len(g.nodes))}


def srgb_lin(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def scan(glb: str) -> dict:
    g = GLTF2().load(glb)
    blob = g.binary_blob()
    images: dict[int, np.ndarray] = {}

    def image(tex_index: int) -> np.ndarray | None:
        src = g.textures[tex_index].source
        if src is None:
            return None
        if src not in images:
            im = g.images[src]
            bv = g.bufferViews[im.bufferView]
            data = blob[(bv.byteOffset or 0):(bv.byteOffset or 0) + bv.byteLength]
            images[src] = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"), np.float32) / 255.0
        return images[src]

    def sample(img: np.ndarray, uv: np.ndarray, info) -> np.ndarray:
        ext = (info.extensions or {}).get("KHR_texture_transform") if info is not None else None
        if ext:
            uv = uv * np.array(ext.get("scale", [1, 1])) + np.array(ext.get("offset", [0, 0]))
        h, w = img.shape[:2]
        x = (np.floor((uv[:, 0] % 1.0) * w).astype(int)) % w
        y = (np.floor((uv[:, 1] % 1.0) * h).astype(int)) % h
        return img[y, x]

    worlds = node_world(g)
    prims = []  # dicts with geometry for the z-fight test
    flags = []
    allpos = []
    for ni, n in enumerate(g.nodes):
        if n.mesh is None:
            continue
        ex = n.extras or {}
        if ex.get("visible") is False or (ex.get("lod") or 0) > 0:
            continue
        mesh = g.meshes[n.mesh]
        for pi, p in enumerate(mesh.primitives):
            if p.material is None:
                continue
            mat = g.materials[p.material]
            mex = mat.extras or {}
            pos = accessor(g, blob, p.attributes.POSITION).astype(np.float64)
            if n.skin is None or p.attributes.JOINTS_0 is None:
                m = worlds[ni]
                pos = pos @ m[:3, :3].T + m[:3, 3]
            else:  # rest pose: sum_k w_k * (joint world @ inverse bind) @ v
                sk = g.skins[n.skin]
                ib = accessor(g, blob, sk.inverseBindMatrices).reshape(-1, 4, 4).transpose(0, 2, 1).astype(np.float64)
                jm = np.stack([worlds[j] for j in sk.joints]) @ ib
                js = accessor(g, blob, p.attributes.JOINTS_0).astype(np.int64)
                ws = accessor(g, blob, p.attributes.WEIGHTS_0).astype(np.float64)
                ws /= np.maximum(ws.sum(1, keepdims=True), 1e-9)
                m = (jm[js] * ws[:, :, None, None]).sum(1)
                pos = np.einsum("nij,nj->ni", m[:, :3, :3], pos) + m[:, :3, 3]
            idx = accessor(g, blob, p.indices).astype(np.int64).reshape(-1, 3) if p.indices is not None \
                else np.arange(len(pos)).reshape(-1, 3)
            allpos.append(pos)
            key = f"{n.name}/{mat.name}"
            effect = bool(mex.get("effect")) or mat.alphaMode == "BLEND"
            pbr = mat.pbrMetallicRoughness
            fac = np.array(pbr.baseColorFactor or [1, 1, 1, 1], np.float32)
            f: dict = {"prim": key, "alpha": mat.alphaMode, "effect": effect}
            uv = accessor(g, blob, p.attributes.TEXCOORD_0) if p.attributes.TEXCOORD_0 is not None else None
            tri = idx
            cent_uv = None
            if uv is not None:
                cent_uv = uv[tri].mean(axis=1)
                uvs = np.concatenate([uv[np.unique(tri)], cent_uv])
            if pbr.baseColorTexture is not None and uv is not None:
                img = image(pbr.baseColorTexture.index)
                if img is not None:
                    s = sample(img, uvs, pbr.baseColorTexture)
                    rgb = s[:, :3] * fac[:3]
                    f["mean"] = [round(float(v), 3) for v in rgb.mean(axis=0)]
                    lum = float((srgb_lin(rgb) @ np.array([0.2126, 0.7152, 0.0722])).mean())
                    f["lum"] = round(float(lum) ** (1 / 2.2), 3)
                    f["chroma"] = round(float((rgb.max(axis=1) - rgb.min(axis=1)).mean()), 3)
                    if mat.alphaMode == "OPAQUE" and img[..., 3].min() < 0.05:
                        f["offAtlas"] = round(float((s[:, 3] < 0.05).mean()), 3)
            else:
                f["flat"] = True
                f["mean"] = [round(float(v), 3) for v in fac[:3]]
                f["lum"] = round(float(srgb_lin(fac[:3]) @ np.array([0.2126, 0.7152, 0.0722])) ** (1 / 2.2), 3)
                f["chroma"] = round(float(fac[:3].max() - fac[:3].min()), 3)
            mr = pbr.metallicRoughnessTexture
            metal = float(pbr.metallicFactor if pbr.metallicFactor is not None else 1.0)
            if mr is not None and uv is not None:
                img = image(mr.index)
                if img is not None:
                    metal *= float(sample(img, uvs, mr)[:, 2].mean())
            f["metal"] = round(metal, 3)
            unite = mex.get("unite") or {}
            f["env"] = bool(unite.get("env") or unite.get("matTex") or mex.get("matcap"))
            issues = []
            if not effect:
                if f.get("flat") and not mex.get("homeColour") and not mex.get("unite", {}).get("colourFrom"):
                    issues.append("flat")
                if f.get("lum", 1) < 0.06:
                    issues.append("black")
                if f.get("lum", 0) > 0.93 and f.get("chroma", 1) < 0.03:
                    issues.append("white")
                if f.get("offAtlas", 0) >= 0.3:
                    issues.append("offAtlas")
                if metal >= 0.5 and not f["env"]:
                    issues.append("metalDark")
            f["issues"] = issues
            flags.append(f)
            if mat.alphaMode != "BLEND" and not effect and len(tri):
                a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
                cr = np.cross(b - a, c - a)
                area = np.linalg.norm(cr, axis=1)
                ok = area > 1e-12
                prims.append({"key": key, "a": a[ok], "b": b[ok], "c": c[ok], "n": cr[ok] / area[ok, None],
                              "area": area[ok] / 2, "two": bool(mat.doubleSided)})
    height = float(np.ptp(np.concatenate(allpos)[:, 1])) if allpos else 1.0
    return {"height": round(height, 3), "prims": flags, "zfight": zfight(prims, height)}


def zfight(prims: list[dict], height: float) -> list[dict]:
    if len(prims) < 2:
        return []
    cell = 0.005 * max(1.0, height)
    tol = 0.0004 * max(1.0, height)
    # B samples: barycentric grid per triangle, spacing <= cell / 2.
    bk, bt, bp = [], [], []
    for pid, p in enumerate(prims):
        e = np.maximum(np.linalg.norm(p["b"] - p["a"], axis=1), np.linalg.norm(p["c"] - p["a"], axis=1))
        k = np.clip(np.ceil(e / (cell / 2)).astype(int), 1, 12)
        for kk in np.unique(k):
            sel = np.nonzero(k == kk)[0]
            ij = [(i, j) for i in range(kk + 1) for j in range(kk + 1 - i)]
            w = np.array([(i / kk, j / kk) for i, j in ij])
            pts = (p["a"][sel, None] * (1 - w[:, 0] - w[:, 1])[None, :, None] + p["b"][sel, None] * w[None, :, 0, None]
                   + p["c"][sel, None] * w[None, :, 1, None]).reshape(-1, 3)
            tri = np.repeat(sel, len(ij))
            bk.append(np.floor(pts / cell).astype(np.int64))
            bt.append(np.stack([np.full(len(tri), pid), tri], 1))
            bp.append(pts)
    bk = np.concatenate(bk)
    bt = np.concatenate(bt)
    key = (bk[:, 0] * 73856093) ^ (bk[:, 1] * 19349663) ^ (bk[:, 2] * 83492791)
    order = np.argsort(key, kind="stable")
    key, bt = key[order], bt[order]
    out = []
    for pid, p in enumerate(prims):
        cen = (p["a"] + p["b"] + p["c"]) / 3
        ck = np.floor(cen / cell).astype(np.int64)
        q = (ck[:, 0] * 73856093) ^ (ck[:, 1] * 19349663) ^ (ck[:, 2] * 83492791)
        lo = np.searchsorted(key, q, "left")
        hi = np.searchsorted(key, q, "right")
        cnt = hi - lo
        if not cnt.sum():
            continue
        cnt = np.minimum(cnt, 64)
        ai = np.repeat(np.arange(len(q)), cnt)
        start = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        other = bt[start]
        keep = other[:, 0] != pid
        ai, other = ai[keep], other[keep]
        if not len(ai):
            continue
        hit = np.zeros(len(ai), bool)
        for qid in np.unique(other[:, 0]):
            s = other[:, 0] == qid
            o = prims[qid]
            ti = other[s, 1]
            nb = o["n"][ti]
            d = np.abs(((cen[ai[s]] - o["a"][ti]) * nb).sum(1))
            al = np.abs((p["n"][ai[s]] * nb).sum(1))
            hit[np.nonzero(s)[0]] = (d < tol) & (al > 0.97)
        for qid in np.unique(other[hit, 0]):
            tris = np.unique(ai[hit & (other[:, 0] == qid)])
            frac = float(p["area"][tris].sum() / p["area"].sum())
            if frac >= 0.005:
                out.append({"a": p["key"], "b": prims[qid]["key"], "areaFrac": round(frac, 4),
                            "area_cm2": round(float(p["area"][tris].sum()) * 1e4, 2), "tris": int(len(tris))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="unite")
    ap.add_argument("--id", action="append", default=[])
    ap.add_argument("--out", default="out/unite_holowear_audit/scan.json")
    a = ap.parse_args()
    ids = a.id or sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"out/{a.game}/*/model.glb"))
    res = {}
    for k, i in enumerate(ids):
        try:
            res[i] = scan(f"out/{a.game}/{i}/model.glb")
        except Exception as e:  # keep going; report
            res[i] = {"error": f"{type(e).__name__}: {e}"}
        if k % 50 == 0:
            print(k, i, file=sys.stderr, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=0)
    n_issue = sum(1 for r in res.values() for p in r.get("prims", []) if p["issues"])
    n_z = sum(len(r.get("zfight", [])) for r in res.values())
    print(f"{len(res)} models, {n_issue} flagged primitives, {n_z} z-fight pairs -> {a.out}")


if __name__ == "__main__":
    main()
