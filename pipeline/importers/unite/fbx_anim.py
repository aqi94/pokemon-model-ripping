"""Skeleton + clips read straight from an AssetStudio FBX (UNITE), without Blender's pose conversion.

Blender bones cannot hold scale, so its FBX importer distorts bones with non-uniform scale: Charmander's `Bone_Mouth01`
is keyed (0.01, 1, 1) in every clip (a flattened, hidden mouth part) and came out of Blender as a sheared basis with
scales 0.95-81, which blew the head up into a band. Unity and glTF compose a node the same way (local = T R S, children
inherit the full matrix), so the file's own Lcl values map one to one onto glTF TRS.

Verified on the UNITE rip (2026-11-01): only `Lcl Translation/Rotation/Scaling` are used (no pivots, offsets,
pre/post rotation, geometric transforms; default rotation order XYZ), units are the same as the vertex data (metres
under a cm UnitScaleFactor), Y up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import fbx as fbxr

TICKS_PER_SECOND = 46186158000


@dataclass
class FbxNode:
    id: int
    name: str
    kind: str  # LimbNode | Null | Mesh ...
    t: tuple = (0.0, 0.0, 0.0)
    r: tuple = (0.0, 0.0, 0.0)  # degrees, XYZ order (R = Rz Ry Rx)
    s: tuple = (1.0, 1.0, 1.0)
    pre: tuple = (0.0, 0.0, 0.0)
    post: tuple = (0.0, 0.0, 0.0)
    parent: int = 0
    bind: np.ndarray | None = None  # cluster TransformLink (global bind matrix), column-vector 4x4


@dataclass
class FbxScene:
    nodes: dict[int, FbxNode]
    # stack name -> node id -> channel ("T"/"R"/"S") -> axis (0..2) -> (times s, values)
    stacks: dict[str, dict[int, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]] = field(default_factory=dict)
    # skinned mesh Model name -> (its Model node id, Geometry vertices Nx3 in the Model's local space)
    mesh_binds: dict[str, tuple[int, np.ndarray]] = field(default_factory=dict)
    # node ids that a cluster deforms vertices with
    clustered: set[int] = field(default_factory=set)
    # skinned mesh Model name -> node id of its first cluster's bone, for skins whose clusters ALL lack weights (Unity:
    # a renderer without blend weights rides its bone 0; Lucario's staff on `Bone001`)
    rigid_skins: dict[str, int] = field(default_factory=dict)
    # skinned mesh Model name -> bone node id -> (that mesh's cluster TransformLink, cluster has weights). Unity keeps
    # bindposes per SkinnedMeshRenderer, and meshes sharing a bone may disagree: Miraidon skin 03's body clusters
    # include its mesh node's +1.5 m offset while the outfit's on the same bones do not. `FbxNode.bind` keeps one.
    mesh_cluster_binds: dict[str, dict[int, tuple[np.ndarray, bool]]] = field(default_factory=dict)
    # set by `importer.apply_fbx_skeleton`: FBX node id of each model joint (same order as Skeleton.joints)
    joint_nodes: list[int] = field(default_factory=list)

    def children(self, nid: int) -> list[int]:
        return [n.id for n in self.nodes.values() if n.parent == nid]


def euler_xyz(deg) -> np.ndarray:
    x, y, z = (math.radians(float(v)) for v in deg)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotation_part(r, pre=(0, 0, 0), post=(0, 0, 0)) -> np.ndarray:
    return euler_xyz(pre) @ euler_xyz(r) @ euler_xyz(post).T


def local_matrix(t, r, s, pre=(0, 0, 0), post=(0, 0, 0)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = rotation_part(r, pre, post) @ np.diag([float(v) for v in s])
    m[:3, 3] = [float(v) for v in t]
    return m


def quat_xyzw(m3: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> (x, y, z, w)."""
    t = np.trace(m3)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m3[2, 1] - m3[1, 2]) / s, (m3[0, 2] - m3[2, 0]) / s, (m3[1, 0] - m3[0, 1]) / s
    elif m3[0, 0] > m3[1, 1] and m3[0, 0] > m3[2, 2]:
        s = math.sqrt(1.0 + m3[0, 0] - m3[1, 1] - m3[2, 2]) * 2
        w, x, y, z = (m3[2, 1] - m3[1, 2]) / s, 0.25 * s, (m3[0, 1] + m3[1, 0]) / s, (m3[0, 2] + m3[2, 0]) / s
    elif m3[1, 1] > m3[2, 2]:
        s = math.sqrt(1.0 + m3[1, 1] - m3[0, 0] - m3[2, 2]) * 2
        w, x, y, z = (m3[0, 2] - m3[2, 0]) / s, (m3[0, 1] + m3[1, 0]) / s, 0.25 * s, (m3[1, 2] + m3[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m3[2, 2] - m3[0, 0] - m3[1, 1]) * 2
        w, x, y, z = (m3[1, 0] - m3[0, 1]) / s, (m3[0, 2] + m3[2, 0]) / s, (m3[1, 2] + m3[2, 1]) / s, 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return (x / n, y / n, z / n, w / n)


def decompose(m: np.ndarray) -> tuple[tuple, tuple, tuple]:
    """TRS of a matrix without shear (an ancestor chain with uniform scales)."""
    t = tuple(float(v) for v in m[:3, 3])
    s = np.linalg.norm(m[:3, :3], axis=0)
    r = m[:3, :3] / np.where(s > 1e-12, s, 1.0)
    if np.linalg.det(r) < 0:
        s[0] = -s[0]
        r[:, 0] = -r[:, 0]
    return t, quat_xyzw(r), tuple(float(v) for v in s)


def parse(path: str) -> FbxScene:
    top = {n.name: n for n in fbxr.parse(path)}
    objs, conns = top["Objects"], top["Connections"]
    nodes: dict[int, FbxNode] = {}
    for m in objs.all("Model"):
        p70 = m.child("Properties70")
        props = {q.props[0]: q.props[4:7] for q in (p70.children if p70 else [])}
        nodes[m.props[0]] = FbxNode(
            id=m.props[0], name=fbxr.object_name(m), kind=str(m.props[2]),
            t=tuple(props.get("Lcl Translation", (0, 0, 0))), r=tuple(props.get("Lcl Rotation", (0, 0, 0))),
            s=tuple(props.get("Lcl Scaling", (1, 1, 1))), pre=tuple(props.get("PreRotation", (0, 0, 0))),
            post=tuple(props.get("PostRotation", (0, 0, 0))),
        )
    clusters = {d.props[0]: d for d in objs.all("Deformer") if len(d.props) > 2 and d.props[2] == "Cluster"}
    curve_nodes = {o.props[0]: fbxr.object_name(o) for o in objs.all("AnimationCurveNode")}
    curves = {o.props[0]: o for o in objs.all("AnimationCurve")}
    layers = {o.props[0] for o in objs.all("AnimationLayer")}
    stacks = {o.props[0]: fbxr.object_name(o) for o in objs.all("AnimationStack")}
    cn_target: dict[int, tuple[int, str]] = {}
    cn_layer: dict[int, int] = {}
    layer_stack: dict[int, int] = {}
    cn_curves: dict[int, dict[int, np.ndarray]] = {}
    for c in conns.children:
        p = c.props
        kind, a, b = p[0], p[1], p[2]
        if kind == "OO":
            if a in nodes and (b in nodes or b == 0):
                nodes[a].parent = b
            elif a in clusters and b in nodes:  # bone -> cluster is OO (bone, cluster)
                pass
            elif b in clusters and a in nodes:
                tl = clusters[b].child("TransformLink")
                if tl is not None:
                    nodes[a].bind = np.asarray(tl.props[0], np.float64).reshape(4, 4).T
            elif a in curve_nodes and b in layers:
                cn_layer[a] = b
            elif a in layers and b in stacks:
                layer_stack[a] = b
        elif kind == "OP":
            if a in curve_nodes and b in nodes:
                prop = p[3]
                ch = {"Lcl Translation": "T", "Lcl Rotation": "R", "Lcl Scaling": "S"}.get(prop)
                if ch:
                    cn_target[a] = (b, ch)
            elif a in curves and b in curve_nodes:
                axis = {"d|X": 0, "d|Y": 1, "d|Z": 2}.get(p[3])
                if axis is not None:
                    cv = curves[a]
                    kt = cv.child("KeyTime").props[0].astype(np.float64) / TICKS_PER_SECOND
                    kv = cv.child("KeyValueFloat").props[0].astype(np.float64)
                    cn_curves.setdefault(b, {})[axis] = (kt, kv)
    # clusters may also be linked bone -> cluster in the other order (OO cluster, bone)
    for c in conns.children:
        p = c.props
        if p[0] == "OO" and p[1] in clusters and p[2] in nodes and nodes[p[2]].bind is None:
            tl = clusters[p[1]].child("TransformLink")
            if tl is not None:
                nodes[p[2]].bind = np.asarray(tl.props[0], np.float64).reshape(4, 4).T
    scene = FbxScene(nodes=nodes)
    # skinned meshes: Cluster -> Skin -> Geometry -> Model
    geoms = {g.props[0]: g for g in objs.all("Geometry") if len(g.props) > 2 and g.props[2] == "Mesh"}
    skins = {d.props[0] for d in objs.all("Deformer") if len(d.props) > 2 and d.props[2] == "Skin"}
    cluster_skin: dict[int, int] = {}
    skin_geom: dict[int, int] = {}
    geom_model: dict[int, int] = {}
    for c in conns.children:
        p = c.props
        if p[0] != "OO":
            continue
        a, b = p[1], p[2]
        if a in clusters and b in skins:
            cluster_skin[a] = b
        elif a in skins and b in geoms:
            skin_geom[a] = b
        elif a in geoms and b in nodes:
            geom_model[a] = b
        elif a in nodes and b in clusters:
            scene.clustered.add(a)
        elif a in clusters and b in nodes:
            scene.clustered.add(b)
    cluster_bone: dict[int, int] = {}
    for c in conns.children:
        p = c.props
        if p[0] == "OO" and p[1] in nodes and p[2] in clusters:
            cluster_bone.setdefault(p[2], p[1])
        elif p[0] == "OO" and p[1] in clusters and p[2] in nodes:
            cluster_bone.setdefault(p[1], p[2])
    skin_clusters: dict[int, list[int]] = {}
    for cid, sid in cluster_skin.items():  # connection order = the skin's bone order
        skin_clusters.setdefault(sid, []).append(cid)
    for sid, cids in skin_clusters.items():
        mid = geom_model.get(skin_geom.get(sid, -1))
        if mid is None or mid not in nodes:
            continue

        def weighted(cid: int) -> bool:
            ix = clusters[cid].child("Indexes")
            return ix is not None and len(ix.props) > 0 and len(ix.props[0]) > 0

        own = scene.mesh_cluster_binds.setdefault(nodes[mid].name, {})
        for c in cids:
            tl = clusters[c].child("TransformLink")
            if c in cluster_bone and tl is not None and cluster_bone[c] not in own:
                own[cluster_bone[c]] = (np.asarray(tl.props[0], np.float64).reshape(4, 4).T, weighted(c))
        if not any(weighted(c) for c in cids):
            first = next((cluster_bone[c] for c in cids if c in cluster_bone), None)
            if first is not None:
                scene.rigid_skins[nodes[mid].name] = first
    for cid, sid in cluster_skin.items():
        gid = skin_geom.get(sid)
        mid = geom_model.get(gid)
        vs = geoms[gid].child("Vertices") if gid in geoms else None
        if mid is None or vs is None or nodes[mid].name in scene.mesh_binds:
            continue
        scene.mesh_binds[nodes[mid].name] = (mid, np.asarray(vs.props[0], np.float64).reshape(-1, 3))
    for cn, (nid, ch) in cn_target.items():
        st = stacks.get(layer_stack.get(cn_layer.get(cn, -1), -1))
        if st is None or cn not in cn_curves:
            continue
        scene.stacks.setdefault(st, {}).setdefault(nid, {})[ch] = cn_curves[cn]
    return scene


def bind_space_bounds(scene: FbxScene) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Skinned mesh name -> bbox (min, max) of its vertices placed by the mesh node's rest world (Lcl chain): the space
    the clusters' inverse binds expect them in, since a mesh's own `TransformLink`s equal its bones' Lcl rest world
    (meshes sharing a bone may disagree with each other: `mesh_cluster_binds`, `importer.apply_mesh_binds`). (The
    cluster `Transform` AssetStudio writes is an unscaled bone-relative matrix, not the mesh's bind placement.)"""
    out = {}
    for name, (mid, v) in scene.mesh_binds.items():
        if len(v):
            tf = world_rest(scene, mid)
            w = v @ tf[:3, :3].T + tf[:3, 3]
            out[name] = (w.min(axis=0), w.max(axis=0))
    return out


def bind_space_vertices(scene: FbxScene) -> dict[str, np.ndarray]:
    """Skinned mesh name -> its Geometry vertices placed by the mesh node's rest world (see `bind_space_bounds`)."""
    out = {}
    for name, (mid, v) in scene.mesh_binds.items():
        tf = world_rest(scene, mid)
        out[name] = v @ tf[:3, :3].T + tf[:3, 3]
    return out


def matched_fraction(points: np.ndarray, ref: np.ndarray, tol: float = 1e-3) -> float:
    """Share of `points` that lie within ~`tol` (per axis, on a `tol` grid) of some point of `ref`."""
    if not len(points):
        return 1.0
    if not len(ref):
        return 0.0

    def key(q):
        q = q.astype(np.int64) + (1 << 20)
        return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]

    refk = np.unique(key(np.floor(ref / tol)))
    cell = np.floor(points / tol)
    hit = np.zeros(len(points), bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                hit |= np.isin(key(cell + (dx, dy, dz)), refk)
    return float(hit.mean())


def world_rest(scene: FbxScene, nid: int) -> np.ndarray:
    m = np.eye(4)
    while nid in scene.nodes:
        n = scene.nodes[nid]
        m = local_matrix(n.t, n.r, n.s, n.pre, n.post) @ m
        nid = n.parent
    return m


def clip_times(channels: dict[int, dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]]) -> tuple[np.ndarray, float]:
    """Uniform sample times over the clip at its key rate (smallest key spacing, rounded to a whole fps)."""
    t_max, dt = 0.0, None
    for chans in channels.values():
        for axes in chans.values():
            for kt, _ in axes.values():
                if len(kt):
                    t_max = max(t_max, float(kt[-1]))
                d = np.diff(kt)
                d = d[d > 1e-6]
                if len(d):
                    dt = float(d.min()) if dt is None else min(dt, float(d.min()))
    fps = float(round(1.0 / dt)) if dt else 30.0
    fps = min(max(fps, 1.0), 120.0)
    n = max(1, int(round(t_max * fps)) + 1)
    return np.arange(n, dtype=np.float64) / fps, fps


def sample_channel(node: FbxNode, chans: dict, ch: str, times: np.ndarray) -> np.ndarray:
    rest = {"T": node.t, "R": node.r, "S": node.s}[ch]
    out = np.tile(np.asarray(rest, np.float64), (len(times), 1))
    for axis, (kt, kv) in chans.get(ch, {}).items():
        if len(kt):
            out[:, axis] = np.interp(times, kt, kv)
    return out
