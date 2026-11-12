"""Minimal binary FBX reader (node tree + arrays), enough to check what Blender's importer cannot tell us.

Used for the UNITE AssetStudio exports: some meshes carry a broken material layer (see `material_layers`), which Blender
silently reads as "every triangle uses material 0".
"""

from __future__ import annotations

import functools
import re as _re
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"Kaydara FBX Binary  "
_SCALAR = {"Y": "<h", "C": "<?", "I": "<i", "F": "<f", "D": "<d", "L": "<q"}
_ARRAY = {"i": np.int32, "d": np.float64, "f": np.float32, "l": np.int64, "b": np.bool_}


@dataclass
class Node:
    name: str
    props: list
    children: list["Node"] = field(default_factory=list)

    def child(self, name: str) -> "Node | None":
        return next((c for c in self.children if c.name == name), None)

    def all(self, name: str) -> list["Node"]:
        return [c for c in self.children if c.name == name]


def _read_array(b: bytes, o: int, t: str) -> tuple[np.ndarray, int]:
    n, enc, clen = struct.unpack_from("<III", b, o)
    o += 12
    raw = b[o : o + clen]
    if enc:
        raw = zlib.decompress(raw)
    return np.frombuffer(raw, _ARRAY[t], count=n), o + clen


@functools.lru_cache(maxsize=4)
def parse(path: str) -> list[Node]:
    """Top-level nodes of a binary FBX. Cached: one import reads the same file for links, blocks, slots, vertex colours
    and the skeleton (5 parses, ~1 s each on a battle model). Callers must not mutate the returned nodes."""
    b = open(path, "rb").read()
    if not b.startswith(MAGIC):
        raise ValueError(f"{path}: not a binary FBX")
    big = struct.unpack_from("<I", b, 23)[0] >= 7500
    head = struct.Struct("<QQQ" if big else "<III")

    def node(o: int) -> tuple[Node | None, int]:
        end, nprops, _ = head.unpack_from(b, o)
        o += head.size
        if end == 0:
            return None, o + 1
        nl = b[o]
        name = b[o + 1 : o + 1 + nl].decode("latin1")
        o += 1 + nl
        props = []
        for _ in range(nprops):
            t = chr(b[o])
            o += 1
            if t in _SCALAR:
                props.append(struct.unpack_from(_SCALAR[t], b, o)[0])
                o += struct.calcsize(_SCALAR[t])
            elif t in "SR":
                ln = struct.unpack_from("<I", b, o)[0]
                v = b[o + 4 : o + 4 + ln]
                o += 4 + ln
                props.append(v.decode("utf-8", "replace") if t == "S" else v)
            else:
                arr, o = _read_array(b, o, t)
                props.append(arr)
        nd = Node(name, props)
        while o < end:
            kid, o2 = node(o)
            if kid is None:
                break
            nd.children.append(kid)
            o = o2
        return nd, end

    out, o = [], 27
    while o < len(b) - head.size:
        nd, o = node(o)
        if nd is None:
            break
        out.append(nd)
    return out


def object_name(nd: Node) -> str:
    """'Name\\x00\\x01Class' -> 'Name'."""
    return str(nd.props[1]).split("\x00")[0] if len(nd.props) > 1 else ""


def geometries(path: str) -> list[dict]:
    """Per mesh geometry: name, triangle count and whether its per-polygon material layer is intact.

    `materials_ok` is False when the layer has fewer entries than polygons (AssetStudio wrote one index per material
    slot and an empty `ByPolygon` element on 71 of 301 UNITE character meshes, all of them meshes with a second UV
    set): Blender then assigns every triangle to material 0."""
    top = {n.name: n for n in parse(path)}
    objs = top.get("Objects")
    out = []
    for g in objs.all("Geometry") if objs else []:
        if len(g.props) < 3 or g.props[2] != "Mesh":
            continue
        pvi = g.child("PolygonVertexIndex")
        npoly = int((pvi.props[0] < 0).sum()) if pvi is not None else 0
        ok = True
        layers = g.all("LayerElementMaterial")
        if layers:
            first = layers[0]
            mapping = first.child("MappingInformationType")
            mats = first.child("Materials")
            idx = mats.props[0] if mats is not None else np.zeros(0, np.int32)
            all_same = mapping is not None and mapping.props[0] == "AllSame"
            # one slot: nothing to lose, whatever the mapping says
            ok = all_same or len(idx) == npoly or len(np.unique(idx)) <= 1
        out.append({"name": object_name(g), "polygons": npoly, "materials_ok": ok})
    return out


def mesh_geometry(path: str, names: set[str] | None = None) -> dict[str, dict]:
    """Mesh Model name -> its raw geometry in the Model's local space: `positions` (N, 3), `uvs` (N, 2, glTF convention:
    V flipped from FBX's bottom-left origin) or None, `indices` (triangle list). One vertex per polygon corner when the
    UV layer is per corner, else the file's vertices; `colours` (N, 4) = the first colour layer per vertex, or None. Polygons are fan-triangulated. For small effect meshes the importer
    borrows by prefab name (Gastly skin 10's gas discs); character meshes go through Blender instead."""
    top = {n.name: n for n in parse(path)}
    objs, conns = top.get("Objects"), top.get("Connections")
    if objs is None or conns is None:
        return {}
    geoms = {g.props[0]: g for g in objs.all("Geometry") if len(g.props) > 2 and g.props[2] == "Mesh"}
    models = {c.props[0]: object_name(c) for c in objs.all("Model") if len(c.props) > 2 and c.props[2] == "Mesh"}
    out = {}
    for c in conns.children:
        p = c.props
        if not (len(p) >= 3 and p[0] == "OO" and p[1] in geoms and p[2] in models):
            continue
        name = models[p[2]]
        if names is not None and name not in names:
            continue
        g = geoms[p[1]]
        pos = np.asarray(g.child("Vertices").props[0], np.float64).reshape(-1, 3)
        pvi = np.asarray(g.child("PolygonVertexIndex").props[0], np.int64)
        corners = np.where(pvi < 0, ~pvi, pvi)
        ends = np.flatnonzero(pvi < 0)
        uv_layer = g.all("LayerElementUV")
        uvs = None
        per_corner = False
        if uv_layer:
            L = uv_layer[0]
            raw = np.asarray(L.child("UV").props[0], np.float64).reshape(-1, 2)
            mapping = str(L.child("MappingInformationType").props[0])
            ref = str(L.child("ReferenceInformationType").props[0])
            if mapping == "ByPolygonVertex":
                per_corner = True
                uvs = raw[np.asarray(L.child("UVIndex").props[0], np.int64)] if ref == "IndexToDirect" else raw
            else:
                uvs = raw[np.asarray(L.child("UVIndex").props[0], np.int64)] if ref == "IndexToDirect" else raw
        tris = []
        start = 0
        for e in ends:
            poly = np.arange(start, e + 1)
            for k in range(1, len(poly) - 1):
                tris.append((poly[0], poly[k], poly[k + 1]))
            start = e + 1
        tris = np.asarray(tris, np.int64).reshape(-1, 3)
        if per_corner:
            positions, idx, points = pos[corners], tris.reshape(-1), corners
        else:
            positions, idx, uvs, points = pos, corners[tris.reshape(-1)], uvs, np.arange(len(pos))
        if uvs is not None:
            uvs = np.stack([uvs[:, 0], 1.0 - uvs[:, 1]], axis=1)
        colours = None
        col_layer = g.all("LayerElementColor")
        if col_layer and col_layer[0].child("Colors") is not None:
            L = col_layer[0]
            raw = np.asarray(L.child("Colors").props[0], np.float64).reshape(-1, 4)
            ix = L.child("ColorIndex")
            if str(L.child("ReferenceInformationType").props[0]) == "IndexToDirect" and ix is not None:
                raw = raw[np.asarray(ix.props[0], np.int64)]
            if str(L.child("MappingInformationType").props[0]) == "ByPolygonVertex" and len(raw) == len(pvi):
                if per_corner:
                    colours = raw
                else:  # one colour per control point: the mean of its corners
                    acc = np.zeros((len(pos), 4))
                    np.add.at(acc, corners, raw)
                    colours = acc / np.maximum(np.bincount(corners, minlength=len(pos)), 1)[:, None]
            elif len(raw) == len(pos):  # ByVertice / ByControlPoint
                colours = raw[points]
        out[name] = {"positions": positions.astype(np.float32), "uvs": None if uvs is None else uvs.astype(np.float32),
                     "indices": idx.astype(np.uint32), "points": points.astype(np.int64),
                     "colours": None if colours is None else colours.astype(np.float32)}
    return out


def mesh_skin_weights(path: str, mesh: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Bone Model id -> (control-point indexes, weights) of the named mesh's skin clusters (empty for unskinned
    meshes). Control points are the geometry's `Vertices`; `mesh_geometry(...)["points"]` maps its vertices to them."""
    top = {n.name: n for n in parse(path)}
    objs, conns = top.get("Objects"), top.get("Connections")
    if objs is None or conns is None:
        return {}
    models = {c.props[0]: object_name(c) for c in objs.all("Model")}
    geoms = {g.props[0] for g in objs.all("Geometry")}
    deformers = {d.props[0]: d for d in objs.all("Deformer")}
    geo = next((p[1] for c in conns.children if len(p := c.props) >= 3 and p[0] == "OO" and p[1] in geoms
                and models.get(p[2]) == mesh), None)
    if geo is None:
        return {}
    skins = {p[1] for c in conns.children if len(p := c.props) >= 3 and p[0] == "OO" and p[2] == geo and p[1] in deformers}
    clusters = {p[1] for c in conns.children if len(p := c.props) >= 3 and p[0] == "OO" and p[2] in skins and p[1] in deformers}
    out = {}
    for c in conns.children:
        p = c.props
        if len(p) >= 3 and p[0] == "OO" and p[2] in clusters and p[1] in models:
            d = deformers[p[2]]
            ix, w = d.child("Indexes"), d.child("Weights")
            if ix is not None and w is not None and len(ix.props[0]):
                out[p[1]] = (np.asarray(ix.props[0], np.int64), np.asarray(w.props[0], np.float64))
    return out


def material_textures(path: str) -> dict[str, dict[str, str]]:
    """Material name -> {FBX property: texture file name} from the Connections (e.g. DiffuseColor, Bump).

    The battle models ship no `Materials/*.json`; the FBX's own texture links are then the only material data."""
    top = {n.name: n for n in parse(path)}
    objs, conns = top.get("Objects"), top.get("Connections")
    if objs is None:
        return {}
    ids: dict[int, tuple[str, str]] = {}
    files: dict[int, str] = {}
    for kind in ("Material", "Texture"):
        for c in objs.all(kind):
            ids[c.props[0]] = (kind, object_name(c))
            if kind == "Texture":
                fn = c.child("RelativeFilename") or c.child("FileName")
                if fn is not None and fn.props and str(fn.props[0]).strip():
                    files[c.props[0]] = str(fn.props[0]).replace("\\", "/").rsplit("/", 1)[-1]
    out: dict[str, dict[str, str]] = {object_name(c): {} for c in objs.all("Material")}
    for c in conns.children if conns is not None else []:
        p = c.props
        if len(p) >= 4 and p[0] == "OP" and ids.get(p[1], ("",))[0] == "Texture" and ids.get(p[2], ("",))[0] == "Material":
            out.setdefault(ids[p[2]][1], {})[p[3]] = files.get(p[1]) or ids[p[1]][1]
    return out




# A texture in nearly every battle folder's first block (821 of 878): a global default, never a material's own colour.
GLOBAL_TEXTURES = {"t_fx_0196_color_01_cj"}
# Texture kinds only lit character shaders use (normal, mro(e), outline colour, matcaps, ramps, masks).
CHARACTER_KINDS = _re.compile(r"^t_(?:n|nm|nn|mro|mroe|more|vc|matcap|mipmap\d*|ramp|m|h|a)(?:_|\d|$)", _re.I)


def _stem(name: str) -> str:
    return _re.sub(r"\.(png|tga|psd|jpg)$", "", name, flags=_re.I)


def material_token(material: str) -> str:
    """`mat_lob_0888_12bodya_lv3` -> `0888_12bodya`; `mat_bat_014320b_lv3` -> `014320b` (the id textures repeat)."""
    m = _re.match(r"^mat_(?:fx_)?(?:bat_|lob_|sce_)?(.+?)(?:_lv\d)?$", material, _re.I)
    return (m.group(1) if m else material).lower()


def names_material(texture: str, material: str) -> bool:
    """`t_matcap_bat_014320a.png` names `mat_bat_014320a_lv3`; `t_matcap_lob_0888_12` names `..._0888_12bodya_..`;
    `t_d_bat_014320fxa_lv3` names `mat_bat_014320fxa_lv3`: the texture's id part and the material's id are prefixes of
    one another (>= 4 characters)."""
    tid, tok = texture_id(texture), material_token(material)
    return len(tid) >= 4 and len(tok) >= 4 and (tok.startswith(tid) or tid.startswith(tok))


def texture_id(texture: str) -> str:
    m = _re.match(r"^t_[a-z]+_(?:fx_)?(?:bat_|lob_)?(.+?)(?:_lv\d)?$", _stem(texture), _re.I)
    return m.group(1).lower() if m else ""


def material_blocks(path: str) -> dict[str, list[str]]:
    """Material name -> the texture files AssetStudio exported for it, linked or not.

    AssetStudio writes the textures as the concatenation, over the Materials in Objects order, of each material's
    `m_TexEnvs` (deduplicated at first use; keys alphabetical): verified exactly on 303 of 325 lobby scene FBXs that have
    the JSONs. It connects only `_MainTex` (DiffuseColor) and `_BumpMap` (Bump), so in the Connections each LINKED
    material's OP links are followed by the Texture<->Video links of the textures it introduced. Materials without any
    link have no anchor: their textures follow the previous linked material's (or lead the file, before the first OP
    link: Charmander's tail flame). Each texture of a run is given to
      1. the unlinked material it is named after (`t_d_bat_014320fxa_lv3` -> `mat_bat_014320fxa_lv3`),
      2. else the linked material it is named after,
      3. else, while no unlinked material has started, the linked material when it is a character kind (n, mro, vc,
         matcap, ramp: Zacian lobby `bodya` = n, d, mroe, vc, matcap),
      4. else the unlinked material whose textures are being written (the last one named, else the first).
    Several unlinked materials sharing a run without names are INFERRED in write order (ledger `fbx:materialBlocks`).
    Global defaults (`GLOBAL_TEXTURES`) are skipped."""
    top = {n.name: n for n in parse(path)}
    objs, conns = top.get("Objects"), top.get("Connections")
    if objs is None or conns is None:
        return {}
    mats = {c.props[0]: object_name(c) for c in objs.all("Material")}
    tex: dict[int, str] = {}
    for c in objs.all("Texture"):
        fn = c.child("RelativeFilename") or c.child("FileName")
        name = str(fn.props[0]) if fn is not None and fn.props and str(fn.props[0]).strip() else object_name(c)
        tex[c.props[0]] = name.replace("\\", "/").rsplit("/", 1)[-1]
    videos = {c.props[0] for c in objs.all("Video")}
    linked: dict[str, list[str]] = {}  # linked material -> its OP-linked textures
    runs: list[tuple[str | None, list[str]]] = [(None, [])]  # (linked material or None for the leading run, introduced)
    for c in conns.children:
        p = c.props
        if len(p) >= 4 and p[0] == "OP" and p[1] in tex and p[2] in mats:
            m = mats[p[2]]
            if runs[-1][0] != m:
                runs.append((m, []))
            if tex[p[1]] not in linked.setdefault(m, []):
                linked[m].append(tex[p[1]])
        elif len(p) >= 3 and p[0] == "OO" and {p[1], p[2]} & tex.keys() and {p[1], p[2]} & videos:
            t = tex[p[1]] if p[1] in tex else tex[p[2]]
            if t not in runs[-1][1]:
                runs[-1][1].append(t)
    order = list(dict.fromkeys(object_name(c) for c in objs.all("Material")))
    return assign_blocks(order, linked, runs)


def assign_blocks(order: list[str], linked: dict[str, list[str]], runs: list[tuple[str | None, list[str]]]) -> dict[str, list[str]]:
    """The assignment step of `material_blocks`: `order` = Materials in Objects order, `linked` = linked material ->
    its OP-linked textures, `runs` = (linked material or None for the textures before the first OP link, the textures
    introduced after it in write order)."""
    out: dict[str, list[str]] = {m: list(v) for m, v in linked.items()}
    # unlinked materials after each linked one (Objects order), and before the first
    followers: dict[str | None, list[str]] = {None: []}
    anchor = None
    for m in order:
        if m in linked:
            anchor = m
            followers.setdefault(m, [])
        else:
            followers.setdefault(anchor, []).append(m)
    for owner, introduced in runs:
        us = followers.get(owner, [])
        cur = None
        for t in introduced:
            if _stem(t).lower() in GLOBAL_TEXTURES:
                continue
            # an unlinked material claims a texture named exactly after it, or one that does not also name the linked
            # material (`t_n_bat_014320_lv3` names both 014320a and 014320b: it is the linked 014320a's)
            exact = [u for u in us if texture_id(t) == material_token(u)]
            loose = [u for u in us if names_material(t, u)] if owner is None or not names_material(t, owner) else []
            u = (exact or loose or [None])[0]
            if u is not None:
                cur = u
                dest = u
            elif owner is not None and (names_material(t, owner) or (cur is None and (CHARACTER_KINDS.match(t) or not us))):
                dest = owner
            elif us:
                dest = cur or us[0]
                cur = dest
            else:
                continue
            if t not in out.setdefault(dest, []):
                out[dest].append(t)
    return out


def mesh_vertex_colours(path: str) -> dict[str, np.ndarray]:
    """Mesh Model name -> the mean RGBA of its geometry's vertex colour layer (`LayerElementColor`). Unity Fx shaders
    multiply their (greyscale) masks by the vertex colour: Charmander's tail flame (1, 0.16, 0), Mega Charizard X's blue
    flames (0.17, 0.33, 0.90), Mega Y's orange (1, 0.16, 0). Lit character meshes carry shader data here instead."""
    top = {n.name: n for n in parse(path)}
    objs, conns = top.get("Objects"), top.get("Connections")
    if objs is None or conns is None:
        return {}
    geoms = {g.props[0]: g for g in objs.all("Geometry") if len(g.props) > 2 and g.props[2] == "Mesh"}
    models = {c.props[0]: object_name(c) for c in objs.all("Model") if len(c.props) > 2 and c.props[2] == "Mesh"}
    out = {}
    for c in conns.children:
        p = c.props
        if len(p) >= 3 and p[0] == "OO" and p[1] in geoms and p[2] in models:
            layers = geoms[p[1]].all("LayerElementColor")
            col = layers[0].child("Colors") if layers else None
            if col is not None and len(col.props) and len(col.props[0]):
                out[models[p[2]]] = np.asarray(col.props[0], np.float64).reshape(-1, 4).mean(0)
    return out


def mesh_materials(path: str) -> dict[str, list[str]]:
    """Mesh Model name -> its material names in slot order (the order of the Material -> Model connections, which is
    the order the geometry's material indices refer to)."""
    top = {n.name: n for n in parse(path)}
    objs, conns = top.get("Objects"), top.get("Connections")
    if objs is None or conns is None:
        return {}
    mats = {c.props[0]: object_name(c) for c in objs.all("Material")}
    models = {c.props[0]: object_name(c) for c in objs.all("Model") if len(c.props) > 2 and c.props[2] == "Mesh"}
    out: dict[str, list[str]] = {}
    for c in conns.children:
        p = c.props
        if len(p) >= 3 and p[0] == "OO" and p[1] in mats and p[2] in models:
            out.setdefault(models[p[2]], []).append(mats[p[1]])
    return out
