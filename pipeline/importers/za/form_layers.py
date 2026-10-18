"""Form layers: mutually exclusive constant-pose layer clips that pick a model's arrangement.

Mega Tatsugiri (Z-A `pm1056_51_00`) is one model holding all three base forms (body_a yellow Stretchy, body_b pink
Droopy, body_c orange Curly), each on its own `root_waist_x` chain at the origin, plus one roe mesh per fish
(`ikura_a/b/c`, hidden by every body clip). The game layers one of `x8201_loop01_loop` / `x8211_loop02_loop` /
`x8221_loop03_loop` on top of the body clip: a constant pose that moves two chains +-0.22 m sideways (the third stays
in the centre) and a `.tracm` that shows that centre fish's roe. The game's own pre-layered battlewaits (hash-named
`0x....tranm`) carry exactly these keys. glTF has no animation layers, so each layer becomes its own model: the
layer's pose is baked into the rest pose, every clip stops keying those bones, and its visibility overrides the clips'.

Rule (no ids): >= 2 clips named `loopNN_loop` at index x82xx whose bone keys are all constant and whose
(pose, visibility) differ from each other. Mega Incineroar's three loop layers (pm0727_51_00) are identical in pose
and visibility (they only differ in effect material tracks), so it is not split.
"""

from __future__ import annotations

import re

import numpy as np

from ...common.model import Model
from . import trinity as T

LAYER_NAME_RE = re.compile(r"^loop\d{2}_loop$")
_TOL = 1e-4


def is_layer_candidate(name: str, index: int) -> bool:
    return bool(LAYER_NAME_RE.match(name)) and (index % 10000) // 100 == 82


def _shape_base(shape: str) -> str:
    s = re.sub(r"_lod\d+$", "", shape)
    return s[: -len("_shape")] if s.endswith("_shape") else s


def layer_signature(tranm: bytes, tracm: bytes | None) -> tuple[dict, dict] | None:
    """(bone -> {kind: value}, shape -> visible) of a constant layer clip; None if anything is animated."""
    tr = T.parse_tranm(tranm)
    pose: dict[str, dict[str, tuple[float, ...]]] = {}
    for bt in tr.tracks:
        for kind in ("translation", "rotation", "scale"):
            ch = getattr(bt, kind)
            if ch is None or not len(ch.values):
                continue
            v = np.asarray(ch.values, np.float64)
            if not np.allclose(v, v[0], atol=_TOL):
                return None
            pose.setdefault(bt.bone, {})[kind] = tuple(float(x) for x in v[0])
    vis: dict[str, bool] = {}
    if tracm:
        for vt in T.parse_tracm_visibility(tracm):
            if not len(vt.values) or vt.values.all() != vt.values.any():
                return None
            vis.setdefault(_shape_base(vt.shape), bool(vt.values[0]))
    return pose, vis


def _same(a: tuple[dict, dict], b: tuple[dict, dict]) -> bool:
    if a[1] != b[1] or a[0].keys() != b[0].keys():
        return False
    for bone, kinds in a[0].items():
        if kinds.keys() != b[0][bone].keys():
            return False
        if any(not np.allclose(kinds[k], b[0][bone][k], atol=_TOL) for k in kinds):
            return False
    return True


def select_form_layers(candidates: dict[str, tuple[bytes, bytes | None]]) -> list[str]:
    """Clip sources ("28201_loop01_loop" -> (tranm, tracm)) that are distinct form layers, sorted; [] if not a split."""
    sigs = {}
    for src, (tranm, tracm) in candidates.items():
        try:
            sig = layer_signature(tranm, tracm)
        except Exception:  # noqa: BLE001 - unreadable = not a layer
            sig = None
        if sig is not None:
            sigs[src] = sig
    if len(sigs) < 2:
        return []
    keys = sorted(sigs)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if _same(sigs[a], sigs[b]):
                return []
    return keys


def apply_form_layer(model: Model, layer: str, layers: list[str]) -> None:
    """Bake clip `layer` (a Clip.source) into `model`: rest pose of the bones and rest/clip visibility of the shapes
    that differ between `layers`; the layer clips themselves are removed (they are now the model)."""
    by_src = {c.source: c for c in model.clips}
    if layer not in by_src:
        raise ValueError(f"form layer {layer} not among the clips")
    layer_clips = [by_src[s] for s in layers if s in by_src]
    sk = model.skeleton
    joints = {j.name: j for j in sk.joints} if sk else {}

    def value(clip, bone: str, kind: str):
        for t in clip.tracks:
            if t.joint == bone and t.kind == kind and len(t.values):
                return tuple(float(x) for x in t.values[0])
        j = joints[bone]
        return tuple(getattr(j, kind))

    # Bones (and channels) whose value differs between the layers.
    keyed = {(t.joint, t.kind) for c in layer_clips for t in c.tracks if t.joint in joints}
    moved: set[str] = set()
    for bone, kind in sorted(keyed):
        vals = [value(c, bone, kind) for c in layer_clips]
        if any(not np.allclose(vals[0], v, atol=_TOL) for v in vals[1:]):
            setattr(joints[bone], kind, value(by_src[layer], bone, kind))
            moved.add(bone)
    shapes = {s for c in layer_clips for s in c.visibility}
    vis = {}
    for s in shapes:
        vals = {bool(c.visibility.get(s, [(0, True)])[0][1]) for c in layer_clips}
        if len(vals) > 1:
            vis[s] = bool(by_src[layer].visibility.get(s, [(0, True)])[0][1])

    model.clips = [c for c in model.clips if c.source not in layers]
    for c in model.clips:
        c.tracks = [t for t in c.tracks if t.joint not in moved]
        for s, v in vis.items():
            c.visibility[s] = [(0, v)]
    for m in model.meshes:
        base = m.name.split("_lod")[0]
        if base in vis:
            m.visible = vis[base]
    if model.default_clip and model.default_clip not in {c.name for c in model.clips}:
        model.default_clip = model.clips[0].name if model.clips else None
    model.extras["form_layer"] = layer
