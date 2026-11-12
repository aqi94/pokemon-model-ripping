"""SV generic-NPC body types: `.trbsm` tables + the per-NPC weights in `.ccdata`, and applying them to a skeleton.

Every base-game generic trainer (`chara/model_tr/tr0330..tr0930`) is authored on the same Adult M skeleton and an
Adult-M-sized mesh (the Youngster's bind pose is 1.76 m tall). The game reshapes them at runtime: the NPC's setup file
`chara/sav_tr/<id>/<id>.ccdata` names a body-type table `chara/bsm/chara_create_value_list_<t>/<...>.trbsm`
(dm = Adult M, identity; df Adult F, f Fit M, ef/em Stocky F/M, a/b/cf/cm kids) and weights its entries.

`.trbsm` root (verified on all 9 tables):
  [0] sliders  {0 name, 1 bone names, 2 max (f32), 3 flags (only `head` has one: 0x2001)}
  [1] mesh scalers {0 `mesh_scaler_<bone>`, 1 bone names, 2 flags (low byte = axis mask: bit0 x, bit1 y, bit2 z;
      the high byte is unexplained), 3 f32[3] per-axis max (0 on axes outside the mask)}
  [2] {0 'foot', 1 ['foot_base']}: read as the ground rule (the `foot` slider re-grounds through `foot_base`, see
      `_reground`); [3] {0 'lower_body'}: unexplained, not used.
`.ccdata` root[1] = {0 trbsm path, 1 mesh scaler weights {0 name, 1..3 x/y/z f32}, 2 slider weights {0 name, 1 f32}}.
Every base-game NPC stores weight 1.0 everywhere.

Model applied here (HYPOTHESIS, checked by renders against the game, see docs/ledger/sv.json):
- a slider scales its bones uniformly by s = 1 + (max - 1) * weight, Trinity segment-scale-compensate style: the
  mesh skinned to the bone scales and its children's offsets scale, the children themselves do not inherit it;
  the flagged `head` slider (0x2001) is inherited by the whole head subtree (face, eyes, hair);
- a mesh scaler is an extra per-axis mesh-only scale on its bones (1 + (max - 1) * weight per masked axis);
  it does not move children.
"""

from __future__ import annotations

import numpy as np

from ...common.model import Clip, Joint, Skeleton
from . import fb

INHERIT_FLAG = 0x2000  # `head` slider flags 0x2001: scale inherited by the subtree (hypothesis)


def parse_trbsm(data: bytes) -> dict:
    r = fb.root(data)
    sliders = [{"name": e.string(0), "bones": e.strings(1), "max": round(e.f32(2, 1.0), 6), "flags": e.u32(3)}
               for e in r.tables(0)]
    scalers = []
    for e in r.tables(1):
        v = e.vector(3)
        vals = [round(float(x), 6) for x in v.scalars("<f4")] if v else [0.0, 0.0, 0.0]
        scalers.append({"name": e.string(0), "bones": e.strings(1), "flags": e.u32(2), "max": vals})
    ground = [{"name": e.string(0), "bones": e.strings(1)} for e in r.tables(2)]
    return {"sliders": sliders, "scalers": scalers, "ground": ground}


def parse_ccdata_body(data: bytes) -> dict | None:
    """{"table": <bsm type>, "sliders": {name: w}, "scalers": {name: [wx, wy, wz]}} or None without a body table."""
    r = fb.root(data)
    b = r.table(1)
    if b is None:
        return None
    path = b.string(0)
    if not path.endswith(".trbsm"):
        return None
    table = path.rsplit("/", 1)[-1][: -len(".trbsm")].replace("chara_create_value_list_", "")
    scalers = {e.string(0): [round(e.f32(i, 1.0), 6) for i in (1, 2, 3)] for e in b.tables(1)}
    sliders = {e.string(0): round(e.f32(1, 1.0), 6) for e in b.tables(2)}
    return {"table": table, "sliders": sliders, "scalers": scalers}


def body_scales(table: dict, weights: dict | None = None) -> tuple[dict[str, float], dict[str, np.ndarray], set[str]]:
    """-> (uniform slider scale per bone, per-axis mesh scale per bone, bones whose slider scale is inherited)."""
    weights = weights or {}
    sw, mw = weights.get("sliders", {}), weights.get("scalers", {})
    slider: dict[str, float] = {}
    inherit: set[str] = set()
    for s in table["sliders"]:
        k = 1.0 + (s["max"] - 1.0) * float(sw.get(s["name"], 1.0))
        for b in s["bones"]:
            slider[b] = slider.get(b, 1.0) * k
            if s["flags"] & INHERIT_FLAG:
                inherit.add(b)
    mesh: dict[str, np.ndarray] = {}
    for m in table["scalers"]:
        mask = m["flags"] & 0xFF
        w = mw.get(m["name"], [1.0, 1.0, 1.0])
        axes = np.array([1.0 + (m["max"][a] - 1.0) * float(w[a]) if mask >> a & 1 else 1.0 for a in range(3)])
        for b in m["bones"]:
            mesh[b] = mesh.get(b, np.ones(3)) * axes
    return slider, mesh, inherit


def apply_body_type(skeleton: Skeleton, clips: list[Clip], table: dict, weights: dict | None = None) -> list[str]:
    """Reshape an Adult-M-authored skeleton (after `split_scale_joints`) into the body type. Returns changed bones."""
    slider, mesh, inherit = body_scales(table, weights)
    joints = skeleton.joints
    idx = {j.name: i for i, j in enumerate(joints)}
    kids: dict[int, list[int]] = {}
    for i, j in enumerate(joints):
        if j.parent >= 0:
            kids.setdefault(j.parent, []).append(i)
    from .composite import _world_matrices

    before = _world_matrices(joints)
    changed: list[str] = []
    for name in sorted(set(slider) | set(mesh)):
        i = idx.get(name)
        if i is None:
            continue
        s = slider.get(name, 1.0)
        m = mesh.get(name, np.ones(3)) * (1.0 if name in inherit else s)
        if abs(s - 1.0) < 1e-6 and np.allclose(m, 1.0):
            continue
        changed.append(name)
        if name in inherit:
            # Inherited (head): scale the bone itself; glTF inheritance carries it to offsets and meshes below.
            _scale_joint(joints[i], clips, np.full(3, s))
            if np.allclose(m, 1.0):
                continue
        proxy = idx.get(f"{name}_scale")
        if proxy is None or joints[proxy].parent != i:
            joints.append(Joint(name=f"{name}_scale", parent=i, inverse_bind=joints[i].inverse_bind,
                                skinning=joints[i].skinning))
            proxy = len(joints) - 1
            idx[joints[proxy].name] = proxy
            skeleton.palette = [proxy if p == i else p for p in skeleton.palette]
            for k in kids.get(i, []):
                if not joints[k].scale_compensate:  # inherits its parent's scale: follow the mesh scale
                    joints[k].parent = proxy
        _scale_joint(joints[proxy], clips, m)
        if name in inherit:
            continue
        for k in kids.get(i, []):
            if k == proxy or joints[k].parent != i or not joints[k].scale_compensate:
                continue
            joints[k].translation = tuple(float(x) * s for x in joints[k].translation)
            for c in clips:
                for t in c.tracks:
                    if t.joint == joints[k].name and t.kind == "translation":
                        t.values = (t.values * s).astype(np.float32)
    if changed:
        _reground(skeleton, clips, before, ground_bone=_ground_bone(table))
    return changed


def _ground_bone(table: dict) -> str:
    return next((b for g in table.get("ground", []) for b in g.get("bones", [])), "foot_base")


def _reground(skeleton: Skeleton, clips: list[Clip], before: dict[int, np.ndarray], ground_bone: str) -> None:
    """Keep the soles on the floor. The point under each ankle on the ground (y = 0 in the authored Adult M pose) is
    carried through the new foot skinning transform; the root `ground_bone` is shifted by minus their mean height
    (the kid's 0.575 feet left it floating 4 cm, the stocky 1.25 feet sank 3 cm)."""
    from .composite import _world_matrices

    joints = skeleton.joints
    idx = {j.name: i for i, j in enumerate(joints)}
    g = idx.get(ground_bone)
    if g is None:
        return
    after = _world_matrices(joints)
    ys = []
    for foot in ("left_foot", "right_foot"):
        f = idx.get(foot)
        if f is None:
            continue
        skin = idx.get(f"{foot}_scale", f)
        p0 = np.array([before[f][0, 3], 0.0, before[f][2, 3], 1.0])
        ys.append(float((after[skin] @ np.linalg.inv(before[f]) @ p0)[1]))
    if not ys:
        return
    dy = -float(np.mean(ys))
    if abs(dy) < 1e-4:
        return
    par = joints[g].parent
    pw = after[par] if par >= 0 else np.eye(4)
    d = np.linalg.solve(pw[:3, :3], np.array([0.0, dy, 0.0]))  # world up -> the ground bone's parent space
    joints[g].translation = tuple(float(a + b) for a, b in zip(joints[g].translation, d))
    for c in clips:
        for t in c.tracks:
            if t.joint == joints[g].name and t.kind == "translation":
                t.values = (t.values + d[None, :]).astype(np.float32)


def _scale_joint(j: Joint, clips: list[Clip], f: np.ndarray) -> None:
    """Multiply a joint's rest scale and every clip scale track on it by `f` (per axis)."""
    j.scale = tuple(float(a) * float(b) for a, b in zip(j.scale, f))
    for c in clips:
        for t in c.tracks:
            if t.joint == j.name and t.kind == "scale":
                t.values = (t.values * f[None, :]).astype(np.float32)
