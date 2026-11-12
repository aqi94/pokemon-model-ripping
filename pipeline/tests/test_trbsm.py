import struct

import numpy as np

from pipeline.common.model import Clip, Joint, Skeleton, Track
from pipeline.importers.za.composite import _world_matrices
from pipeline.importers.za.importer import split_scale_joints
from pipeline.importers.za.trbsm import apply_body_type, body_scales, parse_trbsm
from pipeline.tests.test_trlgt import _Builder


class _B(_Builder):
    def strings(self, items: list[str]) -> int:
        return self.vector([self.string(s) for s in items])

    def floats(self, vals: list[float]) -> int:
        self._align(4)
        self._prepend(struct.pack(f"<{len(vals)}f", *vals))
        return self._prepend(struct.pack("<I", len(vals)))


def _trbsm() -> bytes:
    b = _B()
    head = b.table([("str", b.string("head")), ("vec", b.strings(["head"])), ("f32", 0.5), ("u32", 0x2001)])
    lower = b.table([("str", b.string("lower_body")), ("vec", b.strings(["foot_base", "leg"])), ("f32", 0.5)])
    hips = b.table([("str", b.string("mesh_scaler_hips")), ("vec", b.strings(["hips"])), ("u32", 0x3006),
                    ("vec", b.floats([0.0, 2.0, 3.0]))])
    ground = b.table([("str", b.string("foot")), ("vec", b.strings(["foot_base"]))])
    root = b.table([("vec", b.vector([head, lower])), ("vec", b.vector([hips])), ("vec", b.vector([ground]))])
    return b.finish(root)


def test_parse_trbsm_reads_sliders_scalers_and_ground():
    t = parse_trbsm(_trbsm())
    assert [s["name"] for s in t["sliders"]] == ["head", "lower_body"]
    assert t["sliders"][0]["flags"] == 0x2001 and t["sliders"][1]["bones"] == ["foot_base", "leg"]
    assert t["scalers"][0]["max"] == [0.0, 2.0, 3.0] and t["scalers"][0]["flags"] & 0xFF == 6
    assert t["ground"] == [{"name": "foot", "bones": ["foot_base"]}]
    slider, mesh, inherit = body_scales(t, {"sliders": {"lower_body": 0.5}})
    assert np.isclose(slider["leg"], 0.75) and np.isclose(slider["head"], 0.5) and inherit == {"head"}
    assert np.allclose(mesh["hips"], [1.0, 2.0, 3.0])  # axis x outside the mask stays 1


def _rig() -> Skeleton:
    # origin -> foot_base (0, 0.1, 0) -> leg (0, 0.8, 0) -> foot (0, -0.7, 0); foot_base -> hips; leg -> head
    j = [Joint("origin", -1), Joint("foot_base", 0, (0, 0.1, 0)), Joint("leg", 1, (0, 0.8, 0)),
         Joint("foot", 2, (0, -0.7, 0)), Joint("hips", 1), Joint("head", 2, (0, 0.5, 0)),
         Joint("left_foot", 3), Joint("eye", 5, (0, 0.1, 0.05))]
    w = _world_matrices(j)
    for i, jj in enumerate(j):
        jj.inverse_bind, jj.skinning = np.linalg.inv(w[i]).astype(np.float32), True
    return Skeleton(joints=j, palette=list(range(len(j))))


def test_apply_body_type_compensates_offsets_inherits_head_and_regrounds():
    sk = _rig()
    before = _world_matrices(sk.joints)
    t0 = np.zeros(1, np.float32)
    clip = Clip(name="w", tracks=[Track("foot", "translation", t0, np.array([[0, -0.7, 0]], np.float32)),
                                  Track("head", "scale", t0, np.ones((1, 3), np.float32))], frame_rate=60.0)
    split_scale_joints(sk, [clip])
    table = parse_trbsm(_trbsm())
    changed = apply_body_type(sk, [clip], table)
    assert set(changed) == {"foot_base", "leg", "head", "hips"}
    idx = {j.name: i for i, j in enumerate(sk.joints)}
    # leg (slider 0.5, compensating): its child offset halves, the leg itself is not scaled, its skin leaf is
    assert np.allclose(sk.joints[idx["foot"]].translation, (0, -0.35, 0))
    assert np.allclose(sk.joints[idx["leg"]].scale, 1) and np.allclose(sk.joints[idx["leg_scale"]].scale, 0.5)
    assert np.allclose([t.values for t in clip.tracks if t.joint == "foot" and t.kind == "translation"][0], [[0, -0.35, 0]])
    # mesh scaler: hips skin leaf (1, 2, 3) per axis, children untouched
    assert np.allclose(sk.joints[idx["hips_scale"]].scale, (1, 2, 3))
    assert sk.palette[4] == idx["hips_scale"]
    # head (flag 0x2001): the bone itself is scaled (inherited), including the clip's scale track
    assert np.allclose(sk.joints[idx["head"]].scale, 0.5)
    assert np.allclose([t.values for t in clip.tracks if t.joint == "head" and t.kind == "scale"][0], 0.5)
    # ground: foot_base shifted so the point under the ankle (y = 0 when authored) is back on the floor
    w = _world_matrices(sk.joints)
    lf = idx["left_foot"]
    p0 = np.array([before[lf][0, 3], 0.0, before[lf][2, 3], 1.0])
    sole = w[lf] @ np.linalg.inv(before[lf]) @ p0
    assert abs(sole[1]) < 1e-6
    assert not np.isclose(sk.joints[idx["foot_base"]].translation[1], 0.1)  # it did have to move
