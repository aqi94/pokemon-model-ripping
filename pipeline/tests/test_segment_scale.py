import numpy as np

from pipeline.common.model import Clip, Joint, Skeleton, Track
from pipeline.importers.za.importer import _quat_to_mat, split_scale_joints


def _trs(t, q, s):
    m = np.eye(4)
    m[:3, :3] = _quat_to_mat(np.asarray([q], np.float64))[0] * np.asarray(s)[None, :]
    m[:3, 3] = t
    return m


def _worlds(sk, vals):
    out = {}
    for i, j in enumerate(sk.joints):
        loc = _trs(vals.get((j.name, "translation"), j.translation), vals.get((j.name, "rotation"), j.rotation), vals.get((j.name, "scale"), j.scale))
        out[j.name] = loc if j.parent < 0 else out[sk.joints[j.parent].name] @ loc
    return out


def test_bent_chain_matches_segment_scale_compensate():
    # Binacle/Seviper: a2, a3 scaled non-uniformly along a bent chain; the game's rule is
    # world = parent world without its scale . T(parent_scale * t) . R . S(own scale).
    q = (0.0, 0.0, 0.1045285, 0.9945219)  # 12 deg: not axis-aligned, so per-child compensation would shear
    sk = Skeleton(joints=[Joint("a1", -1), Joint("a2", 0, (0.1, 0, 0), q), Joint("a3", 1, (0.1, 0, 0), q), Joint("head", 2, (0.1, 0, 0), q)], palette=[0, 1, 2, 3])
    own = {"a2": (0.3, 1, 1), "a3": (0.7, 1, 1)}
    t = np.array([0.0], np.float32)
    clip = Clip(name="w", tracks=[Track(n, "scale", t, np.array([s], np.float32)) for n, s in own.items()], frame_rate=60.0)
    game = {}
    for j in sk.joints:
        s = np.asarray(own.get(j.name, (1, 1, 1)), float)
        if j.parent < 0:
            game[j.name] = (_trs(j.translation, j.rotation, s), s)
            continue
        pw, ps = game[sk.joints[j.parent].name]
        ns = pw.copy()
        ns[:3, :3] /= np.linalg.norm(pw[:3, :3], axis=0)
        game[j.name] = (ns @ _trs(np.asarray(j.translation) * ps, j.rotation, s), s)
    split = split_scale_joints(sk, [clip])
    assert split == ["a2", "a3"]
    vals = {(tr.joint, tr.kind): tr.values[0] for tr in clip.tracks}
    gl = _worlds(sk, vals)
    # Skinning uses the proxies for split bones, the bone itself otherwise.
    names = [sk.joints[p].name for p in sk.palette]
    assert names == ["a1", "a2_scale", "a3_scale", "head"]
    for game_name, gl_name in zip(["a1", "a2", "a3", "head"], names):
        assert np.allclose(gl[gl_name], game[game_name][0], atol=1e-6), game_name


def test_uniform_scale_clip_still_hides_the_chain():
    sk = Skeleton(joints=[Joint("a", -1), Joint("b", 0, (0.1, 0, 0))], palette=[0, 1])
    t = np.array([0.0], np.float32)
    squash = Clip(name="s", tracks=[Track("a", "scale", t, np.array([[0.2, 1, 1]], np.float32))], frame_rate=60.0)
    hide = Clip(name="h", tracks=[Track("a", "scale", t, np.zeros((1, 3), np.float32))], frame_rate=60.0)
    split_scale_joints(sk, [squash, hide])
    vals = {(tr.joint, tr.kind): tr.values[0] for tr in hide.tracks}
    assert np.allclose(_worlds(sk, vals)["b"][:3, :3], 0)  # inherited zero scale hides the child


def test_clip_without_scale_key_keeps_rest_scaled_child_offsets():
    # SV trainers: `eyescale` rests at (1.6, 1, 1) with the eyeballs as children. A clip that keys neither the bone's
    # scale nor the eyeball offsets must leave the eyes where the rest pose puts them (it wrote the raw offsets,
    # pulling both eyes towards the centre in every such clip).
    sk = Skeleton(joints=[Joint("eyescale", -1, scale=(1.6, 1, 1)), Joint("eye_r", 0, (-0.014, 0.031, 0.013))], palette=[0, 1])
    t = np.array([0.0], np.float32)
    idle = Clip(name="idle", tracks=[Track("eye_r", "rotation", t, np.array([[0, 0, 0, 1]], np.float32))], frame_rate=60.0)
    split_scale_joints(sk, [idle])
    rest = _worlds(sk, {})["eye_r"][:3, 3]
    vals = {(tr.joint, tr.kind): tr.values[0] for tr in idle.tracks}
    assert np.allclose(_worlds(sk, vals)["eye_r"][:3, 3], rest, atol=1e-6)
    assert np.allclose(rest, (-0.014 * 1.6, 0.031, 0.013), atol=1e-6)


def test_scale_compensate_off_child_inherits_and_rest_skin_is_identity():
    # SV trainers: `eyescale` (1.3, 1, 1) -> `left_eyeball_segoff` (compensate OFF: inherits the scale, its inverse bind
    # includes it) -> `left_pupil`. The rest skin matrix (world x inverse bind) must be identity for every joint.
    sk = Skeleton(joints=[
        Joint("head", -1, skinning=True),
        Joint("eyescale", 0, (0.1, 0, 0.03), scale=(1.3, 1, 1), skinning=True),
        Joint("segoff", 1, (0.008, -0.018, 0.016), skinning=True, scale_compensate=False),
        Joint("pupil", 2, (0.01, 0, 0), skinning=True),
        Joint("brow", 1, (0.02, 0.01, 0), skinning=True),  # compensating sibling
    ], palette=[0, 1, 2, 3, 4])
    game = {}
    for i, j in enumerate(sk.joints):  # game rule: compensating children drop the parent's scale, segoff keeps it
        loc = _trs(j.translation, j.rotation, j.scale)
        if j.parent < 0:
            game[i] = loc
            continue
        pw = game[j.parent]
        if j.scale_compensate:
            ps = np.asarray(sk.joints[j.parent].scale, float)
            pw = pw.copy(); pw[:3, :3] /= ps[None, :]
            loc = _trs(np.asarray(j.translation) * ps, j.rotation, j.scale)
        game[i] = pw @ loc
    for i, j in enumerate(sk.joints):
        j.inverse_bind = np.linalg.inv(game[i]).astype(np.float32)
    split_scale_joints(sk, [])
    w = {}

    def world(i):
        if i not in w:
            j = sk.joints[i]
            loc = _trs(j.translation, j.rotation, j.scale)
            w[i] = loc if j.parent < 0 else world(j.parent) @ loc
        return w[i]

    for p in sk.palette:
        j = sk.joints[p]
        assert np.allclose(world(p) @ j.inverse_bind, np.eye(4), atol=1e-5), j.name
    assert sk.joints[2].parent == sk.index_of("eyescale_scale")
