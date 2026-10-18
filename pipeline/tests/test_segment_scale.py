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
