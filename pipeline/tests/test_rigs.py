"""Rig signatures (pipeline/rigs.py), dropped-track counts (gltf_writer / manifest) and rule-based clip donors."""

import json
import os
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.common.gltf_writer import dropped_tracks, write_glb
from pipeline.common.manifest import build_manifest
from pipeline.common.model import Clip, Joint, Model, Skeleton, Track
from pipeline.rigs import clip_bind_agreement, clip_coverage, compare, rig_signature


def _skel(names=("root", "waist", "spine", "head"), offsets=None) -> Skeleton:
    joints = []
    for i, n in enumerate(names):
        t = offsets.get(n, (0.0, 0.1, 0.0)) if offsets else (0.0, 0.1 * i, 0.0)
        joints.append(Joint(name=n, parent=i - 1, translation=t, inverse_bind=np.eye(4, dtype=np.float32), skinning=True))
    return Skeleton(joints=joints, palette=list(range(len(joints))))


def test_equal_rigs():
    c = compare(_skel(), _skel())
    assert c.jaccard == 1.0 and c.same_order and c.same_palette and c.compatible()
    assert c.max_world == 0.0 and c.length_ratio == 1.0
    assert rig_signature(_skel()).key() == rig_signature(_skel()).key()


def test_renamed_bone():
    c = compare(_skel(), _skel(("root", "waist", "spine01", "head")))
    assert c.only_a == ["spine"] and c.only_b == ["spine01"]
    assert c.parent_mismatch == ["head"]  # head's parent is now spine01
    assert not c.same_order and not c.compatible()


def test_moved_bone():
    moved = _skel(offsets={"root": (0, 0, 0), "waist": (0, 0.1, 0), "spine": (0, 0.25, 0), "head": (0, 0.3, 0)})
    base = _skel(offsets={"root": (0, 0, 0), "waist": (0, 0.1, 0), "spine": (0, 0.2, 0), "head": (0, 0.3, 0)})
    c = compare(base, moved)
    assert c.max_local_t_joint == "spine" and abs(c.max_local_t - 0.05) < 1e-9
    assert c.max_world_joint in ("spine", "head") and abs(c.max_world - 0.05) < 1e-9
    assert not c.same_bind() and c.jaccard == 1.0


def test_subset_and_rotation_sign():
    big = _skel(("root", "waist", "spine", "head", "hat"))
    small = _skel()
    c = compare(small, big)
    assert c.a_subset_b and not c.b_subset_a and c.compatible() and c.palette_prefix
    flipped = _skel()
    flipped.joints[2].rotation = (0.0, 0.0, 0.0, -1.0)  # same rotation as identity
    assert compare(small, flipped).max_rot_deg < 1e-6


def test_clip_coverage_counts_missing_bones():
    clip = Clip(name="a", tracks=[Track("waist", "rotation", np.zeros(1), np.zeros((1, 4))), Track("tail", "rotation", np.zeros(1), np.zeros((1, 4)))])
    cov = clip_coverage(_skel(), [clip, ["head", "tail"]])
    assert (cov.tracks, cov.bound, cov.clips) == (4, 2, 2) and cov.missing["tail"] == 2 and cov.fraction == 0.5


def test_clip_bind_agreement():
    sk = _skel()
    ch = lambda v: SimpleNamespace(values=np.array([v], np.float32))  # noqa: E731
    tr = SimpleNamespace(tracks=[
        SimpleNamespace(bone="spine", translation=ch((0, 0.2, 0))),  # = bind
        SimpleNamespace(bone="head", translation=ch((0, 0.5, 0))),  # moved
        SimpleNamespace(bone="origin", translation=ch((3, 0, 0))),  # root motion: skipped
        SimpleNamespace(bone="waist", translation=None),
    ])
    assert clip_bind_agreement(sk, [tr]) == (1, 2)


def _model(clips) -> Model:
    return Model(id="m", game="za", display_name="m", kind="trainer", skeleton=_skel(), meshes=[], materials={}, textures={}, clips=clips)


def test_dropped_tracks_counted_in_glb_and_manifest(tmp_path, caplog):
    tr = lambda j: Track(j, "rotation", np.array([0.0, 1.0], np.float32), np.tile(np.array([0, 0, 0, 1], np.float32), (2, 1)))  # noqa: E731
    model = _model([Clip(name="ok", tracks=[tr("waist")]), Clip(name="borrowed", tracks=[tr("waist"), tr("tail_01"), tr("tail_02")])])
    assert dropped_tracks(model) == {"borrowed": 2}
    with caplog.at_level("WARNING"):
        summary = write_glb(model, str(tmp_path / "m.glb"))
    assert summary["droppedTracks"] == {"borrowed": 2}
    assert "dropped 2 tracks" in caplog.text and "tail_01" in caplog.text
    data = (tmp_path / "m.glb").read_bytes()
    n = struct.unpack("<I", data[12:16])[0]
    gl = json.loads(data[20 : 20 + n])
    assert [len(a["channels"]) for a in gl["animations"]] == [1, 1]
    man = build_manifest(model, summary)
    assert man["stats"]["droppedTracks"] == 2
    assert [c.get("droppedTracks") for c in man["clips"]] == [None, 2]
    assert build_manifest(model)["stats"]["droppedTracks"] == 2  # no GLB summary: counted from the model


# --------------------------------------------------------------------------- colourway donor rule (fake zips)


class _FakeZip:
    def __init__(self, names):
        self._names = names

    def namelist(self):
        return self._names


def _fake_catalog(sig_of: dict, model_names: list[str], anim_names: list[str]):
    from pipeline.catalog.za import ZaCatalog

    class Cat(ZaCatalog):
        def zips(self, names):
            if names == self.TRAINER_MODEL_ZIPS:
                return [_FakeZip(model_names)]
            if names == self.TRAINER_ANIM_ZIPS:
                return [_FakeZip(anim_names)]
            return []

        def _rig_signature(self, zip_names, folder, stem):
            return sig_of.get(stem)

    return Cat(data_dir="nowhere")


def test_colourway_uses_base_clips_when_rigs_match():
    models = [f"ik_chara/model_ob/{f}/{i}.trmdl" for f, i in [
        ("ob0009_00_flower", "ob0009_00"), ("ob0009_01_flower", "ob0009_01"),
        ("ob0004_00_phone", "ob0004_00"), ("ob0004_01_phone", "ob0004_01"),
        ("ob0002_00_stone", "ob0002_00"), ("ob0002_01_stoneb", "ob0002_01"),
    ]]
    anims = [
        "ik_chara/motion_ob/ob0009_00_flower/ob0009_00_00001_battlewait01_loop.tranm",
        "ik_chara/motion_ob/ob0004_00_phone/ob0004_00_00001_defaultwait01_loop.tranm",
    ]
    same = rig_signature(_skel())
    other = rig_signature(_skel(("root", "body", "lid")))
    sigs = {"ob0009_00": same, "ob0009_01": same, "ob0004_00": same, "ob0004_01": other, "ob0002_00": same, "ob0002_01": same}
    refs = {r.id: r for r in _fake_catalog(sigs, models, anims).objects()}
    assert refs["ob0009_01"].extras.get("clip_id") == "ob0009_00"
    assert refs["ob0009_01"].anim_folder == refs["ob0009_00"].anim_folder
    assert "clip_id" not in refs["ob0004_01"].extras  # rig differs: no borrowing
    assert "clip_id" not in refs["ob0002_01"].extras  # the base has no clips either
    assert "clip_id" not in refs["ob0009_00"].extras


# --------------------------------------------------------------------------- real data (skipped without the zips)

DATA = os.environ.get("POKEMODELS_DATA") or os.path.join(os.path.expanduser("~"), "Documents", "MEGA downloads", "PokéModels")
needs_za = pytest.mark.skipif(
    not (os.path.isfile(os.path.join(DATA, "ZA-TrainersNPCCommon.zip")) and os.path.isfile(os.path.join(DATA, "ZA-TrainersAnim.zip"))),
    reason="Z-A trainer zips not available",
)


@needs_za
def test_cc_skeleton_donors_claim_holds():
    """The female cc bases have clips but no skeleton: their clips' first frames must sit on the donor's bind pose
    (and clearly not on the other bases')."""
    from pipeline.catalog.za import CC_SKELETON_DONORS, ZaCatalog
    from pipeline.importers.za import trinity as T
    from pipeline.rigs import _TrinityRigs

    cat = ZaCatalog(DATA)
    idx = _TrinityRigs(cat).index(cat.TRAINER_ANIM_ZIPS)
    skels = {s: cat._rig_signature(cat.CC_MODEL_ZIPS, f"ik_chara/model_cc_base/{s}/", f"{s}_base") for s in set(CC_SKELETON_DONORS.values()) | {"em", "bu"}}
    checked = 0
    for female, donor in CC_SKELETON_DONORS.items():
        files = [n for n in idx.list(f"{cat.CC_ANIM_ROOT}/{female}/") if n.endswith(".tranm")][:60:6]
        if not files:
            continue  # ff: no clips of its own (plays the fm sets)
        clips = [T.parse_tranm(idx.read(n)) for n in files]
        agree = {s: clip_bind_agreement(sig, clips) for s, sig in skels.items() if sig is not None}
        ok, total = agree[donor]
        assert total > 100 and ok / total > 0.95, (female, donor, agree)
        assert all(a / t < 0.5 for s, (a, t) in agree.items() if s != donor), (female, agree)
        checked += 1
    assert checked >= 2


@needs_za
def test_cc_ir_body_donor_rule():
    from pipeline.catalog.za import ZaCatalog

    cat = ZaCatalog(DATA)
    refs = {r.id: r for r in cat.trainers() if r.extras.get("cc_ir")}
    assert cat.cc_ir_body_donor(refs["tr0030_00"]) == "dm"
    assert cat.cc_ir_body_donor(refs["tr0031_00"]) == "df"
    names = {a.name for a in cat.animations(refs["tr0031_00"])}
    assert "defaultwait01_loop" in names and refs["tr0031_00"].extras["cc_donor"] == "df"
    # the rivals get their body from motion_cc_base/rv (extra layers), never a donor set
    cat.animations(refs["tr0001_00"])
    assert "cc_donor" not in refs["tr0001_00"].extras
