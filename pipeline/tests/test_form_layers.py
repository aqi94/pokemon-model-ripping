"""Form layers (Mega Tatsugiri): constant-pose layer clips become one model each."""

import numpy as np

from pipeline.common.model import Clip, Joint, Mesh, Model, Skeleton, Track
from pipeline.importers.za import form_layers as FL
from pipeline.importers.za.importer import fresnel_facing_color
from pipeline.importers.za.trinity import TrMaterial


def _const(joint, kind, value):
    return Track(joint, kind, np.zeros(1, np.float32), np.asarray([value], np.float32))


def _layer(src, moves, show):
    tracks = [_const("origin", "translation", (0, 0, 0))] + [_const(b, "translation", (0, y, 0)) for b, y in moves.items()]
    vis = {f"ikura_{x}_mesh": [(0, x == show)] for x in "abc"}
    return Clip(name=src.split("_", 1)[1], tracks=tracks, source=src, index=int(src[:5]), visibility=vis)


def _model():
    joints = [Joint("origin", -1)] + [Joint(f"root_waist_{x}", 0, translation=(0.001, 0, 0)) for x in "abc"]
    body = Clip(
        name="battlewait01_loop",
        tracks=[_const("origin", "translation", (0, 0, 0))],
        source="20001_battlewait01_loop",
        visibility={f"ikura_{x}_mesh": [(0, False)] for x in "abc"},
    )
    layers = [
        _layer("28201_loop01_loop", {"root_waist_b": 0.22, "root_waist_c": -0.22}, "a"),
        _layer("28211_loop02_loop", {"root_waist_a": -0.22, "root_waist_c": 0.22}, "b"),
        _layer("28221_loop03_loop", {"root_waist_a": 0.22, "root_waist_b": -0.22}, "c"),
    ]
    meshes = [Mesh(f"ikura_{x}_mesh{s}", [], lod=1 if s else 0, visible=False) for x in "abc" for s in ("", "_lod1")]
    return Model(
        id="m", game="za", display_name="m", kind="pokemon", skeleton=Skeleton(joints, [0, 1, 2, 3]),
        meshes=meshes, materials={}, textures={}, clips=[body] + layers, default_clip="battlewait01_loop",
    )


def test_apply_form_layer_bakes_pose_and_visibility():
    m = _model()
    srcs = ["28201_loop01_loop", "28211_loop02_loop", "28221_loop03_loop"]
    FL.apply_form_layer(m, "28211_loop02_loop", srcs)
    j = {x.name: x for x in m.skeleton.joints}
    assert np.allclose(j["root_waist_a"].translation, (0, -0.22, 0))
    assert j["root_waist_b"].translation == (0.001, 0, 0)  # not keyed by loop02: the centre fish keeps its rest
    assert j["root_waist_c"].translation[1] > 0.2
    assert [c.name for c in m.clips] == ["battlewait01_loop"]  # layer clips are baked away
    assert m.clips[0].visibility["ikura_b_mesh"] == [(0, True)]
    assert m.clips[0].visibility["ikura_a_mesh"] == [(0, False)]
    assert {x.name: x.visible for x in m.meshes} == {
        "ikura_a_mesh": False, "ikura_a_mesh_lod1": False,
        "ikura_b_mesh": True, "ikura_b_mesh_lod1": True,
        "ikura_c_mesh": False, "ikura_c_mesh_lod1": False,
    }
    assert m.default_clip == "battlewait01_loop"


def test_select_form_layers_needs_distinct_layers(monkeypatch):
    sigs = {
        b"1": ({"root_waist_b": {"translation": (0, 0.22, 0)}}, {"ikura_a": True}),
        b"2": ({"root_waist_a": {"translation": (0, -0.22, 0)}}, {"ikura_a": False}),
        b"3": ({"root_waist_b": {"translation": (0, 0.22, 0)}}, {"ikura_a": True}),
    }
    monkeypatch.setattr(FL, "layer_signature", lambda tranm, tracm: sigs[tranm])
    assert FL.select_form_layers({"28211_loop02_loop": (b"2", None), "28201_loop01_loop": (b"1", None)}) == [
        "28201_loop01_loop", "28211_loop02_loop",
    ]
    # Identical layers (Mega Incineroar's three loops) are effect layers, not forms.
    assert FL.select_form_layers({"28201_loop01_loop": (b"1", None), "28211_loop02_loop": (b"3", None)}) == []
    assert FL.select_form_layers({"28201_loop01_loop": (b"1", None)}) == []


def test_layer_candidate_index():
    assert FL.is_layer_candidate("loop02_loop", 28211)
    assert not FL.is_layer_candidate("loop01_loop", 28300)
    assert not FL.is_layer_candidate("walk01_loop", 28211)


def _fresnel(base):
    return TrMaterial(
        name="body_f", shader="FresnelEffect", shader_flags={"EnableFresnelTexture": "True"}, textures=[], samplers=[],
        floats={"FresnelAlphaMin": 1.0}, colors={"BaseColor": base, "BaseColorLayer1": (1.0, 0.06, 0.01, 1.0)},
        ints={}, alpha_type="Opaque", cull=-1,
    )


def test_fresnel_black_base_takes_layer_colour():
    tm = _fresnel((0.0, 0.0, 0.0, 1.0))
    assert fresnel_facing_color(tm, tm.colors["BaseColor"]) == (1.0, 0.06, 0.01, 1.0)
    gem = _fresnel((0.52, 0.03, 0.05, 1.0))  # coloured base: left alone
    assert fresnel_facing_color(gem, gem.colors["BaseColor"]) == (0.52, 0.03, 0.05, 1.0)
