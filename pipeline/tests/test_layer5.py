"""Material colour layer 5 (OpacityMap1): baked when it maps like the albedo (Yveltal's black), an overlay otherwise
(Xerneas' horn glow on UV1, driven by UVScaleOffset5 clip tracks)."""
import numpy as np

from pipeline.importers.za import importer as imp
from pipeline.importers.za import trinity as T
from pipeline.importers.za.importer import layer_overlay_uvs, material_uv_offsets


def _material(uv_set=0, so5=(1.0, 1.0, 0.0, 0.0), layers=6):
    alb = np.full((8, 8, 4), 255, np.uint8)
    lym = np.zeros((8, 8, 4), np.uint8)
    lym[..., 0] = 255  # layer 1 (red) everywhere
    msk = np.zeros((8, 8, 4), np.uint8)
    msk[:4, :, :3] = 255  # layer 5 on the top half of the image
    msk[..., 3] = 255
    texs = {"a.bntx": alb, "m.bntx": lym, "o.bntx": msk}
    tm = T.TrMaterial(
        name="body_a_00", shader="IkCharacter",
        shader_flags={"EnableBaseColorMap": "True", "NumMaterialLayer": str(layers)},
        textures=[T.TextureBinding("BaseColorMap", "a.bntx", 0), T.TextureBinding("LayerMaskMap", "m.bntx", 1),
                  T.TextureBinding("OpacityMap1", "o.bntx", 2)],
        samplers=[("clamp", "clamp")] * 3,
        floats={"EmissionIntensityLayer5": 0.0},
        colors={"UVScaleOffset": (1.0, 1.0, 0.0, 0.0), "BaseColorLayer1": (1.0, 0.0, 0.0, 1.0),
                "BaseColorLayer5": (0.0, 0.0, 0.0, 1.0), "ShadowingColorLayer5": (0.5, 0.5, 0.5, 1.0), "UVScaleOffset5": so5},
        ints={"UVIndexLayer5": uv_set}, alpha_type="Opaque", cull=-1,
    )
    return tm, texs


def test_static_uv0_layer5_is_baked_over_layers_1_to_4():
    tm, texs = _material()
    textures = {}
    ovs = []
    mat = imp._convert_material(tm, "", texs.get, textures, {}, overlays=ovs)
    baked = textures[mat.base_color].rgba
    assert not ovs
    assert baked[:4, :, :3].max() < 5  # layer 5 black covers the red
    assert baked[4:, :, 0].min() > 250 and baked[4:, :, 1].max() < 5  # layer 1 red where layer 5 is off
    shadow = textures[mat.extras["shadow_color"]].rgba
    assert shadow[:4, :, :3].max() < 5


def test_layer5_needs_six_material_layers():
    tm, texs = _material(layers=5)
    textures = {}
    mat = imp._convert_material(tm, "", texs.get, textures, {}, overlays=[])
    assert textures[mat.base_color].rgba[:4, :, 0].min() > 250  # untouched red


def test_uv1_or_moving_layer5_becomes_overlay():
    for kw, moves in (({"uv_set": 1}, False), ({}, True)):
        tm, texs = _material(**kw)
        textures = {}
        ovs = []
        mat = imp._convert_material(tm, "", texs.get, textures, {}, overlays=ovs, layer5_moves=moves)
        assert textures[mat.base_color].rgba[:4, :, 0].min() > 250  # not baked
        assert len(ovs) == 1
        ov = ovs[0]
        assert ov.name == "body_a_00~l5" and ov.alpha_mode == "BLEND"
        assert ov.extras["uv_track_param"] == "UVScaleOffset5" and ov.extras["uv_track_src"] == "body_a_00"
        assert ov.extras["layer_overlay"]["uv_set"] == kw.get("uv_set", 0)
        alpha = textures[ov.base_color].rgba[..., 3]
        assert alpha[:4].min() == 255 and alpha[4:].max() == 0


def test_layer_overlay_uvs_subtract_the_offset():
    # Xerneas' horns: UV1 u 0.81..0.99 at rest offset U 0.5 sample u - 0.5 (the black side of the mask: glow off).
    uv = np.array([[0.81, 0.30], [0.99, 0.90]], np.float32)
    out = layer_overlay_uvs(uv, {"uv_set": 1, "scale": (1.0, 1.0), "offset": (0.5, 0.25)})
    assert np.allclose(out[:, 0], [0.31, 0.49])
    assert np.allclose(out[:, 1], [1 - (0.30 - 0.25), 1 - (0.90 - 0.25)])


def test_layer5_clip_track_drives_the_overlay():
    tm, texs = _material(uv_set=1, so5=(1.0, 1.0, 0.5, 0.0))
    textures = {}
    ovs = []
    imp._convert_material(tm, "", texs.get, textures, {}, overlays=ovs)
    mats = {ovs[0].name: ovs[0]}
    f = np.array([0.0, 30.0], np.float32)
    track = T.MaterialParamTrack("horninc_a_mesh_shape", "body_a_00", "UVScaleOffset5", [
        (f, np.ones(2, np.float32)), (f, np.ones(2, np.float32)),
        (f, np.array([0.5, 0.0], np.float32)), (f, np.array([0.0, 0.4], np.float32)),
    ])
    out = material_uv_offsets([track], mats, 30.0)
    times, xy = out["body_a_00~l5"]
    assert np.allclose(times, [0.0, 1.0])
    # U offset 0.5 -> 0 moves the sample point +0.5 in glTF U; V offset +0.4 moves +0.4 down the image.
    assert np.allclose(xy[-1], [0.5, 0.4])


def test_mode_material_switches_only_chromatic_zero_intensity_layers():
    # Xerneas' Active Mode: horn layers 3/4 carry yellow/cream emission at intensity 0; white is the authoring default.
    tm, _ = _material()
    tm.colors.update({"EmissionColorLayer3": (0.95, 0.81, 0.19, 1.0), "EmissionColorLayer4": (1.0, 1.0, 1.0, 1.0)})
    tm.floats.update({"EmissionIntensityLayer3": 0.0, "EmissionIntensityLayer4": 0.0})
    act = imp._mode_material(tm, (3, 4))
    assert act.name == "body_a_00~active"
    assert act.floats["EmissionIntensityLayer3"] == 1.0 and act.floats["EmissionIntensityLayer4"] == 0.0
    assert tm.floats["EmissionIntensityLayer3"] == 0.0  # original untouched
    tm.colors["EmissionColorLayer3"] = (1.0, 1.0, 1.0, 1.0)
    assert imp._mode_material(tm, (3, 4)) is None


def test_split_mode_meshes_follows_the_mode_meshes():
    from pipeline.common.model import Clip, Mesh, Primitive

    def prim(mat):
        return Primitive(material=mat, indices=np.zeros(3, np.uint32), positions=np.zeros((1, 3), np.float32))

    meshes = [Mesh("x_body_mesh", [prim("body"), prim("horn")]), Mesh("x_horn_a_mesh", [prim("body")], visible=True),
              Mesh("x_inc_mesh", [prim("horn")], visible=False)]
    clip = Clip(name="battlewait01_loop", tracks=[])
    clip.visibility = {"x_horn_a_mesh": [(0, False)], "x_inc_mesh": [(0, True)]}
    imp._split_mode_meshes("x", {"active": ("inc_mesh",), "neutral": "horn_a_mesh"}, meshes, {"horn": 1, "horn~active": 1}, [clip])
    by = {m.name: m for m in meshes}
    assert [p.material for p in by["x_body_mesh"].primitives] == ["body"]
    assert by["x_body_mesh_neutral"].visible and not by["x_body_mesh_active"].visible
    assert by["x_body_mesh_active"].primitives[0].material == "horn~active"
    assert by["x_inc_mesh"].primitives[0].material == "horn~active"
    assert clip.visibility["x_body_mesh_neutral"] == [(0, False)] and clip.visibility["x_body_mesh_active"] == [(0, True)]
