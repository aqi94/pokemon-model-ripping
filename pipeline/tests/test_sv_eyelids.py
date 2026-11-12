"""SV/LA eyes: sliding-lid overlays without Z-A's EnableEyeOptions flag (Sinistcha's expression atlas), the upper lid
driven by UVScaleOffset3, no static lid paint from a flat shadow mask, the overlay's UV tile taken from the eye's own
vertices (shared body/eye vertex buffers), and catchlight parameters for highlight-lobe eyes."""
import numpy as np

from pipeline.importers.za import importer as imp
from pipeline.importers.za import trinity as T
from pipeline.importers.za.importer import lid_overlay_uvs, part_tile


def _eye(flags=None, lid_alpha=None, shadow_mask=None, floats=None, ints=None):
    alb = np.full((8, 8, 4), 255, np.uint8)
    lym = np.zeros((8, 8, 4), np.uint8)
    lym[..., 0] = 255  # layer 1 (yellow) everywhere
    lid = np.full((8, 8, 4), 255, np.uint8)
    lid[..., 3] = 255 if lid_alpha is None else lid_alpha  # opaque lid ...
    if lid_alpha is None:
        lid[2:4, 2:6, 3] = 0  # ... with an eye-shaped hole (Sinistcha's atlas cells)
    msk = np.full((4, 4, 4), 255, np.uint8) if shadow_mask is None else shadow_mask
    texs = {"a.bntx": alb, "m.bntx": lym, "lid.bntx": lid, "s.bntx": msk}
    tm = T.TrMaterial(
        name="r_eye_a", shader="EyeClearCoat",
        shader_flags={"EnableHighlight": "False", "EyelidType": "Upper", **(flags or {})},
        textures=[T.TextureBinding("BaseColorMap", "a.bntx", 0), T.TextureBinding("LayerMaskMap", "m.bntx", 1),
                  T.TextureBinding("EyelidShadowMaskMap", "s.bntx", 2), T.TextureBinding("UpperEyelidColorMap", "lid.bntx", 3)],
        samplers=[("clamp", "clamp")] * 4,
        floats=floats or {},
        colors={"UVScaleOffset": (1.0, 1.0, 0.0, 0.0), "BaseColorLayer1": (0.98, 0.9, 0.24, 1.0),
                "BaseColorLayer7": (0.15, 0.42, 0.15, 1.0), "UVScaleOffset2": (1.0, 1.0, 0.0, 0.0),
                "UVScaleOffset3": (1.0, 1.0, 0.0, 0.0)},
        ints=ints or {}, alpha_type="Blend", cull=-1,
    )
    return tm, texs


def test_sv_upper_lid_overlay_without_eye_options_flag():
    tm, texs = _eye()
    textures, ovs = {}, []
    mat = imp._convert_material(tm, "", texs.get, textures, {}, overlays=ovs)
    assert [o.name for o in ovs] == ["r_eye_a~lidU"]
    assert ovs[0].extras["uv_track_param"] == "UVScaleOffset3"
    # The eye under the lid keeps its yellow (no lid paint over it).
    eye = textures[mat.base_color].rgba
    assert eye[..., 0].min() > 200 and eye[..., 2].max() < 160


def test_flat_shadow_mask_does_not_paint_the_eye():
    # A fully opaque lid gives no overlay (no lid shape), and the flat white EyelidShadowMaskMap must not
    # fill the eye with the lid colour (Sinistcha rendered solid green).
    tm, texs = _eye(lid_alpha=255)
    textures, ovs = {}, []
    mat = imp._convert_material(tm, "", texs.get, textures, {}, overlays=ovs)
    assert not ovs
    eye = textures[mat.base_color].rgba
    assert eye[..., 0].min() > 200  # still yellow, not green (layer 7 = 0.15 linear red)


def test_overlay_tile_comes_from_the_eye_vertices():
    # Shared buffer: body vertices in V tile 1, eye vertices in tile 0 (SV Corphish).
    uv = np.array([[0.5, 1.2], [0.5, 1.5], [0.5, 1.8], [0.5, 1.9], [0.2, 0.1], [0.8, 0.1], [0.5, 0.9]], np.float32)
    eye_idx = np.array([4, 5, 6], np.uint32)
    assert part_tile(uv, eye_idx) == 0.0
    out = lid_overlay_uvs(uv, {"scale": (1.0, 1.0), "offset": (0.0, 0.6), "center": (0.0, 0.0)}, part_tile(uv, eye_idx))
    assert np.allclose(out[eye_idx, 1], [1.5, 1.5, 0.7], atol=1e-6)  # v' = (1 - v) + 0.6


def test_highlight_lobe_eye_gets_catchlight_params():
    tm, texs = _eye(flags={"EnableHighlight": "True", "EyelidType": "None"},
                    floats={"RoughnessHighlight": 0.56, "MetallicHighlight": 1.0, "EmissionIntensityLayer5": 10.0},
                    ints={"PointLightIndex": 2})
    textures = {}
    mat = imp._convert_material(tm, "", texs.get, textures, {}, overlays=[])
    eh = mat.extras["eyeHighlight"]
    assert eh["roughness"] == 0.56 and eh["intensity"] == 10.0 and eh["light"] == 2
    # No glow intensity = no catchlight.
    tm2, texs2 = _eye(flags={"EnableHighlight": "True", "EyelidType": "None"}, floats={"RoughnessHighlight": 0.56})
    assert "eyeHighlight" not in imp._convert_material(tm2, "", texs2.get, {}, {}, overlays=[]).extras


def test_translate_only_uv_mode_ignores_the_scale():
    # Magnemite: UVTransformMode T with UVScaleOffset (2, 4): the eye UVs already address one atlas cell.
    tm, texs = _eye(flags={"EyelidType": "None", "UVTransformMode": "T"})
    tm.colors["UVScaleOffset"] = (2.0, 4.0, 0.0, 0.0)
    mat = imp._convert_material(tm, "", texs.get, {}, {}, overlays=[])
    assert mat.extras["uv_scale_effective"][:2] == (1.0, 1.0)
    # Its clip tracks carry the same (ignored) scale, so offset steps are still exported.
    f = np.array([0.0, 10.0], np.float32)
    tr = T.MaterialParamTrack("s", "r_eye_a", "UVScaleOffset", [(f, np.full(2, 2.0, np.float32)), (f, np.full(2, 4.0, np.float32)),
                                                                 (f, np.array([0.0, 0.5], np.float32)), (f, np.zeros(2, np.float32))])
    out = imp.material_uv_offsets([tr], {"r_eye_a": mat}, 60.0)
    assert np.allclose(out["r_eye_a"][1][:, 0], [0.0, 0.5])


def test_eye_clear_coat_is_written_as_khr_clearcoat():
    tm, texs = _eye(flags={"EyelidType": "None"}, floats={"RoughnessClearCoat": 0.2, "NormalHeight1": 0.2})
    tm.textures.append(T.TextureBinding("NormalMap1", "n1.bntx", 4))
    tm.samplers.append(("clamp", "clamp"))
    texs["n1.bntx"] = np.full((4, 4, 4), 128, np.uint8)
    mat = imp._convert_material(tm, "", texs.get, {}, {}, overlays=[])
    cc = mat.extras["clearcoat"]
    assert cc["factor"] == 1.0 and abs(cc["roughness"] - 0.2) < 1e-6 and abs(cc["normalScale"] - 0.2) < 1e-6 and cc["normal"]
