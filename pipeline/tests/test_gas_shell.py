import numpy as np

from pipeline.importers.za import importer as imp
from pipeline.importers.za import trinity as T


def _smoke_material(height: float, ints: dict | None = None) -> T.TrMaterial:
    # Gastly smoke: white albedo, partition layer mask (R+G+B+A = 1), cloud-density displacement map.
    return T.TrMaterial(
        name="smoke", shader="IkCharacter",
        shader_flags={"EnableBaseColorMap": "True", "EnableDisplacementMap": "True"},
        textures=[T.TextureBinding("BaseColorMap", "a.bntx", 0), T.TextureBinding("LayerMaskMap", "m.bntx", 1), T.TextureBinding("DisplacementMap", "d.bntx", 2)],
        samplers=[("clamp", "clamp")] * 3,
        floats={"DisplacementHeight": height, "ShadowStrength": 0.7},
        colors={
            "UVScaleOffset": (1.0, 1.0, 0.0, 0.0),
            "BaseColorLayer1": (0.2269, 0.0648, 0.4178, 1.0), "BaseColorLayer2": (0.1811, 0.0512, 0.3915, 1.0),
            "BaseColorLayer3": (0.2961, 0.0843, 0.4793, 1.0), "BaseColorLayer4": (0.1169, 0.0343, 0.3049, 1.0),
            "ShadowingColorLayer1": (0.049, 0.054, 0.14, 0.0), "ShadowingColorLayer2": (0.18, 0.056, 0.18, 0.0),
            "ShadowingColorLayer3": (0.22, 0.114, 0.219, 0.0), "ShadowingColorLayer4": (0.205, 0.12, 0.26, 0.0),
        },
        ints=ints or {}, alpha_type="Opaque", cull=1,
    )


def _textures():
    alb = np.full((8, 8, 4), 255, np.uint8)
    lym = np.zeros((8, 8, 4), np.uint8)
    lym[:, :2, 0] = 255   # four layers partition the surface in vertical bands
    lym[:, 2:4, 1] = 255
    lym[:, 4:6, 2] = 255
    lym[:, 6:, 3] = 255
    disp = np.zeros((8, 8, 4), np.uint8)
    disp[..., :3] = np.linspace(0, 255, 8, dtype=np.uint8)[None, :, None]  # density ramps left -> right
    disp[..., 3] = 255
    return {"a.bntx": alb, "m.bntx": lym, "d.bntx": disp}


def test_gas_shell_stays_opaque_and_ships_the_cloud_map():
    texs = _textures()
    textures = {}
    mat = imp._convert_material(_smoke_material(0.3, {"UVIndexLayer3": 1}), "", texs.get, textures, {})
    # Opaque in game: solid lumps, the face shows through a hole where the shell sinks into the body.
    assert mat.alpha_mode == "OPAQUE" and "blend" not in mat.extras
    baked = textures[mat.base_color].rgba
    # The lit colour is violet everywhere: the partition mask averages the layers instead of leaving the white
    # albedo through, and the shadow side is the layers' dark indigo, not light grey.
    assert (baked[..., 2] > baked[..., 1] + 60).all() and (baked[..., 0] < 180).all()
    shadow = textures[mat.extras["shadow_color"]].rgba
    assert shadow[..., :3].mean() < 70
    # Viewer data: the density map (repeat-wrapped by the writer) and the panner parameters.
    gas = textures[mat.extras["gas_density"]].rgba
    assert gas.shape == (8, 8, 4) and gas[0, 7, 0] == 255 and gas[0, 0, 0] == 0
    assert mat.extras["gas"]["height"] == 0.3 and mat.extras["gas"]["texCoord"] == 1
    assert "unlit" in mat.extras["shading"] and not mat.extras["shading"]["unlit"]


def test_small_displacement_is_not_a_gas_shell():
    texs = _textures()
    textures = {}
    mat = imp._convert_material(_smoke_material(0.1), "", texs.get, textures, {})
    assert mat.alpha_mode == "OPAQUE" and "gas" not in mat.extras


def test_gas_displacement_sinks_the_face_band_and_pushes_the_back():
    # Gastly's encoding: vertex colour R 0.5 (face) .. 1.0 (back); mid-grey density everywhere.
    colour = np.array([0.5, 0.6, 1.0], np.float32)
    dens = np.full(3, 0.5, np.float32)
    offset, w = imp.gas_displacement(colour, dens, 0.3)
    assert np.allclose(w, [1e-3, 0.2, 1.0])
    assert np.isclose(offset[0], -0.3 * imp.GAS_RETRACT)          # face: sunk, no lumps
    assert np.isclose(offset[1], 0.3 * (0.2 * 0.5 - imp.GAS_RETRACT * (1 - 0.2 / imp.GAS_RETRACT_END)))
    assert np.isclose(offset[2], 0.3 * 0.5)                        # back: plain outward cloud, no pull
    # Other encodings (Koffing-style 0..0.8) keep a plain weighted push, never negative.
    offset2, w2 = imp.gas_displacement(np.array([0.0, 0.8], np.float32), dens[:2], 0.3)
    assert offset2.min() >= 0 and np.allclose(w2, [1e-3, 0.8])
    # No vertex colours: weight 1.
    offset3, w3 = imp.gas_displacement(None, dens, 0.3)
    assert np.allclose(offset3, 0.15) and np.allclose(w3, 1.0)
