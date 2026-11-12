"""UNITE `LobbyPlayer` shader maths (pipeline/importers/unite/player_shaders.py) on synthetic inputs."""

import numpy as np

from pipeline.importers.unite import player_shaders as ps


def test_shader_names():
    assert ps.player_kind("==PGAME==/LobbyPlayer/m_lob_PlayerPBRHairColor_lv3") == "hair"
    assert ps.player_kind("==PGAME==/LobbyPlayer/m_lob_playerpbrSkinNoShadow_lv3") == "skin"
    assert ps.player_kind("==PGAME==/LobbyPlayer/m_lob_playerpbrRampLight_lv3") == "ramp"
    assert ps.player_kind("==PGAME==/LobbyPlayer/m_lob_PlayerPBRColorEye") == "eye"
    assert ps.player_kind("Hidden/m_lob_PlayerPBRLengcy_lv3") == "lengcy"
    assert ps.player_kind("==PGAME==/LobbyPlayer/m_lob_PlayerPBRHairColorFelt_lv3") is None
    assert ps.player_kind("Unlit/Texture") is None


def test_zone_weights_pick_the_zone():
    for k, a in enumerate((0.0, 85 / 255, 170 / 255, 1.0)):
        w = ps.zone_weights(np.float32(a))
        assert int(np.argmax(w)) == k and w[k] == 1.0
        assert w.sum() < 1.2  # neighbours overlap only at the tent feet (slope 3.19, peak 1.032)


def test_hair_erbie_zone_two():
    # Erbie's mask is (255, 0, 0, 85): zone 1 of 0..3 -> `_color2`; `_Mask` on, M.r = 1 so the texture is used
    mix = np.zeros((2, 2, 4), np.float32)
    mix[..., 0], mix[..., 3] = 1.0, 85 / 255
    main = np.ones((2, 2, 3), np.float32)
    colors = {"_color1": (.05,) * 4, "_color2": (.04,) * 4, "_color3": (.03,) * 4, "_color4": (.5,) * 4,
              "_colorSkin": (0, 0, 0, 1)}
    rgb, alpha, emission = ps.player_albedo("hair", main, mix, None, colors, {"_Mask": 1.0, "_Emissive": 1.0})
    assert np.allclose(rgb, 0.04, atol=1e-3) and np.allclose(alpha, 1.0) and emission is None


def test_hair_mask_blends_to_skin_colour():
    mix = np.zeros((1, 1, 4), np.float32)  # R = 0: fully cut away, colour = _colorSkin
    colors = {"_colorSkin": (0.2, 0.3, 0.4, 1)}
    rgb, alpha, _ = ps.player_albedo("hair", np.ones((1, 1, 3), np.float32), mix, None, colors, {"_Mask": 1.0})
    assert np.allclose(rgb[0, 0], (0.2, 0.3, 0.4)) and alpha[0, 0] == 0.0
    rgb, alpha, _ = ps.player_albedo("hair", np.ones((1, 1, 3), np.float32), mix, None, colors, {"_Mask": 0.0})
    assert alpha is None and not np.allclose(rgb[0, 0], (0.2, 0.3, 0.4))  # no mask: the albedo shows


def test_hair_green_channel_lifts_tint_to_white():
    mix = np.zeros((1, 1, 4), np.float32)
    mix[..., 0], mix[..., 1] = 1.0, 1.0
    colors = {f"_color{i}": (0.1, 0.1, 0.1, 1) for i in range(1, 5)}
    rgb, _, _ = ps.player_albedo("hair", np.full((1, 1, 3), 0.5, np.float32), mix, None, colors, {"_Mask": 1.0})
    assert np.allclose(rgb, 0.5)


def test_skin_tint_and_metalness():
    main = np.full((1, 1, 3), 0.5, np.float32)
    colors = {"_colorSkin": (1.0, 0.8, 0.6, 1)}
    rgb, alpha, emission = ps.player_albedo("skin", main, None, np.zeros((1, 1), np.float32), colors, {})
    assert np.allclose(rgb[0, 0], (0.5, 0.4, 0.3)) and alpha is None and emission is None
    rgb, _, _ = ps.player_albedo("skin", main, None, np.ones((1, 1), np.float32), colors, {})
    assert np.allclose(rgb[0, 0], 0.5)  # normal-map alpha ("metalness") 1 -> untinted


def test_emission_needs_both_toggles_and_mask():
    main = np.full((1, 1, 3), 0.5, np.float32)
    mix = np.zeros((1, 1, 4), np.float32)
    mix[..., 2] = 1.0
    _, _, e = ps.player_albedo("skin", main, mix, None, {}, {"_Emissive": 1.0, "_EmisssionScale": 2.0})
    assert np.allclose(e, 1.0)
    _, _, e = ps.player_albedo("skin", main, mix, None, {}, {"_Emissive": 0.0, "_EmisssionScale": 2.0})
    assert e is None


def test_eye_is_unlit_albedo_times_one_plus_emission():
    mix = np.zeros((1, 1, 4), np.float32)
    mix[..., 0], mix[..., 1], mix[..., 2], mix[..., 3] = 1.0, 1.0, 1.0, 174 / 255
    colors = {f"_color{i}": (3.0, 1.0, 0.4, 1) for i in range(1, 5)}
    rgb, alpha, _ = ps.player_albedo("eye", np.full((1, 1, 3), 0.3, np.float32), mix, None, colors, {})
    assert np.allclose(rgb, 0.6) and alpha[0, 0] == 1.0  # M.g = 1 -> tint 1; x (1 + M.b)
    mix[..., 1] = 0.0  # no white lift: tint * 1.2 (zone 2 = _color3 here)
    rgb, _, _ = ps.player_albedo("eye", np.full((1, 1, 3), 0.3, np.float32), mix, None, colors, {})
    assert rgb[0, 0, 0] > rgb[0, 0, 1] > rgb[0, 0, 2]


def test_roughness():
    a = np.full((1, 1), 0.5, np.float32)
    assert np.isclose(ps.roughness_of("skin", a, None, {}, {"_RoughNessOffset": 0.2})[0, 0], 0.45)
    assert np.isclose(ps.roughness_of("lengcy", np.zeros((1, 1), np.float32), None, {}, {})[0, 0], ps.ROUGH_MIN)
    mix = np.zeros((1, 1, 4), np.float32)  # hair: zone 0, no white lift -> _roughv4[0] + MainTex.a
    r = ps.roughness_of("hair", a, mix, {"_roughv4": (0.25, 1, 1, 1)}, {})
    assert np.isclose(r[0, 0], 0.75)


def test_ramp_emission_only_where_second_uv_x_is_at_most_one():
    ramp = np.zeros((4, 8, 3), np.float32)
    ramp[:, 0] = (0.5, 1.0, 0.8)  # the left end (time 0)
    mask = np.ones((4, 4), np.float32)
    uv0 = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]])
    idx = np.array([0, 1, 2])
    fl = {"_Rampscale": 4.0, "_Rampscale2": 1.0}
    on, cover = ps.ramp_emission(uv0, np.array([[0.2, 0.2], [0.4, 0.2], [0.2, 0.4]]), idx, ramp, mask, fl, (16, 16))
    assert cover.any() and np.allclose(on[cover], (2.0, 4.0, 3.2))  # ramp x (mask 1 -> Rampscale 4)
    off, cover2 = ps.ramp_emission(uv0, np.array([[1.2, 0.2], [1.4, 0.2], [1.2, 0.4]]), idx, ramp, mask, fl, (16, 16))
    assert not cover2.any() and not off.any()


def test_dilate_grows_coverage():
    rgb = np.zeros((6, 6, 3), np.float32)
    cover = np.zeros((6, 6), bool)
    rgb[3, 3], cover[3, 3] = 1.0, True
    out = ps.dilate(rgb, cover, 1)
    assert out[3, 4, 0] == 1.0 and out[2, 3, 0] == 1.0 and out[0, 0, 0] == 0.0
