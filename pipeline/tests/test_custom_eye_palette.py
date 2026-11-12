import numpy as np

from pipeline.common.model import Material
from pipeline.importers.za import trinity as T
from pipeline.importers.za.importer import custom_eye_palette


def _tm(scales=(1.0, 1.0, 1.0, 0.0)):
    return T.TrMaterial(
        name="l_eye", shader="IkCharacter", shader_flags={}, textures=[], samplers=[],
        floats={f"LayerMaskScale{i + 1}": s for i, s in enumerate(scales)}, colors={}, ints={}, alpha_type="Opaque", cull=0,
    )


def _run(own, scales=(1.0, 1.0, 1.0, 0.0)):
    mat = Material(name="l_eye")
    mat.extras["mirror"] = (False, False)
    rgba = np.zeros((4, 4, 4), np.uint8)
    reg = lambda key, img, srgb, mirror=(False, False): key
    custom_eye_palette(mat, _tm(scales), rgba, rgba, "a_alb.bntx", "a_lym.bntx", own + [None], [1.0, 1.0, 1.0, 0.0], reg)
    return mat.extras.get("palette")


def test_pairs_follow_the_original_zones():
    pal = _run([(0.1, 0.1, 0.1), (0.0, 0.25, 1.0), (0.43, 0.6, 0.87)])
    assert pal["custom"] and pal["pairs"] and pal["layers"] == [2]
    assert pal["indices"] == [1, 2, 3, 1]
    assert len(pal["lit"]) == 3 + 2 * (len(pal["pairLabels"]) - 1) and len(pal["shadow"]) == len(pal["lit"])


def test_dark_inner_layer_keeps_its_colour():
    pal = _run([(0.1, 0.1, 0.1), (0.5, 0.25, 0.1), (0.02, 0.02, 0.02)])
    assert all(pal["lit"][3 + 2 * j + 1] == pal["lit"][2] for j in range(len(pal["pairLabels"]) - 1))


def test_white_or_disabled_iris_layer_is_left_alone():
    assert _run([(0.1, 0.1, 0.1), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5)]) is None
    assert _run([(0.1, 0.1, 0.1), (0.0, 0.25, 1.0), (0.4, 0.6, 0.9)], scales=(0, 0, 0, 0)) is None


def test_painted_eye_is_split_into_regions():
    from pipeline.importers.za.importer import paint_eye_regions

    yy, xx = np.mgrid[0:64, 0:64]
    r = np.hypot((xx - 32) / 18.0, (yy - 32) / 24.0)
    alb = np.zeros((64, 64, 4), np.uint8)
    alb[..., :3], alb[..., 3] = 225, 255  # light background
    alb[r < 1.0, :3] = (50, 50, 55)  # outline
    alb[r < 0.8, :3] = (70, 100, 160)  # iris
    alb[(r < 0.8) & (yy > 38), :3] = (130, 170, 230)  # lighter inner iris
    alb[np.hypot(xx - 32, yy - 28) < 4, :3] = (40, 40, 40)  # pupil
    res = paint_eye_regions(alb)
    assert res is not None
    neutral, mask, own = res
    assert mask[28, 32, 0] > 200 and mask[22, 32, 1] > 200 and mask[46, 32, 2] > 200
    assert (neutral[0, 0, :3] == alb[0, 0, :3]).all()  # the background is untouched
    assert own[2][2] > own[1][2] * 0.9 and own[1][2] > own[1][0]  # blue iris, lighter inner iris


def test_painted_eye_without_regions_is_left_alone():
    from pipeline.importers.za.importer import paint_eye_regions

    assert paint_eye_regions(np.full((32, 32, 4), 200, np.uint8)) is None


def test_brow_palette_offers_layer_two_colours():
    from pipeline.importers.za.importer import custom_brow_palette

    mat = Material(name="eyebrow")
    mat.extras["mirror"] = (False, False)
    rgba = np.zeros((4, 4, 4), np.uint8)
    reg = lambda key, img, srgb, mirror=(False, False): key
    custom_brow_palette(mat, _tm((1.0, 1.0, 0.0, 0.0)), rgba, rgba, "a_alb.bntx", "a_lym.bntx",
                        [(0.02, 0.02, 0.02), (0.44, 0.44, 0.44), None, None], [1.0, 1.0, 0.0, 0.0], reg)
    pal = mat.extras["palette"]
    assert pal["custom"] and "pairs" not in pal and pal["layers"] == [2] and pal["indices"] == [1, 2, 1, 1]
    assert len(pal["zoneLabels"]) == len(pal["lit"]) == len(pal["shadow"]) and pal["zoneLabels"][1] == "Original"


def test_brow_palette_keeps_the_mask_resolution():
    """Common faces: 32x32 white placeholder albedo, 32x64 mask (brows G on top, lashes R below). The palette must keep
    the 64 rows; shrinking the mask to the albedo blurred the boundary and a recolour greyed the left lash."""
    from pipeline.importers.za.importer import custom_brow_palette

    mat = Material(name="eyebrow")
    mat.extras["mirror"] = (False, False)
    alb = np.full((32, 32, 4), 255, np.uint8)
    lym = np.zeros((64, 32, 4), np.uint8)
    lym[:32, :, 1] = 255
    lym[32:, :, 0] = 255
    imgs = {}

    def reg(key, img, srgb, mirror=(False, False)):
        imgs[key] = img
        return key

    custom_brow_palette(mat, _tm((1.0, 1.0, 0.0, 0.0)), lym, alb, "sh_white_msk.bntx", "sh_eyebrow_00_lym.bntx",
                        [(0.02, 0.02, 0.02), (0.44, 0.44, 0.44), None, None], [1.0, 1.0, 0.0, 0.0], reg)
    pal = mat.extras["palette"]
    assert imgs[pal["albedo"]].shape[:2] == (64, 32) and pal["albedo"] != "sh_white_msk_raw"
    mask = imgs[pal["mask"]]
    assert mask.shape[:2] == (64, 32)
    assert (mask[31, :, 1] == 255).all() and (mask[32, :, 0] == 255).all() and (mask[32, :, 1] == 0).all()


def test_player_palettes_are_the_custom_options():
    import json, os
    za = os.path.join(os.path.dirname(__file__), "..", "importers", "za")
    load = lambda n: json.load(open(os.path.join(za, n), encoding="utf-8"))
    assert len(load("eyebrow_colors.json")) >= 32  # the first 32 are p0_hairstyle_pallete01 (the player), the rest NPC hair-table extras
    assert [e["label"] for e in load("lip_colors.json")][:2] == ["Beige", "Pink"] and len(load("lip_colors.json")) == 9
    assert all(len(e["lit"]) == 2 for e in load("eye_colors.json"))


def _lips(layer_blob):
    from pipeline.importers.za.importer import lip_layer
    m = np.zeros((100, 100, 4), np.uint8)
    y0, y1, x0, x1 = layer_blob
    m[y0:y1, x0:x1, 1] = 255
    return lip_layer(m, [1, 1, 0, 1], [1, 6, 1, 5])


def test_lip_layer_is_a_small_blob_on_the_mouth():
    assert _lips((72, 78, 45, 56)) == 2  # mouth-sized oval at x .5, y .75
    assert _lips((10, 16, 45, 56)) == 0  # up in the hair
    assert _lips((60, 100, 0, 100)) == 0  # large region: facial hair / makeup


def test_lip_palette_offers_none_original_and_player_colours():
    from pipeline.importers.za.importer import custom_lip_palette
    mat = Material(name="face_skin")
    mat.extras["mirror"] = (False, False)
    rgba = np.zeros((4, 4, 4), np.uint8)
    ct = np.zeros((4, 16, 4), np.uint8)
    custom_lip_palette(mat, rgba, rgba, ct, 8, [1, 6, 1, 5], [1, 1, 0, 1], 2, "a_alb.bntx", "a_lym.bntx", lambda k, i, s, m=(False, False): k)
    pal = mat.extras["palette"]
    assert pal["skinLink"] and pal["layers"] == [2] and pal["offer"][:2] == [0, 6] and pal["offer"][2] == 9
    assert len(pal["lit"]) == 8 + 9 and pal["zoneLabels"][pal["offer"][-1] - 1] == "Black" and pal["noneLabel"] == "No lipstick"


def _beard(y0, y1, x0, x1, scl=(1, 1, 0, 1), idxs=(1, 5, 1, 5)):
    from pipeline.importers.za.importer import beard_paint_layer
    m = np.zeros((100, 100, 4), np.uint8)
    m[y0:y1, x0:x1, 1] = 255
    return beard_paint_layer(m, list(scl), list(idxs))


def test_beard_paint_layer_is_a_wide_mass_on_layer_2():
    assert _beard(60, 95, 10, 90) == 2  # jaw-wide mass (Jogger M)
    assert _beard(90, 100, 25, 75) == 0  # small neck patch: under 5% of the texture
    assert _beard(70, 100, 35, 65) == 0  # narrow (jewellery)
    assert _beard(60, 95, 10, 90, idxs=(5, 5, 1, 5)) == 0  # painted in the skin zone: not a beard
    assert _beard(60, 95, 10, 90, scl=(1, 0, 0, 1)) == 0  # layer off


def test_beard_palette_offers_no_beard_and_hair_colours():
    from pipeline.importers.za.importer import custom_lip_palette, _hair_colors
    mat = Material(name="face_skin")
    mat.extras["mirror"] = (False, False)
    rgba = np.zeros((4, 4, 4), np.uint8)
    ct = np.zeros((4, 16, 4), np.uint8)
    custom_lip_palette(mat, rgba, rgba, ct, 8, [1, 5, 1, 5], [1, 1, 0, 1], 2, "a_alb.bntx", "a_lym.bntx", lambda k, i, s, m=(False, False): k, colors=_hair_colors(), none_label="No beard")
    pal = mat.extras["palette"]
    assert pal["noneLabel"] == "No beard" and len(pal["lit"]) == 8 + len(_hair_colors()) and pal["offer"][:2] == [0, 5]


def test_single_layer_iris_gets_a_row_on_layer_one():
    # Adult M Common A-F: layer 1 = iris, layer 2 white, layer 3 unused.
    mat = Material(name="l_eye")
    mat.extras["mirror"] = (False, False)
    rgba = np.zeros((4, 4, 4), np.uint8)
    reg = lambda key, img, srgb, mirror=(False, False): key
    custom_eye_palette(mat, _tm((1.0, 1.0, 0.0, 0.0)), rgba, rgba, "a_alb.bntx", "a_lym.bntx",
                       [(0.25, 0.24, 0.27), (1.0, 1.0, 1.0), None, None], [1.0, 1.0, 0.0, 0.0], reg)
    pal = mat.extras["palette"]
    assert pal["custom"] and pal["layers"] == [1] and pal["indices"] == [2, 1, 1, 1]
    assert len(pal["lit"]) == len(pal["shadow"]) == len(pal["zoneLabels"])
