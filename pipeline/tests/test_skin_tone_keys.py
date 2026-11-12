import numpy as np

from pipeline.importers.za import importer as imp
from pipeline.importers.za import trinity as T


def _skin(lym_file: str, stroke: bool = False):
    # Two outfits share the albedo but paint different layers of body_skin (Artist's hands vs a plain body).
    alb = np.full((4, 4, 4), 200, np.uint8)
    lym = np.zeros((4, 4, 4), np.uint8)
    lym[..., 0] = 255
    if lym_file == "paint.bntx":
        lym[:, :2, 1] = 255  # layer 2 = paint
    if stroke:
        lym[0, 3, 2] = 255  # layer 3 = a thin stroke
    table = np.zeros((4, 8, 4), np.uint8)
    table[..., 3] = 255
    table[:2, 0] = (180, 140, 130, 255)  # zone 1 lit (deep skin)
    table[:2, 1] = (204, 162, 146, 255)  # zone 2
    table[:2, 2] = (245, 215, 191, 255)  # zone 3
    table[:2, 3] = (252, 222, 220, 255)  # zone 4 lit (pale skin)
    table[:2, 4] = (254, 134, 20, 255)   # zone 5 lit (orange paint)
    texs = {"alb.bntx": alb, lym_file: lym, "ct.bntx": table}
    tm = T.TrMaterial(
        name="body_skin", shader="IkCharacter",
        shader_flags={"EnableBaseColorMap": "True", "EnableColorTableMap": "True"},
        textures=[T.TextureBinding("BaseColorMap", "alb.bntx", 0), T.TextureBinding("LayerMaskMap", lym_file, 1),
                  T.TextureBinding("ColorTableMap", "ct.bntx", 2)],
        samplers=[("clamp", "clamp")] * 3, floats={f"LayerMaskScale{i}": 1.0 for i in range(1, 5)},
        colors={}, ints={"ColorTableDivideNumber": 8, "BaseColorIndex1": 1, "BaseColorIndex2": 5, "BaseColorIndex3": 4 if stroke else 1, "BaseColorIndex4": 1},
        alpha_type="Opaque", cull=-1,
    )
    textures = {}
    mat = imp._convert_material(tm, "", texs.get, textures, {}, skin_tones=True)
    return mat, textures


def test_skin_tone_textures_are_keyed_by_mask():
    plain, tex_plain = _skin("plain.bntx")
    paint, tex_paint = _skin("paint.bntx")
    key_plain = plain.extras["skin_tones"][0][1]
    key_paint = paint.extras["skin_tones"][0][1]
    assert key_plain != key_paint
    # the painted outfit's tone texture really carries the paint (orange: low blue)
    assert (tex_paint[key_paint].rgba[..., 2] < 80).any()
    assert not (tex_plain[key_plain].rgba[..., 2] < 80).any()


def test_painted_body_skin_layer_gets_skin_linked_palette():
    # Occult F's leggings: body_skin layer 2 in its own zone -> a runtime palette row; plain skin gets none.
    paint, _ = _skin("paint.bntx")
    pal = paint.extras["palette"]
    assert pal["layers"] == [2] and pal["skinLink"] and pal["indices"][0] == imp.SKIN_DEFAULT_TONE and pal["indices"][1] == 5
    plain, _ = _skin("plain.bntx")
    assert "palette" not in plain.extras


def test_skin_zone_stroke_is_one_shade_deeper_at_every_tone():
    # Rust Grunt M / Jub F: thin layer-3 strokes in a skin zone. The game shows the Grunt's furrow darker than his skin:
    # the stroke follows the selected tone one shade deeper (tone t -> zone t-1), never a pale line.
    mat, textures = _skin("plain.bntx", stroke=True)
    tones = {z: textures[key].rgba for z, key, _ in mat.extras["skin_tones"]}
    for z in (2, 3, 4):
        assert (tones[z][0, 3, :3] == tones[z - 1][0, 2, :3]).all()
    stroke, skin = tones[1][0, 3, :3].astype(int), tones[1][0, 2, :3].astype(int)
    assert (stroke < skin).all()  # below the deepest tone: the same step again
    assert not imp.skin_zone_layers(np.zeros((4, 4, 4), np.uint8), [1.0] * 4, [3, 5, 1, 1])


def test_thin_skin_zone_stroke_is_followed():
    # Rust Grunt F: strokes cover 0.03% of the texture (one texel of 64x64 here); no coverage floor.
    lym = np.zeros((64, 64, 4), np.uint8)
    lym[..., 0] = 255
    lym[10, 10, 2] = 200
    assert imp.skin_zone_layers(lym, [1.0, 0.0, 1.0, 1.0], [3, 1, 3, 5]) == [3]

