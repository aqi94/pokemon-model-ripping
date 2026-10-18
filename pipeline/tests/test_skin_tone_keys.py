import numpy as np

from pipeline.importers.za import importer as imp
from pipeline.importers.za import trinity as T


def _skin(lym_file: str):
    # Two outfits share the albedo but paint different layers of body_skin (Artist's hands vs a plain body).
    alb = np.full((4, 4, 4), 200, np.uint8)
    lym = np.zeros((4, 4, 4), np.uint8)
    lym[..., 0] = 255
    if lym_file == "paint.bntx":
        lym[:, :2, 1] = 255  # layer 2 = paint
    table = np.zeros((4, 4, 4), np.uint8)
    table[..., 3] = 255
    table[:2, 1] = (245, 215, 191, 255)  # zone 1 lit (skin)
    table[:2, 3] = (254, 134, 20, 255)   # zone 2 lit (orange paint)
    texs = {"alb.bntx": alb, lym_file: lym, "ct.bntx": table}
    tm = T.TrMaterial(
        name="body_skin", shader="IkCharacter",
        shader_flags={"EnableBaseColorMap": "True", "EnableColorTableMap": "True"},
        textures=[T.TextureBinding("BaseColorMap", "alb.bntx", 0), T.TextureBinding("LayerMaskMap", lym_file, 1),
                  T.TextureBinding("ColorTableMap", "ct.bntx", 2)],
        samplers=[("clamp", "clamp")] * 3, floats={f"LayerMaskScale{i}": 1.0 for i in range(1, 5)},
        colors={}, ints={"ColorTableDivideNumber": 2, "BaseColorIndex1": 1, "BaseColorIndex2": 2, "BaseColorIndex3": 1, "BaseColorIndex4": 1},
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
