import numpy as np

from pipeline.importers.za import importer as imp
from pipeline.importers.za import trinity as T


def test_full_width_clamp_mask_is_not_doubled():
    # Croconaw body_b: half-width mirrored albedo, full-width clamp-sampled layer mask.
    alb = np.full((4, 8, 4), 255, np.uint8)
    lym = np.zeros((8, 16, 4), np.uint8)
    lym[:, :8, 0] = 255   # left half: layer 1
    lym[:, 8:, 1] = 255   # right half: layer 2 (a symmetric mask would mirror this; here it differs)
    texs = {"a.bntx": alb, "m.bntx": lym}
    tm = T.TrMaterial(
        name="body_b", shader="IkCharacter", shader_flags={"EnableBaseColorMap": "True"},
        textures=[T.TextureBinding("BaseColorMap", "a.bntx", 0), T.TextureBinding("LayerMaskMap", "m.bntx", 1)],
        samplers=[("mirror", "clamp"), ("clamp", "clamp")], floats={},
        colors={"UVScaleOffset": (2.0, 1.0, 0.0, 0.0), "BaseColorLayer1": (1.0, 0.0, 0.0, 1.0), "BaseColorLayer2": (0.0, 1.0, 0.0, 1.0)},
        ints={}, alpha_type="Opaque", cull=-1,
    )
    textures = {}
    mat = imp._convert_material(tm, "", texs.get, textures, {})
    baked = textures[mat.base_color].rgba
    # Only the left (layer 1, red) half survives, then mirrors: no green anywhere.
    assert baked.shape[1] == 2 * 8
    assert baked[..., 1].max() < 10 and baked[..., 0].min() > 100
