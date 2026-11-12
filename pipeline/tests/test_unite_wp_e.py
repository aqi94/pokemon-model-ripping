import numpy as np
from PIL import Image

from pipeline.importers.unite import fx
from pipeline.importers.unite.importer import build_effect_material


def _info(shader, tex, floats=None, colors=None):
    return {"shader": shader, "source": "json", "floats": floats or {}, "colors": colors or {},
            "textures": {k: {"name": v} for k, v in tex.items()}}


def test_color_alpha_ignored_and_emission(tmp_path):
    Image.new("RGBA", (2, 2), (200, 100, 50, 255)).save(tmp_path / "a.png")
    info = _info("==PGAME==/Fx/m_1P_Base", {"_Tex1": "a"}, {"_Alpha1": 0.5, "_Emission1": 0.5, "_BlendMode": 3.0},
                 {"_Color1": [1.0, 1.0, 1.0, 0.0]})
    m = build_effect_material("m", info, str(tmp_path), {})
    assert m.base_color_factor[3] == 0.5  # `_Color1.a` 0 no longer hides it
    assert abs(m.emissive_factor[0] - 0.5) < 1e-6
    assert m.extras["unite"]["fx"]["family"] == "base"


def test_mask_and_dissolve_bake():
    p = {"family": "mask", "power": 1.0, "st": {}, "roles": {"_Tex2": "mask"}}
    main = np.full((4, 4, 4), 255, np.uint8)
    mask = np.zeros((4, 4, 4), np.uint8)
    mask[:, :2] = 255
    out = fx.bake(p, main, {"_Tex2": mask})
    assert out[0, 0, 3] == 255 and out[0, 3, 3] == 0
    assert fx.smoothstep_dissolve(np.array([1.0]), 0.5, 1.0)[0] == 1.0
    assert fx.smoothstep_dissolve(np.array([0.0]), 0.5, 1.0)[0] == 0.0


def test_color_alpha_shaders():
    assert fx.uses_color_alpha(_info("==PGAME==/Fx/m_1P_Modulate", {}))
    assert not fx.uses_color_alpha(_info("==PGAME==/Fx/m_2P_mask", {}))
    assert fx.plan(_info("==PGAME==/Fx/m_3P_2Mask_Overlay", {"_Dtex": "x"}))["colorSlot"] == "_Dtex"
