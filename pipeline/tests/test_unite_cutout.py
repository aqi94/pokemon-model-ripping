"""UNITE cut-out albedo rule (`flag_cutouts`): solid-majority bimodal alpha + UVs landing on alpha 0 -> MASK."""
import numpy as np
from PIL import Image

from pipeline.common.model import Material, Mesh, Model, Primitive
from pipeline.importers.unite import importer as I


def _prim(uv_box, mat="m"):
    (u0, v0), (u1, v1) = uv_box
    uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], np.float32)
    return Primitive(material=mat, indices=np.array([0, 1, 2, 0, 2, 3], np.uint32),
                     positions=np.zeros((4, 3), np.float32), uvs=[uv])


def _model(tmp_path, alpha, uv_box):
    rgba = np.zeros(alpha.shape + (4,), np.uint8)
    rgba[..., :3] = 200
    rgba[..., 3] = alpha
    Image.fromarray(rgba).save(tmp_path / "t_d_x.png")
    mat = Material(name="m", base_color="t_d_x")
    mat.extras = {"unity": {"source": "fbx", "textures": {"_MainTex": "t_d_x"}, "floats": {}}}
    model = Model(id="x", game="unite", display_name="x", kind="pokemon", skeleton=None, meshes=[], materials={}, textures={})
    model.materials = {"m": mat}
    model.textures = {}
    model.meshes = [Mesh(name="a", primitives=[_prim(uv_box)])]
    return model


def _run(tmp_path, alpha, uv_box):
    model = _model(tmp_path, alpha, uv_box)
    out = I.flag_cutouts(model, str(tmp_path), None)
    return model, out


def test_binary_strip_under_uvs_is_mask(tmp_path):
    a = np.full((100, 100), 255, np.uint8)
    a[40:60] = 0  # a clean cut-out strip
    model, out = _run(tmp_path, a, ((0.0, 0.3), (1.0, 0.7)))
    assert out == ["m"]
    mat = model.materials["m"]
    assert mat.alpha_mode == "MASK" and mat.alpha_cutoff == 0.5
    assert model.textures[mat.base_color].rgba[50, 50, 3] == 0


def test_cloth_mask_stays_opaque(tmp_path):
    a = np.zeros((100, 100), np.uint8)
    a[:, :20] = 255  # mostly alpha 0
    model, out = _run(tmp_path, a, ((0.0, 0.0), (1.0, 1.0)))
    assert out == [] and model.materials["m"].alpha_mode == "OPAQUE"


def test_padding_outside_uvs_stays_opaque(tmp_path):
    a = np.full((100, 100), 255, np.uint8)
    a[80:] = 0  # alpha-0 padding the UVs never reach
    model, out = _run(tmp_path, a, ((0.0, 0.0), (1.0, 0.5)))
    assert out == [] and model.materials["m"].alpha_mode == "OPAQUE"


def test_soft_alpha_stays_opaque(tmp_path):
    a = np.full((100, 100), 255, np.uint8)
    a[40:60] = 0
    a[:, :30] = 128  # a gradient-like band: not bimodal
    model, out = _run(tmp_path, a, ((0.0, 0.3), (1.0, 0.7)))
    assert out == []


def test_json_materials_are_not_cut_by_statistics(tmp_path):
    a = np.full((100, 100), 255, np.uint8)
    a[40:60] = 0
    model = _model(tmp_path, a, ((0.0, 0.3), (1.0, 0.7)))
    model.materials["m"].extras["unity"]["source"] = "json"
    assert I.flag_cutouts(model, str(tmp_path), None) == []
