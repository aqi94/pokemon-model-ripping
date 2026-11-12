"""Texture-less UNITE effect materials borrow their texture from the same species' other FBX (same material name)."""
import numpy as np
from PIL import Image

from pipeline.importers.unite import importer as imp


def test_effect_key_ignores_artist_tag_and_case():
    assert imp.effect_key("mat_fx_rig_bat_0911_04_hb") == imp.effect_key("MAT_FX_RIG_BAT_0911_04")
    assert imp.effect_key("mat_fx_rig_bat_0911_03_hb") != imp.effect_key("mat_fx_rig_bat_0911_01")
    assert imp.effect_key("mat_fx_x.001") == "mat_fx_x"


def _share(tmp_path):
    root = tmp_path / "3D Models"
    for kind, name in (("Battle Models", "pref_char_bat_091100_skeledirge_lv3"),
                       ("Lobby Models", "pref_char_lob_091100_skeledirge_lv3"),
                       ("Lobby Models", "pref_char_lob_000100_other_lv3")):
        d = root / kind / name
        d.mkdir(parents=True)
        (d / (name + ".fbx")).write_bytes(b"")
    Image.fromarray(np.full((4, 4, 4), 200, np.uint8)).save(root / "Battle Models" / "pref_char_bat_091100_skeledirge_lv3" / "t_fx_a.png")
    return root


def test_borrow_from_other_fbx_of_same_dex(tmp_path, monkeypatch):
    root = _share(tmp_path)
    bat = str(root / "Battle Models" / "pref_char_bat_091100_skeledirge_lv3" / "pref_char_bat_091100_skeledirge_lv3.fbx")
    lob = str(root / "Lobby Models" / "pref_char_lob_091100_skeledirge_lv3")
    monkeypatch.setattr(imp.fbxr, "material_textures", lambda p: {"mat_fx_rig_bat_0911_04_hb": {"DiffuseColor": "t_fx_a.png"}})
    monkeypatch.setattr(imp.fbxr, "material_blocks", lambda p: {"mat_fx_rig_bat_0911_04_hb": ["t_fx_a.png"]})
    monkeypatch.setattr(imp.fbxr, "mesh_materials", lambda p: {"m": ["mat_fx_rig_bat_0911_04_hb"]})
    monkeypatch.setattr(imp.fbxr, "mesh_vertex_colours", lambda p: {"m": np.array([0.8, 0.4, 0.1, 1.0])})
    assert imp.sibling_fbxs(lob) == [bat]  # another dex is not a sibling, the file itself is excluded
    got = imp.borrow_effect_info("mat_fx_rig_bat_0911_04", lob, None, {})
    assert got is not None and got[2] == "mat_fx_rig_bat_0911_04_hb"
    assert got[0]["textures"]["_MainTex"]["name"].startswith("t_fx_a")
    assert np.allclose(got[3], [0.8, 0.4, 0.1, 1.0])
    assert imp.borrow_effect_info("mat_fx_rig_bat_0911_01", lob, None, {}) is None  # a different material: no borrow
