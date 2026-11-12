from pipeline.importers.unite import shading as sh

CLOTH = "==PGAME==/LobbyPokemon/m_lob_basepbrCloth_lv3"


def test_json_material_constants():
    i = {"shader": CLOTH, "floats": {"_SHScale": 0.8, "_RoughNessOffset": -0.1}, "colors": {"_SHTopColor": (4.0, 3.0, 2.0, 1.0), "_RimlightColor": (51, 13, 5, 0.8)}, "textures": {}}
    p = sh.shading_params(i, "cloth")
    assert p["form"] == "cloth" and p["specTerm"] == "nov"
    assert p["shTop"] == [4.0, 3.0, 2.0] and p["shScale"] == 0.8 and p["roughOffset"] == -0.1
    assert p["shBot"] == [2.5, 2.5, 2.5]  # shader default for a key the JSON lacks
    assert "rim" not in p and "inferred" not in p  # the Cloth rim is a global, the JSON value is stale


def test_fbx_only_uses_defaults_and_is_marked():
    p = sh.shading_params({"shader": "", "source": "fbx", "floats": {}, "colors": {}, "textures": {}}, "actorbase")
    assert p["inferred"] and p["form"] == "battle" and p["specTerm"] == "ndh" and p["lightClamp"] == 1.2
    assert p["shTop"] == [2.5, 2.5, 2.5] and p["specColor"] == [1.0, 1.0, 1.0]


def test_only_lit_families_and_fur_rim():
    assert sh.shading_params({"shader": "", "textures": {}}, "fx") is None
    assert sh.shading_params({"shader": "", "textures": {}}, "glass") is None
    i = {"shader": "Hidden/Lobby/m_lob_basepbr5pass", "floats": {"_RimlightScale": 6.44}, "colors": {"_RimlightColor": (51.4, 13.2, 4.8, 0.8)}, "textures": {}}
    p = sh.shading_params(i, "fur")
    assert p["rim"]["scale"] == 6.44 and p["rim"]["color"][0] == 51.4
