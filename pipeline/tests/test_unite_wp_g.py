from pipeline.importers.unite import outline


def info(shader, cols=None, fl=None, tex=None, block=None, **kw):
    return {"shader": shader, "floats": fl or {}, "colors": cols or {}, "textures": tex or {}, "block": block or [], **kw}


def test_json_outline_from_data():
    i = info("==PGAME==/LobbyPokemon/m_lob_basepbrCloth_lv3", {"_OutlineColor": (0.2, 0.3, 0.4, 1), "_lightDir": (-4.5, 7.3, 1.3, 0)},
             {"_VTEX": 1.0, "_Offset": -5.0}, {"_OutlineMap": {"name": "t_vc_lob_000303"}})
    o = outline.outline_params(i)
    assert o["color"] == [0.2, 0.3, 0.4] and o["lightDir"][0] == -4.5 and o["mapName"] == "t_vc_lob_000303"
    assert abs(o["width_ndc"] - 0.00462963) < 1e-7 and o["vtex"] and o["source"] == "json"


def test_no_pass_shaders():
    assert outline.outline_params(info("==PGAME==/Fx/m_1P_scroll_LerpColor_CODE", {"_OutlineColor": (0, 0, 0, 1)})) is None
    assert outline.outline_params(info("==PGAME==/Battle/m_bat_actorbase_lv3", {"_OutlineColor": (1, 1, 1, 1)})) is None


def test_fbx_only_needs_vc_map():
    assert outline.outline_params(info("", block=["t_d_x"], family="cloth")) is None
    o = outline.outline_params(info("", block=["t_vc_lob_0003"], family="cloth"))
    assert o["source"] == "fbx" and o["vtex"] and o["mapName"] == "t_vc_lob_0003" and o["color"] == [0.5, 0.5, 0.5]
    assert outline.outline_params(info("", block=["t_vc_x"], family="actorbase")) is None
