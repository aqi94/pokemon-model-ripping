from pipeline.importers.unite import families as fam
from pipeline.importers.unite.importer import info_from_fbx


def test_family_of_json_shaders():
    f = fam.family_of
    assert f("==PGAME==/LobbyPokemon/m_lob_basepbr_lv3") == "basepbr"
    assert f("==PGAME==/LobbyPokemon/m_lob_basepbreye_lv3") == "basepbreye"
    assert f("==PGAME==/LobbyPokemon/m_lob_basepbrmouth_lv3") == "basepbrmouth"
    assert f("==PGAME==/LobbyPokemon/m_lob_basepbrCloth_lv3") == "cloth"
    assert f("Hidden/Lobby/m_lob_basepbrClothLengcy_lv3") == "lengcy"
    assert f("==PGAME==/Lobby/m_lob_basepbrMatcapMask_lv3") == "matcapMask"
    assert f("Hidden/Lobby/m_lob_basepbrClothMatcapMask_lv3") == "matcapMask"
    assert f("==PGAME==/Lobby/m_lob_basepbrClothFur5Pass_lv3") == "fur"
    assert f("==PGAME==/Battle/m_bat_actorMatcapMask_lv3") == "actorMatcap"
    assert f("==PGAME==/Battle/m_bat_actorbase_lv3") == "actorbase"
    assert f("==PGAME==/Fx/m_2P_mask") == "fx"
    assert f("Lobby/m_lob_basepbrBubble_lv3") == "bubble"
    assert f("==PGAME==/LobbyPokemon/Extra/m_lob_ghostGlass_lv3") == "glass"
    assert f("Hidden/m_lob_PlayerPBRAlpha_lv3") == "player"
    assert f("Some/Unknown") == "Some/Unknown"
    assert f("") == ""


def test_every_survey_shader_has_a_family():
    import json, os
    names = json.load(open(os.path.join(os.path.dirname(fam.__file__), "shader_files.json")))
    assert all(fam.family_of(n) != n for n in names)


def test_infer_family():
    i = fam.infer_family
    assert i("mat_lob_0025eye_lv3", ["t_d_lob_0025eye_lv3.png", "t_n_x.png"], "lobby") == "basepbreye"
    assert i("mat_lob_0025mouth_lv3", [], "lobby") == "basepbrmouth"
    assert i("mat_lob_0025body_lv3", ["t_d_lob_0025body_lv3.png", "t_mro_x.png", "t_vc_x.png"], "lobby") == "basepbr"
    assert i("mat_lob_003811a_lv3", ["t_d_a.png", "t_matcap_lob_003811a.png", "t_mipmap03.png"], "lobby") == "cloth"
    assert i("mat_lob_082007b_lv2", ["t_d_a.png", "t_noise_lob_082007.png"], "lobby") == "fur"
    assert i("mat_lob_0425eye_lv3", ["t_d_a.png", "t_nm_a.png"], "lobby") == "player"
    assert i("mat_lob_000906_lv2", ["t_d_a.png", "t_fx_changeskin_flicker_wave_02.png", "t_mro_a.png", "t_vc_a.png"], "lobby") == "lengcy"
    assert i("mat_fx_077806_bat_mask_01_hb", ["t_fx_a.png"], "battle") == "fx"
    assert i("mat_bat_0122mouth_lv3", [], "battle") == "actorbasemouth"
    assert i("mat_bat_021201_lv3", ["t_d_bat_021201_lv3.png", "t_matcap_000709.png"], "battle") == "actorMatcap"
    assert i("mat_bat_000905_lv3", ["t_d_bat_000905_lv3.png"], "battle") == "actorbase"


def test_info_from_fbx_sets_family():
    info = info_from_fbx({"DiffuseColor": "t_d_bat_000905_lv3.png"}, ".", None, ["t_d_bat_000905_lv3.png"], "mat_bat_000905_lv3")
    assert info["family"] == "actorbase" and info["familyInferred"] is True
    info = info_from_fbx({}, ".", None, [], "mat_lob_0025eye_lv3")
    assert info["family"] == "basepbreye"


def test_default_params():
    p = fam.default_params("basepbr")
    assert p.get("_actorscale") == 1.0 and p.get("_SHScale") == 1.0
    assert fam.default_params("==PGAME==/LobbyPokemon/m_lob_basepbr_lv3") == p
    assert fam.default_params("nope") == {}
