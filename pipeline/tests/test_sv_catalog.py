"""SvCatalog discovery against a tiny synthetic zip set (no real data needed)."""

import zipfile

from pipeline.catalog.sv import SvCatalog
from pipeline.catalog.za import ZaCatalog


def _zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr(n, b"")


def test_sv_discovery(tmp_path):
    _zip(tmp_path / "SV-Poke.zip", [
        "pokemon/data/pm0025/pm0025_00_00/pm0025_00_00.trmdl",
        "pokemon/data/pm0025/pm0025_00_00/pm0025_00_00.trmsh",
        "pokemon/data/pm0128/pm0128_13_51/pm0128_13_51.trmdl",
    ])
    _zip(tmp_path / "SV-TrainersNPC.zip", [
        "chara/model_uq/tr0200_normal/tr0200_00.trmdl",
        "chara/model_tr/tr0330_karate/tr0330_00.trmdl",
    ])
    _zip(tmp_path / "SV-Misc.zip", ["chara/model_ob/ob0001_phone/ob0001_00.trmdl"])
    cat = SvCatalog(str(tmp_path))
    pk = cat.pokemon()
    assert [r.id for r in pk] == ["pm0025_00_00", "pm0128_13_51"]
    assert pk[0].game == "sv" and pk[0].folder == "pokemon/data/pm0025/pm0025_00_00/"
    assert pk[1].display_name.endswith("(13_51)")
    tr = cat.trainers()
    assert [(r.id, r.display_name) for r in tr] == [("tr0200_00", "Normal"), ("tr0330_00", "Karate")]
    assert tr[0].folder == "chara/model_uq/tr0200_normal/" and tr[0].anim_folder == ""
    ob = cat.objects()
    assert [(r.id, r.display_name) for r in ob] == [("ob0001_00", "Phone")]
    assert cat.find("tr0330_00").kind == "trainer"


def test_za_catalog_unchanged_by_refactor(tmp_path):
    _zip(tmp_path / "ZA-Poke.zip", ["ik_pokemon/data/pm0001/pm0001_00_00/pm0001_00_00.trmdl"])
    _zip(tmp_path / "ZA-TrainersNPC.zip", [
        "ik_chara/model_uq/tr0003_00_foo_bar/tr0003_00.trmdl",
        "ik_chara/model_ob/ob0204_00_monsterball/ob0204_00.trmdl",
    ])
    cat = ZaCatalog(str(tmp_path))
    assert cat.pokemon()[0].folder == "ik_pokemon/data/pm0001/pm0001_00_00/"
    t = cat.trainers()[0]
    assert (t.display_name, t.folder, t.anim_folder) == ("Foo Bar", "ik_chara/model_uq/tr0003_00_foo_bar/", "ik_chara/motion_uq/tr0003_00_foo_bar/")
    assert cat.objects()[0].display_name == "Poké Ball"


def test_sv_generic_trainer_body_sets(tmp_path, monkeypatch):
    import pipeline.catalog.sv as sv

    _zip(tmp_path / "SV-TrainersNPC.zip", [
        "chara/model_tr/tr0340_artist/tr0340_00.trmdl",
        "chara/model_tr/tr0420_mountain/tr0420_00.trmdl",
        "chara/model_tr/tr1500_villagerm/tr1500_00.trmdl",
    ])
    with zipfile.ZipFile(tmp_path / "SV-TrainersAnim.zip", "w") as z:
        for n in [
            "chara/motion_tr/tr0340_artist/tr0340_00_04505_art01_loop.tranm",  # cloth layer
            "chara/motion_tr/tr0340_artist/tr0340_00_08200_face01.tranm",
            "chara/motion_cc_base/tr0340_artist/tr0340_00_04505_art01_loop.tranm",  # body
            "chara/motion_cc_base/cc0003_cut_f/cc0003_cut_f_00000_defaultwait01_loop.tranm",
            "chara/motion_cc_base/cc0007_pos_f/cc0007_pos_f_00000_defaultwait01_loop.tranm",
            "chara/motion_cc_base/cc0007_pos_f/cc0007_pos_f_00030_walk01_loop.tranm",
            "chara/motion_cc_base/tr0420_mountain/tr0420_00_00000_defaultwait01_loop.tranm",
            "chara/motion_cc_base/cc0005_adl_f/cc0005_adl_f_00000_defaultwait01_loop.tranm",
            "chara/motion_cc_base/cc0004_adl_m/cc0004_adl_m_00000_defaultwait01_loop.tranm",
        ]:
            z.writestr(n, n.encode())
    monkeypatch.setattr(sv, "motion_sets", lambda: {
        "tr0340_00": {"own": "tr0340_artist", "sets": ["cc0003_cut_f", "cc0007_pos_f"], "clips": []},
        "tr0420_00": {"own": "tr0420_mountain", "sets": ["cc0005_adl_f"], "clips": []},
        "tr1500_00": {"own": "", "sets": ["cc0004_adl_m"], "clips": [], "guess": True},
    })
    cat = sv.SvCatalog(str(tmp_path))

    clips = {(a.name, a.index, a.group): a for a in cat.animations(cat.find("tr0340_00"))}
    assert set(clips) == {
        ("art01_loop", 4505, ""), ("face01", 8200, None),
        ("defaultwait01_loop", 0, ""),  # first set keeps bare names
        ("defaultwait01_loop", 100000, "pos_f"), ("walk01_loop", 100030, "pos_f"),
    }
    art = clips[("art01_loop", 4505, "")]
    assert art.data.endswith(b"motion_cc_base/tr0340_artist/tr0340_00_04505_art01_loop.tranm")
    assert art.extra.endswith(b"motion_tr/tr0340_artist/tr0340_00_04505_art01_loop.tranm")

    # own full set -> every shared set is an alternative
    hiker = {(a.name, a.index, a.group) for a in cat.animations(cat.find("tr0420_00"))}
    assert hiker == {("defaultwait01_loop", 0, ""), ("defaultwait01_loop", 100000, "adl_f")}

    # DLC borrower: no motion_tr folder, still gets the borrowed set
    ref = cat.find("tr1500_00")
    assert ref.anim_folder == "" and ref.extras["sv_motion"]["guess"]
    assert [(a.name, a.index) for a in cat.animations(ref)] == [("defaultwait01_loop", 0)]


def test_sv_body_clips_leave_eyes_to_the_face_layer():
    from pipeline.catalog.sv import BODY_SKIP_BONES

    for b in ["eyescale", "left_pupil_01", "right_pupil_02", "left_eyehilight_02", "right_eyeball_segoff"]:
        assert BODY_SKIP_BONES.search(b)
    for b in ["head", "neck", "eyebrow_mesh", "left_skirt_c_01", "left_arm_width"]:
        assert not BODY_SKIP_BONES.search(b)


def test_sv_model_folders_map_to_dex_through_the_game_catalog():
    from pipeline.catalog.sv import SvCatalog, model_dex

    table = model_dex()
    assert table["pm1013"] == 909 and table["pm1012"] == 908 and table["pm0909"] == 819  # Fuecoco, Meowscarada, Skwovet
    cat = SvCatalog.__new__(SvCatalog)
    assert cat.species_dex("pm1013") == (909, True)
    assert cat.species_dex("pm0025") == (25, True)
    assert cat.species_dex("pm9998") == (9998, False)  # DLC folder outside the base-game table: unnamed


def test_sv_gen9_internal_species_convert_to_national_dex():
    from pipeline.catalog.sv import SvCatalog
    from pipeline.catalog.sv_species import TABLE9_INTERNAL_TO_NATIONAL, national9

    assert sorted(national9(i) for i in range(917, 917 + len(TABLE9_INTERNAL_TO_NATIONAL))) == list(range(917, 1026))
    assert national9(916) == 916 and national9(917) == 982
    cat = SvCatalog.__new__(SvCatalog)
    assert cat.species_dex("pm1020") == (916, True)  # Oinkologne
    assert cat.species_dex("pm1021") == (982, True)  # Dudunsparce (internal 917), not Tarountula
    # DLC folders, not in the base-game table: internal + 109
    assert cat.species_dex("pm1120") == (1017, True)  # Ogerpon
    assert cat.species_dex("pm1130") == (1024, True)  # Terapagos
    assert cat.species_dex("pm1134") == (1013, True)  # Sinistcha
