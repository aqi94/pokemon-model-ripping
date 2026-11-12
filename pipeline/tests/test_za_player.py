"""Z-A player composite (`pc_<body>`): catalog regexes, default-part rule, group ordering, clip-name dedupe."""

import zipfile

from pipeline.catalog.za import PC_BASE_RE, PC_PART_RE, ZaCatalog
from pipeline.common.model import Clip
from pipeline.importers.za.composite import GROUP_LABELS, group_rank
from pipeline.importers.za.importer import dedupe_clip_names


def test_pc_regexes():
    m = PC_PART_RE.match("ik_chara/model_pc/p1_upb0020_00_tshirt/p1_upb0020_00.trmdl")
    assert (m["body"], m["grp"], m["num"], m["var"], m["name"]) == ("p1", "upb", "0020", "00", "tshirt")
    assert PC_PART_RE.match("ik_chara/model_pc/p0_bag9999_00_none/p0_bag9999_00.trmdl")["num"] == "9999"
    assert not PC_PART_RE.match("ik_chara/model_pc/p1_upb0020_00_tshirt/p1_upb0020_00_lod1.trmsh")
    assert PC_BASE_RE.match("ik_chara/model_pc_base/model/p0_base.trmdl").group(1) == "p0"
    assert not PC_BASE_RE.match("ik_chara/model_pc_base/model/p0_base_loc.trskl")


def _player_zip(path):
    files = [
        "ik_chara/model_pc_base/model/p0_base.trmdl",
        "ik_chara/model_pc/p0_face0001_00_slant/p0_face0001_00.trmdl",
        "ik_chara/model_pc/p0_face0000_00_round/p0_face0000_00.trmdl",
        "ik_chara/model_pc/p0_bag0000_00_round/p0_bag0000_00.trmdl",
        "ik_chara/model_pc/p0_bag9999_00_none/p0_bag9999_00.trmdl",
        "ik_chara/model_pc/p1_upb0003_00_cardiganshirt/p1_upb0003_00.trmdl",
        "ik_chara/model_pc/p1_upb0000_00_blousonshirt/p1_upb0000_00.trmdl",
        "ik_chara/model_pc/p1_upb9999_00_none/p1_upb9999_00.trmdl",
        "ik_chara/model_pc/p2_upb0000_00_blousonshirt/p2_upb0000_00.trmdl",
    ]
    with zipfile.ZipFile(path, "w") as z:
        for f in files:
            z.writestr(f, b"")


def test_pc_models_one_per_body_with_shared_parts(tmp_path):
    _player_zip(tmp_path / "ZA-TrainersPlayer.zip")
    cat = ZaCatalog(str(tmp_path))
    refs = cat.pc_models()
    assert [r.id for r in refs] == ["pc_p1", "pc_p2"]
    assert refs[0].anim_folder == "ik_chara/motion_pc/base/" and refs[0].extras["pc_skeleton"] == "p0_base"
    skel, folder, bodies = cat._pc_parts()
    assert (skel, folder) == ("p0", "ik_chara/model_pc_base/model/")
    p1 = {p.id: p.default for p in bodies["p1"]}
    # lowest-numbered part is the default, the "none" part (9999) stays selectable; other parts are not imported
    assert p1 == {"p0_bag0000_00": True, "p0_bag9999_00": False, "p0_face0000_00": True, "p1_upb0000_00": True, "p1_upb9999_00": False}
    assert {p.id for p in bodies["p2"]} == {"p0_bag0000_00", "p0_bag9999_00", "p0_face0000_00", "p2_upb0000_00"}


def test_pc_models_absent_without_player_zip(tmp_path):
    assert ZaCatalog(str(tmp_path)).pc_models() == []


def test_group_rank_orders_known_then_unknown():
    groups = ["zzz", "ftw", "face", "upb", "hrs"]
    assert sorted(groups, key=group_rank) == ["face", "hrs", "upb", "ftw", "zzz"]
    assert GROUP_LABELS["upb"] == "Tops"


def test_dedupe_clip_names_keeps_lowest_index_bare():
    clips = [Clip(name="ballthrow01", tracks=[], index=1020), Clip(name="ballthrow01", tracks=[], index=352), Clip(name="walk01", tracks=[], index=51)]
    dedupe_clip_names(clips)
    assert sorted(c.name for c in clips) == ["ballthrow01", "ballthrow01 [01020]", "walk01"]
    assert next(c for c in clips if c.index == 352).name == "ballthrow01"
