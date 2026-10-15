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
