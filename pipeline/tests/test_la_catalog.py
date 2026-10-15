"""LaCatalog discovery against a tiny synthetic zip set (no real data needed)."""

import zipfile

from pipeline.catalog.la import LaCatalog, dex_of, form_label


def _zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr(n, b"x")


def _mk(tmp_path):
    base = "bin/pokemon/pm0025/pm0025_00_00/"
    _zip(tmp_path / "LA-Poke.zip", [
        base + "mdl/pm0025_00_00.trmdl",
        "bin/pokemon/pm0025/pm0025_01_00/mdl/pm0025_01_00.trmdl",
        "bin/pokemon/pm0059/pm0059_71_41/mdl/pm0059_71_41.trmdl",
        "bin/pokemon/pm0751/pm0751_00_41/mdl/pm0751_00_41.trmdl",
        "bin/pokemon/pm1002/pm1002_71_00/mdl/pm1002_71_00.trmdl",
        "bin/pokemon/pm0201/pm0201_12_00/mdl/pm0201_12_00.trmdl",
        "bin/chara/data/pm/pm0025_00_00/mdl/pm0025_00_00.trmdl",  # legacy duplicate tree: ignored
    ])
    _zip(tmp_path / "LA-PokeAnim.zip", [
        base + "anm/pm0025_00_00.tranm",  # shared template without a clip index: not a clip
        base + "anm/pm0025_00_00_00000_defaultwait01_loop.tranm",
        base + "anm/pm0025_00_00_00000_defaultwait01_loop.tracm",
        base + "anm/pm0025_00_00_00030_walk01_loop.tranm",
        base + "anm/pm0025_00_00_00030_walk01_loop.traef",
    ])
    cat = LaCatalog(str(tmp_path))
    cat.extract_dir = str(tmp_path / "none")
    return cat


def test_la_discovery(tmp_path):
    pk = {r.id: r for r in _mk(tmp_path).pokemon()}
    assert sorted(pk) == ["pm0025_00_00", "pm0025_01_00", "pm0059_71_41", "pm0201_12_00", "pm0751_00_41", "pm1002_71_00"]
    r = pk["pm0025_00_00"]
    assert r.game == "la" and r.folder == "bin/pokemon/pm0025/pm0025_00_00/mdl/"
    assert r.anim_folder == "bin/pokemon/pm0025/pm0025_00_00/anm/"
    assert pk["pm0025_01_00"].display_name == "Pikachu (Female)"
    assert pk["pm0059_71_41"].display_name == "Arcanine (Hisuian, Noble)"
    assert pk["pm0751_00_41"].display_name == "Avalugg (Hisuian)"  # internal id 751 = dex 713
    assert pk["pm1002_71_00"].display_name == "Kleavor (Noble)"  # pm1001.. = dex 899..
    assert pk["pm0201_12_00"].display_name == "Unown (B)"
    assert pk["pm0751_00_41"].extras["dex"] == 713


def test_la_animations_skip_template(tmp_path):
    cat = _mk(tmp_path)
    anims = cat.animations(cat.find("pm0025_00_00"))
    assert [(a.name, a.index) for a in anims] == [("defaultwait01_loop", 0), ("walk01_loop", 30)]
    assert anims[0].config == b"x" and anims[1].config is None
    assert cat.trainers() == [] and cat.objects() == []


def test_dex_and_labels():
    assert dex_of("pm1007") == 905 and dex_of("pm0843") == 724 and dex_of("pm0025") == 25
    assert form_label(25, "00_00") == ""
    assert form_label(550, "13_41") == "White-Striped"


def test_la_trainers_and_items_from_extracted_folders(tmp_path):
    ex = tmp_path / "extract"
    tr = ex / "trainers/chara/data/tr/tr0003_00_ouji"
    tr.mkdir(parents=True)
    (tr / "tr0003_00_ouji.trmdl").write_bytes(b"x")
    (ex / "trainers/chara/data/tr/tr0099_00_empty").mkdir()  # no trmdl: skipped
    it = ex / "misc/chara/data/item/item_201"
    it.mkdir(parents=True)
    (it / "item_201.trmdl").write_bytes(b"x")
    cat = LaCatalog(str(tmp_path))
    cat.extract_dir = str(ex)
    t = cat.trainers()
    assert [(r.id, r.display_name, r.extras["stem"]) for r in t] == [("tr0003_00", "Ouji", "tr0003_00_ouji")]
    o = cat.objects()
    assert [(r.id, r.kind, r.display_name) for r in o] == [("item_201", "object", "Item 201")]
    assert cat.find("item_201").folder == str(it)
