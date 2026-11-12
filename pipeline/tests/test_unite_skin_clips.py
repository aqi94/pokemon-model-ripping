"""UNITE skins inherit their form's skin-00 clip set (override controller): donor folder rule + merge rule."""

from pipeline.catalog.unite_battle import skin_keys
from pipeline.importers.unite.importer import _typo_variant, merge_base_clips


def _base_of(folder, others):
    keyed = {skin_keys(f)[0]: f for f in others}
    return keyed.get(skin_keys(folder)[1])


def test_donor_folder():
    folders = ["pref_char_bat_002500_pikachu_lv3", "pref_char_bat_002500_pikachuF_lv3", "pref_char_bat_035900_Absol_lv3",
               "pref_char_bat_068100_aegislash_lv3", "pref_char_bat_068100_aegislash_sword_lv3",
               "pref_char_bat_015000_51_mewtwo_lv3", "pref_char_bat_015000_mewtwo_lv3",
               "pref_char_bat_089200A_urshifu_lv3", "pref_char_bat_089200B_urshifu_lv3",
               "pref_char_bat_0870_falinks_lv3", "pref_char_bat_c_0870_falinks_lv3"]
    assert _base_of("pref_char_bat_002513_pikachu_lv3", folders) == "pref_char_bat_002500_pikachu_lv3"
    assert _base_of("pref_char_bat_035902_absol_lv3", folders) == "pref_char_bat_035900_Absol_lv3"  # name case
    assert _base_of("pref_char_bat_068102_aegislash_sword_lv3", folders) == "pref_char_bat_068100_aegislash_sword_lv3"
    assert _base_of("pref_char_bat_068102_aegislash_lv3", folders) == "pref_char_bat_068100_aegislash_lv3"
    assert _base_of("pref_char_bat_015008_51_mewtwo_lv3", folders) == "pref_char_bat_015000_51_mewtwo_lv3"
    assert _base_of("pref_char_bat_089204A_urshifu_lv3", folders) == "pref_char_bat_089200A_urshifu_lv3"  # style kept
    assert _base_of("pref_char_bat_087004_falinks_lv3", folders) == "pref_char_bat_0870_falinks_lv3"  # bare = skin 00
    assert _base_of("pref_char_bat_c_087004_falinks_lv3", folders) == "pref_char_bat_c_0870_falinks_lv3"  # prefix kept
    # `0006_51` is a form (Mega X), not skin 51 of Charizard 00: forms keep their own clips
    assert skin_keys("pref_char_bat_0006_51_charizard_lv3")[1] is None


def test_typo_variant():
    assert _typo_variant("akt1", "atk1") and _typo_variant("spll1a", "spell1a") and _typo_variant("sepll1b1", "spell1b1")
    assert not _typo_variant("atk2", "atk1")  # digit changed
    assert not _typo_variant("atk1", "atk11")  # digit added
    assert not _typo_variant("run", "ran")  # letter replaced
    assert not _typo_variant("idle", "idle")


def test_merge_base_clips():
    add, aliases = merge_base_clips(["born", "score", "akt1"], ["idle", "score", "atk1", "atk2"])
    assert add == ["idle", "atk2"]  # own `score` replaces the base one, `akt1` overrides `atk1`
    assert aliases == {"atk1": "akt1"}
    # an own clip that is also a real base name is never taken as another clip's typo
    add, aliases = merge_base_clips(["atk1"], ["atk1", "akt1"])
    assert add == ["akt1"] and not aliases
