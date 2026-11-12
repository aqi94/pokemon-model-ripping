import numpy as np

from pipeline.importers.unite import emission as em


def info(shader, fl=None, col=None, mix=True):
    return {"shader": shader, "floats": fl or {}, "colors": col or {}, "textures": {"_MixMap": {"name": "t_mroe_x"}} if mix else {}}


def test_basepbr_scale_form():
    p = em.emission_params(info("==PGAME==/LobbyPokemon/m_lob_basepbr_lv3", {"_Emissive": 1.0, "_EmisssionScale": 0.4}))
    assert p["scale"] == 0.4 and p["form"] == "scale"
    assert em.emission_params(info("==PGAME==/LobbyPokemon/m_lob_basepbr_lv3", {"_Emissive": 0.0, "_EmisssionScale": 1})) is None


def test_cloth_uses_emiss_not_emissive():
    i = info("==PGAME==/LobbyPokemon/m_lob_basepbrCloth_lv3", {"_Emissive": 1.0, "_Emiss": 0.0}, {"_EmissColor": (2, 0, 4, 1)})
    assert em.emission_params(i) is None
    i["floats"]["_Emiss"] = 1.0
    assert em.emission_params(i)["colour"] == [2.0, 0.0, 4.0]
    f, strength = em.factors(em.emission_params(i))
    assert strength == 4.0 and max(f) == 1.0


def test_emissmask_with_ramp_is_a_mask():
    i = info("==PGAME==/LobbyPokemon/m_lob_basepbrCloth_lv3", {"_Emiss": 1.0, "_EMISSMASK": 1.0, "_IFMAT": 1.0})
    assert em.emission_params(i) is not None and em.mask_kind(i) == "matcapMask"  # the code still emits (Cloth_lv3.txt:27174+404)


def test_fbx_only_is_off_and_ao_scopes():
    assert em.emission_params(info("", mix=True)) is None
    assert em.ao_params(info("==PGAME==/Battle/m_bat_actorbase_lv3"))["scope"] == "total"
    assert em.ao_params(info("==PGAME==/LobbyPokemon/m_lob_basepbr_lv3", {"_AOScale": 2.0})) == {"scope": "indirect", "power": 2.0}


def test_bake_is_albedo_times_alpha():
    alb = np.full((2, 2, 4), 255, np.uint8)
    out = em.bake_emission(alb, np.array([[0, 255], [255, 0]], np.uint8))
    assert out[0, 0, 0] == 0 and out[0, 1, 0] == 255
