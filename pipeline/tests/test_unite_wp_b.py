from pipeline.importers.unite.alpha import decide

CLOTH = "==PGAME==/LobbyPokemon/m_lob_basepbrCloth_lv3"


def test_cloth_lerp():
    assert decide(CLOTH, {"_DstBlendFactor": 10.0, "_ALPHAMASK": 0.0})["mode"] == "BLEND"
    assert decide(CLOTH, {"_DstBlendFactor": 0.0})["mode"] == "OPAQUE"
    assert decide(CLOTH, {"_DstBlendFactor": 10.0, "_ALPHAMASK": 1.0})["mode"] == "OPAQUE"
    # the stale test is gone
    assert decide(CLOTH, {"_SrcBlend": 5.0, "_DstBlend": 10.0})["mode"] == "OPAQUE"


def test_mask_bubble_actor():
    d = decide(CLOTH, {"_BlendMode": 1.0, "_Cutoff": 0.3, "_DstBlendFactor": 10.0})
    assert (d["mode"], d["cutoff"]) == ("MASK", 0.3)
    assert decide("x/Lobby/m_lob_basepbrBubble_lv3", {"_IFTrans": 0.0})["mode"] == "BLEND"
    d = decide("x/Battle/m_bat_actorbase_lv3", {"_CustomAlpha": 0.5})
    assert d["mode"] == "BLEND" and d["alpha_factor"] == 0.5
