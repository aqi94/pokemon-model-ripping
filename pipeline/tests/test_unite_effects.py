"""UNITE rules from the 2026-11-01 spot check: weightless skins, effect parts, placeholders, overlay shells, glass."""

import numpy as np
from PIL import Image

from pipeline.common.model import Joint, Material, Mesh, Model, Primitive, Skeleton, TextureImage
from pipeline.importers.unite import fbx_anim
from pipeline.importers.unite.importer import (
    ATTACK_PREFAB, bind_rigid_skins, build_effect_material, build_glass_material, build_material, material_role,
    matcap_of, overlay_shells,
)


def _info(shader="", textures=None, floats=None, colors=None, source="json"):
    tx = {k: {"name": v, "scale": (1.0, 1.0), "offset": (0.0, 0.0)} for k, v in (textures or {}).items()}
    return {"name": "", "shader": shader, "textures": tx, "floats": floats or {}, "colors": colors or {}, "source": source}


def _prim(pos, material="m", skinned=True):
    pos = np.asarray(pos, np.float32)
    n = len(pos)
    return Primitive(material, np.arange(n - n % 3, dtype=np.uint32), pos,
                     joints=np.zeros((n, 4), np.uint16) if skinned else None,
                     weights=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)) if skinned else None)


def test_weightless_skin_rides_its_first_cluster_bone():
    """Lucario's staff: cluster on `Bone001` with no weights; Blender's dump put it on joint 0 (the static root)."""
    nodes = {1: fbx_anim.FbxNode(1, "Bip001", "LimbNode"), 2: fbx_anim.FbxNode(2, "Bone001", "LimbNode", parent=1),
             3: fbx_anim.FbxNode(3, "weapon", "Mesh")}
    scene = fbx_anim.FbxScene(nodes=nodes, rigid_skins={"weapon": 2})
    eye = np.eye(4, dtype=np.float32)
    sk = Skeleton(joints=[Joint("Bip001", -1, inverse_bind=eye), Joint("Bone001", 0, inverse_bind=eye)], palette=[0, 1])
    staff, body = _prim(np.zeros((3, 3))), _prim(np.zeros((3, 3)))
    m = Model(id="t", game="unite", display_name="t", kind="pokemon", skeleton=sk,
              meshes=[Mesh("weapon", [staff]), Mesh("body", [body])], materials={}, textures={})
    assert bind_rigid_skins(m, scene) == ["weapon"]
    assert (staff.joints[:, 0] == 1).all() and (staff.weights[:, 0] == 1).all()
    assert (body.joints[:, 0] == 0).all()  # weighted meshes untouched


def test_material_roles():
    assert material_role("mat_lob_npc01eye1_lv3", _info("Unlit/Texture", {"_MainTex": "t"})) == "backdrop"
    assert material_role("x", _info("==PGAME==/Fx/m_2P_dissolve_ColorEdge", {"_Tex1": "t"})) == "effect"
    assert material_role("mat_bat_0077hari_fx_lv3", _info("", {"_MainTex": "t_d_bat_0077body_lv3"}, source="fbx")) == "effect"
    assert material_role("mat_bat_0077body_lv3", _info("", {"_MainTex": "t_d_bat_0077body_lv3"}, source="fbx")) == "character"
    assert ATTACK_PREFAB.search("pref_fx_Dhelmise_natk1_atk1")
    assert not ATTACK_PREFAB.search("pref_fx_Gastly_idle_lv3") and not ATTACK_PREFAB.search("pref_fx_0004_battlefire_lv3")


def test_effect_material_alpha_rules(tmp_path):
    Image.new("RGBA", (2, 2), (90, 90, 90, 255)).save(tmp_path / "t_fx_grey.png")  # grey opaque mask -> its own alpha
    Image.new("RGBA", (2, 2), (200, 50, 20, 255)).save(tmp_path / "t_fx_col.png")  # opaque colour -> additive
    Image.new("RGBA", (2, 2), (200, 50, 20, 0)).save(tmp_path / "t_d_body.png")  # character atlas: alpha is a mask
    Image.new("RGBA", (2, 2), (10, 200, 20, 128)).save(tmp_path / "t_d_fx_flag.png")
    tex: dict[str, TextureImage] = {}
    g = build_effect_material("g", _info(textures={"_MainTex": "t_fx_grey"}, source="fbx"), str(tmp_path), tex)
    assert tex[g.base_color].rgba[0, 0, 3] == 90 and g.extras["blend"] == "Blend" and g.extras["effect"]
    c = build_effect_material("c", _info(textures={"_MainTex": "t_fx_col"}, source="fbx"), str(tmp_path), tex)
    assert c.extras["blend"] == "Add" and c.alpha_mode == "BLEND" and c.emissive == c.base_color
    a = build_effect_material("a", _info(textures={"_MainTex": "t_d_body"}, source="fbx"), str(tmp_path), tex)
    # an opaque character atlas is solid paint (depth-tested, not additive): Crocalor's flame atlas, manes
    assert tex[a.base_color].rgba[0, 0, 3] == 255 and a.alpha_mode == "OPAQUE" and a.extras["solidEffect"]
    f = build_effect_material("f", _info("==PGAME==/Fx/m_2P_dissolve_ColorEdge", {"_Tex1": "t_d_fx_flag"},
                                         {"_BlendMode": 1.0, "_Alpha1": 0.5, "_Cull": 2.0},
                                         {"_Color1": (1.0, 0.5, 1.0, 1.0)}), str(tmp_path), tex)
    assert f.extras["blend"] == "Blend" and f.base_color_factor[3] == 0.5 and f.emissive_factor == (1.0, 0.5, 1.0)
    assert not f.double_sided
    assert build_effect_material("n", _info(source="fbx"), str(tmp_path), tex) is None  # nothing to draw


def test_see_through_character_shaders(tmp_path):
    Image.new("RGBA", (2, 2), (200, 100, 50, 60)).save(tmp_path / "t_d.png")
    tex: dict[str, TextureImage] = {}
    m = build_material("o", _info("==PGAME==/Lobby/m_lob_basepbrOutlineAlpha_lv3", {"_MainTex": "t_d"}, {"_ZWrite": 0.0}), str(tmp_path), tex)
    assert m.alpha_mode == "BLEND" and tex["t_d~a"].rgba[0, 0, 3] == 60
    tex = {}
    m = build_material("c", _info("==PGAME==/LobbyPokemon/m_lob_basepbrCloth_lv3", {"_MainTex": "t_d"}, {"_ZWrite": 1.0}), str(tmp_path), tex)
    assert m.alpha_mode == "OPAQUE" and tex["t_d"].rgba[0, 0, 3] == 255  # cloth mask, not opacity


def test_glass_from_matcap(tmp_path):
    img = np.zeros((4, 4, 4), np.uint8)
    img[..., :3] = 200
    img[..., 3] = 26
    Image.fromarray(img).save(tmp_path / "t_matcap_009401_lv3.png")
    # battle: no texture links at all, matcap found by the material's dex+skin
    info = _info(source="fbx")
    p = matcap_of(info, str(tmp_path), None, "mat_bat_009401a_lv3")
    assert p and p.endswith("t_matcap_009401_lv3.png")
    assert matcap_of(info, str(tmp_path), None, "mat_off_Lit") is None
    tex: dict[str, TextureImage] = {}
    g = build_glass_material("glass", info, p, tex)
    gl = g.extras["unite"]["glass"]
    assert g.alpha_mode == "BLEND" and gl["texture"] == {"texture": "t_matcap_009401_lv3", "uv": 0} and g.base_color is None
    assert g.base_color_factor == (0.0, 0.0, 0.0, 0.0) and gl["kind"] == "ghost" and gl["top"] == [2.5, 2.5, 2.5, 1.0]


def test_overlay_shell_dropped_but_distinct_geometry_kept():
    coat = np.random.default_rng(0).random((30, 3)).astype(np.float32)
    mats = {"coat": Material("coat", base_color="t"), "twinkle": Material("twinkle"), "hair": Material("hair")}
    m = Model(id="t", game="unite", display_name="t", kind="pokemon", skeleton=None,
              meshes=[Mesh("rig", [_prim(coat, "coat"), _prim(coat[:27], "twinkle"), _prim(coat[:27] + 0.5, "hair")])],
              materials=mats, textures={})
    assert overlay_shells(m) == [("rig", "twinkle")]
