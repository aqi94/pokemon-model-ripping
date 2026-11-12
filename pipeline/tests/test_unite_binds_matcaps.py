"""UNITE rules from the 2026-11-01 issue list: per-mesh bindposes (torn Miraidon), metal reflections (black Zacian),
glass helmets (Snorlax skin 20), matcap / mask albedo links (white Litwick fire)."""

import json
import struct

import numpy as np
from PIL import Image

from pipeline.common.gltf_writer import write_glb
from pipeline.common.model import Joint, Material, Mesh, Model, Primitive, Skeleton, TextureImage
from pipeline.importers.unite import fbx_anim
from pipeline.importers.unite.fbx import assign_blocks
from pipeline.importers.unite.importer import (
    apply_mesh_binds, build_effect_material, build_material, clear_albedo, enclosure, flag_lenses, info_from_fbx,
    is_character_material, matcap_of, matcap_only, names_material, particle_emitters, sibling_colour,
    split_missing_texture_parts,
)


def _prim(pos, joint=0, material="m"):
    pos = np.asarray(pos, np.float32)
    n = len(pos)
    j = np.zeros((n, 4), np.uint16)
    j[:, 0] = joint
    return Primitive(material, np.arange(n - n % 3, dtype=np.uint32), pos, joints=j,
                     weights=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)))


def _translate(x):
    m = np.eye(4)
    m[0, 3] = x
    return m


def _model(meshes):
    sk = Skeleton(joints=[Joint("root", -1, inverse_bind=np.eye(4, dtype=np.float32)),
                          Joint("spine", 0, inverse_bind=np.eye(4, dtype=np.float32))], palette=[0, 1])
    return Model(id="t", game="unite", display_name="t", kind="pokemon", skeleton=sk, meshes=meshes,
                 materials={"m": Material("m")}, textures={})


def test_mesh_with_consistently_offset_binds_is_baked():
    """Miraidon skin 02's `est`: every cluster's TransformLink carries the same 1.15 m offset -> move the vertices."""
    body = Mesh("body", [_prim([[1.5, 0, 0]] * 3, joint=1)])
    m = _model([body, Mesh("outfit", [_prim(np.zeros((3, 3)), joint=1)])])
    scene = fbx_anim.FbxScene(nodes={}, joint_nodes=[10, 11],
                              mesh_cluster_binds={"body": {11: (_translate(1.5), True)}, "outfit": {11: (np.eye(4), True)}})
    assert apply_mesh_binds(m, scene) == ["body (baked)"]
    assert np.allclose(body.primitives[0].positions[:, 0], 0.0)  # inv(K) @ R @ p = inv(TL_own) @ p
    assert body.inverse_binds is None and m.meshes[1].primitives[0].positions[0, 0] == 0.0


def test_mesh_owning_its_other_bones_is_baked_and_rebinds_them():
    """Fuecoco skin 01's tuft: node scale 1, the 1/40 sits in every cluster TransformLink. Its feeler bones are its
    alone (kept bind = its own TL), only the shared head disagrees -> bake R and rebind the feelers to R @ TL."""
    s = np.diag([40.0, 40.0, 40.0, 1.0])
    tuft = Mesh("tuft", [_prim([[40.0, 0, 0]] * 3, joint=0), _prim([[40.0, 0, 0]] * 3, joint=1)])
    m = _model([tuft])
    m.skeleton.joints[1].inverse_bind = np.linalg.inv(s).astype(np.float32)  # the tuft's own cluster set it
    scene = fbx_anim.FbxScene(nodes={}, joint_nodes=[10, 11],
                              mesh_cluster_binds={"tuft": {10: (s, True), 11: (s, True)}, "body": {10: (np.eye(4), True)}})
    assert apply_mesh_binds(m, scene) == ["tuft (baked, 1 own bones rebound)"]
    assert tuft.inverse_binds is None
    assert np.allclose(tuft.primitives[1].positions[:, 0], 1.0)
    # skinned result unchanged: joint1 inverse bind @ new p == inv(TL) @ old p
    assert np.allclose(m.skeleton.joints[1].inverse_bind, np.eye(4), atol=1e-6)


def test_mesh_with_mixed_binds_gets_its_own_skin(tmp_path):
    """Miraidon skin 03's body: only the bones it shares with the outfit disagree -> its own inverse binds."""
    p0, p1 = _prim([[2.0, 0, 0]] * 3, joint=0), _prim([[2.0, 0, 0]] * 3, joint=1)
    body = Mesh("body", [p0, p1])
    m = _model([body])
    scene = fbx_anim.FbxScene(nodes={}, joint_nodes=[10, 11],
                              mesh_cluster_binds={"body": {10: (np.eye(4), True), 11: (_translate(2.0), True)},
                                                  "outfit": {10: (np.eye(4), True)}})
    assert apply_mesh_binds(m, scene) == ["body (own skin, 1 of 2 bones)"]
    assert set(body.inverse_binds) == {1} and np.allclose(body.inverse_binds[1], _translate(-2.0))
    out = tmp_path / "m.glb"
    write_glb(m, str(out))
    b = out.read_bytes()
    n = struct.unpack("<I", b[12:16])[0]
    g = json.loads(b[20:20 + n])
    assert len(g["skins"]) == 2 and g["skins"][0]["joints"] == g["skins"][1]["joints"]
    assert next(nd for nd in g["nodes"] if nd.get("name") == "body")["skin"] == 1


def test_block_names():
    assert names_material("t_matcap_bat_014320a.png", "mat_bat_014320a_lv3")
    assert names_material("t_matcap_lob_0888_12.png", "mat_lob_0888_12bodya_lv3")
    assert names_material("t_d_bat_014320fxa_lv3.png", "mat_bat_014320fxa_lv3")
    assert not names_material("t_matcap_009401_lv3.png", "mat_bat_0094eye_lv3")  # Gengar 01: the helmet's


def test_blocks_follow_the_write_order():
    """Snorlax 20 battle: 014320a is linked; 014320b (glass), fxa and fxb are not, and follow it in Objects order."""
    order = ["mat_bat_0143body_lv3", "mat_bat_014320a_lv3", "mat_bat_014320b_lv3", "mat_bat_014320fxa_lv3", "mat_bat_014320fxb_lv3"]
    linked = {"mat_bat_0143body_lv3": ["t_d_bat_0143body_lv3.png"], "mat_bat_014320a_lv3": ["t_d_bat_014320_lv3.png"]}
    runs = [(None, []), ("mat_bat_0143body_lv3", ["t_fx_0196_color_01_cj.png", "t_mro_bat_0143body_lv3.png"]),
            ("mat_bat_014320a_lv3", ["t_n_bat_014320_lv3.png", "t_matcap_bat_014320a.png", "t_mroe_bat_014320_lv3.png",
                                     "t_matcap_bat_014320b.png", "t_d_bat_014320fxa_lv3.png", "t_fx_scoresettlement_triangulo_01.png",
                                     "t_d_bat_014320fxb_lv3.png"])]
    b = assign_blocks(order, linked, runs)
    assert "t_fx_0196_color_01_cj.png" not in b["mat_bat_0143body_lv3"]  # global default skipped
    assert b["mat_bat_014320a_lv3"][1:] == ["t_n_bat_014320_lv3.png", "t_matcap_bat_014320a.png", "t_mroe_bat_014320_lv3.png"]
    assert b["mat_bat_014320b_lv3"] == ["t_matcap_bat_014320b.png"]
    assert b["mat_bat_014320fxa_lv3"] == ["t_d_bat_014320fxa_lv3.png", "t_fx_scoresettlement_triangulo_01.png"]
    assert b["mat_bat_014320fxb_lv3"] == ["t_d_bat_014320fxb_lv3.png"]
    # textures written before the first OP link belong to the leading unlinked materials (Charmander's tail flame)
    b = assign_blocks(["mat_fx_0006_fire_10_gp", "mat_bat_0004body_lv3"], {"mat_bat_0004body_lv3": ["t_d_bat_0004body_lv3.png"]},
                      [(None, ["t_fx_0006_fire_01_gp.png", "t_fx_0006_fire_02_gp.png"]), ("mat_bat_0004body_lv3", [])])
    assert b["mat_fx_0006_fire_10_gp"] == ["t_fx_0006_fire_01_gp.png", "t_fx_0006_fire_02_gp.png"]
    info = info_from_fbx({}, ".", None, b["mat_fx_0006_fire_10_gp"], "mat_fx_0006_fire_10_gp")
    assert info["textures"]["_Tex1"]["name"] == "t_fx_0006_fire_01_gp"


def test_metal_has_a_matcap_term_only_where_the_shader_has_one(tmp_path):
    Image.new("RGBA", (4, 4), (210, 200, 133, 255)).save(tmp_path / "t_d_lob_x_lv3.png")
    Image.new("RGBA", (4, 4), (250, 120, 220, 255)).save(tmp_path / "t_mroe_lob_x_lv3.png")  # R = metal
    Image.new("RGBA", (4, 4), (223, 200, 72, 255)).save(tmp_path / "t_matcap_lob_x.png")
    Image.new("RGBA", (4, 4), (220, 220, 220, 255)).save(tmp_path / "t_d_lob_w_lv3.png")
    Image.new("RGBA", (4, 4), (5, 140, 180, 255)).save(tmp_path / "t_mro_lob_w_lv3.png")  # wax: not metal
    links = {"DiffuseColor": "t_d_lob_x_lv3.png"}
    info = info_from_fbx(links, str(tmp_path), None, ["t_d_lob_x_lv3.png", "t_mroe_lob_x_lv3.png", "t_matcap_lob_x.png"], "mat_lob_x_lv3")
    tex: dict[str, TextureImage] = {}
    info["family"] = "cloth"  # FBX-only: keyword terms follow the shader default (`_IFMAT` 0), so no matTex (WP-D)
    m = build_material("mat_lob_x_lv3", info, str(tmp_path), tex)
    assert "matTex" not in m.extras.get("unite", {}) and "matcapMode" not in m.extras and m.metallic_factor == 0.0
    info = info_from_fbx(links, str(tmp_path), None, [], "mat_lob_x_lv3")
    m = build_material("mat_lob_x_lv3", info, str(tmp_path), {})
    # no matcap named by the material: no invented reflection, metal shows its albedo (Rapidash vs Ponyta)
    assert "matcap" not in m.extras and m.metallic_factor == 0.0 and m.metallic_roughness
    info = info_from_fbx({"DiffuseColor": "t_d_lob_w_lv3.png"}, str(tmp_path), None, [], "mat_lob_w_lv3")
    m = build_material("mat_lob_w_lv3", info, str(tmp_path), {})
    assert "matcap" not in m.extras and m.metallic_factor == 0.0


def test_matcap_and_mask_albedo_links(tmp_path):
    """Litwick's `fireB` links its matcap as DiffuseColor; NPC hair can link a `t_m_` mask."""
    info = info_from_fbx({"DiffuseColor": "t_matcap_bat_0607fireD.png"}, str(tmp_path), None, [], "mat_bat_0607fireB_lv3")
    assert "_MainTex" not in info["textures"] and info["textures"]["_MatCapTex"]["name"] == "t_matcap_bat_0607fireD"
    assert matcap_only(info)
    info = info_from_fbx({"DiffuseColor": "t_m_lob_h.png"}, str(tmp_path), None, ["t_m_lob_h.png", "t_d_lob_h_lv3.png"], "mat_lob_h_lv3")
    assert info["textures"]["_MainTex"]["name"] == "t_d_lob_h_lv3"


def test_glass_rules(tmp_path):
    cap = np.full((4, 4, 4), 200, np.uint8)
    cap[..., 3] = 40
    Image.fromarray(cap).save(tmp_path / "t_matcap_bat_014320b.png")  # translucent: glass
    Image.new("RGBA", (4, 4), (80, 80, 80, 255)).save(tmp_path / "t_matcap_bat_014320a.png")  # opaque: chrome
    # (a) battle: no links, translucent matcap named after the material (with the bat_ prefix)
    p = matcap_of(info_from_fbx({}, str(tmp_path), None, [], "mat_bat_014320b_lv3"), str(tmp_path), None, "mat_bat_014320b_lv3")
    assert p and p.endswith("t_matcap_bat_014320b.png")
    assert matcap_of(info_from_fbx({}, str(tmp_path), None, [], "mat_bat_014320a_lv3"), str(tmp_path), None, "mat_bat_014320a_lv3") is None
    # (b) lobby: albedo only, transparent under the material's UVs; glass image = a translucent sphere in its block
    alb = np.full((8, 8, 4), 212, np.uint8)
    alb[:4, :, 3] = 0
    Image.fromarray(alb).save(tmp_path / "t_d_lob_014320a_lv3.png")
    Image.fromarray(cap).save(tmp_path / "t_d_lob_014320.png")
    info = info_from_fbx({"DiffuseColor": "t_d_lob_014320a_lv3.png"}, str(tmp_path), None,
                         ["t_d_lob_014320a_lv3.png", "t_d_lob_014320.png"], "mat_lob_014320b_lv3")
    helmet = Primitive("m", np.arange(3, dtype=np.uint32), np.zeros((3, 3), np.float32), uvs=[np.full((3, 2), 0.2, np.float32)])
    suit = Primitive("m", np.arange(3, dtype=np.uint32), np.zeros((3, 3), np.float32), uvs=[np.full((3, 2), 0.8, np.float32)])
    assert clear_albedo(info, helmet, str(tmp_path), None) and not clear_albedo(info, suit, str(tmp_path), None)
    assert matcap_of(info, str(tmp_path), None, "mat_lob_014320b_lv3").endswith("t_d_lob_014320.png")
    info["textures"]["_BumpMap"] = {"name": "t_n", "scale": (1, 1), "offset": (0, 0)}
    assert not clear_albedo(info, helmet, str(tmp_path), None)  # a normal-mapped surface: the alpha is a cloth mask


def _fbx_info(**slots):
    return {"name": "", "shader": "", "floats": {}, "colors": {}, "source": "fbx",
            "textures": {k: {"name": v, "scale": (1, 1), "offset": (0, 0)} for k, v in slots.items()}}


def test_normal_map_means_lit_material_whatever_the_name():
    assert is_character_material("mat_bat_0077hari_fx_lv3", _fbx_info(_MainTex="t_d_bat_0077body_lv3", _BumpMap="t_n_bat_0077body_lv3"))
    assert not is_character_material("mat_bat_0077hari_fx_lv3", _fbx_info(_MainTex="t_d_bat_0077body_lv3"))
    lobby = {**_fbx_info(_MainTex="t_d_lob_0078bodyb_lv3"), "block": ["t_d_lob_0078bodyb_lv3.png", "t_vc_lob_0078body.png"]}
    assert is_character_material("mat_lob_0078haria_fx_lv3", lobby)


def test_opaque_atlas_effect_is_solid_not_additive(tmp_path):
    Image.new("RGBA", (2, 2), (250, 200, 60, 255)).save(tmp_path / "t_d_bat_0910_fx_lv3.png")
    m = build_effect_material("f", _fbx_info(_MainTex="t_d_bat_0910_fx_lv3"), str(tmp_path), {})
    assert m.alpha_mode == "OPAQUE" and m.extras["solidEffect"] and "blend" not in m.extras


def test_enclosure_separates_shells_from_one_sided_parts():
    body = np.random.default_rng(0).normal(size=(400, 3))
    assert enclosure(body * 2.0, body, np.zeros(3)) == 1.0
    mane = body[body[:, 1] > 0.5] * 1.5
    assert enclosure(mane, body, np.zeros(3)) < 0.9


def _tri(u, material):
    return Primitive(material, np.arange(3, dtype=np.uint32), np.zeros((3, 3), np.float32), uvs=[np.full((3, 2), u, np.float32)])


def test_sibling_slot_colour_and_hidden_untextured_parts(tmp_path):
    atlas = np.zeros((4, 4, 4), np.uint8)
    atlas[:, :2] = (250, 230, 40, 255)  # yellow half
    atlas[:, 2:] = (250, 120, 20, 255)  # orange half
    Image.fromarray(atlas).save(tmp_path / "t_d_bat_0910_fx_lv3.png")
    info = {"mat_a": _fbx_info(_MainTex="t_d_bat_0910_fx_lv3"), "mat_b": _fbx_info()}
    a, b = _tri(0.2, "mat_a"), _tri(0.8, "mat_b")
    mesh = Mesh("fx", [a, b])
    assert sibling_colour(mesh, b, lambda n: info[n], str(tmp_path), None) == {"_MainTex": info["mat_a"]["textures"]["_MainTex"]}
    m = _model([mesh])
    m.materials = {"mat_a": Material("mat_a"), "mat_b": Material("mat_b", extras={"missingTexture": True})}
    assert split_missing_texture_parts(m) == 1
    assert [x.name for x in m.meshes] == ["fx", "fx~notex"] and not m.meshes[1].visible


def test_particle_emitters_are_prefabs_without_meshes():
    N = fbx_anim.FbxNode
    nodes = {1: N(1, "Bone_Tail06", "LimbNode"), 2: N(2, "pref_fx_0004_battlefire_lv3", "Null", parent=1),
             3: N(3, "huo 1", "Null", parent=2), 4: N(4, "pref_fx_Gastly_idle_lv3", "Null", parent=1),
             5: N(5, "mesh huo1", "Mesh", parent=4), 6: N(6, "pref_fx_Dhelmise_natk1_atk1", "Null", parent=1)}
    assert particle_emitters(fbx_anim.FbxScene(nodes=nodes)) == ["pref_fx_0004_battlefire_lv3"]


def test_tinted_lens_slot(tmp_path):
    atlas = np.full((8, 8, 4), 200, np.uint8)
    atlas[..., 3] = 255
    atlas[:4, :4] = (40, 40, 45, 234)  # the lens region: dark, uniform alpha below 255
    Image.fromarray(atlas).save(tmp_path / "t_d_lob_000902_lv3.png")
    u = {"source": "fbx", "textures": {"_MainTex": "t_d_lob_000902_lv3"}}
    m = _model([Mesh("rig", [_tri(0.8, "body"), _tri(0.2, "lens")])])
    m.materials = {"body": Material("body", extras={"unity": u}), "lens": Material("lens", extras={"unity": u})}
    assert flag_lenses(m, str(tmp_path), None) == ["lens"]
    assert m.materials["lens"].alpha_mode == "BLEND" and m.materials["body"].alpha_mode == "OPAQUE"


def test_clip_names_keep_the_files_own_mode():
    from pipeline.importers.unite.importer import clip_names
    # Morpeko's Hangry file (_12) also carries Full Belly (_11) clips; idle is the file's own mode
    assert clip_names(["ani_run_bat_0877_12", "ani_idle_bat_0877_12", "ani_run_bat_0877_11", "ani_atk1_bat_0877_11"]) == \
        ["run", "idle", "run [0877_11]", "atk1"]


def test_form_tokens_only_where_folders_collide():
    from pipeline.catalog.unite_battle import form_token
    assert form_token("11_morpeko", "Morpeko") == "11"
    assert form_token("aegislash_shield", "Aegislash") == "shield"
    assert form_token("box", "Dragapult") == "box"
    assert form_token("00_hoopa", "Hoopa") == "" and form_token("mewtwo", "Mewtwo") == ""


def _sheet(tmp_path, name, clear):
    a = np.full((8, 8, 4), 180, np.uint8)
    a[..., 3] = 255
    if clear:
        a[:6, :, 3] = 0  # mostly transparent cut-out sheet
    Image.fromarray(a).save(tmp_path / f"{name}.png")


def test_overlay_decal_and_alternate_state(tmp_path):
    from pipeline.importers.unite.importer import flag_overlay_decals, hide_alternate_states
    rng = np.random.default_rng(1)
    face = rng.random((60, 3)).astype(np.float32)
    body = Primitive("body", np.arange(60, dtype=np.uint32), rng.random((60, 3)).astype(np.float32) + 2)
    eye = Primitive("eye", np.arange(60, dtype=np.uint32), face)
    decal = Primitive("cheek", np.arange(60, dtype=np.uint32), face + 0.0005)  # on the eye slot, within 3 mm
    _sheet(tmp_path, "t_d_cheek", True)
    m = _model([Mesh("rig", [body, eye]), Mesh("rigb", [decal])])
    m.materials = {"body": Material("body"), "eye": Material("eye"),
                   "cheek": Material("cheek", base_color="t_d_cheek", extras={"unity": {"textures": {"_MainTex": "t_d_cheek"}}})}
    m.textures = {"t_d_cheek": TextureImage("t_d_cheek", np.full((8, 8, 4), 255, np.uint8))}
    assert flag_overlay_decals(m, str(tmp_path), None) == ["rigb"]
    assert not m.meshes[1].visible and m.materials["cheek"].alpha_mode == "MASK"
    assert m.textures["t_d_cheek"].rgba[0, 0, 3] == 0  # real alpha restored
    # a partial overlap is not an alternate state; a whole copy is
    part = Mesh("teeth", [Primitive("body", np.arange(60, dtype=np.uint32), np.concatenate([face[:20], face[20:] + 1.0]))])
    copy = Mesh("copy", [Primitive("body", np.arange(120, dtype=np.uint32), np.concatenate([body.positions, face]))])
    m2 = _model([Mesh("rig", [body, eye]), part, copy])
    m2.materials = {"body": Material("body"), "eye": Material("eye")}
    assert hide_alternate_states(m2) == ["copy"]
