"""UNITE importer pieces that need no Blender: material JSON -> Material, shader rule, FBX material-layer check."""

import json
import struct

import numpy as np
from PIL import Image

from pipeline.catalog.unite import LOBBY_RE, _theme_label
from pipeline.common.model import TextureImage
from pipeline.importers.unite import fbx as fbxr
from pipeline.importers.unite.importer import build_material, is_character_shader, material_jsons


def _tex(name):
    return {"m_Texture": {"Name": name, "IsNull": False}, "m_Scale": {"X": 1.0, "Y": 1.0}, "m_Offset": {"X": 0.0, "Y": 0.0}}


def test_build_material(tmp_path):
    Image.new("RGBA", (2, 2), (200, 100, 50, 0)).save(tmp_path / "t_d.png")
    Image.new("RGBA", (2, 2), (10, 20, 30, 40)).save(tmp_path / "t_mroe.png")
    (tmp_path / "Materials").mkdir()
    (tmp_path / "Materials" / "mat_body.json").write_text(json.dumps({
        "m_Shader": {"Name": "==PGAME==/LobbyPokemon/m_lob_basepbr_lv3"},
        "m_Name": "mat_body",
        "m_SavedProperties": {
            "m_TexEnvs": {"_MainTex": _tex("t_d"), "_MixMap": _tex("t_mroe"), "_BumpMap": _tex("t_missing")},
            "m_Floats": {"_Cull": 0.0, "_SrcBlend": 1.0, "_DstBlend": 0.0, "_AlphaClip": 0.0},
            "m_Colors": {"_OutlineColor": {"r": 0.5, "g": 0.5, "b": 0.5, "a": 1.0}},
        },
    }))
    info = material_jsons(str(tmp_path))["mat_body"]
    textures: dict[str, TextureImage] = {}
    m = build_material("mat_body", info, str(tmp_path), textures)
    assert m.alpha_mode == "OPAQUE" and m.double_sided
    assert textures["t_d"].rgba[0, 0, 3] == 255  # opaque material: albedo alpha is not opacity
    assert m.normal is None  # texture file absent
    orm = textures[m.metallic_roughness].rgba[0, 0]
    assert list(orm[:3]) == [30, 20, 10]  # mro (R metal, G rough, B occl) -> glTF (R occl, G rough, B metal)
    # no matcap named by the material: metal shows its albedo (no invented reflection)
    assert m.occlusion == m.metallic_roughness and m.metallic_factor == 0.0
    assert m.extras["unity"]["colors"]["_OutlineColor"] == [0.5, 0.5, 0.5, 1.0]


def test_character_shader_rule():
    assert is_character_shader("==PGAME==/LobbyPokemon/m_lob_basepbrCloth_lv3")
    assert is_character_shader("Hidden/Lobby/m_lob_basepbrClothLengcy_lv3")
    assert is_character_shader("==PGAME==/Battle/m_bat_actorbase")
    assert not is_character_shader("==PGAME==/Fx/m_1P_Base")
    assert not is_character_shader("==PGAME==/Fx2/m_holographic2Pass")
    assert not is_character_shader("Unlit/Transparent Cutout")
    assert not is_character_shader("==PGAME==/LobbyScene/m_lob_LobbyUnlit")


def test_lobby_names():
    m = LOBBY_RE.match("pref_sce_lobby_068105_king_shield_lv3")
    assert m.groups() == ("0681", "05", "king_shield", "3")
    assert _theme_label("professorphorus") == "Professor Phorus"


# --- a tiny binary FBX 7300 writer, just enough for fbx.geometries ---------------------------------------------

def _prop(v):
    if isinstance(v, str):
        b = v.encode()
        return b"S" + struct.pack("<I", len(b)) + b
    if isinstance(v, int):
        return b"I" + struct.pack("<i", v)
    a = np.asarray(v, np.int32)
    return b"i" + struct.pack("<III", len(a), 0, a.nbytes) + a.tobytes()


def _node(name, props=(), children=(), offset=0):
    body = b"".join(_prop(p) for p in props)
    head_len = 13 + len(name)
    kids, pos = b"", offset + head_len + len(body)
    for c in children:
        k = c(pos)
        kids += k
        pos += len(k)
    if children:
        kids += b"\0" * 13
    end = offset + head_len + len(body) + len(kids)
    return struct.pack("<III", end, len(props), len(body)) + bytes([len(name)]) + name.encode() + body + kids


def N(name, props=(), children=()):
    return lambda off: _node(name, props, children, off)


def _write_fbx(path, geoms):
    data = b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", 7300)
    objs = []
    for name, npoly, mapping, mats in geoms:
        pvi = np.arange(npoly * 3, dtype=np.int32).reshape(-1, 3)
        pvi[:, 2] = ~pvi[:, 2]
        objs.append(N("Geometry", [1, f"{name}\x00\x01Geometry", "Mesh"], [
            N("PolygonVertexIndex", [pvi.ravel()]),
            N("LayerElementMaterial", [0], [N("MappingInformationType", [mapping]), N("Materials", [mats])]),
        ]))
    data += N("Objects", [], objs)(len(data)) + b"\0" * 13 + b"\0" * 160
    path.write_bytes(data)


def test_lost_material_layer(tmp_path):
    p = tmp_path / "x.fbx"
    _write_fbx(p, [
        ("good", 4, "ByPolygon", [0, 0, 1, 1]),
        ("lost", 4, "NoMappingInformation", [0, 1, 2]),  # one entry per slot: AssetStudio's broken layer
        ("single", 4, "NoMappingInformation", [0]),  # one slot: nothing lost
    ])
    got = {g["name"]: (g["polygons"], g["materials_ok"]) for g in fbxr.geometries(str(p))}
    assert got == {"good": (4, True), "lost": (4, False), "single": (4, True)}


def test_effect_materials_and_clip_names():
    from pipeline.importers.unite.importer import clean_clip_name, is_character_material

    def info(d="t_d_bat_0025body_lv3", shader=""):
        return {"shader": shader, "textures": {"_MainTex": {"name": d}}}

    for fx in ("mat_fx_0006_fire_10_gp", "mat_bat_000912fx_glow_lv3", "mat_bat_009411fxa_lv3", "mat_lob_fx_0146_wei"):
        assert not is_character_material(fx, info())
    assert not is_character_material("mat_bat_049102_lv3", info("t_fx_049102_bat_mask_04_yjz"))
    assert is_character_material("mat_bat_0815fix_lv3", info())
    assert is_character_material("mat_fx_named_but_json", info(shader="==PGAME==/LobbyPokemon/m_lob_basepbr_lv3"))
    assert clean_clip_name("ani_spell2a1_bat_000609") == "spell2a1"
    assert clean_clip_name("ani_idle_lob_003") == "idle"


def test_info_from_fbx_finds_mix_map_by_name(tmp_path):
    from pipeline.importers.unite.importer import info_from_fbx

    Image.new("RGBA", (2, 2)).save(tmp_path / "t_mroe_bat_0025body_lv3.png")
    i = info_from_fbx({"DiffuseColor": "t_d_bat_0025body_lv3.png", "Bump": "t_n_default.png"}, str(tmp_path))
    assert {k: v["name"] for k, v in i["textures"].items()} == {
        "_MainTex": "t_d_bat_0025body_lv3", "_BumpMap": "t_n_default", "_MixMap": "t_mroe_bat_0025body_lv3"}
    # elsewhere in the tree: found through the index
    other = tmp_path / "x"
    other.mkdir()
    i = info_from_fbx({"DiffuseColor": "t_d_bat_0025body_lv3.png"}, str(other),
                      {"t_mroe_bat_0025body_lv3.png": str(tmp_path / "t_mroe_bat_0025body_lv3.png")})
    assert i["textures"]["_MixMap"]["name"] == "t_mroe_bat_0025body_lv3"


def test_fbx_transform_math():
    from pipeline.importers.unite import fbx_anim

    # XYZ Euler order: R = Rz Ry Rx; a 90 deg Z turn maps +X to +Y
    r = fbx_anim.euler_xyz((0, 0, 90))
    assert np.allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-9)
    # non-uniform scale survives exactly: local = T R S, no decomposition involved
    m = fbx_anim.local_matrix((1, 2, 3), (10, 20, 30), (0.01, 1, 1))
    assert np.allclose(np.linalg.norm(m[:3, :3], axis=0), [0.01, 1, 1])
    t, q, s = fbx_anim.decompose(fbx_anim.local_matrix((1, 2, 3), (0, 0, 90), (0.5, 0.5, 0.5)))
    assert np.allclose(t, (1, 2, 3)) and np.allclose(s, (0.5, 0.5, 0.5))
    assert np.allclose(np.abs(q), (0, 0, np.sqrt(0.5), np.sqrt(0.5)), atol=1e-6)
    times, fps = fbx_anim.clip_times({1: {"R": {0: (np.array([0.0, 1 / 30, 2 / 30]), np.zeros(3))}}})
    assert fps == 30.0 and len(times) == 3


def test_prune_unused_joints():
    from pipeline.common.model import Clip, Joint, Mesh, Model, Primitive, Skeleton, Track
    from pipeline.importers.unite.importer import prune_unused_joints

    eye = np.eye(4, dtype=np.float32)
    joints = [Joint("root", -1, inverse_bind=eye), Joint("body", 0, inverse_bind=eye),
              Joint("fx", 0, inverse_bind=eye), Joint("fx_child", 2, inverse_bind=eye)]
    sk = Skeleton(joints=joints, palette=[0, 1, 2, 3])
    prim = Primitive(material="m", indices=np.array([0, 1, 2], np.uint32), positions=np.zeros((3, 3), np.float32),
                     joints=np.array([[1, 3, 0, 0]] * 3, np.uint16),
                     weights=np.array([[1, 0, 0, 0]] * 3, np.float32))  # fx_child listed but weight 0
    m = Model(id="t", game="unite", display_name="t", kind="pokemon", skeleton=sk, meshes=[Mesh("m", [prim])], materials={}, textures={},
              clips=[Clip("idle", [Track("fx", "scale", np.zeros(1, np.float32), np.ones((1, 3), np.float32))])])
    assert prune_unused_joints(m) == 2
    assert [j.name for j in m.skeleton.joints] == ["root", "body"] and m.skeleton.palette == [0, 1]
    assert prim.joints[0, 0] == 1 and not m.clips[0].tracks


def _scaled_rig_scene(mesh_offset=(0.0, 0.0, 0.0), root_t=(0.0, 0.0, 0.0)):
    """Root Null -> `reference` (scale 0.745) -> `Bip001` (at 0.4 m) + a skinned mesh under `reference`. Returns
    (scene, Model as Blender dumps it: bone WITHOUT the reference scale, vertices in bind space + `mesh_offset`)."""
    from pipeline.common.model import Joint, Mesh, Model, Primitive, Skeleton
    from pipeline.importers.unite import fbx_anim

    N = fbx_anim.FbxNode
    nodes = {1: N(1, "root", "Null", t=root_t), 2: N(2, "reference", "LimbNode", s=(0.745, 0.745, 0.745), parent=1),
             3: N(3, "Bip001", "LimbNode", t=(0.0, 0.4, 0.0), parent=2), 4: N(4, "body", "Mesh", parent=2)}
    scene = fbx_anim.FbxScene(nodes=nodes)
    nodes[3].bind = fbx_anim.world_rest(scene, 3)
    verts = np.array([[0, 0, 0], [0.1, 0.5, 0], [0, 0.6, 0.1]], np.float64)
    scene.mesh_binds["body"] = (4, verts)
    scene.clustered.add(3)
    blender_bone = np.eye(4)
    blender_bone[1, 3] = 0.4  # Blender: unscaled 0.4 instead of 0.298
    sk = Skeleton(joints=[Joint("Bip001", -1, inverse_bind=np.linalg.inv(blender_bone), skinning=True)], palette=[0])
    pos = (verts * 0.745 + 2 * np.asarray(root_t) + np.asarray(mesh_offset)).astype(np.float32)
    prim = Primitive("m", np.array([0, 1, 2], np.uint32), pos, joints=np.zeros((3, 4), np.uint16),
                     weights=np.array([[1, 0, 0, 0]] * 3, np.float32))
    return scene, Model(id="t", game="unite", display_name="t", kind="pokemon", skeleton=sk,
                        meshes=[Mesh("body", [prim])], materials={}, textures={})


def test_fbx_skeleton_trusts_vertices_not_blender_bones():
    """Blender drops the `reference` scale from its bones (Pikachu `Bip001` 0.24 m off) but dumps the vertices right:
    the FBX skeleton must still be used."""
    from pipeline.importers.unite.importer import apply_fbx_skeleton

    scene, model = _scaled_rig_scene()
    assert apply_fbx_skeleton(model, scene) is None
    ib = model.skeleton.joints[0].inverse_bind
    assert np.allclose(np.linalg.inv(ib)[1, 3], 0.4 * 0.745, atol=1e-6)


def test_fbx_skeleton_root_transform_applied_twice():
    """Charizard battle: the root Null sits at x -1 m and Blender dumps the mesh at -2 m. The mesh is re-placed from
    the dump's own matrix, and the model ends up in the root node's space (centred), binds included."""
    from pipeline.importers.unite.importer import apply_fbx_skeleton

    scene, model = _scaled_rig_scene(root_t=(-1.0, 0.0, 0.0))
    placed = np.diag([0.745, 0.745, 0.745, 1.0])
    placed[0, 3] = -2.0  # the matrix Blender's dump used for this mesh
    assert apply_fbx_skeleton(model, scene, {"body": placed.tolist()}) is None
    assert np.allclose(model.meshes[0].primitives[0].positions.min(axis=0), [0, 0, 0], atol=1e-6)
    assert np.allclose(np.linalg.inv(model.skeleton.joints[0].inverse_bind)[:3, 3], [0, 0.4 * 0.745, 0], atol=1e-6)
    assert any("re-placed meshes" in w for w in model.extras["warnings"])


def test_fbx_skeleton_refused_when_geometry_differs():
    from pipeline.importers.unite.importer import apply_fbx_skeleton

    scene, model = _scaled_rig_scene()
    model.meshes[0].primitives[0].positions[1] *= 1.2  # stretched: not a pure translation
    why = apply_fbx_skeleton(model, scene)
    assert why and "bind space" in why



def test_fbx_skeleton_accepts_mesh_with_dropped_primitives():
    """Ponyta: effect primitives (flame mane) are dropped, so the dumped mesh is a subset of the FBX geometry."""
    from pipeline.importers.unite import fbx_anim
    from pipeline.importers.unite.importer import apply_fbx_skeleton

    scene, model = _scaled_rig_scene()
    mid, v = scene.mesh_binds["body"]
    scene.mesh_binds["body"] = (mid, np.vstack([v, [[0, 2.0, -1.5]]]))  # an extra (dropped) far vertex
    assert apply_fbx_skeleton(model, scene) is None
    assert fbx_anim.matched_fraction(np.zeros((1, 3)), np.ones((1, 3))) == 0.0


def test_other_folder_characters(tmp_path):
    """Loose `Other/pref_char_*` FBXs join the catalog unless a Battle/Lobby model has the same dex + skin + kind."""
    from pipeline.catalog.unite import UniteCatalog
    from pipeline.catalog.za import ModelRef
    other = tmp_path / "Other"
    for folder in ("pref_char_bat_038400_rayquaza_scene_lv0", "pref_char_lob_000906_blastoise_lv1",
                   "pref_char_bat_065600_Sub", "pref_char_bat_011100_rhyhorn_lv2"):
        (other / folder).mkdir(parents=True)
        (other / folder / f"{folder}.fbx").write_bytes(b"")
    (other / "pref_char_bat_011100_rhyhorn_lv2_nofbx").mkdir()
    known = [ModelRef(game="unite", kind="pokemon", id="pm0009_06_x", folder="", display_name="", form="06",
                      extras={"dex": 9, "holowear": "000906"})]
    refs = {r.id: r for r in UniteCatalog(str(tmp_path), str(tmp_path / "none"), str(tmp_path / "none")).others(known)}
    assert set(refs) == {"pm0384_00_scene", "pm0111_00"}  # Blastoise lobby exists; `_Sub` is not a character folder
    r = refs["pm0384_00_scene"]
    assert r.display_name == "Rayquaza (scene)" and r.extras["battle"] and r.extras["standalone"]
    assert r.folder == "Other/pref_char_bat_038400_rayquaza_scene_lv0/pref_char_bat_038400_rayquaza_scene_lv0.fbx"


def test_marcy_wild_pokemon(tmp_path):
    """Marcy4000/Unite: only game-named FBXs, the largest copy per name, nothing the Drive share already has."""
    from pipeline.catalog.unite import UniteCatalog
    from pipeline.catalog.za import ModelRef
    files = {"Bunnelby/pref_char_bat_065900_bunnelby_lv3.fbx": 300, "pref_char_bat_065900_bunnelby_lv3/pref_char_bat_065900_bunnelby_lv3.fbx": 10,
             "Vespiqueen/pref_char_bat_041600_vespiquen_lv3.fbx": 50, "Gigantamax Pikachu/pref_char_bat_0025_81_pikachu_lv3.fbx": 50,
             "Pikachu/pref_char_bat_002500_pikachu_lv3.fbx": 50, "Flygon/flygon.fbx": 50}
    for rel, size in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x" * size)
    known = [ModelRef(game="unite", kind="pokemon", id="pm0025_00", folder="", display_name="", form="00",
                      extras={"dex": 25, "skin": "00", "battle": True})]
    refs = {r.id: r for r in UniteCatalog(str(tmp_path / "none"), str(tmp_path / "none"), str(tmp_path)).marcy(known)}
    assert set(refs) == {"pm0659_00", "pm0416_00", "pm0025_81"}
    assert refs["pm0659_00"].folder.endswith("Bunnelby" + __import__("os").sep + "pref_char_bat_065900_bunnelby_lv3.fbx")
    assert refs["pm0416_00"].display_name == "Vespiquen" and refs["pm0025_81"].display_name == "Gigantamax Pikachu"


def test_npc_clips_from_marcy(tmp_path):
    """Lobby NPCs take their clips from the Marcy4000/Unite copy of the same prefab (the MEGA FBX has no curves)."""
    from pipeline.catalog.unite import UniteCatalog
    folder = "pref_npc_lob_003_zirce_lv3"
    (tmp_path / "mega" / "NPCs" / "Lv3" / folder).mkdir(parents=True)
    (tmp_path / "mega" / "NPCs" / "Lv3" / folder / f"{folder}.fbx").write_bytes(b"")
    (tmp_path / "marcy" / "NPCs (Shopkeepers, professors)").mkdir(parents=True)
    donor = tmp_path / "marcy" / "NPCs (Shopkeepers, professors)" / f"{folder}.fbx"
    donor.write_bytes(b"")
    (ref,) = UniteCatalog(str(tmp_path / "mega"), str(tmp_path / "none"), str(tmp_path / "marcy")).trainers()
    assert ref.id == "npc003_zirce" and ref.extras["baseClips"] == str(donor)
