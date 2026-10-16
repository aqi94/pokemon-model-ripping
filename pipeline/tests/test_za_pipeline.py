"""End-to-end tests on the pm0001_00_00 fixture (Bulbasaur, Legends: Z-A)."""

import json
import os
import struct

import numpy as np
import pytest

from pipeline.common.gltf_writer import write_glb
from pipeline.common.manifest import build_manifest
from pipeline.common.model import Clip, Joint, Material, Mesh, Model, Primitive, Skeleton, TextureImage, Track
from pipeline.importers.za import trinity as T
from pipeline.importers.za.importer import FolderSource, bake_layer_specular, bake_layers, collect_animations, import_model

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "pm0001_00_00")
MID = "pm0001_00_00"


def rd(name: str) -> bytes:
    with open(os.path.join(FIX, name), "rb") as f:
        return f.read()


def read_glb_json(path: str) -> dict:
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"glTF"
    n = struct.unpack_from("<I", data, 12)[0]
    return json.loads(data[20 : 20 + n])


# ---------------------------------------------------------------- readers


def test_trmdl_lists_lods_and_links():
    m = T.parse_trmdl(rd(f"{MID}.trmdl"))
    assert m.meshes[0] == f"{MID}.trmsh"
    assert len(m.meshes) == 4
    assert m.skeleton == f"{MID}.trskl"
    assert m.materials == [f"{MID}.trmtr"]


def test_trmsh_vertex_layout_and_counts():
    s = T.parse_trmsh(rd(f"{MID}.trmsh"))
    b = T.parse_trmbf(rd(f"{MID}.trmbf"))
    assert s.buffer_file == f"{MID}.trmbf"
    assert len(s.shapes) == len(b) == 4
    body = s.shapes[0]
    assert body.mesh_name == f"{MID}_body_mesh"
    usages = {e.usage for e in body.declarations[0].elements}
    assert {T.USAGE_POSITION, T.USAGE_NORMAL, T.USAGE_TEXCOORD, T.USAGE_BLEND_INDICES, T.USAGE_BLEND_WEIGHTS} <= usages
    v = T.decode_vertices(body.declarations[0], b[0])
    assert v.count == 2234
    assert v.positions.shape == (2234, 3)
    assert np.allclose(np.linalg.norm(v.normals, axis=1), 1.0, atol=2e-3)
    assert v.weights.shape == (2234, 4) and np.allclose(v.weights.sum(axis=1), 1.0, atol=1e-3)
    for part in body.parts:
        idx = T.read_indices(body, b[0], part)
        assert len(idx) == part.index_count and len(idx) % 3 == 0
        assert idx.max() < v.count


def test_trskl_hierarchy_and_palette():
    k = T.parse_trskl(rd(f"{MID}.trskl"))
    assert len(k.nodes) == 114
    assert len(k.joints) == 91
    assert k.nodes[0].parent == -1
    assert all(n.parent < i for i, n in enumerate(k.nodes))  # parents come first
    rig = sorted(n.joint_info for n in k.nodes if n.joint_info >= 0)
    assert rig == list(range(91))
    assert k.joints[0].inverse_bind is not None and k.joints[0].inverse_bind.shape == (4, 4)


def test_trmtr_materials_and_layer_colours():
    mats = {m.name: m for m in T.parse_trmtr(rd(f"{MID}.trmtr"))}
    assert set(mats) == {"body_a", "body_b", "r_eye", "l_eye"}
    body = mats["body_a"]
    assert body.shader == "IkCharacter"
    slots = {t.slot: t.file for t in body.textures}
    assert slots["BaseColorMap"] == f"{MID}_body_a_alb.bntx"
    assert slots["LayerMaskMap"] == f"{MID}_body_a_lym.bntx"
    assert "BaseColorLayer1" in body.colors and len(body.colors["BaseColorLayer1"]) == 4
    rare = {m.name: m for m in T.parse_trmtr(rd(f"{MID}_rare.trmtr"))}
    assert rare["body_a"].colors["BaseColorLayer1"] != body.colors["BaseColorLayer1"]


def test_tranm_tracks_and_quaternions():
    a = T.parse_tranm(rd(f"anim/{MID}_00030_walk01_loop.tranm"))
    assert a.looped and a.frame_count == 61 and a.frame_rate == 60
    assert len(a.tracks) > 40
    rots = [t.rotation for t in a.tracks if t.rotation is not None]
    assert rots
    for ch in rots:
        assert ch.values.shape[1] == 4
        assert np.allclose(np.linalg.norm(ch.values, axis=1), 1.0, atol=1e-4)
        assert (np.diff(ch.frames) >= 0).all()


def test_packed_quaternion_identity():
    # Three 15-bit fields at mid-scale (= 0 rad) with W omitted (index 3) -> identity
    mid = 16383
    word = (mid << 3) | (mid << 18) | (mid << 33) | 3
    packed = np.array([[word & 0xFFFF, (word >> 16) & 0xFFFF, (word >> 32) & 0xFFFF]], np.uint16)
    q = T.unpack_quaternions(packed)[0]
    assert abs(q[3]) > 0.999 and np.allclose(q[:3], 0, atol=1e-3)


# ---------------------------------------------------------------- baking


def test_bake_layers_uses_palette_where_mask_set():
    alb = np.full((4, 4, 4), 255, np.uint8)
    mask = np.zeros((4, 4, 4), np.uint8)
    mask[..., 0] = 255  # layer 1 everywhere
    out = bake_layers(alb, mask, [(1.0, 0.0, 0.0), (0, 1, 0), (0, 0, 1), (1, 1, 1)], [1, 1, 1, 1])
    assert out[0, 0, 0] == 255 and out[0, 0, 1] == 0 and out[0, 0, 2] == 0
    assert out[0, 0, 3] == 255
    # linear 0.214 (= sRGB 0.5) over white albedo encodes back to ~128
    out = bake_layers(alb, mask, [(0.214, 0.214, 0.214), (0, 1, 0), (0, 0, 1), (1, 1, 1)], [1, 1, 1, 1])
    assert 125 <= out[0, 0, 0] <= 131


def test_bake_layers_resizes_mask_without_premultiplying_alpha():
    alb = np.full((4, 4, 4), 255, np.uint8)
    mask = np.zeros((8, 8, 4), np.uint8)
    mask[..., 0] = 255  # layer 1 everywhere, alpha stays 0
    out = bake_layers(alb, mask, [(1.0, 0.0, 0.0), (0, 1, 0), (0, 0, 1), (1, 1, 1)], [1, 1, 1, 1])
    assert out[2, 2, 0] == 255 and out[2, 2, 1] == 0


# ---------------------------------------------------------------- model -> glb


@pytest.fixture(scope="module")
def model():
    src = FolderSource(FIX)
    anims = collect_animations(src, MID, "anim")
    return import_model(src, MID, "Bulbasaur", animations=anims)


def test_import_model_structure(model):
    assert [m.name for m in model.meshes if m.lod == 0] == [
        f"{MID}_body_mesh",
        f"{MID}_eye_mesh",
        f"{MID}_left_tuta_mesh",
        f"{MID}_right_tuta_mesh",
    ]
    assert {m.lod for m in model.meshes} == {0, 1, 2, 3}
    assert model.skeleton and len(model.skeleton.joints) == 114 and len(model.skeleton.palette) == 91
    assert set(model.materials) == {"body_a", "body_b", "r_eye", "l_eye"}
    assert model.materials["body_a"].base_color.endswith("_baked")
    assert [v.name for v in model.variants] == ["normal", "rare"]
    assert [c.name for c in model.clips] == ["defaultwait01_loop", "defaultidle01", "walk01_loop"]
    assert model.default_clip == "defaultwait01_loop"
    walk = next(c for c in model.clips if c.name == "walk01_loop")
    assert walk.loop and walk.category == "move" and abs(walk.duration - 1.0) < 1e-6
    # baked body colour is Bulbasaur teal, shiny is yellow-green
    body = model.textures[model.materials["body_a"].base_color].rgba[..., :3].reshape(-1, 3).mean(axis=0)
    assert body[1] > body[0] and body[1] > 100
    shiny = model.textures[model.variants[1].materials["body_a"].base_color].rgba[..., :3].reshape(-1, 3).mean(axis=0)
    assert shiny[1] > shiny[2]  # more green than blue


def test_material_extras_keep_uv_scale(model):
    # The primitive UV bake reads these; losing them silently breaks scaled axes.
    for m in model.materials.values():
        assert "uv_scale_effective" in m.extras and "mirror" in m.extras and m.extras.get("shader")


def test_write_glb_is_valid_structure(model, tmp_path):
    out = str(tmp_path / "model.glb")
    s = write_glb(model, out)
    g = read_glb_json(out)
    assert g["asset"]["version"] == "2.0"
    assert s["animations"] == 3 and len(g["animations"]) == 3
    assert len(g["skins"]) == 1 and len(g["skins"][0]["joints"]) == 91
    assert "KHR_materials_variants" in g["extensionsUsed"]
    # one glTF material per (name, variant): 4 materials x (default + rare) = 8
    assert len(g["materials"]) == len({m["name"] for m in g["materials"]}) == 8
    assert [v["name"] for v in g["extensions"]["KHR_materials_variants"]["variants"]] == ["normal", "rare"]
    # every accessor's referenced bufferView lies inside the buffer
    total = g["buffers"][0]["byteLength"]
    for bv in g["bufferViews"]:
        assert bv["byteOffset"] + bv["byteLength"] <= total
    for acc in g["accessors"]:
        assert acc["count"] > 0
    # tangents are VEC4, texcoords VEC2
    prim = g["meshes"][0]["primitives"][0]
    assert g["accessors"][prim["attributes"]["TANGENT"]]["type"] == "VEC4"
    assert g["accessors"][prim["attributes"]["TEXCOORD_0"]]["type"] == "VEC2"
    # animation inputs strictly increasing
    for anim in g["animations"]:
        for smp in anim["samplers"]:
            acc = g["accessors"][smp["input"]]
            assert acc["min"][0] <= acc["max"][0]


def test_manifest_shape(model):
    m = build_manifest(model, {"bytes": 1})
    assert m["id"] == MID and m["game"] == "za" and m["defaultClip"] == "defaultwait01_loop"
    assert m["variantGroups"][0]["name"] == "coloring"
    assert sum(o["default"] for o in m["variantGroups"][0]["options"]) == 1
    assert {c["name"] for c in m["clips"]} == {"defaultwait01_loop", "defaultidle01", "walk01_loop"}
    assert any(p["lod"] == 0 for p in m["parts"]) and m["stats"]["triangles"] > 0


def test_material_shading_params_exported(model, tmp_path):
    sh = model.materials["body_a"].extras["shading"]
    assert 0 < sh["halfLambertBias"] <= 1 and len(sh["shadowColor"]) == 3 and sh["unlit"] is False
    out = str(tmp_path / "s.glb")
    write_glb(model, out)
    g = read_glb_json(out)
    assert any("shading" in m.get("extras", {}) and "halfLambertBias" in m["extras"]["shading"] for m in g["materials"])


def test_manifest_props_from_attach_bones():
    joints = [Joint("root", -1, skinning=True), Joint("right_hand", 0, skinning=True), Joint("right_attach", 1, skinning=True)]
    sk = Skeleton(joints=joints, palette=[0, 1, 2])
    shown = Clip("advent01_loop", [Track("right_attach", "scale", np.array([0.0], np.float32), np.ones((1, 3), np.float32))], frame_count=2, frame_rate=1)
    plain = Clip("walk01_loop", [Track("right_hand", "rotation", np.array([0.0], np.float32), np.array([[0, 0, 0, 1]], np.float32))], frame_count=2, frame_rate=1)
    model = Model("t", "test", "T", "trainer", sk, [], {}, {}, clips=[shown, plain])
    props = build_manifest(model)["props"]
    assert props == [{"bone": "right_attach", "side": "right", "clips": ["advent01_loop"]}]


def test_synthetic_model_round_trip(tmp_path):
    """The common format must not depend on any game importer."""
    joints = [Joint("root", -1, skinning=True, inverse_bind=np.eye(4, dtype=np.float32)), Joint("tip", 0, translation=(0, 1, 0), skinning=True, inverse_bind=np.eye(4, dtype=np.float32))]
    sk = Skeleton(joints=joints, palette=[0, 1])
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    prim = Primitive(
        material="m",
        indices=np.array([0, 1, 2], np.uint32),
        positions=pos,
        normals=np.tile([0, 0, 1], (3, 1)).astype(np.float32),
        uvs=[np.zeros((3, 2), np.float32)],
        joints=np.zeros((3, 4), np.uint16),
        weights=np.tile([1, 0, 0, 0], (3, 1)).astype(np.float32),
    )
    tex = TextureImage("t", np.zeros((2, 2, 4), np.uint8))
    clip = Clip("idle", [Track("tip", "rotation", np.array([0, 1], np.float32), np.array([[0, 0, 0, 1], [0, 0, 0, 1]], np.float32))], frame_count=2, frame_rate=1)
    model = Model("x", "test", "X", "pokemon", sk, [Mesh("tri", [prim])], {"m": Material("m", base_color="t")}, {"t": tex}, clips=[clip], default_clip="idle")
    out = str(tmp_path / "x.glb")
    s = write_glb(model, out)
    g = read_glb_json(out)
    assert s["meshes"] == 1 and s["joints"] == 2 and s["animations"] == 1
    assert g["nodes"][g["skins"][0]["joints"][1]]["name"] == "tip"
    assert build_manifest(model)["clips"][0]["default"] is True


def test_layer_strength_placeholder_vs_real_alpha_zero():
    from pipeline.importers.za.importer import _layer_strength
    assert _layer_strength((0.03, 0.03, 0.06, 0.0)) == 1.0  # Tarragon eyelashes
    assert _layer_strength((1.0, 0.0, 0.0, 0.0)) == 0.0  # placeholder red
    assert _layer_strength((0.023, 0.972, 0.875, 0.0)) == 0.0  # placeholder cyan
    assert _layer_strength((0.2, 0.5, 0.4, 1.0)) == 1.0


def _material(name: str, colors: dict) -> T.TrMaterial:
    return T.TrMaterial(
        name=name, shader="Hair", shader_flags={"EnableBaseColorMap": "True"},
        textures=[T.TextureBinding("BaseColorMap", "hair_alb.bntx", 0)], samplers=[("repeat", "repeat")],
        floats={}, colors=colors, ints={}, alpha_type="Opaque", cull=-1,
    )


def test_shadow_texture_carries_unbaked_base_color():
    # No layer mask: BaseColor stays the glTF factor, so the shadow-side texture must be tinted
    # by it too (the Game shader uses it instead of factor x map). Rika's hair: near-white strands,
    # teal BaseColor; the untinted shadow bake made the hair grey-sage.
    from pipeline.importers.za.importer import _convert_material, linear_to_srgb

    alb = np.full((4, 4, 4), 255, np.uint8)
    textures: dict = {}
    bc = (0.0075, 0.171, 0.168, 1.0)
    load = lambda f: alb.copy()  # noqa: E731
    teal = _convert_material(_material("hairstyle", {"BaseColor": bc}), "", load, textures, {})
    plain = _convert_material(_material("hairstyle_obj", {}), "", load, textures, {})
    assert tuple(round(x, 4) for x in teal.base_color_factor[:3]) == tuple(round(x, 4) for x in bc[:3])
    sh = textures[teal.extras["shadow_color"]].rgba[0, 0]
    lit = linear_to_srgb(np.asarray(bc[:3], np.float32)) * 255
    assert sh[1] > 3 * sh[0] and sh[2] > 3 * sh[0]  # teal, not grey
    assert sh[1] < lit[1] and sh[2] < lit[2]  # darker than the lit colour
    # Same albedo, different BaseColor -> separate shadow textures (no key collision).
    assert teal.extras["shadow_color"] != plain.extras["shadow_color"]
    assert textures[plain.extras["shadow_color"]].rgba[0, 0, 0] > 150


def test_layer_specular_per_texel_offset():
    # Courier outfit (ff_drs0002_00 body): base 0.14 / offset 0.4 on unpainted texels, layer 2
    # (black trousers) 0.03 / offset 0. The trousers must keep their own offset in the mask.
    floats = {"SpecularIntensity": 0.14, "SpecularOffset": 0.4, "SpecularContrast": 0.0,
              "SpecularLayer2Intensity": 0.03, "SpecularLayer2Offset": 0.0, "SpecularLayer2Contrast": 0.0}
    mask = np.zeros((2, 2, 4), np.uint8)
    mask[0, :, 1] = 255  # top row: layer 2
    rgba, peak, off, con = bake_layer_specular(floats, mask, None, [1.0] * 4)
    assert peak == pytest.approx(0.14)
    assert rgba[0, 0, 0] == round(0.03 / 0.14 * 255) and rgba[1, 0, 0] == 255
    assert rgba[0, 0, 1] == 128 and abs(int(rgba[1, 0, 1]) - 0.7 * 255) <= 1  # offset 0 / 0.4 as (o+1)/2
    assert (rgba[..., 2] == 0).all()
    assert 0.0 < off < 0.4
