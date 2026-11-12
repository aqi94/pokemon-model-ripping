"""LGPE field maps / props (`field/<area>/*.gfbmdl`): parser additions (UV2, samplers, groups, cull/blend), static
import rules and catalog discovery, on tiny synthetic files (no real data needed)."""

import os

import flatbuffers
import numpy as np

from pipeline.catalog.lgpe import LgpeCatalog, field_label
from pipeline.common.gltf_writer import write_glb
from pipeline.importers.lgpe import field as F
from pipeline.importers.lgpe.gfbmdl import parse_gfbmdl


def _vec_of(b, offsets):
    b.StartVector(4, len(offsets), 4)
    for o in reversed(offsets):
        b.PrependUOffsetTRelative(o)
    return b.EndVector()


def _vec3(b, v):
    b.Prep(4, 12)
    for x in reversed(v):
        b.PrependFloat32(x)
    return b.Offset()


def _named_u32(b, items):
    out = []
    for k, v in items.items():
        n = b.CreateString(k)
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(0, n, 0)
        b.PrependUint32Slot(1, v, 0xFFFFFFFF)
        out.append(b.EndObject())
    return _vec_of(b, out)


def _material(b, tex_names, name, shader, slots, floats=None, colors=None, switches=None, cull=None, blend=None, render=None):
    """slots: [(slot name, texture index, wrap u, wrap v)]; render: (toggles, values) for field 20"""
    rs = None
    if render:
        t0, t1 = _named_u32(b, render[0]), _named_u32(b, render[1])
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(0, t0, 0)
        b.PrependUOffsetTRelativeSlot(1, t1, 0)
        rs = b.EndObject()
    slot_offs = []
    for sname, idx, wu, wv in slots:
        b.StartObject(6)
        b.PrependUint32Slot(2, wv, 0)
        b.PrependUint32Slot(1, wu, 0)
        samp = b.EndObject()
        n = b.CreateString(sname)
        b.StartObject(3)
        b.PrependUOffsetTRelativeSlot(0, n, 0)
        b.PrependUint32Slot(1, idx, 0)
        b.PrependUOffsetTRelativeSlot(2, samp, 0)
        slot_offs.append(b.EndObject())
    fl = []
    for k, v in (floats or {}).items():
        n = b.CreateString(k)
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(0, n, 0)
        b.PrependFloat32Slot(1, v, 0)
        fl.append(b.EndObject())
    cl = []
    for k, v in (colors or {}).items():
        n = b.CreateString(k)
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(0, n, 0)
        b.PrependStructSlot(1, _vec3(b, v), 0)
        cl.append(b.EndObject())
    sw = []
    for k, v in (switches or {}).items():
        n = b.CreateString(k)
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(0, n, 0)
        b.PrependUint8Slot(1, v, 0)
        sw.append(b.EndObject())
    tv, fv, cv, swv = _vec_of(b, slot_offs), _vec_of(b, fl), _vec_of(b, cl), _vec_of(b, sw)
    nm, sh = b.CreateString(name), b.CreateString(shader)
    b.StartObject(21)
    b.PrependUOffsetTRelativeSlot(0, nm, 0)
    b.PrependUOffsetTRelativeSlot(1, sh, 0)
    if blend is not None:
        b.PrependUint32Slot(6, blend, 99)
    if cull is not None:
        b.PrependUint32Slot(7, cull, 99)
    b.PrependUOffsetTRelativeSlot(11, tv, 0)
    b.PrependUOffsetTRelativeSlot(12, swv, 0)
    b.PrependUOffsetTRelativeSlot(13, fv, 0)
    b.PrependUOffsetTRelativeSlot(14, cv, 0)
    if rs is not None:
        b.PrependUOffsetTRelativeSlot(20, rs, 0)
    return b.EndObject()


def build_field_gfbmdl(vertex_rgb=(128, 255, 255)) -> bytes:
    """2 meshes on identity bones: mesh 0 = a 1 m quad (cm) with UV0/UV1/UV2 + colour, polygons using material 0
    (tiled object texture, two-sided cut-out) and 1 (shadow only); mesh 1 = a quad with material 2 (ramp tree)."""
    b = flatbuffers.Builder(2048)
    tex_names = ["wall_com", "shadowtable_t", "leaf_mask", "leaf_ramp"]
    tex_vec = _vec_of(b, [b.CreateString(t) for t in tex_names])
    m0 = _material(b, tex_names, "wall", "FieldObjectShader", [("Texture01", 0, 0, 0), ("ShadowToonTable", 1, 1, 1)],
                   floats={"DiscardValuie": 0.85, "Tex01_Scale_U": 2.0, "Tex01_Scale_V": 1.0}, cull=0)
    m1 = _material(b, tex_names, "shadow", "FieldShadowOnlyShader", [], switches={"SkipMainRendering": 1})
    m2 = _material(b, tex_names, "tree", "FieldTreeShader05", [("Texture01", 1, 0, 0), ("Texture02", 3, 1, 1), ("Texture03", 2, 0, 0)],
                   floats={"DiscardValuie": 0.5}, cull=1, render=({"DiscardEnable": 1025}, {"CullMode": 2, "BlendMode": 0, "UV_tex01": 0}))
    mats = _vec_of(b, [m0, m1, m2])

    def mesh(mats_idx, uv_sets, rgb):
        dt = [("p", "<f4", 3), ("n", "<f2", 4), ("t", "<f2", 4)] + [(f"uv{k}", "<f4", 2) for k in range(uv_sets)] + [("c", "u1", 4)]
        v = np.zeros(4, dtype=dt)
        v["p"] = [(0, 0, 0), (100, 0, 0), (100, 0, 100), (0, 0, 100)]
        v["n"] = [(0, 1, 0, 0)] * 4
        for k in range(uv_sets):
            v[f"uv{k}"] = [(0, 0), (5 + k, 0), (5 + k, 3), (0, 3)]  # tiles far outside [0, 1]
        v["c"] = [(*rgb, 77)] * 4
        vb = b.CreateByteVector(v.tobytes())
        polys = []
        for mi, idx in mats_idx:
            iv = b.CreateNumpyVector(np.array(idx, np.uint16))
            b.StartObject(2)
            b.PrependUint32Slot(0, mi, 99)
            b.PrependUOffsetTRelativeSlot(1, iv, 0)
            polys.append(b.EndObject())
        pv = _vec_of(b, polys)
        attrs = []
        spec = [(0, 0, 3), (1, 1, 4), (2, 1, 4)] + [(3 + k, 0, 2) for k in range(uv_sets)] + [(7, 3, 4)]
        for usage, typ, cnt in spec:
            b.StartObject(3)
            b.PrependUint32Slot(0, usage, 99)
            b.PrependUint32Slot(1, typ, 99)
            b.PrependUint32Slot(2, cnt, 99)
            attrs.append(b.EndObject())
        av = _vec_of(b, attrs)
        b.StartObject(3)
        b.PrependUOffsetTRelativeSlot(0, pv, 0)
        b.PrependUOffsetTRelativeSlot(1, av, 0)
        b.PrependUOffsetTRelativeSlot(2, vb, 0)
        return b.EndObject()

    meshes = _vec_of(b, [mesh([(0, [0, 1, 2]), (1, [0, 2, 3])], 3, vertex_rgb), mesh([(2, [0, 1, 2, 0, 2, 3])], 1, (0, 0, 0))])
    groups = []
    for bone, mi in ((1, 0), (2, 1)):
        b.StartObject(3)
        b.PrependUint32Slot(0, bone, 99)
        b.PrependUint32Slot(1, mi, 99)
        b.PrependFloat32Slot(2, 12.5, 0)
        groups.append(b.EndObject())
    gv = _vec_of(b, groups)
    bones = []
    for name, parent in (("map_root", -1), ("house01", 0), ("tree01", 0)):
        n = b.CreateString(name)
        b.StartObject(10)
        b.PrependUOffsetTRelativeSlot(0, n, 0)
        b.PrependInt32Slot(2, parent, 99)
        b.PrependStructSlot(5, _vec3(b, (1, 1, 1)), 0)
        b.PrependStructSlot(6, _vec3(b, (0, 0, 0)), 0)
        b.PrependStructSlot(7, _vec3(b, (0, 0, 0)), 0)
        bones.append(b.EndObject())
    bv = _vec_of(b, bones)
    b.StartObject(10)
    b.PrependUint32Slot(0, 0x18020511, 0)
    b.PrependUOffsetTRelativeSlot(2, tex_vec, 0)
    b.PrependUOffsetTRelativeSlot(6, mats, 0)
    b.PrependUOffsetTRelativeSlot(7, gv, 0)
    b.PrependUOffsetTRelativeSlot(8, meshes, 0)
    b.PrependUOffsetTRelativeSlot(9, bv, 0)
    b.Finish(b.EndObject())
    return bytes(b.Output())


def _textures():
    wall = np.full((4, 4, 4), 200, np.uint8)
    wall[0, 0, 3] = 0  # a hole -> cut-out
    mask = np.full((4, 4, 4), 255, np.uint8)
    mask[..., 3] = 0
    mask[1:3, 1:3, 3] = 255
    ramp = np.zeros((8, 2, 4), np.uint8)  # W x 4W vertical ramp: green opaque top half, transparent red bottom
    ramp[:4] = (0, 200, 100, 255)
    ramp[4:] = (255, 0, 0, 0)
    return {"wall_com": wall, "shadowtable_t": np.full((1, 8, 4), 255, np.uint8), "leaf_mask": mask, "leaf_ramp": ramp}


def _model(**kw):
    g = parse_gfbmdl(build_field_gfbmdl(**kw))
    tex = _textures()
    return g, F.build_field_model(g, tex.get, "town999_00", "Test Town", "map")


def test_parser_reads_uv2_samplers_groups_cull():
    g = parse_gfbmdl(build_field_gfbmdl())
    assert sorted(g.meshes[0].uv_sets) == [0, 1, 2]
    assert np.allclose(g.meshes[0].uv_sets[2][1], (7, 0))
    assert len(g.meshes[0].uvs) == 2  # the Pokemon path still sees UV0/UV1 only
    assert g.materials[0].samplers["Texture01"] == ("repeat", "repeat")
    assert g.materials[0].samplers["ShadowToonTable"] == ("clamp", "clamp")
    assert g.materials[0].cull == 0 and g.materials[2].cull == 1
    assert [(gr.bone, gr.mesh) for gr in g.groups] == [(1, 0), (2, 1)]
    assert g.materials[2].render == {"DiscardEnable": 1025, "CullMode": 2, "BlendMode": 0, "UV_tex01": 0}


def test_static_import_units_uvs_and_materials():
    _, m = _model()
    assert m.skeleton is None and m.kind == "map"
    assert [x.name for x in m.meshes] == ["house01", "tree01"]  # named after the group's bone
    wall = m.meshes[0].primitives
    assert len(wall) == 1  # the shadow-only polygon is not drawn
    p = wall[0]
    assert np.allclose(p.positions.max(0), (1.0, 0.0, 1.0))  # cm -> m, no transform (identity bones)
    assert len(p.uvs) == 3  # UV0, UV1, UV2 kept as TEXCOORD_0..2 (lightmap-ish sets survive)
    assert np.allclose(p.uvs[0][2], (5.0, 1.0 - 3.0))  # V flipped, tiling kept (no wrap baked)
    mat = m.materials["wall"]
    assert mat.wrap["base_color"] == ("repeat", "repeat")
    assert mat.uv_transform["base_color"] == (2.0, 1.0, 0.0, 0.0)  # Tex01_Scale_U
    assert mat.alpha_mode == "MASK" and abs(mat.alpha_cutoff - 0.85) < 1e-6  # DiscardValuie
    assert mat.double_sided  # cull 0
    assert p.colors is not None and abs(p.colors[0, 0] - 128 / 255) < 1e-3 and p.colors[0, 3] == 1.0
    assert m.extras["shadowOnlyTriangles"] == 1
    assert m.extras["sizeMetres"] == [1.0, 0.0, 1.0]


def test_ramp_tree_bake_and_data_vertex_colours():
    _, m = _model()
    tree = m.materials["tree"]
    assert tree.extras.get("rampTree")
    assert tree.double_sided  # field 20 CullMode 2 wins over field 7
    assert tree.alpha_mode == "MASK"  # DiscardEnable 1025
    img = m.textures[tree.base_color].rgba
    assert img[0, 0, 3] == 0 and img[1, 1, 3] == 255  # Texture03's alpha is the cut-out
    assert tuple(img[1, 1, :3]) == (0, 200, 100)  # x the ramp's opaque (alpha-weighted) colour
    assert m.meshes[1].primitives[0].colors is None  # tree vertex RGB is data, not a tint


def test_black_vertex_colour_is_unpainted():
    _, m = _model(vertex_rgb=(0, 0, 0))
    assert m.meshes[0].primitives[0].colors is None


def test_glb_writes_static_scene(tmp_path):
    _, m = _model()
    s = write_glb(m, str(tmp_path / "model.glb"))
    assert s["joints"] == 0 and s["meshes"] == 2 and s["materials"] == 2


def test_field_labels():
    assert field_label("town001_00", "area02") == "Pallet Town"
    assert field_label("road002_1_00", "area02") == "Route 2 (1)"
    assert field_label("enc_grass01", "buildmodel") == "enc_grass01"
    assert field_label("t001r01_00", "t001r01") == "t001r01_00 (t001r01)"


def test_catalog_maps_props_and_duplicate_stems(tmp_path):
    root = tmp_path / "fieldmaps" / "field"
    for folder, stem in (("area02", "town001_00"), ("area02", "sea_00"), ("area03", "sea_00"), ("buildmodel", "enc_grass01")):
        os.makedirs(root / folder, exist_ok=True)
        (root / folder / f"{stem}.gfbmdl").write_bytes(build_field_gfbmdl())
    cat = LgpeCatalog(data_dir=str(tmp_path), trainer_dir=str(tmp_path / "none"), field_dir=str(tmp_path / "fieldmaps"), misc_dir=str(tmp_path / "none"))
    assert sorted(r.id for r in cat.maps()) == ["area02_sea_00", "area03_sea_00", "town001_00"]
    assert [r.id for r in cat.props()] == ["enc_grass01"]
    ref = cat.find("town001_00")
    assert ref.display_name == "Pallet Town"
    m = cat.import_ref(ref)
    assert m.kind == "map" and m.skeleton is None and m.meshes


def test_base_slot_uv_set_from_render_states():
    from pipeline.importers.lgpe.gfbmdl import GfMaterial

    rock = GfMaterial("r", "FieldRockShader", {"Rock_tex": "rock01", "GroundTex01": "grass01"}, render={"Tex00_UV": 0, "Tex01_UV": 1})
    assert F.base_slot(rock) == ("Rock_tex", 1)
    grass = GfMaterial("g", "FieldGrassShader02", {"TextureMap01": "grass01"}, render={"UVSet0": 0, "UVSet01": 1})
    assert F.base_slot(grass) == ("TextureMap01", 0)
    cliff = GfMaterial("c", "FieldCliffShader01", {"CliffTex01": "cliff01"})  # no field 20: the default
    assert F.base_slot(cliff) == ("CliffTex01", 1)


def test_window_glass_screens_texture_over_color01():
    """FieldWindowShader: colour = screen(Color01, texture) = Color01 + (1 - Color01) * texture. House windows (pale blue +
    grey mask) used to export the near-black mask alone; Center/Gym door glass (near-black Color01 + coloured logo) keeps
    its logo; a bright mask (Mart window body) stays blue instead of saturating to white (WP 2.5d)."""
    from pipeline.importers.lgpe.gfbmdl import GfMaterial

    gm = GfMaterial(name="window01_com", shader="FieldWindowShader", textures={"TextureMap01": "window01_com"},
                    colors={"Color01": (0.6, 0.8, 1.0), "Color02": (0.4, 0.8, 1.0)})
    tx = np.zeros((2, 2, 4), np.uint8)
    tx[..., 3] = 255
    tx[1, 1, :3] = (51, 51, 51)  # a lighter mask pixel
    tex = F._Tex({"window01_com": tx}.get)
    mat, _ = F.convert_field_material(gm, tex)
    img = tex.textures[mat.base_color].rgba
    assert mat.extras["glass"] and mat.alpha_mode == "OPAQUE"
    assert tuple(img[0, 0]) == (153, 204, 255, 255)  # mask 0 -> Color01
    assert tuple(img[1, 1]) == (173, 214, 255, 255)  # screen with 0.2: 0.6 + 0.4 * 0.2, 0.8 + 0.2 * 0.2, 1
    gm.colors["Color01"] = (0.0, 0.0, 0.0)
    tx[0, 0, :3] = (10, 120, 200)  # door glass logo colour survives
    tex = F._Tex({"window01_com": tx}.get)
    mat, _ = F.convert_field_material(gm, tex)
    assert tuple(tex.textures[mat.base_color].rgba[0, 0]) == (10, 120, 200, 255)


def _solid(rgba):
    img = np.zeros((2, 2, 4), np.uint8)
    img[...] = rgba
    return img


def test_field_blend_recipes_by_slot_layout():
    """WP 2.5d: the layer recipes are chosen by the material's shader-input slots, with the UV sets its render states name
    (cliff: ground cap over the rock strip, BorderTex on GroundBlend 2; ground: grass over soil, BlendTex on UV_tex02;
    lawn: TextureMap02 + green_blend, green_hikari cut-out on UVSet01 cut at DiscardValuie; roofs: Texture02 by vertex
    alpha), and the glTF writer turns every layer into a texture reference with its UV set."""
    import json
    import struct
    import tempfile

    from pipeline.common.model import Mesh, Model, Primitive
    from pipeline.importers.lgpe.gfbmdl import GfMaterial

    imgs = {n: _solid((100, 150, 50, 255)) for n in ("cliff01", "grass01", "grass02", "soil01", "soil08", "noise", "mask",
                                                      "tm2", "hik", "roof1", "roof2", "ll")}
    tex = F._Tex(imgs.get)
    cliff = GfMaterial("c", "FieldCliffShader01", {"CliffTex01": "cliff01", "GroundTex01": "grass01", "GroundTex02": "grass02",
                                                   "BlendTex": "noise", "BorderTex": "mask"},
                       render={"Tex00_UV": 0, "Tex01_UV": 1, "GroundBlend": 2})
    mat, uv = F.convert_field_material(cliff, tex)
    fb = mat.extras["fieldBlend"]
    assert uv == 1 and [o["texture"] for o in fb["over"]] == ["grass01", "grass02"] and fb["over"][0]["uv"] == 0
    assert fb["mask"] == {"texture": "mask", "uv": 2, "wrap": ["repeat", "repeat"]} and fb["noise"]["texture"] == "noise"
    assert "base2" not in fb

    ground = GfMaterial("g", "FieldGroundShader01", {"GroundTex01": "soil01", "GroundTex02": "soil08", "GrassTex01": "grass01",
                                                     "GrassTex02": "grass02", "GrassBlendTex": "noise", "BlendTex": "mask"},
                        render={"UV_tex0": 0, "UV_tex02": 2})
    mat, _ = F.convert_field_material(ground, tex)
    fb = mat.extras["fieldBlend"]
    assert fb["base2"]["texture"] == "soil08" and fb["base2By"] == "noise"
    assert [o["texture"] for o in fb["over"]] == ["grass01", "grass02"] and fb["mask"]["uv"] == 2

    lawn = GfMaterial("l", "FieldGrassShader02", {"TextureMap01": "grass01", "TextureMap02": "tm2", "green_blend": "noise",
                                                  "green_hikari": "hik"},
                      floats={"DiscardValuie": 0.85}, render={"UVSet0": 0, "UVSet01": 1, "DiscardEnable": F.ON, "CullMode": 2})
    mat, _ = F.convert_field_material(lawn, tex)
    fb = mat.extras["fieldBlend"]
    assert fb["cutout"]["texture"] == "hik" and fb["cutout"]["uv"] == 1 and fb["cutoff"] == 0.85
    assert mat.alpha_mode == "MASK" and abs(mat.alpha_cutoff - 0.85) < 1e-6 and fb["base2"]["texture"] == "tm2"
    lawn.render["DiscardEnable"] = 0  # no discard: the outline layer is not a cut-out
    mat, _ = F.convert_field_material(lawn, tex)
    assert "cutout" not in mat.extras["fieldBlend"] and mat.alpha_mode == "OPAQUE" and "opacity" not in mat.extras["fieldBlend"]
    lawn.render["BlendMode"] = 5  # blended decal (rockmask01): the outline is its opacity
    mat, _ = F.convert_field_material(lawn, tex)
    assert mat.extras["fieldBlend"]["opacity"]["uv"] == 1 and mat.alpha_mode == "BLEND"
    tuft = GfMaterial("t", "FieldGrassShader04", {"Texture01": "grass01", "Texture02": "tm2", "Texture03": "ll"},
                      render={"Tex01_UV": 0, "Tex02_UV": 1})
    mat, _ = F.convert_field_material(tuft, tex)
    assert mat.extras["fieldBlend"]["lightLine"]["texture"] == "ll"

    roof = GfMaterial("r", "FieldDefaultShader", {"Texture01": "roof1", "Texture02": "roof2"},
                      render={"TexBlendType": 1, "VertexColor": F.ON, "Tex02_UV": 0})
    mat, _ = F.convert_field_material(roof, tex)
    assert mat.extras["fieldBlend"] == {"base2": {"texture": "roof2", "uv": 0, "wrap": ["repeat", "repeat"]}, "base2By": "vertexAlpha"}
    plain = GfMaterial("o", "FieldObjectShader", {"Texture01": "roof1", "Texture02": "roof2"}, render={"Tex02_UV": 1})
    mat, _ = F.convert_field_material(plain, tex)
    assert "fieldBlend" not in mat.extras  # an object's second texture on UV1 is not a vertex-alpha blend
    pine = GfMaterial("p", "FieldTreeShader02", {"Texture01": "roof1", "Texture02": "roof2"},
                      colors={"DirlightColor": (0.088, 0.366, 0.09)}, render={"UV_tex01": 0})
    mat, _ = F.convert_field_material(pine, tex)
    fb = mat.extras["fieldBlend"]
    assert fb["base2By"] == "lightAdd" and fb["base2"]["texture"] == "roof2" and abs(fb["lightColor"][1] - 0.366) < 1e-6

    # writer: layer references become glTF texture references (index + texCoord + name), images stored
    mat, _ = F.convert_field_material(ground, tex)
    prim = Primitive(material=mat.name, indices=np.array([0, 1, 2], np.uint32), positions=np.eye(3, dtype=np.float32),
                     normals=None, uvs=[np.zeros((3, 2), np.float32)] * 3, colors=None)
    model = Model(id="m", game="lgpe", display_name="m", kind="map", skeleton=None, meshes=[Mesh(name="m", primitives=[prim])],
                  materials={mat.name: mat}, textures=tex.textures)
    path = os.path.join(tempfile.mkdtemp(), "m.glb")
    write_glb(model, path)
    with open(path, "rb") as f:
        data = f.read()
    gj = json.loads(data[20:20 + struct.unpack_from("<I", data, 12)[0]])
    fb = gj["materials"][0]["extras"]["fieldBlend"]
    names = [gj["images"][gj["textures"][o["index"]]["source"]]["name"] for o in fb["over"]]
    assert names == ["grass01", "grass02"] and fb["mask"]["texCoord"] == 2 and fb["mask"]["name"] == "mask"


def test_ramp_tree_and_water_layers():
    """Ramp trees export leaf/ramp/silhouette with the ramp on UVSet01 (UV1); sea water exports its nets, light blobs and
    colours; shore foam its two scrolling layers."""
    from pipeline.importers.lgpe.gfbmdl import GfMaterial

    ramp = np.zeros((8, 2, 4), np.uint8)
    ramp[..., 1] = 200
    ramp[..., 3] = 255
    imgs = {"t1": _solid((0, 5, 5, 255)), "t2": ramp, "t3": _solid((240, 240, 240, 255)), "net1": _solid((20, 20, 20, 255)),
            "net2": _solid((20, 20, 20, 255)), "light": _solid((60, 60, 60, 255)), "sky": _solid((90, 247, 255, 255)),
            "foam": _solid((233, 233, 233, 50))}
    tex = F._Tex(imgs.get)
    tree = GfMaterial("t", "FieldTreeShader05", {"Texture01": "t1", "Texture02": "t2", "Texture03": "t3"},
                      render={"UV_tex01": 0, "UVSet01": 1, "DiscardEnable": F.ON})
    mat, _ = F.convert_field_material(tree, tex)
    fr = mat.extras["fieldRamp"]
    assert mat.extras["rampTree"] and fr["ramp"]["uv"] == 1 and fr["leaf"]["texture"] == "t3" and fr["silhouette"]["uv"] == 0

    sea = GfMaterial("sea", "FieldWaterShader01", {"WaterNetTex01": "net1", "WaterNetTex02": "net2", "WaterLightTex01": "light",
                                                   "WaterLightTex02": "light", "sky_tex": "sky"},
                     colors={"Base_Color1": (0.0, 0.19, 0.9), "Base_Color2": (0.0, 0.29, 1.0), "WaterNetTex01_Color": (0.87, 0.87, 0.87)},
                     render={"BlendMode": 5, "UV": 0})
    mat, _ = F.convert_field_material(sea, tex)
    fw = mat.extras["fieldWater"]
    assert [n["texture"] for n in fw["net"]] == ["net1", "net2"] and fw["netColor"][0] == [0.87, 0.87, 0.87]
    assert fw["colors"][0] == [0.0, 0.19, 0.9] and fw["sky"]["texture"] == "sky" and mat.alpha_mode == "BLEND"

    wave = GfMaterial("wave", "FieldWaterShader02", {"Texture01": "foam", "Texture02": "foam"},
                      floats={"Tex01_Translate_U": 0.25, "Tex02_Translate_V": -0.1}, render={"BlendMode": 5})
    mat, _ = F.convert_field_material(wave, tex)
    fw = mat.extras["fieldWater"]
    assert len(fw["foam"]) == 2 and fw["foamOffset"] == [[0.25, 0.0], [0.0, -0.1]]
