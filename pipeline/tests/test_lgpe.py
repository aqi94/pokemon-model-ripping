"""Let's Go Pikachu/Eevee: gfbmdl / gfbanm / DDS reading, material baking and catalog discovery on tiny
synthetic files (no real data needed)."""

import struct
import zipfile

import flatbuffers
import numpy as np

from pipeline.catalog.lgpe import LgpeCatalog, form_label
from pipeline.importers.lgpe import importer as L
from pipeline.importers.lgpe.gfbmdl import parse_gfbmdl

# ------------------------------------------------------------------ synthetic FlatBuffers helpers


def _str(b, s):
    return b.CreateString(s)


def _vec3(b, v):
    b.Prep(4, 12)
    for x in reversed(v):
        b.PrependFloat32(x)
    return b.Offset()


def _table(b, n, fill):
    b.StartObject(n)
    fill()
    return b.EndObject()


def _vec_of(b, offsets):
    b.StartVector(4, len(offsets), 4)
    for o in reversed(offsets):
        b.PrependUOffsetTRelative(o)
    return b.EndVector()


def _named(b, name, slot_fill=None):
    n = _str(b, name)

    def fill():
        b.PrependUOffsetTRelativeSlot(0, n, 0)
        if slot_fill:
            slot_fill()

    return _table(b, 2, fill)


def build_gfbmdl(uv_scale_u=2.0, col_name="body_col") -> bytes:
    b = flatbuffers.Builder(1024)
    # textures / material
    tex_names = [col_name, "body_sdw"]
    tex_off = _vec_of(b, [_str(b, t) for t in tex_names])
    col_slot = _named(b, "Col0Tex", lambda: b.PrependUint32Slot(1, 0, 0))
    sdw_slot = _named(b, "ColSdw0Tex", lambda: b.PrependUint32Slot(1, 1, 0))
    sw = _named(b, "Layer1Enable", lambda: b.PrependUint8Slot(1, 1, 0))
    fl = _named(b, "ColorUVScaleU", lambda: b.PrependFloat32Slot(1, uv_scale_u, 0))
    cname = _str(b, "ConstantColor0")
    b.StartObject(2)
    b.PrependUOffsetTRelativeSlot(0, cname, 0)
    b.PrependStructSlot(1, _vec3(b, (1.0, 1.0, 1.0)), 0)
    col = b.EndObject()
    texs, sws, fls, cols = _vec_of(b, [col_slot, sdw_slot]), _vec_of(b, [sw]), _vec_of(b, [fl]), _vec_of(b, [col])
    mname, shader = _str(b, "Body"), _str(b, "PokeDefaultShader")

    def mat_fill():
        b.PrependUOffsetTRelativeSlot(0, mname, 0)
        b.PrependUOffsetTRelativeSlot(1, shader, 0)
        b.PrependUOffsetTRelativeSlot(11, texs, 0)
        b.PrependUOffsetTRelativeSlot(12, sws, 0)
        b.PrependUOffsetTRelativeSlot(13, fls, 0)
        b.PrependUOffsetTRelativeSlot(14, cols, 0)

    mat = _table(b, 15, mat_fill)
    mats = _vec_of(b, [mat])

    # mesh: 4 vertices, one triangle fan of 2 triangles. layout: pos f32x3 | normal f16x4 | uv f32x2 | joints u8x4 | weights u8x4
    verts = np.zeros(4, dtype=[("p", "<f4", 3), ("n", "<f2", 4), ("uv", "<f4", 2), ("j", "u1", 4), ("w", "u1", 4)])
    verts["p"] = [(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0)]  # centimetres
    verts["n"] = [(0, 0, 1, 1)] * 4
    verts["uv"] = [(0.0, 3.0), (0.5, 3.0), (0.5, 3.5), (0.0, 3.5)]  # V tile 3, as Pikachu BodyB
    verts["j"] = [(1, 2, 2, 0), (1, 0, 0, 0), (2, 0, 0, 0), (1, 2, 0, 0)]
    verts["w"] = [(128, 127, 0, 0), (255, 0, 0, 0), (200, 0, 0, 0), (100, 100, 0, 0)]  # row 2: sum != 255
    vb = b.CreateByteVector(verts.tobytes())
    idx = np.array([0, 1, 2, 0, 2, 3], np.uint16)
    iv = b.CreateNumpyVector(idx)
    poly = _table(b, 2, lambda: (b.PrependUOffsetTRelativeSlot(1, iv, 0), b.PrependUint32Slot(0, 0, 0)))
    polys = _vec_of(b, [poly])
    attrs = []
    for usage, typ, cnt in [(0, 0, 3), (1, 1, 4), (3, 0, 2), (11, 3, 4), (12, 8, 4)]:
        attrs.append(
            _table(
                b,
                3,
                lambda u=usage, t=typ, c=cnt: (b.PrependUint32Slot(2, c, 0), b.PrependUint32Slot(1, t, 0), b.PrependUint32Slot(0, u, 0)),
            )
        )
    attr_vec = _vec_of(b, attrs)
    mesh = _table(b, 3, lambda: (b.PrependUOffsetTRelativeSlot(2, vb, 0), b.PrependUOffsetTRelativeSlot(1, attr_vec, 0), b.PrependUOffsetTRelativeSlot(0, polys, 0)))
    meshes = _vec_of(b, [mesh])

    # bones: root, Origin, Waist (90 deg about Z, 10 cm up), Head
    def bone(name, parent, euler=(0, 0, 0), trans=(0, 0, 0)):
        n = _str(b, name)

        def fill():
            b.PrependStructSlot(7, _vec3(b, trans), 0)
            b.PrependStructSlot(6, _vec3(b, euler), 0)
            b.PrependStructSlot(5, _vec3(b, (1, 1, 1)), 0)
            b.PrependInt32Slot(2, parent, 99)
            b.PrependUOffsetTRelativeSlot(0, n, 0)

        return _table(b, 10, fill)

    bones = _vec_of(
        b,
        [bone("root", -1), bone("Origin", 0), bone("Waist", 1, (0, 0, np.pi / 2), (0, 10, 0)), bone("Head", 2, trans=(5, 0, 0))],
    )

    def root_fill():
        b.PrependUOffsetTRelativeSlot(9, bones, 0)
        b.PrependUOffsetTRelativeSlot(8, meshes, 0)
        b.PrependUOffsetTRelativeSlot(6, mats, 0)
        b.PrependUOffsetTRelativeSlot(2, tex_off, 0)
        b.PrependUint32Slot(0, 0x18020511, 0)

    b.Finish(_table(b, 10, root_fill))
    return bytes(b.Output())


def build_gfbanm(frames=3, fps=30, looped=True, translation_cm=(0.0, 10.0, 0.0), scale_hidden=False) -> bytes:
    b = flatbuffers.Builder(512)

    def u16_struct(vals):
        b.Prep(2, 6)
        for v in reversed(vals):
            b.PrependUint16(v)
        return b.Offset()

    def track(name, scale=None, rot=None, trans=None):
        n = _str(b, name)
        parts = {}
        if scale is not None:
            parts[(1, 2)] = ("fixed_vec", scale)
        if rot is not None:
            parts[(3, 4)] = ("fixed_quat", rot)
        if trans is not None:
            parts[(5, 6)] = ("dyn_vec", trans)
        tables = {}
        for key, (kind, val) in parts.items():
            if kind == "fixed_vec":
                tables[key] = (1, _table(b, 1, lambda v=val: b.PrependStructSlot(0, _vec3(b, v), 0)))
            elif kind == "fixed_quat":
                tables[key] = (1, _table(b, 1, lambda v=val: b.PrependStructSlot(0, u16_struct(v), 0)))
            else:
                b.StartVector(12, len(val), 4)
                for v in reversed(val):
                    for x in reversed(v):
                        b.PrependFloat32(x)
                vec = b.EndVector()
                tables[key] = (2, _table(b, 1, lambda vec=vec: b.PrependUOffsetTRelativeSlot(0, vec, 0)))

        def fill():
            for (ti, vi), (kind, tab) in tables.items():
                b.PrependUOffsetTRelativeSlot(vi, tab, 0)
                b.PrependUint8Slot(ti, kind, 0)
            b.PrependUOffsetTRelativeSlot(0, n, 0)

        return _table(b, 7, fill)

    s = (0.1, 1.0, 1.0) if scale_hidden else (1.0, 1.0, 1.0)
    tracks = _vec_of(
        b,
        [
            track("Waist", scale=s, rot=(3, 2, 0x8001), trans=[translation_cm] * frames),  # identity quaternion
            track("NotInSkeleton", scale=(1, 1, 1)),
        ],
    )
    skel = _table(b, 1, lambda: b.PrependUOffsetTRelativeSlot(0, tracks, 0))
    info = _table(b, 3, lambda: (b.PrependUint32Slot(2, fps, 0), b.PrependUint32Slot(1, frames, 0), b.PrependUint32Slot(0, int(looped), 0)))
    b.Finish(_table(b, 4, lambda: (b.PrependUOffsetTRelativeSlot(1, skel, 0), b.PrependUOffsetTRelativeSlot(0, info, 0))))
    return bytes(b.Output())


def dds_rgba(w, h, pixels: np.ndarray) -> bytes:
    """Uncompressed 32-bit BGRA DDS."""
    hdr = bytearray(128)
    hdr[:4] = b"DDS "
    struct.pack_into("<7I", hdr, 4, 124, 0x1007, h, w, w * 4, 0, 0)
    struct.pack_into("<2I4sI4I", hdr, 76, 32, 0x41, b"\0\0\0\0", 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    bgra = pixels[..., [2, 1, 0, 3]].astype(np.uint8)
    return bytes(hdr) + bgra.tobytes()


# ------------------------------------------------------------------ tests


def test_parse_gfbmdl_fields():
    g = parse_gfbmdl(build_gfbmdl())
    assert g.textures == ["body_col", "body_sdw"]
    assert [b.name for b in g.bones] == ["root", "Origin", "Waist", "Head"]
    assert [b.parent for b in g.bones] == [-1, 0, 1, 2]
    assert abs(g.bones[2].euler[2] - np.pi / 2) < 1e-6 and g.bones[2].translation == (0.0, 10.0, 0.0)
    m = g.materials[0]
    assert m.shader == "PokeDefaultShader" and m.textures == {"Col0Tex": "body_col", "ColSdw0Tex": "body_sdw"}
    assert m.switches == {"Layer1Enable": True} and m.floats == {"ColorUVScaleU": 2.0} and m.colors["ConstantColor0"] == (1.0, 1.0, 1.0)
    mesh = g.meshes[0]
    assert mesh.positions.shape == (4, 3) and mesh.positions[1].tolist() == [100.0, 0.0, 0.0]
    assert mesh.normals[0].tolist() == [0.0, 0.0, 1.0]
    assert mesh.joints[0].tolist() == [1, 2, 2, 0] and abs(mesh.weights[0][0] - 128 / 255) < 1e-6
    assert mesh.polygons[0].material == 0 and mesh.polygons[0].indices.tolist() == [0, 1, 2, 0, 2, 3]


def _files(**extra):
    col = np.zeros((4, 8, 4), np.uint8)
    col[..., :] = (200, 100, 50, 255)
    sdw = col.copy()
    sdw[..., :3] = (100, 50, 25)
    files = {"m.gfbmdl": build_gfbmdl(), "body_col.dds": dds_rgba(8, 4, col), "body_sdw.dds": dds_rgba(8, 4, sdw)}
    files.update(extra)
    return files


def _import(files, **kw):
    return L.import_model(lambda n: files[n], lambda n: n in files, "m", "Test", model_file="m.gfbmdl", **kw)


def test_import_geometry_skeleton_and_uv_bake():
    model = _import(_files())
    sk = model.skeleton
    assert [j.name for j in sk.joints] == ["root", "Origin", "Waist", "Head"] and sk.palette == [0, 1, 2, 3]
    # centimetres -> metres; Waist is 10 cm above Origin, Head 5 cm along Waist's local X (= world up after 90 deg Z)
    assert np.allclose(sk.joints[2].translation, (0, 0.1, 0))
    world_head = np.linalg.inv(sk.joints[3].inverse_bind)
    assert np.allclose(world_head[:3, 3], (0, 0.15, 0), atol=1e-6)
    prim = model.meshes[0].primitives[0]
    assert np.allclose(prim.positions[1], (1.0, 0.0, 0.0))
    assert prim.indices.tolist() == [0, 1, 2, 0, 2, 3]
    # zero-weight joint slots are cleared, weights are renormalised per vertex
    assert prim.joints[0].tolist() == [1, 2, 0, 0] and np.allclose(prim.weights.sum(1), 1.0)
    assert np.allclose(prim.weights[2], (1.0, 0, 0, 0))
    # ColorUVScaleU = 2 -> mirrored texture doubled to 16 px wide, UV u unchanged (scale/2); V tile 3 dropped and flipped
    mat = model.materials["Body"]
    assert model.textures[mat.base_color].rgba.shape == (4, 16, 4)
    uv = prim.uvs[0]
    assert np.allclose(uv[:, 0], [0.0, 0.5, 0.5, 0.0]) and np.allclose(uv[:, 1], [1.0, 1.0, 0.5, 0.5])
    # right half of the baked texture is the mirror image
    tex = model.textures[mat.base_color].rgba
    assert (tex[:, :8] == tex[:, 8:][:, ::-1]).all()
    # the shadow texture is baked the same way for the viewer's Game shading
    assert model.textures[mat.extras["shadow_color"]].rgba.shape == (4, 16, 4)
    assert model.textures[mat.extras["shadow_color"]].rgba[0, 0, 0] == 100


def test_uv_scale_one_keeps_texture_size():
    files = _files(**{"m.gfbmdl": build_gfbmdl(uv_scale_u=1.0)})
    model = _import(files)
    assert model.textures[model.materials["Body"].base_color].rgba.shape == (4, 8, 4)


def test_layer_composite_shows_iris_through_holes():
    col = np.zeros((4, 4, 4), np.uint8)
    col[..., :] = (255, 0, 0, 255)
    col[1:3, 1:3, 3] = 0  # transparent eye opening
    iris = np.zeros((2, 2, 4), np.uint8)
    iris[..., :] = (0, 0, 255, 255)
    g = parse_gfbmdl(build_gfbmdl())
    out = L.composite_layer(col, iris, g.materials[0], L.UvBake(), (False, False))
    assert out[1, 1].tolist() == [0, 0, 255, 255] and out[0, 0].tolist() == [255, 0, 0, 255]


def test_dds_decoders():
    px = np.random.default_rng(1).integers(0, 255, (4, 8, 4), dtype=np.uint8)
    assert (L.decode_dds(dds_rgba(8, 4, px)) == px).all()
    # BC1 solid block: both end colours white, indices 0
    hdr = bytearray(128)
    hdr[:4] = b"DDS "
    struct.pack_into("<7I", hdr, 4, 124, 0x1007, 4, 4, 8, 0, 0)
    struct.pack_into("<2I4s", hdr, 76, 32, 4, b"DXT1")
    block = struct.pack("<HHI", 0xFFFF, 0xFFFF, 0)
    out = L.decode_dds(bytes(hdr) + block)
    assert out.shape == (4, 4, 4) and (out[..., :3] == 255).all()


def test_animation_conversion_and_rest_scale():
    anim = L.LgpeAnimation("ba10_waitA01", build_gfbanm(frames=3, fps=60, looped=True, scale_hidden=True))
    model = _import(_files(), animations=[anim, L.LgpeAnimation("eye01", b"ignored")])
    assert [c.name for c in model.clips] == ["ba10_waitA01"] and model.extras["materialOnlyClips"] == ["eye01"]
    clip = model.clips[0]
    assert clip.frame_rate == 60 and clip.frame_count == 3 and clip.loop and clip.category == "idle"
    assert {t.joint for t in clip.tracks} == {"Waist"}  # NotInSkeleton dropped
    tr = next(t for t in clip.tracks if t.kind == "translation")
    assert np.allclose(tr.values[0], (0, 0.1, 0))  # cm -> m
    rot = next(t for t in clip.tracks if t.kind == "rotation")
    assert np.allclose(rot.values[0], (0, 0, 0, 1), atol=1e-3)
    assert model.default_clip == "ba10_waitA01"
    # constant non-unit scale of the default clip = the game's hidden state: becomes the joint's rest scale
    assert np.allclose(model.skeleton.joints[2].scale, (0.1, 1.0, 1.0))


def test_shiny_variant_from_rare_model():
    rare = np.zeros((4, 8, 4), np.uint8)
    rare[..., :] = (10, 200, 10, 255)
    files = _files(**{"m_rare.gfbmdl": build_gfbmdl(col_name="body_col_rare"), "body_col_rare.dds": dds_rgba(8, 4, rare)})
    model = _import(files)
    assert [(v.name, v.label, v.default) for v in model.variants] == [("normal", "Normal", True), ("rare", "Shiny", False)]
    over = model.variants[1].materials["Body"]
    assert over.base_color != model.materials["Body"].base_color
    assert model.textures[over.base_color].rgba[0, 0, :3].tolist() == [10, 200, 10]
    # a rare model whose textures are identical adds no variant
    same = _files(**{"m_rare.gfbmdl": build_gfbmdl()})
    assert _import(same).variants == []


def test_clip_names_and_categories():
    assert L.clip_name("bin/pokemon/pm0025_00/anm/pm0025_00_ba10_waitA01.gfbanm", "pm0025_00") == "ba10_waitA01"
    assert L.clip_name("x/pm0025_00.gfbanm", "pm0025_00") == "base"
    assert L.clip_name("bin/pokemon/pm0059_00/anm/ri0059_00_ri0059_walk01_loop.gfbanm", "pm0059_00") == "ri0059_walk01_loop"
    assert L.lgpe_category("fi20_walk01") == "move" and L.lgpe_category("kw32_happyA01") == "emote" and L.lgpe_category("ba41_down01") == "battle"


# ------------------------------------------------------------------ catalog


def _zip(path, names):
    with zipfile.ZipFile(path, "w") as z:
        for n in names:
            z.writestr(n, b"x")


def test_catalog_discovery(tmp_path):
    base = "bin/pokemon/"
    _zip(
        tmp_path / "LGPE-Poke.zip",
        [
            base + "pm0025_00/mdl/pm0025_00.gfbmdl",
            base + "pm0025_00/mdl/pm0025_00_rare.gfbmdl",  # not a model of its own
            base + "pm0025_01/mdl/pm0025_01.gfbmdl",
            base + "pm0006_00/mdl/pm0006_00.gfbmdl",
            base + "pm0006_51/mdl/pm0006_51.gfbmdl",
            base + "pm0006_52/mdl/pm0006_52.gfbmdl",
            base + "pm0019_61/mdl/pm0019_61.gfbmdl",
            base + "pm0891_00/mdl/pm0891_00.gfbmdl",
            base + "pm0000_00/mdl/pm0000_00.gfbmdl",
            "bin/chara/data/bd/bd0001_00_pikachu/mdl/bd0001_00.gfbmdl",
            "bin/chara/data/bd/bd0001_00_pikachu_col/mdl/bd0001_00_col.gfbmdl",  # collision helper: ignored
            "bin/chara/data/pd/pd0001_00_cap/mdl/pd0001_00.gfbmdl",
        ],
    )
    _zip(
        tmp_path / "LGPE-PokeAnim.zip",
        [
            "bin/pokemon/pm0025_00/anm/pm0025_00.gfbanm",
            "bin/pokemon/pm0025_00/anm/pm0025_00_ba10_waitA01.gfbanm",
            "bin/pokemon/pm0025_00/anm/pm0025_00_app01.gfbanmcfg",  # animation state config, not a clip
            "bin/pokemon/pm0025_01/anm/pm0025_01_fi20_walk01.gfbanm",
        ],
    )
    cat = LgpeCatalog(str(tmp_path), trainer_dir=str(tmp_path / "none"))
    pk = {r.id: r for r in cat.pokemon()}
    assert sorted(pk) == ["bd0001_00", "pm0000_00", "pm0006_00", "pm0006_51", "pm0006_52", "pm0019_61", "pm0025_00", "pm0025_01", "pm0891_00"]
    assert pk["pm0025_00"].display_name == "Pikachu" and pk["pm0025_01"].display_name == "Pikachu (Female)"
    assert pk["pm0006_51"].display_name == "Mega X Charizard" and pk["pm0006_52"].display_name == "Mega Y Charizard"
    assert pk["pm0019_61"].display_name == "Alolan Rattata"
    assert pk["pm0891_00"].display_name == "Meltan" and pk["pm0000_00"].display_name == "Egg"
    assert pk["bd0001_00"].display_name == "Partner Pikachu" and pk["bd0001_00"].extras["dex"] == 25
    assert pk["pm0025_00"].extras["dex"] == 25
    assert [r.id for r in cat.objects()] == ["pd0001_00"]
    anims = cat.animations(pk["pm0025_00"])
    assert sorted(a.name for a in anims) == ["ba10_waitA01", "base"]
    assert cat.find("pm0025_01") is pk["pm0025_01"] or cat.find("pm0025_01").id == "pm0025_01"
    assert form_label("51", {"51"}) == "Mega"


def test_catalog_trainers(tmp_path):
    root = tmp_path / "chara" / "data" / "tr"
    for d in ("tr0001_00_hero", "tr0026_02_bugcatcher", "tr0099_00_nomodel"):
        (root / d).mkdir(parents=True)
    (root / "tr0001_00_hero" / "tr0001_00_hero.gfbmdl").write_bytes(b"x")
    (root / "tr0026_02_bugcatcher" / "tr0026_02_bugcatcher.gfbmdl").write_bytes(b"x")
    cat = LgpeCatalog(str(tmp_path), trainer_dir=str(tmp_path))
    tr = {r.id: r for r in cat.trainers()}
    assert sorted(tr) == ["tr0001_00", "tr0026_02"]
    assert tr["tr0001_00"].display_name == "Player (boy)" and tr["tr0026_02"].display_name == "Bugcatcher (02)"
    assert tr["tr0001_00"].kind == "trainer"
