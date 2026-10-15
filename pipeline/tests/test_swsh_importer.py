"""Sword/Shield importer against tiny synthetic .gfbmdl / .gfbanm buffers and a synthetic zip set (no real data)."""

import math
import zipfile

import numpy as np
import pytest

flatbuffers = pytest.importorskip("flatbuffers")

from pipeline.catalog.swsh import SwshCatalog  # noqa: E402
from pipeline.common.gltf_writer import write_glb  # noqa: E402
from pipeline.importers.swsh import gfbmdl as G  # noqa: E402
from pipeline.importers.swsh.gfbanm import parse_gfbanm  # noqa: E402
from pipeline.importers.swsh.importer import clip_from_gfbanm, import_gfbmdl  # noqa: E402


# ----------------------------------------------------------------------------- builders
def _vec_tables(b, offsets):
    b.StartVector(4, len(offsets), 4)
    for o in reversed(offsets):
        b.PrependUOffsetTRelative(o)
    return b.EndVector()


def _vec_strings(b, strings):
    return _vec_tables(b, [b.CreateString(s) for s in strings])


def _vec_bytes(b, data: bytes):
    return b.CreateByteVector(data)


def _vec_u16(b, vals):
    b.StartVector(2, len(vals), 2)
    for v in reversed(vals):
        b.PrependUint16(v)
    return b.EndVector()


def _vec3_struct(b, xyz):
    b.Prep(4, 12)
    for v in reversed(xyz):
        b.PrependFloat32(v)
    return b.Offset()


def _bone(b, name, parent, rigged, t, r=(0.0, 0.0, 0.0), typ=1):
    n = b.CreateString(name)
    b.StartObject(11)
    b.PrependUOffsetTRelativeSlot(0, n, 0)
    b.PrependUint32Slot(1, typ, 0)
    b.PrependInt32Slot(2, parent, 0)
    b.PrependStructSlot(5, _vec3_struct(b, (1.0, 1.0, 1.0)), 0)
    b.PrependStructSlot(6, _vec3_struct(b, r), 0)
    b.PrependStructSlot(7, _vec3_struct(b, t), 0)
    b.PrependBoolSlot(10, rigged, True)
    return b.EndObject()


def _attr(b, vtype, fmt, count):
    b.StartObject(3)
    b.PrependUint8Slot(0, vtype, 0)
    b.PrependUint8Slot(1, fmt, 0)
    b.PrependUint32Slot(2, count, 0)
    return b.EndObject()


def _tex_slot(b, sampler, index, wrap_u):
    name = b.CreateString(sampler)
    b.StartObject(6)
    b.PrependUint32Slot(1, wrap_u, 0)  # RepeatU
    st = b.EndObject()
    b.StartObject(3)
    b.PrependUOffsetTRelativeSlot(0, name, 0)
    b.PrependUint32Slot(1, index, 0)
    b.PrependUOffsetTRelativeSlot(2, st, 0)
    return b.EndObject()


def make_gfbmdl(tex_name="pm9999_00_Body_col", su=2.0) -> bytes:
    """One quad mesh (centimetres) skinned to 2 joints, 1 material with a mirror-U albedo of scale su."""
    b = flatbuffers.Builder(2048)
    bones = [
        _bone(b, "root", -1, False, (0, 0, 0), typ=0),
        _bone(b, "Waist", 0, True, (0, 10, 0)),
        _bone(b, "Head", 1, True, (0, 10, 0), r=(0.0, 0.0, math.pi / 2)),
        _bone(b, "Eff01", 2, False, (0, 1, 0), typ=0),
    ]
    # attributes: pos f32x3, normal f32x3, uv f32x2, joints u8x4, weights fixed(u8)x4
    attrs = [_attr(b, G.V_POS, 0, 3), _attr(b, G.V_NRM, 0, 3), _attr(b, G.V_UV0, 0, 2), _attr(b, G.V_JOINTS, 3, 4), _attr(b, G.V_WEIGHTS, 8, 4)]
    verts = bytearray()
    quad = [((-5, 0, 0), (0.0, 0.0)), ((5, 0, 0), (1.0, 0.0)), ((5, 10, 0), (1.0, 1.0)), ((-5, 10, 0), (0.0, 1.0))]
    import struct

    for pos, uv in quad:
        verts += struct.pack("<3f3f2f", *pos, 0.0, 0.0, 1.0, uv[0], uv[1] + 2.0)  # V tile 2
        verts += bytes([0, 1, 0, 0, 255 if pos[1] == 0 else 128, 0 if pos[1] == 0 else 127, 0, 0])
    vb = _vec_bytes(b, bytes(verts))
    idx = _vec_u16(b, [0, 1, 2, 0, 2, 3])
    b.StartObject(2)
    b.PrependUint32Slot(0, 0, 0)
    b.PrependUOffsetTRelativeSlot(1, idx, 0)
    poly = b.EndObject()
    polys = _vec_tables(b, [poly])
    attr_v = _vec_tables(b, attrs)
    b.StartObject(3)
    b.PrependUOffsetTRelativeSlot(0, polys, 0)
    b.PrependUOffsetTRelativeSlot(1, attr_v, 0)
    b.PrependUOffsetTRelativeSlot(2, vb, 0)
    shape = b.EndObject()

    # material "Body": Col0Tex (mirror U), UV scale su x 1
    def fparam(name, v):
        n = b.CreateString(name)
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(0, n, 0)
        b.PrependFloat32Slot(1, v, 0.0)
        return b.EndObject()

    vals = _vec_tables(b, [fparam("ColorUVScaleU", su), fparam("ColorUVScaleV", 1.0)])
    texs = _vec_tables(b, [_tex_slot(b, "Col0Tex", 0, 2)])
    mname, sname = b.CreateString("Body"), b.CreateString("PokeDefaultShader")
    b.StartObject(21)
    b.PrependUOffsetTRelativeSlot(0, mname, 0)
    b.PrependUOffsetTRelativeSlot(1, sname, 0)
    b.PrependUint32Slot(7, 1, 0)  # cull back
    b.PrependUOffsetTRelativeSlot(11, texs, 0)
    b.PrependUOffsetTRelativeSlot(13, vals, 0)
    mat = b.EndObject()

    b.StartObject(3)
    b.PrependUint32Slot(0, 3, 0)  # bone id (Eff01)
    mesh = b.EndObject()

    tex_files = _vec_strings(b, [tex_name])
    bones_v = _vec_tables(b, bones)
    mesh_v = _vec_tables(b, [mesh])
    shape_v = _vec_tables(b, [shape])
    mat_v = _vec_tables(b, [mat])
    b.StartObject(12)
    b.PrependUOffsetTRelativeSlot(2, tex_files, 0)
    b.PrependUOffsetTRelativeSlot(6, mat_v, 0)
    b.PrependUOffsetTRelativeSlot(7, mesh_v, 0)
    b.PrependUOffsetTRelativeSlot(8, shape_v, 0)
    b.PrependUOffsetTRelativeSlot(9, bones_v, 0)
    b.Finish(b.EndObject())
    return bytes(b.Output())


def _pack_quat(q):
    """Inverse of the 48-bit unpack: q = (x, y, z, w) -> three u16."""
    q = np.asarray(q, np.float64)
    drop = int(np.argmax(np.abs(q)))
    sign = q[drop] < 0
    if sign:
        q = -q
    rest = [q[i] for i in range(4) if i != drop]
    vals = [int(round((v + math.pi / 4) / ((math.pi / 2) / 32767.0))) for v in rest]
    pack = drop | (4 if sign else 0) | (vals[0] << 3) | (vals[1] << 18) | (vals[2] << 33)
    return [pack & 0xFFFF, (pack >> 16) & 0xFFFF, (pack >> 32) & 0xFFFF]


def _track(b, bone, rot=None, trans=None):
    name = b.CreateString(bone)
    rot_off = tr_off = None
    if rot is not None:  # Dynamic rotation track (union type 2), one key per frame
        packed = [c for q in rot for c in _pack_quat(q)]
        b.StartVector(6, len(rot), 2)  # struct of 3 u16 = 6 bytes
        for c in reversed(packed):
            b.PrependUint16(c)
        co = b.EndVector()
        b.StartObject(1)
        b.PrependUOffsetTRelativeSlot(0, co, 0)
        rot_off = b.EndObject()
    if trans is not None:  # Framed16 translation track (type 3)
        frames = _vec_u16(b, [f for f, _ in trans])
        b.StartVector(12, len(trans), 4)
        for _, v in reversed(trans):
            for c in reversed(v):
                b.PrependFloat32(c)
        co = b.EndVector()
        b.StartObject(2)
        b.PrependUOffsetTRelativeSlot(0, frames, 0)
        b.PrependUOffsetTRelativeSlot(1, co, 0)
        tr_off = b.EndObject()
    b.StartObject(7)
    b.PrependUOffsetTRelativeSlot(0, name, 0)
    if rot_off is not None:
        b.PrependUint8Slot(3, 2, 0)
        b.PrependUOffsetTRelativeSlot(4, rot_off, 0)
    if tr_off is not None:
        b.PrependUint8Slot(5, 3, 0)
        b.PrependUOffsetTRelativeSlot(6, tr_off, 0)
    return b.EndObject()


def make_gfbanm(frames=4, fps=30, loop=1) -> bytes:
    b = flatbuffers.Builder(1024)
    ang = [i * 0.3 for i in range(frames)]
    rots = [(0.0, 0.0, math.sin(a / 2), math.cos(a / 2)) for a in ang]
    t_head = _track(b, "Head", rot=rots)
    t_waist = _track(b, "Waist", trans=[(0, (0.0, 10.0, 0.0)), (frames - 1, (0.0, 20.0, 0.0))])
    t_root = _track(b, "root")  # empty track
    tracks = _vec_tables(b, [t_root, t_waist, t_head])
    b.StartObject(2)
    b.PrependUOffsetTRelativeSlot(0, tracks, 0)
    sk = b.EndObject()
    b.StartObject(3)
    b.PrependUint32Slot(0, loop, 0)
    b.PrependUint32Slot(1, frames, 0)
    b.PrependUint32Slot(2, fps, 0)
    info = b.EndObject()
    b.StartObject(5)
    b.PrependUOffsetTRelativeSlot(0, info, 0)
    b.PrependUOffsetTRelativeSlot(1, sk, 0)
    b.Finish(b.EndObject())
    return bytes(b.Output())


# ----------------------------------------------------------------------------- tests
def test_parse_gfbmdl():
    m = G.parse_gfbmdl(make_gfbmdl())
    assert [b.name for b in m.bones] == ["root", "Waist", "Head", "Eff01"]
    assert [b.rigged for b in m.bones] == [False, True, True, False]
    assert m.bones[2].parent == 1 and m.bones[2].translation == (0.0, 10.0, 0.0)
    s = m.shapes[0]
    assert s.attrs[G.V_POS].shape == (4, 3) and s.attrs[G.V_JOINTS].shape == (4, 4)
    assert np.allclose(s.attrs[G.V_WEIGHTS][0], [1, 0, 0, 0])
    assert s.polygons[0][1].tolist() == [0, 1, 2, 0, 2, 3]
    mat = m.materials[0]
    assert mat.tex("Col0Tex").file == "pm9999_00_Body_col" and mat.tex("Col0Tex").wrap[0] == "mirror"
    assert mat.values["ColorUVScaleU"] == 2.0


def test_import_model_units_skin_and_uv_bake():
    albedo = np.zeros((8, 16, 4), np.uint8)
    albedo[..., 0] = np.arange(16, dtype=np.uint8)[None, :] * 10
    albedo[..., 3] = 255
    model = import_gfbmdl(make_gfbmdl(), None, lambda name: albedo, "pm9999_00", "Test")
    sk = model.skeleton
    assert sk.palette == [1, 2]  # rigged bones only
    head = sk.joints[2]
    assert head.translation == pytest.approx((0.0, 0.1, 0.0))  # cm -> m
    # inverse bind = inverse of the bind world matrix: Waist at y=0.1, Head at y=0.2
    assert np.linalg.inv(sk.joints[2].inverse_bind)[1, 3] == pytest.approx(0.2, abs=1e-6)
    mesh = model.meshes[0]
    assert mesh.name == "Eff01"
    p = mesh.primitives[0]
    assert p.positions.max() == pytest.approx(0.10)  # 10 cm -> 0.1 m
    assert p.joints.max() == 1 and np.allclose(p.weights.sum(axis=1), 1.0)
    # mirror U with scale 2: the texture is baked to twice the width and UVs stay in [0,1]
    tex = model.textures[model.materials["Body"].base_color]
    assert tex.rgba.shape == (8, 32, 4)
    assert np.array_equal(tex.rgba[:, 16:, :], tex.rgba[:, :16, :][:, ::-1])
    uv = p.uvs[0]
    assert uv.min() >= -1e-6 and uv.max() <= 1 + 1e-6
    # V tile dropped and flipped to the top-left origin: gf v=0 (bottom) -> glTF v=1
    assert uv[0, 1] == pytest.approx(1.0) and uv[2, 1] == pytest.approx(0.0)
    assert mesh.visible


def test_rare_variant_and_missing_texture():
    def load(name):
        return None if name == "missing" else np.full((4, 4, 4), 200 if name.endswith("rare") else 100, np.uint8)

    model = import_gfbmdl(make_gfbmdl("pm9999_00_Body_col"), make_gfbmdl("pm9999_00_Body_col_rare"), load, "pm9999_00", "T")
    assert [v.name for v in model.variants] == ["normal", "rare"]
    assert model.variants[1].materials["Body"].base_color != model.materials["Body"].base_color
    # a model whose albedo cannot be loaded keeps working: mesh is hidden (effect/no-albedo meshes)
    model = import_gfbmdl(make_gfbmdl("missing"), None, load, "pm9999_00", "T")
    assert model.materials["Body"].base_color is None and not model.meshes[0].visible


def test_parse_and_convert_clip():
    anm = parse_gfbanm(make_gfbanm())
    assert anm.looped and anm.frame_count == 4 and anm.frame_rate == 30
    by = {t.bone: t for t in anm.tracks}
    assert by["Head"].rotation.values.shape == (4, 4)
    assert by["Head"].rotation.values[3, 2] == pytest.approx(math.sin(0.9 / 2), abs=1e-3)
    assert by["Waist"].translation.frames.tolist() == [0, 3]
    model = import_gfbmdl(make_gfbmdl(), None, lambda n: np.full((4, 4, 4), 255, np.uint8), "pm9999_00", "T")
    clip = clip_from_gfbanm(make_gfbanm(), "ba10_waitA01", model.skeleton)
    kinds = {(t.joint, t.kind) for t in clip.tracks}
    assert kinds == {("Head", "rotation"), ("Waist", "translation")}  # empty root track dropped
    wt = next(t for t in clip.tracks if t.joint == "Waist")
    assert wt.values[-1].tolist() == pytest.approx([0.0, 0.2, 0.0])  # cm -> m
    assert wt.times[-1] == pytest.approx(3 / 30) and clip.loop and clip.category == "battle"


def test_glb_roundtrip(tmp_path):
    model = import_gfbmdl(make_gfbmdl(), None, lambda n: np.full((4, 4, 4), 255, np.uint8), "pm9999_00", "T")
    model.clips = [clip_from_gfbanm(make_gfbanm(), "ba10_waitA01", model.skeleton)]
    summary = write_glb(model, str(tmp_path / "m.glb"))
    assert summary["bytes"] > 0


def test_catalog_discovery_and_clips(tmp_path):
    def z(name, members):
        with zipfile.ZipFile(tmp_path / name, "w") as f:
            for n, d in members.items():
                f.writestr(n, d)

    mdl = make_gfbmdl("pm0025_00_Body_col")
    z("SwSh-PokeGen1.zip", {
        "bin/pokemon/pm0025_00/mdl/pm0025_00.gfbmdl": mdl,
        "bin/pokemon/pm0025_00/mdl/pm0025_00_rare.gfbmdl": mdl,
        "bin/pokemon/pm0025_00/mdl/pm0025_00_Body_col.bntx": b"",
        "bin/pokemon/pm0025_61/mdl/pm0025_61.gfbmdl": mdl,
    })
    z("SwSh-PokeAnim.zip", {
        "bin/pokemon/pm0025_00/anm/pm0025_00.gfbanm": make_gfbanm(),
        "bin/pokemon/pm0025_00/anm/pm0025_00_ba20_buturi01.gfbanm": make_gfbanm(),
        "bin/pokemon/pm0025_00/anm/pm0025_00_ba10_waitA01.gfbanm": make_gfbanm(),
        "bin/pokemon/pm0025_00/anm/pm0025_00_battle.gfbanmcfg": b"",
    })
    cat = SwshCatalog(str(tmp_path))
    refs = cat.pokemon()
    assert [r.id for r in refs] == ["pm0025_00", "pm0025_61"]
    assert refs[0].game == "swsh" and refs[0].extras["dex"] == 25 and refs[0].folder == "bin/pokemon/pm0025_00/"
    assert cat.find("pm0025_61").form == "61"
    model = cat.import_ref(refs[0])
    assert [c.name for c in model.clips] == ["ba10_waitA01", "ba20_buturi01"]  # wait first, bare id skipped
    assert model.default_clip == "ba10_waitA01"


# ----------------------------------------------------------------------------- DDS / trainers
def _dds(w, h, body, fourcc=b"DX10", dxgi=72, rgba_mask=None):
    import struct

    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    struct.pack_into("<I", hdr, 4, 124)
    struct.pack_into("<II", hdr, 12, h, w)
    struct.pack_into("<I", hdr, 76, 32)
    if rgba_mask:
        struct.pack_into("<II", hdr, 80, 0x41, 0)
        struct.pack_into("<I", hdr, 88, 32)
        struct.pack_into("<4I", hdr, 92, *rgba_mask)
    else:
        struct.pack_into("<I4s", hdr, 80, 4, fourcc)
    out = bytes(hdr)
    if fourcc == b"DX10" and not rgba_mask:
        out += struct.pack("<5I", dxgi, 3, 0, 1, 0)
    return out + body


def test_dds_bc1_srgb_and_rgba():
    from pipeline.importers.swsh.dds import parse_dds

    # BC1 block: both endpoints pure red (0xF800), all indices 0 -> solid red
    block = bytes([0x00, 0xF8, 0x00, 0xF8, 0, 0, 0, 0])
    rgba, srgb = parse_dds(_dds(4, 4, block, dxgi=72))
    assert srgb and rgba.shape == (4, 4, 4) and tuple(rgba[0, 0]) == (255, 0, 0, 255)
    px = bytes([10, 20, 30, 40]) * 4
    rgba, srgb = parse_dds(_dds(2, 2, px, rgba_mask=(0xFF, 0xFF00, 0xFF0000, 0xFF000000)))
    assert not srgb and tuple(rgba[1, 1]) == (10, 20, 30, 40)
    with pytest.raises(ValueError):
        parse_dds(b"nope" * 40)


def test_catalog_trainers_and_objects(tmp_path):
    mdl = make_gfbmdl("tr0001_00_body_col")
    for arch, rel in [
        ("trainers", "chara/data/tr/tr0001_00_champion/tr0001_00_champion"),
        ("trainers_dlc1", "chara/data/tr_m/mdl/tr0160_00_master/tr0160_00_master"),
        ("misc", "chara/data/ob/ob0204_00_monsterball/ob0204_00_monsterball"),
    ]:
        base = tmp_path / arch / rel
        base.parent.mkdir(parents=True)
        (str(base) + ".gfbmdl") and open(str(base) + ".gfbmdl", "wb").write(mdl)
    # the base game's `_m` variant must not be catalogued
    open(str(tmp_path / "trainers/chara/data/tr/tr0001_00_champion/tr0001_00_champion_m.gfbmdl"), "wb").write(mdl)
    cat = SwshCatalog(str(tmp_path / "none"), raw_dir=str(tmp_path))
    assert [(r.id, r.display_name) for r in cat.trainers()] == [("tr0001_00", "Champion"), ("tr0160_00", "Master")]
    assert [(r.id, r.kind) for r in cat.objects()] == [("ob0204_00", "object")]
    # no DDS textures next to the model: imports (mesh hidden, no albedo) instead of failing
    m = cat.import_ref(cat.find("tr0001_00"))
    assert m.kind == "trainer" and m.clips == [] and len(m.meshes) == 1
