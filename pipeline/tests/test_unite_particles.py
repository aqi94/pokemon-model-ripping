"""UNITE particle stand-ins: emitter records (bone + rest offset) and HOME colours."""

import numpy as np

from pipeline.common.model import Joint, Model, Skeleton
from pipeline.importers.unite import fbx_anim
from pipeline.importers.unite.importer import colour_particles, particle_emitters, particle_records


def _scene():
    N = fbx_anim.FbxNode
    nodes = {
        1: N(1, "root", "Null"),
        2: N(2, "Bone_Tail05", "LimbNode", t=(0.0, 1.0, 0.0), s=(2.0, 2.0, 2.0), parent=1),
        # unweighted bone pruned from the final skeleton: the emitter must hang off Bone_Tail05
        3: N(3, "Bone_Tail06", "Null", t=(0.1, 0.0, 0.0), parent=2),
        4: N(4, "pref_fx_0004_battlefire_lv3", "Null", t=(0.0, 0.05, 0.0), r=(0.0, 0.0, 90.0), parent=3),
        5: N(5, "huo 1", "Null", s=(1.3, 1.3, 1.3), parent=4),
        6: N(6, "CTFIre (5)", "Null", s=(1.5, 1.5, 1.5), parent=5),
        # attack prefabs and prefabs holding meshes are not idle particle emitters
        7: N(7, "pref_fx_0004_natk_atk01", "Null", parent=3),
        8: N(8, "pref_fx_mesh_lv3", "Null", parent=3),
        9: N(9, "flame", "Mesh", parent=8),
    }
    return fbx_anim.FbxScene(nodes=nodes)


def _model():
    sk = Skeleton(joints=[Joint(name="root", parent=-1), Joint(name="Bone_Tail05", parent=0)], palette=[0, 1])
    return Model(id="t", game="unite", display_name="t", kind="pokemon", skeleton=sk, meshes=[], materials={}, textures={})


def test_emitter_detection():
    assert particle_emitters(_scene()) == ["pref_fx_0004_battlefire_lv3"]


def test_record_relative_to_nearest_kept_joint():
    (r,) = particle_records(_scene(), _model())
    assert r["bone"] == "Bone_Tail05"
    # offset in Bone_Tail05's local space: (0.1 + 0, 0 + 0.05, 0), unaffected by the bone's own scale
    assert np.allclose(r["translation"], (0.1, 0.05, 0.0), atol=1e-6)
    assert np.allclose(r["scale"], (1, 1, 1), atol=1e-6)
    assert abs(r["subtreeScale"] - 1.95) < 1e-3  # 1.3 x 1.5
    # 90 deg about Z
    assert np.allclose(np.abs(r["rotation"]), (0, 0, np.sqrt(0.5), np.sqrt(0.5)), atol=1e-6)


def test_colour_from_home_table():
    m = _model()
    m.extras["particles"] = particle_records(_scene(), m)
    assert colour_particles(m, 6) == 1  # Charizard: HOME FireCore
    r = m.extras["particles"][0]
    assert r["kind"] == "fire" and r["source"].startswith("home:")
    assert np.allclose(r["baseColor"][:3], (1.0, 0.1961, 0.102), atol=1e-3)
    m2 = _model()
    m2.extras["particles"] = particle_records(_scene(), m2)
    assert colour_particles(m2, 448) == 0  # Lucario: no HOME effect material
    assert "kind" not in m2.extras["particles"][0]


def test_prefab_key_matches_nested_copies():
    from pipeline.importers.unite.importer import prefab_key
    assert prefab_key("pref_fx_0004_lobbyfire_atk_lv3 (1)") == prefab_key("pref_fx_0004_lobbyfire_atk")
    assert prefab_key("pref_fx_0004_battlefire_lv3") != prefab_key("pref_fx_0004_lobbyfire_atk")


# ---- round 3 (2026-11-10): HOME motion, borrow coverage, noise sheens, multi-mesh HOME matches

def test_noise_direction_follows_root_to_tip():
    from pipeline.importers.unite.importer import noise_direction
    t = np.linspace(0, 1, 20)
    up = np.stack([np.zeros(20), t], 1)  # V grows towards the tip
    assert noise_direction(up, t) == 1.0
    assert noise_direction(np.stack([np.zeros(20), 1 - t], 1), t) == -1.0
    assert noise_direction(None, t) == 1.0


def test_borrow_coverage_counts_drawn_texels():
    from pipeline.common.model import Primitive, TextureImage
    from pipeline.importers.unite.importer import borrow_coverage
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[:, :2] = 255  # left half drawn
    tex = TextureImage(key="k", rgba=rgba, srgb=True)
    uv = np.array([[0.1, 0.5], [0.3, 0.5], [0.7, 0.5], [0.9, 0.5]], np.float32)
    p = Primitive(material="m", indices=np.zeros(0, np.uint32), positions=np.zeros((4, 3), np.float32), uvs=[uv])
    assert borrow_coverage(p, tex) == 0.5
    assert borrow_coverage(p, None) == 0.0


def test_near_opaque_grey_noise_is_an_additive_sheen(tmp_path):
    from PIL import Image
    from pipeline.importers.unite.importer import build_effect_material
    rng = np.random.default_rng(0)
    g = (rng.random((32, 32)) * 160 + 20).astype(np.uint8)  # tiling grey noise, never empty or solid
    a = np.full((32, 32), 255, np.uint8)
    a[0, :8] = 248  # noise_06 ships alpha 0.973..1.0
    Image.fromarray(np.dstack([g, g, g, a])).save(tmp_path / "t_fx_0001_noise_06.png")
    info = {"floats": {}, "colors": {}, "shader": "", "source": "fbx",
            "textures": {"_MainTex": {"name": "t_fx_0001_noise_06", "scale": (1, 1), "offset": (0, 0)}}}
    textures = {}
    m = build_effect_material("mat_fx_lob_0006_51_X_wenli", info, str(tmp_path), textures)
    assert m is not None and m.extras.get("noiseSheen") and m.extras["blend"] == "Add"
    # the grey became the alpha (not the near-solid 248..255 alpha)
    assert textures[m.base_color].rgba[..., 3].min() < 200


def test_home_match_unions_meshes_one_part_holds():
    from pipeline.importers.unite.home_parts import match_part
    rng = np.random.default_rng(1)
    body = rng.random((500, 3)) * [1.0, 1.0, 1.0]
    img = np.full((4, 4, 4), 200, np.uint8)

    def blob(c):
        return np.asarray(c) + rng.normal(0, 0.01, (60, 3))
    birds = [blob((0.2, 2.0, 0.0)), blob((0.5, 2.0, 0.0)), blob((0.8, 2.0, 0.0))]
    home = [{"name": f"Bird{i}", "pos": b, "uv": np.zeros((60, 2)), "image": img, "factor": np.ones(4, np.float32),
             "blend": False} for i, b in enumerate(birds)] + \
           [{"name": "Body", "pos": body, "uv": np.zeros((500, 2)), "image": img, "factor": np.ones(4, np.float32), "blend": False}]
    part = np.concatenate(birds)  # one UNITE mesh holding all three birds
    r = match_part(part, body, home)
    assert r is not None and r["ok"] and r["cover"] == 1.0
    assert len(r["homeMeshes"]) == 3


def test_transfer_uvs_picks_seam_copy_agreeing_with_neighbours():
    from pipeline.importers.unite.home_parts import transfer_uvs
    # a strip of 2 triangles; HOME duplicates vertex 1 at a seam (u 0.9 and u 0.1)
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float64)
    hpos = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float64)
    huv = np.array([[0.0, 0], [0.9, 0], [0.1, 0], [0.0, 0.2], [0.1, 0.2]], np.float32)
    idx = np.array([0, 1, 3, 4])  # nearest picks the u 0.9 copy for vertex 1
    dist = np.zeros(4)
    uv, _, _, seam = transfer_uvs(pos, np.array([0, 1, 2, 1, 3, 2]), {"pos": hpos, "uv": huv}, idx, dist, 1.0, 1.0)
    assert abs(uv[1, 0] - 0.1) < 1e-6 and seam == 0.0
