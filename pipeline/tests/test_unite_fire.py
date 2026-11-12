"""UNITE flame materials: silhouette-mask rule and which effect materials get the fire look."""

import numpy as np
from PIL import Image

from pipeline.common.model import Material, Mesh, Model, Primitive, TextureImage
from pipeline.importers.unite import importer as imp


def _save(tmp_path, name, grey):
    Image.fromarray(np.stack([grey] * 3 + [np.full_like(grey, 255)], -1).astype(np.uint8)).save(tmp_path / name)
    return str(tmp_path / name)


def test_is_cutout(tmp_path):
    sil = np.zeros((64, 64), np.uint8)
    sil[:, :30] = 255  # half solid, half empty: a flame silhouette
    noise = np.full((64, 64), 128, np.uint8)  # a tiling noise/distortion map cuts nothing
    blobs = np.zeros((64, 64), np.uint8)
    blobs[10:14, 10:14] = 255  # sparse blobs: not a silhouette
    assert imp.is_cutout(_save(tmp_path, "s.png", sil))
    assert not imp.is_cutout(_save(tmp_path, "n.png", noise))
    assert not imp.is_cutout(_save(tmp_path, "b.png", blobs))


def _model(rgb, colours):
    tex = TextureImage(key="t", rgba=np.dstack([np.full((4, 4, 3), rgb, np.uint8), np.full((4, 4), 255, np.uint8)]))
    mat = Material(name="m", alpha_mode="BLEND")
    mat.base_color = "t"
    mat.extras = {"effect": True, "blend": "Blend"}
    p = Primitive(material="m", indices=np.arange(3, dtype=np.uint32), positions=np.zeros((3, 3), np.float32))
    p.colors = np.asarray(colours, np.float32)
    return Model(id="x", game="unite", display_name="x", kind="pokemon", skeleton=None,
                 meshes=[Mesh(name="fx", primitives=[p])], materials={"m": mat}, textures={"t": tex}), p


def test_fire_effects_grey_masks_only(monkeypatch):
    core = {"source": "home:x", "baseColor": [1.0, 0.2, 0.1, 1.0], "layerColor": [0.94, 0.81, 0.0, 1.0], "lerpProfile": [0] * 8}
    monkeypatch.setattr(imp, "home_fire", lambda dex, form=None: core)
    orange = [[1.0, 0.27, 0.0, 1.0], [1.0, 0.075, 0.0, 1.0], [1.0, 0.2, 0.0, 1.0]]
    m, p = _model((200, 200, 200), orange)  # grey mask x vertex colours: the fire look, colours kept
    assert imp.fire_effects(m, 6) == ["m"] and p.colors is not None
    m, p = _model((250, 120, 20), orange)  # painted flame (Moltres): untouched, colours dropped
    assert imp.fire_effects(m, 146) == [] and p.colors is None and "fire" not in m.materials["m"].extras
    m, p = _model((200, 200, 200), [[1, 1, 1, 1]] * 3)  # white vertex colours carry no hue: rim from HOME
    assert imp.fire_effects(m, 146) == ["m"] and p.colors is None and m.materials["m"].extras["fire"]["rimFromHome"]


def test_flame_coordinate_root_near_joint():
    # a vertical card: root = the end nearest the emitter point, whichever way the axis is signed
    pos = np.array([[0, y, 0.01 * (i % 2)] for i, y in enumerate(np.linspace(0.0, 1.0, 9))], np.float32)
    t = imp.flame_coordinate(pos, [0.0, -0.1, 0.0])
    assert abs(float(t[0])) < 1e-6 and abs(float(t[-1]) - 1) < 1e-6 and np.all(np.diff(t) > 0)
    t = imp.flame_coordinate(pos, [0.0, 1.2, 0.0])  # emitter at the other end: reversed
    assert abs(float(t[-1])) < 1e-6 and abs(float(t[0]) - 1) < 1e-6
    assert imp.flame_coordinate(np.zeros((5, 3)), [0, 0, 0]) is None


def test_fire_effects_along_flame_uses_home_pair(monkeypatch):
    from pipeline.common.model import Joint, Skeleton
    core = {"source": "home:x", "baseColor": [1.0, 0.2, 0.1, 1.0], "layerColor": [0.94, 0.81, 0.0, 1.0], "lerpProfile": [0] * 8}
    monkeypatch.setattr(imp, "home_fire", lambda dex, form=None: core)
    m, p = _model((200, 200, 200), [[1.0, 0.27, 0.0, 1.0]] * 3)
    p.positions = np.array([[0, 0.0, 0], [0, 0.5, 0], [0, 1.0, 0]], np.float32)
    p.joints = np.zeros((3, 4), np.uint16)
    p.weights = np.tile(np.array([1, 0, 0, 0], np.float32), (3, 1))
    ib = np.eye(4)
    ib[1, 3] = -0.0  # root joint at the origin, end of the card with t = 0
    m.skeleton = Skeleton(joints=[Joint(name="root", parent=-1, inverse_bind=ib, skinning=True)], palette=[0])
    assert imp.fire_effects(m, 6) == ["m"]
    assert p.colors is None and m.materials["m"].extras["fire"]["alongFlame"]
    assert np.allclose(p.flame_t, [0, 0.5, 1])
