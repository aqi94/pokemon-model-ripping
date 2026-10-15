"""GO importer: IR construction from a synthetic Blender dump (no Blender needed)."""

import numpy as np
from PIL import Image

from pipeline.catalog.go import pretty_name
from pipeline.common.gltf_writer import write_glb
from pipeline.importers.go.importer import model_from_dump


def _dump():
    joints = [
        {"name": "root", "parent": -1, "t": [0, 0, 0], "r": [1, 0, 0, 0], "s": [1, 1, 1]},
        {"name": "arm", "parent": 0, "t": [0, 1, 0], "r": [1, 0, 0, 0], "s": [1, 1, 1]},
    ]
    ibm = np.stack([np.eye(4, dtype=np.float32), np.array([[1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)])
    pre = "m0_s0_"
    arrays = {
        "ibm": ibm,
        pre + "pos": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        pre + "nrm": np.tile(np.array([0, 0, 1], np.float32), (3, 1)),
        pre + "uv": np.array([[0, 0], [1, 0], [0, 1]], np.float32),
        pre + "idx": np.array([0, 1, 2], np.uint32),
        pre + "joints": np.zeros((3, 4), np.uint16),
        pre + "weights": np.tile(np.array([1, 0, 0, 0], np.float32), (3, 1)),
    }
    frames = 5
    loc = np.zeros((frames, 2, 3), np.float32)
    loc[:, 1] = [0, 1, 0]
    loc[:, 0, 0] = np.linspace(0, 1, frames)  # root moves in x
    rot = np.zeros((frames, 2, 4), np.float32)
    rot[..., 0] = 1  # identity (w first)
    scl = np.ones((frames, 2, 3), np.float32)
    arrays.update({"a0_loc": loc, "a0_rot": rot, "a0_scl": scl})
    meta = {
        "fps": 30,
        "joints": joints,
        "meshes": [{"name": "body", "slots": [{"key": pre, "material": "Mat", "skinned": True}]}],
        "materials": [{"name": "Mat", "base": "tex.png", "images": ["tex.png"]}],
        "clips": [{"name": "pvp_idle", "key": "a0_", "frames": frames, "start": 1}],
    }
    return meta, arrays


def test_model_from_dump(tmp_path):
    Image.new("RGBA", (4, 4), (200, 10, 10, 255)).save(tmp_path / "tex.png")
    meta, arrays = _dump()
    m = model_from_dump(meta, arrays, str(tmp_path), "go_test", "Test")
    assert m.game == "go" and m.kind == "trainer"
    assert [j.name for j in m.skeleton.joints] == ["root", "arm"]
    assert m.materials["Mat"].base_color == "tex"
    (prim,) = m.meshes[0].primitives
    assert np.allclose(prim.uvs[0], [[0, 1], [1, 1], [0, 0]])  # V flipped to top-left origin
    (clip,) = m.clips
    assert clip.loop and m.default_clip == "pvp_idle"
    # only the root's x translation varies; the constant arm channels equal rest and are dropped
    assert [(t.joint, t.kind) for t in clip.tracks] == [("root", "translation")]
    out = tmp_path / "m.glb"
    assert write_glb(m, str(out))["bytes"] > 0


def test_pretty_name():
    assert pretty_name("blanchev2_rig") == "Blanche V2"
    assert pretty_name("spark_rig_quest") == "Spark Quest"


def test_deep_translation_tracks_are_dropped(tmp_path):
    Image.new("RGBA", (4, 4), (200, 10, 10, 255)).save(tmp_path / "tex.png")
    meta, arrays = _dump()
    # a third joint, three levels deep, whose animated translation disagrees with its rest (other rig's proportions)
    meta["joints"].append({"name": "eye", "parent": 1, "t": [0, 0, 0], "r": [1, 0, 0, 0], "s": [1, 1, 1]})
    meta["joints"].append({"name": "lid", "parent": 2, "t": [0, 0, 0], "r": [1, 0, 0, 0], "s": [1, 1, 1]})
    arrays["ibm"] = np.stack([np.eye(4, dtype=np.float32)] * 4)
    for k in ("loc", "rot", "scl"):
        a = arrays["a0_" + k]
        arrays["a0_" + k] = np.concatenate([a, a[:, :1], a[:, :1]], axis=1)
    arrays["a0_loc"][:, 3] = [0.5, 0.5, 0.5]
    m = model_from_dump(meta, arrays, str(tmp_path), "go_test", "Test")
    assert "lid" not in {t.joint for t in m.clips[0].tracks if t.kind == "translation"}
