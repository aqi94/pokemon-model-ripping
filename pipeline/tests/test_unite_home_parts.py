"""Texture-less UNITE parts take their colour from the same species in HOME (`home_parts`, ledger fx:homeColouredParts)."""

import numpy as np

from pipeline.common.model import Material, Mesh, Model, Primitive
from pipeline.importers.unite import home_parts as hp


def _tri(pos, material="m"):
    pos = np.asarray(pos, np.float32)
    return Primitive(material, np.arange(len(pos) - len(pos) % 3, dtype=np.uint32), pos)


def _body_box(scale=1.0):
    # bottom centre at the origin, footprint 1 x 1, height 1 (times scale)
    return np.array([[-.5, 0, -.5], [.5, 0, .5], [.5, 1, .5], [-.5, 1, -.5]], np.float32) * scale


def _home(tuft_colour):
    img = np.zeros((4, 4, 4), np.uint8)
    img[..., :3] = tuft_colour
    img[..., 3] = 255
    tuft = np.array([[0, 1, 0], [0.02, 1.2, 0], [-0.02, 1.4, 0]], np.float32) * 0.5  # HOME is half the size
    uv = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], np.float32)
    body = {"name": "Body", "pos": _body_box(0.5), "uv": np.zeros((4, 2), np.float32),
            "factor": np.ones(4, np.float32), "image": np.full((2, 2, 4), 255, np.uint8)}
    return [body, {"name": "Hair", "pos": tuft, "uv": uv, "factor": np.ones(4, np.float32), "image": img}]


def test_match_picks_nearest_mesh_and_samples_texture():
    part = np.array([[0, 1, 0], [0.02, 1.2, 0], [-0.02, 1.4, 0]], np.float32)  # UNITE scale 1
    r = hp.match_part(part, _body_box(), _home((255, 200, 0)))
    assert r["ok"] and r["homeMesh"] == "Hair" and r["matchDistance"] < 0.01
    assert np.allclose(r["colors"][0, :3], [1.0, hp._srgb_to_linear(np.float32(200 / 255)), 0.0], atol=1e-3)


def test_gate_rejects_a_part_far_from_every_home_mesh():
    part = np.array([[3, 3, 3], [3.1, 3, 3], [3, 3.1, 3]], np.float32)
    r = hp.match_part(part, _body_box(), _home((255, 0, 0)))
    assert r is None or not r["ok"]


def test_colour_from_home_rewrites_material_and_warnings(tmp_path, monkeypatch):
    body = _tri(_body_box(), "body")
    tuft = _tri([[0, 1, 0], [0.02, 1.2, 0], [-0.02, 1.4, 0]], "fx")
    model = Model(id="t", game="unite", display_name="t", kind="pokemon", skeleton=None,
                  meshes=[Mesh("body", [body]), Mesh("fx", [tuft])],
                  materials={"body": Material("body"), "fx": Material("fx", extras={"effect": True, "missingTexture": True})},
                  textures={})
    model.extras["warnings"] = ["effect parts with no texture in the rip, imported hidden: fx/fx"]
    monkeypatch.setattr(hp, "home_glb", lambda dex, ensure=True: tmp_path / "pm0001_00_00" / "model.glb")
    monkeypatch.setattr(hp, "load_home_meshes", lambda glb: _home((255, 200, 0)))
    res = hp.colour_from_home(model, 1)
    assert len(res["done"]) == 1 and not res["failed"]
    mat = model.materials[tuft.material]
    assert not mat.extras.get("missingTexture") and not mat.extras.get("effect")
    assert mat.extras["unite"]["colourFrom"]["source"] == "home" and tuft.colors.shape == (3, 4)
    assert model.extras["homeColouredParts"][0]["homeMesh"] == "Hair"
    assert not any(w.startswith("effect parts with no texture") for w in model.extras["warnings"])
