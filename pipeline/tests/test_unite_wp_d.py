"""WP-D (2026-11-09 shader survey): the matcap family is four shader terms, read with their real constants."""

import numpy as np
from PIL import Image

from pipeline.common.gltf_writer import write_glb
from pipeline.common.model import Material, Mesh, Model, Primitive, TextureImage
from pipeline.importers.unite import matcap_terms
from pipeline.importers.unite.importer import build_material


def _png(tmp_path, name, rgba=(120, 120, 120, 255)):
    Image.new("RGBA", (4, 4), rgba).save(tmp_path / f"{name}.png")


def _info(shader="==PGAME==/LobbyPokemon/m_lob_basepbrCloth_lv3", floats=None, colors=None, **slots):
    return {"name": "m", "shader": shader, "floats": floats or {}, "colors": colors or {}, "source": "json",
            "textures": {k: {"name": v, "scale": (1.0, 1.0), "offset": (0.0, 0.0)} for k, v in slots.items()}}


def _terms(info, tmp_path, material="m"):
    from pipeline.importers.unite.importer import _png as png
    return matcap_terms.lit_terms(info, str(tmp_path), None, {}, png, material)


def test_cloth_terms_follow_the_keywords(tmp_path):
    for n in ("mt", "rt", "mc"):
        _png(tmp_path, n)
    info = _info(_MatTex="mt", _RampTex="rt", _MatcapTex="mc",
                 floats={"_IFMAT": 1.0, "_IFRAMP": 0.0, "_Matcap": 0.0, "_MATREFL": 1.0},
                 colors={"_MatTexColor": (0.5, 0.6, 0.7, 0.0), "_MatcapColor": (4.84,) * 3 + (0.0,)})
    t = _terms(info, tmp_path)
    assert set(t) == {"matTex"} and t["matTex"]["color"] == [0.5, 0.6, 0.7]  # `_MATREFL` is not a Cloth property
    info["floats"].update({"_IFMAT": 0.0, "_IFRAMP": 1.0, "_Matcap": 1.0})
    info["textures"]["_RampTex"]["scale"], info["textures"]["_RampTex"]["offset"] = (0.5, 1.0), (0.25, 0.0)
    t = _terms(info, tmp_path)
    assert set(t) == {"rampLight", "env"}
    assert t["rampLight"]["scale"] == 0.5 and t["rampLight"]["offset"] == 0.25
    assert t["env"]["kind"] == "cloth" and t["env"]["color"] == [4.84] * 3


def test_pbr_env_and_battle_env(tmp_path):
    _png(tmp_path, "mm")
    info = _info("Hidden/Lobby/m_lob_basepbrClothLengcy_lv3", floats={"_MATREFL": 1.0, "_MatMapMul": 2.0}, _MatMap="mm")
    assert _terms(info, tmp_path)["env"] == {"kind": "pbr", "texture": {"texture": "mm", "uv": 0}, "color": [2.0] * 3}
    info["floats"]["_MATREFL"] = 0.0  # no keyword = the scene probe, not in the rip: nothing
    assert _terms(info, tmp_path) == {}
    own = {"name": "", "shader": "", "floats": {}, "colors": {}, "source": "fbx", "family": "matcapMask",
           "textures": {"_MatCapTex": {"name": "mm", "scale": (1, 1), "offset": (0, 0)}}}
    assert _terms(own, tmp_path) == {}  # FBX-only: `_MATREFL` default is 0, no assumed keyword
    fbx = {"name": "", "shader": "", "floats": {}, "colors": {}, "source": "fbx", "family": "actorMatcap",
           "textures": {"_MatCapTex": {"name": "mm", "scale": (1, 1), "offset": (0, 0)}}}
    e = _terms(fbx, tmp_path, "mat_bat_0025body_lv3")["env"]
    assert e["kind"] == "battle" and e["color"] == [1.0, 1.0, 1.0] and e["inferred"]


def test_glass_constants_and_kinds():
    ghost = _info("==PGAME==/LobbyPokemon/Extra/m_lob_ghostGlass_lv3",
                  floats={"_ShadowOffset": -0.59, "_ShadowPow": 0.49, "_ShadowScale": 0.0, "_Cutoff": 1.0},
                  colors={"_SHTopColor": (1, 1, 1, 0), "_SHBotColor": (1.35, 2.1, 2.27, 1)})
    g = matcap_terms.glass_terms(ghost, "cap", "m")
    assert g["kind"] == "ghost" and g["offset"] == -0.59 and g["pow"] == 0.49 and g["scale"] == 0.0
    assert g["top"] == [1.0, 1.0, 1.0, 0.0] and g["bot"][3] == 1.0 and "inferred" not in g
    fbx = {"name": "", "shader": "", "floats": {}, "colors": {}, "source": "fbx", "textures": {}}
    a = matcap_terms.glass_terms(fbx, "cap", "mat_bat_014320b_lv3")
    assert a["kind"] == "actor" and a["pow"] == 2.0 and a["scale"] == 1.0 and a["inferred"]


def test_terms_survive_the_glb_writer(tmp_path):
    _png(tmp_path, "mt")
    info = _info(_MatTex="mt", floats={"_IFMAT": 1.0})
    tex: dict[str, TextureImage] = {}
    m = build_material("m", info, str(tmp_path), tex)
    assert m.extras["unite"]["matTex"]["texture"] == {"texture": "mt", "uv": 0}
    mesh = Mesh("a", [Primitive("m", np.arange(3, dtype=np.uint32), np.zeros((3, 3), np.float32))])
    model = Model(id="t", game="unite", display_name="t", kind="pokemon", skeleton=None, meshes=[mesh],
                  materials={"m": m}, textures=tex)
    out = tmp_path / "t.glb"
    write_glb(model, str(out))
    import json, struct
    b = out.read_bytes()
    j = json.loads(b[20:20 + struct.unpack("<I", b[12:16])[0]])
    ref = j["materials"][0]["extras"]["unite"]["matTex"]["texture"]
    assert isinstance(ref["index"], int) and ref["name"] == "mt"
