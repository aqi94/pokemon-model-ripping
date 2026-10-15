import numpy as np

from pipeline.catalog.za import CC_BASE_RE, CC_PART_RE
from pipeline.common.manifest import build_manifest
from pipeline.common.model import Mesh, Model, Primitive
from pipeline.importers.za.composite import _remap_joints, _PartRig, part_label


def test_cc_regexes():
    m = CC_PART_RE.match("ik_chara/model_cc_vr/drs/dm_drs0001_01_adult_b_m/dm_drs0001_01.trmdl")
    assert (m["base"], m["grp"], m["num"], m["var"], m["name"]) == ("dm", "drs", "0001", "01", "adult_b_m")
    assert not CC_PART_RE.match("ik_chara/model_cc_vr/drs/dm_drs0001_01_adult_b_m/dm_drs0001_01_lod1.trmsh")
    assert CC_BASE_RE.match("ik_chara/model_cc_base/dm/dm_base.trmdl")


def test_part_label_variants():
    assert part_label("adult_a_m", 0) == "Adult A M"
    assert part_label("adult_a_m", 2) == "Adult A M 3"


def test_manifest_part_groups():
    def mesh(name, option, visible):
        return Mesh(name=name, primitives=[], visible=visible, extras={"group": "hrs", "option": option})

    model = Model(
        id="cc_x", game="za", display_name="x", kind="trainer", skeleton=None,
        meshes=[mesh("a__hair_mesh", "a", True), mesh("b__hair_mesh", "b", False)],
        materials={}, textures={},
        extras={"part_groups": [{"name": "hrs", "label": "Hair", "options": [{"id": "a", "label": "A", "default": True}, {"id": "b", "label": "B", "default": False}]}]},
    )
    man = build_manifest(model)
    assert [o["nodes"] for o in man["partGroups"][0]["options"]] == [["a__hair_mesh"], ["b__hair_mesh"]]
    assert "part_groups" not in man["extras"]
    assert man["parts"][0]["label"] == "hair" and man["parts"][1]["group"] == "hrs"


def test_remap_joints_into_combined_palette():
    prim = Primitive(
        material="m", indices=np.zeros(3, np.uint32), positions=np.zeros((2, 3), np.float32),
        joints=np.array([[3, 0, 0, 0], [204, 205, 0, 0]], np.uint16),
        weights=np.array([[1, 0, 0, 0], [0.5, 0.5, 0, 0]], np.float32),
    )
    bad = _remap_joints(prim, _PartRig({204: 80, 205: 81}), base_count=74)
    assert bad == 0 and prim.joints.tolist() == [[3, 0, 0, 0], [80, 81, 0, 0]]
