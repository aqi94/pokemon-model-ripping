import numpy as np

from pipeline.importers.za import trinity as T
from pipeline.importers.za.importer import _patch_params, _same_params, _variation_label


def _track(param, keys):
    """keys: list of (frame, (x, y, z, w))."""
    fr = np.array([k for k, _ in keys], np.float32)
    vals = np.array([v for _, v in keys], np.float32)
    return T.MaterialParamTrack("body_mesh_shape", "body_skin", param, [(fr, vals[:, c]) for c in range(vals.shape[1])])


def test_track_value_holds_the_last_key_at_or_below_the_id():
    # SV Karate skin: key 0 = authored default, then one key per colour id (held), 305 = end marker
    tr = _track("BaseColorLayer1", [(0, (0.24, 0.14, 0.13, 1)), (301, (0.55, 0.33, 0.28, 1)), (303, (0.9, 0.65, 0.63, 1)), (305, (0.9, 0.65, 0.63, 1))])
    assert np.allclose(T.track_value(tr, 300), (0.24, 0.14, 0.13, 1))
    assert np.allclose(T.track_value(tr, 301), (0.55, 0.33, 0.28, 1))
    assert np.allclose(T.track_value(tr, 302), (0.55, 0.33, 0.28, 1))
    assert np.allclose(T.track_value(tr, 303), (0.9, 0.65, 0.63, 1))


def test_patch_params_overrides_colours_and_floats_and_detects_the_authored_option():
    tm = T.TrMaterial("hairstyle", "Hair", {}, [], [], {"Metallic": 0.1}, {"BaseColor": (0.2, 0.1, 0.05, 1.0)}, {}, "Opaque", 0)
    p = _patch_params(tm, {"BaseColor": (0.9, 0.5, 0.1, 1.0), "Metallic": (0.4,)})
    assert p.colors["BaseColor"] == (0.9, 0.5, 0.1, 1.0) and p.floats["Metallic"] == 0.4
    assert tm.colors["BaseColor"] == (0.2, 0.1, 0.05, 1.0)  # the parsed material is not mutated
    assert _same_params(tm, {"BaseColor": (0.2, 0.1, 0.05, 1.0), "Metallic": (0.1,)})
    assert not _same_params(tm, {"BaseColor": (0.9, 0.5, 0.1, 1.0)})


def test_variation_label_names_the_recoloured_parts():
    assert _variation_label({"body_skin", "face_skin", "hairstyle", "hairup", "eye", "eyebrow"}) == "Skin / Hair / Eyes"
    assert _variation_label({"body_skin", "face_skin"}) == "Skin"
    assert _variation_label({"obj00"}) == "Accessory colour"
