import numpy as np

from pipeline.common.model import Material
from pipeline.importers.za import trinity as T
from pipeline.importers.za.importer import material_uv_offsets


def _track(material, ov_frames, ov_values, scale=(2.0, 1.0)):
    f0 = np.array([0.0, 60.0], np.float32)
    return T.MaterialParamTrack(
        "body_a_mesh_shape",
        material,
        "UVScaleOffset",
        [
            (f0, np.full(2, scale[0], np.float32)),
            (f0, np.full(2, scale[1], np.float32)),
            (f0, np.zeros(2, np.float32)),
            (np.array(ov_frames, np.float32), np.array(ov_values, np.float32)),
        ],
    )


def _mat(name, mirror=(False, False)):
    m = Material(name=name)
    m.extras["uv_scale_offset"] = (2.0, 1.0, 0.0, 0.0)
    m.extras["mirror"] = mirror
    return m


def test_blink_steps_held_and_shared_with_shiny():
    # Falinks' blink: V offset 0 -> 0.25 -> 0.5 -> 0 through the eye-atlas rows, held between keys.
    mats = {"r_eye_a": _mat("r_eye_a", (True, False)), "r_eye_a@rare": _mat("r_eye_a@rare", (True, False))}
    out = material_uv_offsets([_track("r_eye_a", [0, 57, 60, 64, 90], [0, 0.25, 0.5, 0, 0])], mats, 60.0)
    assert set(out) == {"r_eye_a", "r_eye_a@rare"}
    times, xy = out["r_eye_a"]
    assert np.allclose(times * 60.0, [0, 57, 60, 64])
    assert np.allclose(xy[:, 1], [0, 0.25, 0.5, 0])
    assert np.allclose(xy[:, 0], 0)


def test_static_or_scale_changing_tracks_are_skipped():
    mats = {"eye": _mat("eye")}
    assert material_uv_offsets([_track("eye", [0, 60], [0, 0])], mats, 60.0) == {}
    assert material_uv_offsets([_track("eye", [0, 10], [0, 0.25], scale=(3.0, 1.0))], mats, 60.0) == {}


def test_lid_atlas_gate(monkeypatch):
    from pipeline.importers.za import importer as I

    def mk(mat, param, ov):
        f = np.array([0.0, 10.0], np.float32)
        one = (f, np.ones(2, np.float32))
        return T.MaterialParamTrack("s", mat, param, [one, one, (f, np.zeros(2, np.float32)), (f, np.array(ov, np.float32))])

    lockstep = [mk("falinks", "UVScaleOffset", [0, 0.25]), mk("falinks", "UVScaleOffset4", [0, 0.25])]
    parked = [mk("froakie", "UVScaleOffset4", [2.0, 2.0])]  # lid moved alone (off its atlas)
    monkeypatch.setattr(I.T, "parse_tracm_materials", lambda cfg: cfg)
    anims = [type("A", (), {"config": lockstep + parked})()]
    assert I._lid_atlas_unfit(anims) == {"froakie"}


def test_lid_overlay_follows_its_eye_param():
    # Greninja's upper lid: overlay `r_eye_a~lidU` follows r_eye_a's UVScaleOffset2 (scale (0, 1): only V moves).
    eye = _mat("r_eye_a")
    lid = Material(name="r_eye_a~lidU")
    lid.extras.update({"uv_scale_offset": (0.0, 1.0, 0.0, -0.5), "uv_track_param": "UVScaleOffset2", "uv_track_src": "r_eye_a", "mirror": (False, False)})
    f = np.array([0.0, 10.0], np.float32)
    tr = T.MaterialParamTrack("s", "r_eye_a", "UVScaleOffset2", [(f, np.zeros(2, np.float32)), (f, np.ones(2, np.float32)),
                                                                 (f, np.zeros(2, np.float32)), (f, np.array([0.206, -0.253], np.float32))])
    out = material_uv_offsets([tr], {"r_eye_a": eye, "r_eye_a~lidU": lid}, 60.0)
    assert set(out) == {"r_eye_a~lidU"}
    assert np.allclose(out["r_eye_a~lidU"][1][:, 1], [0.706, 0.247], atol=1e-5)
    assert np.allclose(out["r_eye_a~lidU"][1][:, 0], 0)


def test_lid_overlay_uvs_rest_transform():
    from pipeline.importers.za.importer import lid_overlay_uvs

    # Trinity UV in V tile 2; scale-0 U collapses onto the centre column; +V offset samples further down the image.
    uv = np.array([[0.2, 2.25], [0.8, 2.75]], np.float32)
    out = lid_overlay_uvs(uv, {"scale": (0.0, 1.0), "offset": (0.0, 0.2), "center": (0.5, 0.5)})
    assert np.allclose(out[:, 0], 0.5)
    assert np.allclose(out[:, 1], [0.95, 0.45])
