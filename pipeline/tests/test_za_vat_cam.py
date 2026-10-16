import os
import zipfile

import numpy as np
import pytest

from pipeline.importers.za.trcma import parse_trcma
from pipeline.importers.za.trvat import read_rgba16f, vat_deltas
from pipeline.catalog.za import DEFAULT_DATA_DIR


def _zip(name):
    p = os.path.join(DEFAULT_DATA_DIR, name)
    if not os.path.isfile(p):
        pytest.skip(f"{name} not available")
    return zipfile.ZipFile(p)


def test_vat_deltas_centre():
    tex = np.full((2, 3, 4), 0.5, np.float32)
    assert np.allclose(vat_deltas(tex, 0.5), 0.0)
    tex[..., 0] = 1.0
    assert np.allclose(vat_deltas(tex, 0.5)[..., 0], 0.5)


def test_vat_texture_real():
    z = _zip("ZA-TrainersAnim.zip")
    tex = read_rgba16f(z.read("ik_chara/motion_uq/tr0566_00_adult_a_m/tr0566_00_00551_speak01_loop_body.bntx"))
    assert tex.shape == (116, 951, 4)
    assert np.allclose(tex[..., 3], 1.0)
    assert 0.0 <= tex[..., :3].min() and tex[..., :3].max() <= 1.0


def test_trcma_real():
    z = _zip("ZA-Demo.zip")
    cam = parse_trcma(z.read("ik_demo/ev/d004/d004_cam_c04.trcma"))
    assert (cam.frame_count, cam.frame_rate) == (241, 60)
    assert len(cam.rotation.frames) == 241 and abs(np.linalg.norm(cam.rotation.values[0]) - 1) < 1e-3
    assert abs(cam.fov_y - 0.4975) < 1e-3
