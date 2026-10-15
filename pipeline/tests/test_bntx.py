from pathlib import Path

import numpy as np
import pytest

from pipeline.importers.bntx import decode_bntx_file, to_image

FIX = Path(__file__).parent / "fixtures" / "pm0001_00_00"
FILES = sorted(FIX.glob("*.bntx"))


def _one(name):
    texs = decode_bntx_file(FIX / name)
    assert len(texs) == 1
    return texs[0]


def _pow2(n):
    return n > 0 and n & (n - 1) == 0


def test_fixtures_present():
    assert FILES


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_decode_all(path):
    for t in decode_bntx_file(path):
        assert _pow2(t.width) and _pow2(t.height)
        assert t.rgba.shape == (t.height, t.width, 4) and t.rgba.dtype == np.uint8
        assert to_image(t).size == (t.width, t.height)


def test_albedo_is_structured_not_noise():
    t = _one("pm0001_00_00_body_a_alb.bntx")
    assert t.srgb and t.format_name == "BC7_SRGB"
    rgb = t.rgba[..., :3].astype(np.float32)
    # ZA albedo maps are near-white/neutral (tint comes from material colour)
    assert rgb.mean() > 150
    # Swizzle errors give high-frequency noise: neighbouring pixels must be similar.
    grey = rgb.mean(-1)
    assert np.abs(np.diff(grey, axis=1)).mean() < 5
    # ...yet the image is not flat: coarse 8x8 block means vary.
    h, w = grey.shape
    coarse = grey.reshape(8, h // 8, 8, w // 8).mean((1, 3))
    assert 1.0 < coarse.std() < 60


def test_body_b_albedo_has_colour_regions():
    t = _one("pm0001_00_00_body_b_alb.bntx")
    rgb = t.rgba[..., :3].astype(np.float32)
    assert (rgb[..., 0] - rgb[..., 2]).max() > 50  # reddish patches (mouth/skin)


def test_normal_map():
    t = _one("pm0001_00_00_body_a_nrm.bntx")
    assert t.format_name == "BC5_UNORM"
    assert t.rgba[..., 2].mean() / 255 > 0.5
    assert abs(t.rgba[..., 0].mean() / 255 - 0.5) < 0.1
    assert np.abs(np.diff(t.rgba[..., 0].astype(np.float32), axis=1)).mean() < 5
