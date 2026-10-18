import numpy as np

from pipeline.importers.za.importer import clear_near_opaque_lens


def _tex(lens_alpha):
    t = np.full((4, 4, 4), 255, np.uint8)
    t[:2, :, 3] = lens_alpha
    return t


def test_near_opaque_lens_is_cleared_and_frame_kept():
    src = _tex(235)
    out = clear_near_opaque_lens(src, "Blend")
    assert (out[:2, :, 3] < 60).all() and (out[2:, :, 3] == 255).all()
    assert (src[:2, :, 3] == 235).all()  # shared decode cache untouched


def test_other_lenses_and_non_blend_untouched():
    for t, kind in ((_tex(77), "Blend"), (_tex(30), "Blend"), (_tex(235), "Opaque"), (_tex(235), None)):
        assert clear_near_opaque_lens(t, kind) is t
