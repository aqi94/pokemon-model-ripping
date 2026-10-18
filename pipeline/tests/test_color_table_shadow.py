import numpy as np

from pipeline.importers.za.importer import apply_color_table, table_shadow_cell, table_shadow_rgb


def test_bright_shadow_under_black_becomes_dark():
    # Schoolkids F (cc_bu face0003): lit #252626 brows over a green (80,138,90) shadow cell.
    out = table_shadow_rgb([37, 38, 38], [80, 138, 90])
    assert max(out) <= 40
    assert max(out) - min(out) <= 2  # the lit grey, darker; no green tint


def test_black_lit_gives_black_shadow():
    assert table_shadow_rgb([0, 0, 0], [89, 67, 62]) == [0, 0, 0]


def test_darker_shadow_is_unchanged():
    lit = np.asarray([0.5, 0.4, 0.3], np.float32)
    sh = np.asarray([0.2, 0.1, 0.1], np.float32)
    assert np.array_equal(table_shadow_cell(lit, sh), sh)


def test_apply_color_table_clamps_shadow_row():
    # 2 zones x 4 rows: zone 1 dark grey lit over a green shadow, zone 2 tan over a darker tan.
    table = np.zeros((4, 2, 4), np.uint8)
    table[:2, 0, :3] = (37, 38, 38)
    table[2:, 0, :3] = (80, 138, 90)
    table[:2, 1, :3] = (160, 142, 128)
    table[2:, 1, :3] = (131, 91, 87)
    table[..., 3] = 255
    alb = np.full((2, 2, 4), 255, np.uint8)
    mask = np.zeros((2, 2, 4), np.uint8)
    mask[..., 0] = 255  # layer 1 everywhere
    lit, shadow = apply_color_table(alb, mask, table, 2, [1, 1, 1, 1], masked_only=True, scales=[1, 0, 0, 0])
    assert tuple(lit[0, 0, :3]) == (37, 38, 38)
    assert int(shadow[0, 0, :3].max()) <= 40  # not green
    _, shadow2 = apply_color_table(alb, mask, table, 2, [2, 1, 1, 1], masked_only=True, scales=[1, 0, 0, 0])
    assert tuple(shadow2[0, 0, :3]) == (131, 91, 87)  # real shadow tone untouched
