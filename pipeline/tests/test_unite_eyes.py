import numpy as np
import pytest

from pipeline.importers.unite.importer import eye_cell_shifts


def _sheet(cols, rows, cell=16, blank=(), red=()):
    a = np.full((rows * cell, cols * cell, 4), (200, 160, 60, 255), np.uint8)
    for r in range(rows):
        for c in range(cols):
            y, x = r * cell, c * cell
            if (c, r) in red:
                a[y:y + cell, x:x + cell, :3] = (255, 0, 0)
            elif (c, r) not in blank:
                a[y + 4:y + 8, x + 4 + c % 3:x + 10, :3] = (10, 10, 10 * (r + 1))  # a drawing unique per cell
                a[y + 5:y + 7, x + 3:x + 5, :3] = (r * 20, c * 20, 0)
    return a


def test_two_by_two_single_eye_shifts_over_every_cell():
    cols, rows, shifts = eye_cell_shifts([(0.05, 0.05, 0.45, 0.45)], _sheet(2, 2))
    assert (cols, rows) == (2, 2)
    assert sorted(shifts) == [(0, 1), (1, 0), (1, 1)]


def test_pair_spanning_both_columns_only_moves_by_rows():
    cols, rows, shifts = eye_cell_shifts([(0.05, 0.02, 0.45, 0.2), (0.55, 0.02, 0.95, 0.2)], _sheet(2, 4))
    assert (cols, rows) == (2, 4)
    assert sorted(shifts) == [(0, 1), (0, 2), (0, 3)]


def test_blank_and_unused_marker_cells_are_skipped():
    _, _, shifts = eye_cell_shifts([(0.05, 0.02, 0.45, 0.2)], _sheet(2, 4, blank={(1, 0)}, red={(1, 3)}))
    assert (1, 0) not in shifts and (1, 3) not in shifts
    assert len(shifts) == 5


def test_rest_jaws_drops_small_jaw_motion_only():
    from pipeline.common.model import Clip, Joint, Model, Skeleton, Track
    from pipeline.importers.unite.importer import rest_jaws

    def q(deg):
        h = np.radians(deg) / 2
        return np.array([[0, 0, np.sin(h), np.cos(h)]], np.float32)

    j = Joint(name="Jaw_ACC", parent=-1, translation=(0, 0, 0), rotation=(0, 0, 0, 1), scale=(1, 1, 1), inverse_bind=np.eye(4, dtype=np.float32))
    m = Model(id="x", game="unite", display_name="x", kind="pokemon", skeleton=Skeleton(joints=[j], palette=[0]), meshes=[], materials={}, textures={})
    t = np.zeros(1, np.float32)
    m.clips = [Clip(name="idle", tracks=[Track("Jaw_ACC", "rotation", t, q(15))]), Clip(name="atk", tracks=[Track("Jaw_ACC", "rotation", t, q(50))])]
    assert rest_jaws(m) == 1
    assert not m.clips[0].tracks and len(m.clips[1].tracks) == 1


def _jaw_model(clips):
    from pipeline.common.model import Clip, Joint, Model, Skeleton, Track

    def q(deg):
        h = np.radians(np.atleast_1d(deg)) / 2
        return np.stack([np.zeros_like(h), np.sin(h), np.zeros_like(h), np.cos(h)], -1).astype(np.float32)

    j = Joint(name="Bone_Jaw", parent=-1, translation=(0, 0, 0), rotation=(0, 0, 0, 1), scale=(1, 1, 1), inverse_bind=np.eye(4, dtype=np.float32))
    m = Model(id="x", game="unite", display_name="x", kind="pokemon", skeleton=Skeleton(joints=[j], palette=[0]), meshes=[], materials={}, textures={})
    m.clips = [Clip(name=n, tracks=[Track("Bone_Jaw", "rotation", np.arange(len(np.atleast_1d(d)), dtype=np.float32), q(d))]) for n, d in clips]
    return m


def test_rest_jaws_open_bind_keeps_closed_clips():
    # Gengar: the bind pose is ~9 deg open, every clip holds the jaw shut at -9, attacks open to +35
    from pipeline.importers.unite.importer import rest_jaws
    m = _jaw_model([("idle", [-9.0] * 10), ("run", [-9.0] * 10), ("scoreidle", [-9.0, -3.0, 6.0]),
                    ("atk", [-9.0, 10.0, 35.0, -9.0])])
    assert rest_jaws(m) == 1
    idle, run, score, atk = (c.tracks[0] for c in m.clips)
    assert len(idle.values) == 10 and len(run.values) == 10 and len(atk.values) == 4
    assert len(score.values) == 1  # held at the closed key
    assert np.degrees(2 * np.arcsin(score.values[0][1])) == pytest.approx(-9.0, abs=0.1)
    jaw = m.extras["jaws"][0]
    assert jaw["degrees"] == pytest.approx(44.0, abs=0.1)
    assert np.degrees(2 * np.arcsin(jaw["open"][1])) == pytest.approx(44.0, abs=0.1)  # from -9 toward +35
    assert m.extras["jawClosed"]["Bone_Jaw"] == pytest.approx(-9.0, abs=0.1)


def test_rest_jaws_without_an_opening_clip_changes_nothing():
    # lobby: one idle clip near the closed pose; which side opens is unknown
    from pipeline.importers.unite.importer import rest_jaws
    m = _jaw_model([("idle", [-6.8, -5.6, -6.5])])
    assert rest_jaws(m) == 0
    assert len(m.clips[0].tracks[0].values) == 3 and "jaws" not in m.extras


def test_grid_comes_from_the_sheet_not_the_island_size():
    # 2 x 4 sheet (one drawn eye per cell, opaque background) but the eye mesh only fills 1/8 of the height: the grid
    # must stay 2 x 4 (the island alone would suggest 2 x 8 and offer half-cell duds). Lapras lobby.
    from pipeline.importers.unite.importer import eye_cell_shifts
    img = np.zeros((512, 256, 4), np.uint8)
    img[...] = (98, 181, 230, 255)
    for r in range(4):
        for c in range(2):
            img[r * 128 + 30:r * 128 + 70, c * 128 + 30:c * 128 + 70 + 6 * (r * 2 + c)] = (10, 10, 10, 255)
    img[384:, 128:] = (255, 0, 0, 255)
    cols, rows, shifts = eye_cell_shifts([(0.05, 0.06, 0.45, 0.19)], img)
    assert (cols, rows) == (2, 4)
    assert sorted(shifts) == [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
