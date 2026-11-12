import numpy as np

from pipeline.importers.za import trinity as T
from pipeline.importers.za.importer import blend_shape_weights


def _shape(regions=(), names=(), decls=()):
    return T.MeshShape("face_base_mesh_shape", "face_base_mesh", 1, list(decls), [], [], list(regions), list(names))


def _decl():
    return T.VertexDeclaration([T.VertexElement(0, 1, 0, 51, 0), T.VertexElement(0, 2, 0, 43, 12), T.VertexElement(0, 3, 0, 43, 20)], [28])


def _records(pos: np.ndarray) -> bytes:
    raw = np.zeros((len(pos), 28), np.uint8)
    raw[:, :12] = np.frombuffer(pos.astype("<f4").tobytes(), np.uint8).reshape(-1, 12)
    return raw.tobytes()


def test_sparse_region_targets_are_deltas_and_skip_padding():
    base = np.zeros((5, 3), np.float32)
    idx = np.array([1, 3, 0xFFFFFFFF], "<u4")
    delta = np.array([[0, 0.01, 0], [0, -0.02, 0], [9, 9, 9]], np.float32)
    shape = _shape(regions=[("face_m1_mesh_shape", [T.VertexDeclaration([], [4]), _decl()], ["mouth_a_m1_face_mesh_shape"])])
    bufs = T.MeshBuffers([], [], [[memoryview(idx.tobytes()), memoryview(_records(delta))]])
    (name, d), = T.morph_targets(shape, bufs, base)
    assert name == "mouth_a_m1_face_mesh_shape"
    assert np.allclose(d[1], (0, 0.01, 0)) and np.allclose(d[3], (0, -0.02, 0)) and np.allclose(d[[0, 2, 4]], 0)


def test_dense_targets_store_absolute_positions():
    base = np.array([[0, 1.5, 0], [0.1, 1.5, 0]], np.float32)
    target = base + np.array([[0, 0.01, 0], [0, 0, 0]], np.float32)
    shape = _shape(names=["battle_00_m4_eyebrow_mesh_shape"], decls=[_decl(), _decl()])
    bufs = T.MeshBuffers([], [memoryview(b""), memoryview(_records(target))])
    (name, d), = T.morph_targets(shape, bufs, base)
    assert np.allclose(d, [[0, 0.01, 0], [0, 0, 0]])


def test_weights_follow_the_deformer_region():
    # blend_mouth covers region m1, blend_eyebrow m4: the same target name in both drives different morphs
    tracks = [
        T.BlendShapeTrack("blend_mouth:battle_00", np.array([0, 0, 30]), np.array([0.0, 0.0, 1.0], np.float32), ["face_m1_mesh_shape", "tooth_mesh_shape"]),
        T.BlendShapeTrack("blend_eyebrow:battle_00", np.array([0]), np.array([0.7], np.float32), ["face_m4_mesh_shape", "eyebrow_mesh_shape"]),
    ]
    out = blend_shape_weights(tracks, {"face_base_mesh": ["battle_00_m1", "battle_00_m4", "other_m2"], "tooth_mesh": ["battle_00_m1"]}, 30.0)
    times, w = out["face_base_mesh"]
    assert np.allclose(times, [0, 1]) and np.allclose(w[:, 0], [0, 1]) and np.allclose(w[:, 1], 0.7) and np.allclose(w[:, 2], 0)
    assert np.allclose(out["tooth_mesh"][1][:, 0], [0, 1])
