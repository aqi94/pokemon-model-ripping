import zlib
from types import SimpleNamespace as NS

import numpy as np

from pipeline.importers.bdsp import importer as B


def test_mirror_x_conversion():
    assert B._p(NS(x=1.0, y=2.0, z=3.0)) == (-1.0, 2.0, 3.0)
    assert B._q(NS(x=0.1, y=0.2, z=0.3, w=0.9)) == (0.1, -0.2, -0.3, 0.9)
    m = NS(**{f"e{r}{c}": float(r * 4 + c) for r in range(4) for c in range(4)})
    out = B._mat(m)
    assert out[0, 0] == 0.0 and out[0, 1] == -1.0 and out[1, 0] == -4.0 and out[1, 1] == 5.0 and out[1, 3] == 7.0


def test_bake_tint_modes():
    albedo = np.full((2, 2, 4), 200, np.uint8)
    mask = np.zeros((2, 2, 4), np.uint8)
    mask[0, :, 0] = 255  # R region
    mask[1, :, 1] = 255  # G region
    P, S, K = (0.5, 0.5, 0.5), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    z = B.bake_tint(albedo, mask, P, S, K, "zones")
    assert tuple(z[0, 0, :3]) == (100, 100, 100) and tuple(z[1, 0, :3]) == (200, 0, 0)
    assert tuple(B.bake_tint(albedo, mask, P, S, K, "hair")[1, 1, :3]) == (100, 100, 100)
    sk = B.bake_tint(albedo, mask, P, S, K, "skin")
    assert tuple(sk[0, 0, :3]) == (0, 200, 0) and tuple(sk[1, 0, :3]) == (200, 200, 200)
    assert z[0, 0, 3] == 200  # alpha untouched


def test_streamed_clip_keys_and_negative_first_frame():
    f = lambda x: int(np.float32(x).view(np.uint32))  # noqa: E731
    words = [f(-3.4e38), 1, 0, 0, 0, 0, f(1.0),  # frame 0 stamped -FLT_MAX: curve 0 = 1.0
             f(0.0), 1, 0, 0, 0, 0, f(2.0),      # frame at t=0 replaces it
             f(0.5), 1, 0, 0, 0, 0, f(3.0)]
    clip = NS(m_MuscleClip=NS(m_Clip=NS(data=NS(m_StreamedClip=NS(data=words)))))
    ts, vs = B._streamed(clip)[0]
    assert ts == [0.0, 0.5] and vs == [2.0, 3.0]


def test_path_hash_matches_unity_crc32():
    assert zlib.crc32(b"Waist") == zlib.crc32("Origin/Waist".split("/", 1)[1].encode())
