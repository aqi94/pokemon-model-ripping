"""Z-A cutscene camera tracks (`ik_demo/**/*.trcma`), decoded but not imported anywhere.

root[0] = info {[1] frame_count, [2] fps (60)}; root[1] = camera table:
  [2] transform {slot 0 rotation (packed quaternion, same union as .tranm), slot 2 translation,
      slots 4/6 fixed zero vec3 (scale / pivot, unverified)},
  [1] lens {slot 0 near 0.1, slot 2 far 10000, slot 4 vertical FOV in radians (0.4975 = 28.5 deg)}.
The camera node is called `ubercam1`. Frame counts match the character clips of the same cut
(`d004_cam_c04` = 241 frames = `d004_tr0057_anim_c04`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fb
from .trinity import Channel, _channel


@dataclass
class Trcma:
    frame_count: int
    frame_rate: int
    rotation: Channel | None  # quaternions (x, y, z, w)
    translation: Channel | None
    near: float
    far: float
    fov_y: float  # radians


def parse_trcma(data: bytes) -> Trcma:
    root = fb.root(data)
    info, cam = root.table(0), root.table(1)
    if info is None or cam is None:
        raise ValueError("TRCMA missing info/camera tables")
    tf, lens = cam.table(2), cam.table(1)
    rot = _channel(tf, 0, True) if tf is not None and tf._field_offset(0) else None
    pos = _channel(tf, 2, False) if tf is not None and tf._field_offset(2) else None

    def lens_value(slot: int, default: float) -> float:
        ch = _channel(lens, slot, False) if lens is not None and lens._field_offset(slot) else None
        return float(ch.values[0][0]) if ch is not None else default

    return Trcma(info.u32(1), info.u32(2) or 60, rot, pos, lens_value(0, 0.1), lens_value(2, 10000.0), lens_value(4, 0.5))
