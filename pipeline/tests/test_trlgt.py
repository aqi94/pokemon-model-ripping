"""`.trlgt` light-rig reader, on a FlatBuffer synthesised here (no game data needed)."""

import math
import struct

from pipeline.importers.za.trlgt import parse_trlgt, dump_rig


class _Builder:
    """Tiny back-to-front FlatBuffers builder: enough for tables of scalars/structs/strings/vectors."""

    def __init__(self):
        self.buf = bytearray()

    def _prepend(self, b: bytes) -> int:
        self.buf[0:0] = b
        return len(self.buf)  # "offset" = distance from buffer end

    def _align(self, n: int):
        while len(self.buf) % n:
            self.buf[0:0] = b"\0"

    def string(self, s: str) -> int:
        b = s.encode()
        while (len(self.buf) + len(b) + 1) % 4:  # pad after the terminator so the length prefix stays aligned
            self.buf[0:0] = b"\0"
        self._prepend(b"\0")
        self._prepend(b)
        return self._prepend(struct.pack("<I", len(b)))

    def table(self, fields: list) -> int:
        """fields: list of (kind, value) in schema order; kind in u32/f32/str/tbl/vec/f3/none."""
        # table body: inline scalars/structs and uoffsets; layout each field at increasing offset
        body = bytearray()
        offs = []
        for kind, val in fields:
            if kind == "none":
                offs.append(0)
                continue
            while len(body) % 4:
                body += b"\0"
            offs.append(len(body) + 4)
            if kind == "u32":
                body += struct.pack("<I", val)
            elif kind == "f32":
                body += struct.pack("<f", val)
            elif kind == "f3":
                body += struct.pack("<3f", *val)
            elif kind == "f4":
                body += struct.pack("<4f", *val)
            else:  # offset to an object already written (val = end-distance)
                body += struct.pack("<I", 0)  # patched below
        while len(body) % 4:
            body += b"\0"
        tlen = len(body) + 4
        vt = struct.pack("<HH", 4 + 2 * len(fields), tlen) + b"".join(struct.pack("<H", o) for o in offs)
        if len(vt) % 4:
            vt += b"\0\0"
        # write body then vtable in front of it; soffset in the table points back to the vtable
        self._align(4)
        self._prepend(bytes(body))
        table_end = len(self.buf)
        self._prepend(struct.pack("<i", 0))  # placeholder soffset
        table_off = len(self.buf)
        self._prepend(vt)
        vt_off = len(self.buf)
        # patch soffset: vtable position = table position - soffset
        tpos = len(self.buf) - table_off
        vpos = len(self.buf) - vt_off
        struct.pack_into("<i", self.buf, tpos, tpos - vpos)
        # patch uoffset fields
        for (kind, val), o in zip(fields, offs):
            if kind in ("str", "tbl", "vec"):
                fpos = tpos + o
                target = len(self.buf) - val
                struct.pack_into("<I", self.buf, fpos, target - fpos)
        del table_end
        return table_off

    def vector(self, offsets: list[int]) -> int:
        self._align(4)
        for off in reversed(offsets):
            pos_after = len(self.buf) + 4
            self._prepend(struct.pack("<I", 0))
            # uoffset from this slot to the target
            slot = len(self.buf) - pos_after
            target = len(self.buf) - off
            struct.pack_into("<I", self.buf, slot, target - slot)
        return self._prepend(struct.pack("<I", len(offsets)))

    def finish(self, root: int) -> bytes:
        self._align(4)
        self._prepend(struct.pack("<I", 0))
        struct.pack_into("<I", self.buf, 0, len(self.buf) - root)
        return bytes(self.buf)


def _rig_bytes() -> bytes:
    b = _Builder()
    f_int = b.table([("str", b.string("Intensity")), ("f32", 4.0)])
    f_cat = b.table([("str", b.string("CategoryLabel")), ("f32", 6.0)])
    v_col = b.table([("str", b.string("Color")), ("f4", (0.76, 0.85, 0.9, 0.0))])
    s_tex = b.table([("str", b.string("DiffuseProbe")), ("str", b.string("probemain_diffuse.bntx"))])
    srt = b.table([("f3", (1.0, 1.0, 1.0)), ("f3", (-0.7, 3.0, 0.0)), ("f3", (0.0, 0.0, 0.0))])
    sun = b.table([
        ("str", b.string("DirectionalMain")), ("str", b.string("Directional")), ("tbl", srt),
        ("vec", b.vector([f_int])), ("vec", b.vector([v_col])), ("vec", b.vector([s_tex])), ("u32", 4),
    ])
    d6 = b.table([
        ("str", b.string("dir_6")), ("str", b.string("Selectable")), ("none", None),
        ("vec", b.vector([f_cat, f_int])), ("vec", b.vector([v_col])), ("none", None), ("u32", 4),
    ])
    root = b.table([("u32", 0), ("vec", b.vector([sun, d6]))])
    return b.finish(root)


def test_parse_synthetic_rig():
    rig = parse_trlgt(_rig_bytes())
    assert rig.types() == ["DirectionalMain", "dir_6"]
    sun = rig.get("DirectionalMain")
    assert sun.name == "Directional" and sun.flags == 4
    assert sun.floats == {"Intensity": 4.0}
    assert all(abs(a - b) < 1e-6 for a, b in zip(sun.colors["Color"][:3], (0.76, 0.85, 0.9)))
    assert sun.textures == {"DiffuseProbe": "probemain_diffuse.bntx"}
    assert sun.scale == (1.0, 1.0, 1.0) and sun.translation == (0.0, 0.0, 0.0)
    assert abs(sun.rotation[0] + 0.7) < 1e-6 and abs(sun.rotation[1] - 3.0) < 1e-6
    d6 = rig.get("dir_6")
    assert d6.category == 6 and d6.rotation is None and d6.direction() is None
    assert rig.category(6)["dir"] is d6 and rig.category(6)["rim"] is None


def test_direction_and_angles():
    rig = parse_trlgt(_rig_bytes())
    sun = rig.get("DirectionalMain")
    d = sun.direction()
    assert abs(math.hypot(*d) - 1) < 1e-6
    # pitch -0.7 rad tilts the -Z forward axis down; yaw 3.0 rad swings it almost to +Z
    assert d[1] < -0.6 and d[2] > 0.7
    az, el = sun.angles()
    assert abs(el - math.degrees(0.7)) < 1e-3
    assert abs(az) > 165  # yaw 3.0 rad: the sun sits almost straight behind (+Z side is the model's front)


def test_dump_text():
    text = dump_rig(parse_trlgt(_rig_bytes()))
    assert "LIGHT type='DirectionalMain'" in text and "f Intensity = 4" in text and "el 40.1" in text
