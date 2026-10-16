"""Tests for the pure-numpy BC6H decoder (pipeline/importers/bntx/bc6h.py)."""
import time

import numpy as np
import pytest

from pipeline.importers.bntx import bc6h
from pipeline.importers.bntx.bc6h import FIXUP, MODES, PARTITIONS, decode_bc6h, decode_bc6h_blocks

# Real BC6H_SF16 blocks from Switch BNTX light probes (format 0x1F05).
DIFFUSE_PROBE = [
    "4b7ef9e5f7c71f7f04006824ac79ffce",
    "0b7ef8e1efa79f7e38127856ab9aefcd",
    "8b7efae9efa79f7e4c129a57cd9affce",
    "2bfef8e3efa79f7e3a127946bd8affce",
]
SPECULAR_PROBE = [
    "8b4ca89f1fe000002a017734ab68efab",
    "ebccab9f37a080002a258956ce8bffde",
]
SAMPLES = DIFFUSE_PROBE + SPECULAR_PROBE

F16MAX = 0x7BFF


# ---------------------------------------------------------------------------
# helpers: a bit writer and the spec's quantizer so blocks can be hand-encoded
# ---------------------------------------------------------------------------
class _BitWriter:
    def __init__(self):
        self.value = 0
        self.pos = 0

    def write(self, v, n):
        assert 0 <= v < (1 << n), (v, n)
        self.value |= v << self.pos
        self.pos += n

    def bytes(self):
        assert self.pos == 128, self.pos
        return self.value.to_bytes(16, "little")


def _quantize(half_bits, prec, signed):
    """DirectXTex Quantize(): half-float bit pattern -> prec-bit endpoint value."""
    if signed:
        s = half_bits & 0x8000
        mag = min(half_bits & 0x7FFF, F16MAX)
        q = mag if prec >= 16 else (mag << (prec - 1)) // (F16MAX + 1)
        return (-q) & ((1 << prec) - 1) if s else q
    assert not half_bits & 0x8000
    return half_bits if prec >= 15 else (half_bits << prec) // (F16MAX + 1)


def _half_bits(x):
    return int(np.array(x, dtype="<f2").view("<u2"))


def _mode11_block(ep0, ep1, indices, signed):
    """Mode 11 (0b00011): 10.10, one subset, no delta.  ep0/ep1 = RGB floats, indices = 16 ints 0..15
    (texel 0 has only 3 bits: its MSB is implicitly 0, so it can address at most index 7)."""
    w = _BitWriter()
    w.write(0b00011, 5)
    for c in range(3):
        w.write(_quantize(_half_bits(ep0[c]), 10, signed), 10)
    for c in range(3):
        w.write(_quantize(_half_bits(ep1[c]), 10, signed), 10)
    w.write(indices[0], 3)
    for i in indices[1:]:
        w.write(i, 4)
    return w.bytes()


WEIGHTS4 = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64]


def _ref_texel(x0, x1, bits, signed, index):
    """Scalar transcription of the spec: value -> quantize -> unquantize -> interpolate -> finish -> half."""

    def unq(q):
        if not signed:
            if bits >= 15:
                return q
            if q == 0:
                return 0
            if q == (1 << bits) - 1:
                return 0xFFFF
            return ((q << 16) + 0x8000) >> bits
        if bits >= 16:
            return q
        s, q = q < 0, abs(q)
        if q == 0:
            u = 0
        elif q >= (1 << (bits - 1)) - 1:
            u = 0x7FFF
        else:
            u = ((q << 15) + 0x4000) >> (bits - 1)
        return -u if s else u

    def sext(q):
        return q - (1 << bits) if q & (1 << (bits - 1)) else q

    q0, q1 = _quantize(_half_bits(x0), bits, signed), _quantize(_half_bits(x1), bits, signed)
    if signed:
        q0, q1 = sext(q0), sext(q1)
    a, b = unq(q0), unq(q1)
    w = WEIGHTS4[index]
    v = (a * (64 - w) + b * w + 32) >> 6
    if signed:
        m = (abs(v) * 31) >> 5
        h = (0x8000 | m) if v < 0 else m
    else:
        h = (v * 31) >> 6
    return float(np.frombuffer(np.array([h], "<u2").tobytes(), "<f2")[0])


# ---------------------------------------------------------------------------
# (1) real probe blocks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hexblock", SAMPLES)
def test_sample_blocks_decode_to_plausible_radiance(hexblock):
    img = decode_bc6h(bytes.fromhex(hexblock), 4, 4, signed=True)
    assert img.shape == (4, 4, 3) and img.dtype == np.float32
    assert np.isfinite(img).all()
    assert img.min() >= -1.0 and img.max() <= 10.0
    # ambient / sky probes: positive, soft radiance
    assert img.min() > 0.0 and img.max() < 2.0


def test_sample_blocks_are_not_flat():
    spreads = []
    for hexblock in SAMPLES:
        img = decode_bc6h(bytes.fromhex(hexblock), 4, 4, signed=True)
        spreads.append(float(img.max() - img.min()))
    assert max(spreads) > 0.001
    # the probe blocks are all mode 13 (12.8) with varying indices: none is flat
    assert all(s > 1e-6 for s in spreads)


def test_sample_blocks_are_sf16_not_uf16():
    # Decoding the same bits as unsigned gives near-black (~0.006), signed gives ~1.2: the probes are SF16.
    block = bytes.fromhex(DIFFUSE_PROBE[0])
    assert decode_bc6h(block, 4, 4, signed=True).mean() > 1.0
    assert decode_bc6h(block, 4, 4, signed=False).mean() < 0.05


def test_diffuse_probe_is_neutral_and_smooth():
    for hexblock in DIFFUSE_PROBE:
        img = decode_bc6h(bytes.fromhex(hexblock), 4, 4, signed=True)
        assert np.allclose(img[..., 0], img[..., 1]) and np.allclose(img[..., 1], img[..., 2])
        # neighbouring texels differ by less than 1 %: a gradient, not noise
        assert np.abs(np.diff(img[..., 0], axis=0)).max() < 0.02
        assert np.abs(np.diff(img[..., 0], axis=1)).max() < 0.02


def test_specular_probe_is_sky_blue():
    for hexblock in SPECULAR_PROBE:
        rgb = decode_bc6h(bytes.fromhex(hexblock), 4, 4, signed=True).reshape(-1, 3).mean(0)
        assert rgb[2] > rgb[1] > rgb[0]
        assert 0.8 < rgb[2] < 1.0


# ---------------------------------------------------------------------------
# (2) hand-encoded one-subset blocks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("signed", [False, True])
def test_mode11_endpoints_round_trip(signed):
    lo, hi = (0.5, 0.5, 0.5), (1.0, 1.0, 1.0)
    img0 = decode_bc6h(_mode11_block(lo, hi, [0] * 16, signed), 4, 4, signed=signed)
    img15 = decode_bc6h(_mode11_block(lo, hi, [7] + [15] * 15, signed), 4, 4, signed=signed)
    # exact agreement with the scalar spec maths
    assert (img0 == _ref_texel(0.5, 1.0, 10, signed, 0)).all()
    assert (img15.reshape(-1, 3)[1:] == _ref_texel(0.5, 1.0, 10, signed, 15)).all()
    assert (img15[0, 0] == _ref_texel(0.5, 1.0, 10, signed, 7)).all()
    if not signed:
        # unsigned 10-bit: 1.0 -> q 495 -> 0x3C00 exactly; 0.5 -> q 462 -> 0x3801 (one half ulp off,
        # 0x3800 is not reachable through the (x*31)>>6 finish step)
        assert (img15.reshape(-1, 3)[1:] == 1.0).all()
        assert np.abs(img0 - 0.5).max() <= 2 ** -11
    else:
        # signed 10-bit keeps 9 magnitude bits (step 62 half-bit units ~ 3 % at 0.5): 0.5 -> 0x3811, 1.0 -> 0x3BF1
        assert np.abs(img0 - 0.5).max() < 0.5 * 0.035
        assert np.abs(img15.reshape(-1, 3)[1:] - 1.0).max() < 0.035


@pytest.mark.parametrize("signed", [False, True])
def test_mode11_interpolation_is_monotonic(signed):
    lo, hi = (0.25, 0.5, 0.75), (1.0, 2.0, 4.0)
    block = _mode11_block(lo, hi, [0] + list(range(1, 16)), signed)
    tex = decode_bc6h_blocks(np.frombuffer(block, np.uint8), signed)[0]  # (16,3)
    assert (np.diff(tex, axis=0) > 0).all()
    for c in range(3):
        for i in range(16):
            assert tex[i, c] == _ref_texel(lo[c], hi[c], 10, signed, i)
    assert np.allclose(tex[0], lo, rtol=0.035) and np.allclose(tex[15], hi, rtol=0.035)


def test_mode14_exact_values_and_delta():
    # 16-bit endpoints: unsigned unquantize is the identity, so 29598 * 31 >> 6 == 0x3800 == 0.5 exactly
    q = 29598
    assert (q * 31) >> 6 == 0x3800
    w = _BitWriter()
    w.write(0b01111, 5)
    for _ in range(3):
        w.write(q & 0x3FF, 10)
    for _ in range(3):
        w.write(3, 4)  # delta +3 -> 29601 -> 29601*31>>6 = 14337 = 0x3801
        for bit in range(15, 9, -1):  # rw[10:15] is stored MSB first
            w.write((q >> bit) & 1, 1)
    w.write(0, 3)
    for i in range(1, 16):
        w.write(15 if i >= 8 else 0, 4)
    img = decode_bc6h(w.bytes(), 4, 4, signed=False)
    assert (img[:2] == 0.5).all()
    assert (img[2:] == np.float32(np.frombuffer(np.array([0x3801], "<u2").tobytes(), "<f2")[0])).all()


def test_mode14_negative_delta_wraps_mod_2_16_unsigned():
    # delta -1 from endpoint 0 == 0 wraps to 0xFFFF (65535 * 31 >> 6 = 0x7BFF = 65504, half max)
    w = _BitWriter()
    w.write(0b01111, 5)
    for _ in range(3):
        w.write(0, 10)
    for _ in range(3):
        w.write((-1) & 0xF, 4)
        for _ in range(6):
            w.write(0, 1)
    w.write(0, 3)
    for _ in range(15):
        w.write(15, 4)
    img = decode_bc6h(w.bytes(), 4, 4, signed=False)
    assert img[0, 0, 0] == 0.0 and img[3, 3, 0] == 65504.0


def test_signed_negative_values():
    lo, hi = (-0.5, -1.0, -2.0), (0.5, 1.0, 2.0)
    img0 = decode_bc6h(_mode11_block(lo, hi, [0] * 16, True), 4, 4, signed=True)
    img15 = decode_bc6h(_mode11_block(lo, hi, [7] + [15] * 15, True), 4, 4, signed=True)
    assert np.allclose(img0[0, 0], lo, rtol=0.035)
    assert np.allclose(img15[3, 3], hi, rtol=0.035)
    for c in range(3):
        assert img0[0, 0, c] == _ref_texel(lo[c], hi[c], 10, True, 0)
        assert img15[3, 3, c] == _ref_texel(lo[c], hi[c], 10, True, 15)
    # the midpoint of a symmetric pair is ~0 (index 8 = weight 34/64)
    mid = decode_bc6h(_mode11_block(lo, hi, [0] + [8] * 15, True), 4, 4, signed=True)[3, 3]
    assert np.abs(mid).max() < 0.1


# ---------------------------------------------------------------------------
# (3) unsigned and signed agree for positive values
# ---------------------------------------------------------------------------
def test_unsigned_and_signed_agree_for_positive_values():
    # The same bit pattern does NOT mean the same thing in UF16 and SF16 (a signed 10-bit endpoint
    # keeps only 9 magnitude bits, and the finish step scales by 31/32 instead of 31/64), so encode
    # the same positive colours with each variant's quantizer and check both decoders reproduce
    # them to within the coarser (signed) quantization step.
    lo, hi = (0.5, 0.25, 0.125), (1.0, 1.5, 3.0)
    for indices in ([0] * 16, [7] + [15] * 15, [0] + list(range(1, 16))):
        u = decode_bc6h(_mode11_block(lo, hi, indices, False), 4, 4, signed=False)
        s = decode_bc6h(_mode11_block(lo, hi, indices, True), 4, 4, signed=True)
        assert np.allclose(u, s, rtol=0.04, atol=1e-4)
        assert (s >= 0).all()
    # unsigned quantization has one more magnitude bit than signed, so it sits closer to the target
    u0 = decode_bc6h(_mode11_block(lo, hi, [0] * 16, False), 4, 4, signed=False)[0, 0]
    s0 = decode_bc6h(_mode11_block(lo, hi, [0] * 16, True), 4, 4, signed=True)[0, 0]
    assert np.abs(u0 / np.array(lo) - 1).max() < 0.02
    assert np.abs(s0 / np.array(lo) - 1).max() < 0.035


# ---------------------------------------------------------------------------
# tables, layouts, robustness, performance
# ---------------------------------------------------------------------------
def test_mode_layout_tables_self_consistent():
    # _ModeInfo validates header bit counts and per-field widths on construction
    assert sorted(MODES) == list(range(1, 15))
    for m, info in MODES.items():
        assert info.header_bits == (82 if info.subsets == 2 else 65)


def test_partition_and_fixup_tables():
    assert PARTITIONS.shape == (64, 16) and FIXUP.shape == (64,)
    # texel 0 always belongs to subset 0, the fix-up texel to subset 1, both subsets used
    assert (PARTITIONS[:, 0] == 0).all()
    assert (PARTITIONS[np.arange(64), FIXUP] == 1).all()
    assert (PARTITIONS.sum(axis=1) > 0).all() and (PARTITIONS.sum(axis=1) < 16).all()
    assert len({tuple(r) for r in PARTITIONS}) == 64
    assert set(FIXUP.tolist()) == {2, 6, 8, 15}


def test_random_blocks_all_modes_finite():
    rng = np.random.default_rng(1234)
    blocks = rng.integers(0, 256, (4096, 16), dtype=np.uint8)
    modes = bc6h._mode_of(blocks)
    assert set(range(1, 15)) <= set(modes.tolist())
    for signed in (False, True):
        out = decode_bc6h_blocks(blocks, signed)
        assert out.shape == (4096, 16, 3) and np.isfinite(out).all()
        assert (np.abs(out) <= 65504.0).all()
        if not signed:
            assert (out >= 0).all()
    # reserved mode patterns decode to zero
    reserved = np.zeros((4, 16), dtype=np.uint8)
    reserved[:, 0] = [0b10011, 0b10111, 0b11011, 0b11111]
    assert (decode_bc6h_blocks(reserved, True) == 0).all()


def test_image_layout_and_cropping():
    rng = np.random.default_rng(7)
    data = rng.integers(0, 256, 2 * 3 * 16, dtype=np.uint8).tobytes()  # 8x12 texels = 2x3 blocks
    img = decode_bc6h(data, 8, 12, signed=True)
    tex = decode_bc6h_blocks(np.frombuffer(data, np.uint8), True)
    assert img.shape == (12, 8, 3)
    # block (by=2, bx=1) texel (3,2) -> pixel (11, 6)
    assert np.array_equal(img[11, 6], tex[2 * 2 + 1, 3 * 4 + 2])
    assert decode_bc6h(data, 6, 10, signed=True).shape == (10, 6, 3)
    with pytest.raises(ValueError):
        decode_bc6h(data[:-1], 8, 12)


def test_performance_64x64_face():
    rng = np.random.default_rng(99)
    data = rng.integers(0, 256, 256 * 16, dtype=np.uint8).tobytes()
    decode_bc6h(data, 64, 64, signed=True)  # warm up
    t0 = time.perf_counter()
    decode_bc6h(data, 64, 64, signed=True)
    assert time.perf_counter() - t0 < 0.5
