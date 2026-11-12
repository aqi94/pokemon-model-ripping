"""fb recorder (touched vs present fields), FieldAudit on a synthetic buffer, ledger merge keeping hand notes."""
from types import SimpleNamespace

import pytest

flatbuffers = pytest.importorskip("flatbuffers")

from pipeline import audit_fields as A  # noqa: E402
from pipeline.importers.za import fb  # noqa: E402


def make_buf() -> bytes:
    """root {0: u32 7, 1: string "hello", 2: [ {0: f32 1.5, 1: u8 9}, {0: f32 2.5, 1: u8 9} ], 3: f32 0.25}"""
    b = flatbuffers.Builder(256)
    s = b.CreateString("hello")
    kids = []
    for f in (1.5, 2.5):
        b.StartObject(2)
        b.PrependFloat32Slot(0, f, 0.0)
        b.PrependUint8Slot(1, 9, 0)
        kids.append(b.EndObject())
    b.StartVector(4, len(kids), 4)
    for k in reversed(kids):
        b.PrependUOffsetTRelative(k)
    vec = b.EndVector()
    b.StartObject(4)
    b.PrependUint32Slot(0, 7, 0)
    b.PrependUOffsetTRelativeSlot(1, s, 0)
    b.PrependUOffsetTRelativeSlot(2, vec, 0)
    b.PrependFloat32Slot(3, 0.25, 0.0)
    b.Finish(b.EndObject())
    return bytes(b.Output())


def parse_testfmt(data):
    """Reads root field 0 and the children's field 0 only."""
    r = fb.root(data)
    return r.u32(0), [t.f32(0) for t in r.tables(2)]


def test_recorder_paths_and_restore():
    orig = fb.Table._field_offset
    with fb.recording() as rec:
        assert parse_testfmt(make_buf()) == (7, [1.5, 2.5])
    assert fb.Table._field_offset is orig and fb.root is fb._ORIG["root"]
    assert ("testfmt:", 0) in rec.touched and ("testfmt:", 2) in rec.touched
    assert ("testfmt:/2[]", 0) in rec.touched
    assert ("testfmt:", 1) not in rec.touched and ("testfmt:/2[]", 1) not in rec.touched
    (info,) = rec.take()
    assert info.fmt == "testfmt"
    assert sorted(set(info.tables.values())) == ["testfmt:", "testfmt:/2[]"]


def test_off_means_untouched_reader():
    # outside recording() nothing is logged and results are identical
    assert parse_testfmt(make_buf()) == (7, [1.5, 2.5])
    assert fb._REC is None


def test_field_layout_and_guess():
    data = make_buf()
    r = fb.root(data)
    lay = {i: (off, size) for i, off, size in fb.field_layout(r.buf, r.pos)}
    assert set(lay) == {0, 1, 2, 3}
    assert fb.guess_field(r.buf, *lay[1]) == ("str", "hello")
    assert fb.guess_field(r.buf, *lay[3]) == ("f32", 0.25)
    typ, (n, _) = fb.guess_field(r.buf, *lay[2])
    assert typ == "vec<table>" and n == 2


def test_audit_reports_present_but_unread():
    data = make_buf()
    cat = SimpleNamespace(import_ref=lambda ref, with_anim=True: parse_testfmt(data))
    refs = [SimpleNamespace(id="m1", kind="object"), SimpleNamespace(id="m2", kind="object")]
    audit = A.FieldAudit("test")
    audit.run(cat, refs, log=lambda *_: None)
    unread = dict(audit.unread_fields())
    assert set(unread) == {"testfmt:/1", "testfmt:/3", "testfmt:/2[]/1"}
    st = unread["testfmt:/2[]/1"]
    assert st.models == {"m1", "m2"} and st.instances == 4 and dict(st.values) == {"9": 4}
    assert A._auto_status(st) == "unread"  # only 2 models: not yet "constant"
    assert dict(unread["testfmt:/1"].values) == {"hello": 2}


def test_recdict_logs_lookups_only():
    A._RecDict.reads = set()
    d = A._RecDict({"A": 1, "B": 2}, ("trmtr", "float"))
    assert d.get("A") == 1 and "Z" not in d
    _ = {**d}  # copies are not reads
    assert A._RecDict.reads == {(("trmtr", "float"), "A"), (("trmtr", "float"), "Z")}


def test_neutral():
    assert A._neutral("flag", "False") and A._neutral("switch", False)
    assert A._neutral("color", (1, 1, 1, 0)) and A._neutral("color", (0, 0, 0, 0))
    assert not A._neutral("color", (0.5, 1, 1, 1)) and not A._neutral("float", 0.1)


def test_merge_keeps_hand_notes():
    old = {"entries": [
        {"key": "trmtr:/1[]/16", "status": "unknown_meaning", "notes": "seen on eyes", "hypothesis": "eye flag", "models": 3},
        {"key": "trmtr:/1[]/17", "status": "unread", "status_auto": True, "notes": "", "hypothesis": "", "models": 9},
        {"key": "trvat:/3", "status": "unread", "status_auto": True, "notes": "1.0 everywhere", "models": 1},
        {"key": "gone", "status": "unread", "status_auto": True, "notes": "", "models": 1},
    ]}
    new = [
        {"key": "trmtr:/1[]/16", "status": "unread", "status_auto": True, "models": 5, "values": [["1", 5]]},
        {"key": "trmtr:/1[]/17", "status": "constant", "status_auto": True, "models": 12, "values": [["0", 12]]},
    ]
    fitted = [{"key": "fitted:x.py:K", "games": ("za",), "where": "x.py:3", "text": "K on pm0001", "code": "K = 2"}]
    led = A.merge_ledger(old, new, {"game": "za"}, fitted)
    e = {x["key"]: x for x in led["entries"]}
    assert e["trmtr:/1[]/16"]["status"] == "unknown_meaning" and e["trmtr:/1[]/16"]["notes"] == "seen on eyes"
    assert e["trmtr:/1[]/16"]["hypothesis"] == "eye flag" and e["trmtr:/1[]/16"]["models"] == 5
    assert "status_auto" not in e["trmtr:/1[]/16"]
    assert e["trmtr:/1[]/17"]["status"] == "constant"  # scanner-chosen status follows the scan
    assert e["trvat:/3"]["seen"] is False and e["trvat:/3"]["notes"] == "1.0 everywhere"
    assert "gone" not in e
    assert e["fitted:x.py:K"]["status"] == "fitted" and e["fitted:x.py:K"]["comment"] == "K on pm0001"


def test_fitted_scan(tmp_path):
    p = tmp_path / "pipeline" / "importers" / "swsh"
    p.mkdir(parents=True)
    (p / "m.py").write_text("GAIN = 0.3  # FITTED: gain on pm0025 + pm0006, 2026-10-01\n", encoding="utf-8")
    (tmp_path / "viewer" / "src").mkdir(parents=True)
    (tmp_path / "viewer" / "src" / "a.ts").write_text("const B = 2 // FITTED(za,sv): band edge\n", encoding="utf-8")
    got = {f["where"]: f for f in A.scan_fitted(str(tmp_path))}
    assert got["pipeline/importers/swsh/m.py:1"]["games"] == ("swsh",)
    assert got["viewer/src/a.ts:1"]["games"] == ("za", "sv") and got["viewer/src/a.ts:1"]["text"] == "band edge"
