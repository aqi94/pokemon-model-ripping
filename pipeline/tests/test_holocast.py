from pipeline.common.model import Material
from pipeline.importers.za import trinity as T
from pipeline.importers.za.importer import attach_holocast, dedupe_holocast, holocast_label, holocast_name, holocast_params, is_holocast_set


def _tm(name, color=(0.223, 3.392, 3.392, 1.0), on=True, **floats):
    f = {"ScanlineSpeed": 2.0, "ScanlineDepth": 0.7, "ScanlineTiling": 260.0, "ScanGradationTiling": 1.0, "OcclusionStrength": 1.8, **floats}
    return T.TrMaterial(name, "IkCharacter", {"EnableHolocastEffect": "True"} if on else {}, [], [], f,
                        {"HolocastColor": color}, {}, "Opaque", -1)


def test_params_only_when_enabled():
    assert holocast_params(_tm("body", on=False)) is None
    p = holocast_params(_tm("body", color=(6.191, 0.442, 0.393, 1.0), ScanlineSpeed=1.0))
    assert p["color"] == [6.191, 0.442, 0.393] and p["scanlineSpeed"] == 1.0 and p["scanlineTiling"] == 260.0
    assert p["occlusionStrength"] == 1.8


def test_set_names_and_labels():
    assert all(is_holocast_set(n) for n in ("holo", "holo_red", "horo", "horo_red", "holo00", "holo01"))
    assert not any(is_holocast_set(n) for n in ("normal", "rare", "tex_00"))
    assert holocast_name("horo_red") == "holo_red" and holocast_name("holo00") == "holo"
    assert holocast_label("holo") == "Hologram" and holocast_label("holo_red") == "Hologram (red)" and holocast_label("holo01") == "Hologram 2"


class _Src:
    def __init__(self, files):
        self.files = files

    def exists(self, f):
        return f in self.files

    def read(self, f):
        return self.files[f]


def test_attach_uses_majority_and_tags_materials(monkeypatch):
    eye = (0.084, 2.946, 2.946, 1.0)
    tms = [_tm("body"), _tm("hair"), _tm("l_eye", color=eye)]
    monkeypatch.setattr(T, "parse_trmtr", lambda data: tms)
    mats = {n: Material(name=n) for n in ("body", "hair", "l_eye")}
    sets = [T.MaterialVariantSet("normal", [], {}), T.MaterialVariantSet("holo", ["m_00.trmtr"], {})]
    presets = attach_holocast(mats, sets, _Src({"m_00.trmtr": b""}), "m")
    assert [p["name"] for p in presets] == ["holo"] and presets[0]["color"] == [0.223, 3.392, 3.392]
    assert mats["l_eye"].extras["holocast"]["holo"]["color"] == [0.084, 2.946, 2.946]


def test_dedupe_near_identical_presets():
    a = {"name": "holo", "color": [0.2234, 3.3921, 3.3921], "scanlineSpeed": 2.0, "scanlineDepth": 0.7, "scanlineTiling": 260.0, "scanGradationTiling": 1.0}
    b = {**a, "name": "holo01", "color": [0.2239, 3.392, 3.392]}
    c = {**a, "name": "holo_red", "color": [6.19, 0.44, 0.39]}
    assert [p["name"] for p in dedupe_holocast([a, b, c])] == ["holo", "holo_red"]


def test_plain_set_label_names_non_cyan_hue():
    assert holocast_label("holo", [0.223, 3.392, 3.392]) == "Hologram"
    assert holocast_label("holo", [5.552, 0.399, 0.355]) == "Hologram (red)"
    assert holocast_label("holo", [3.181, 0.843, 3.629]) == "Hologram (magenta)"
