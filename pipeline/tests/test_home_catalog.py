from pipeline.catalog.home import HomeCatalog


def test_home_catalog_pairs_shiny_and_lists_objects(tmp_path):
    (tmp_path / "pokemons").mkdir()
    (tmp_path / "objects").mkdir()
    for n in ("pm0025_00_00", "pm0025_00_00_rare", "pm0006_51_00"):
        (tmp_path / "pokemons" / f"{n}.unity3d").write_bytes(b"")
    (tmp_path / "pokemons" / "dependencies").mkdir()
    (tmp_path / "objects" / "ob0100_00.unity3d").write_bytes(b"")
    cat = HomeCatalog(str(tmp_path))
    refs = {r.id: r for r in cat.pokemon()}
    assert set(refs) == {"pm0025_00_00", "pm0006_51_00"}  # `_rare` is a variant, dependencies/ is not a model
    assert refs["pm0025_00_00"].extras["rare"] == "pokemons/pm0025_00_00_rare.unity3d"
    assert refs["pm0006_51_00"].extras["rare"] is None
    assert refs["pm0025_00_00"].extras["dex"] == 25
    assert [r.id for r in cat.objects()] == ["ob0100_00"]
    assert cat.find("ob0100_00").kind == "object"
