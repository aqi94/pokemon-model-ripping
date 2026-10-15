Per-character import overrides for BDSP trainers: `<model id>.json`, e.g. `tr1037_00.json`. Read by `importer.py` (`_overrides`).

Keys (all optional):
- `lens_opacity`: opacity factor for clear lens meshes (`*glassSkin` / `*gogglelensSkin` with a high-alpha lens area). Default 0.25.
- `lens_tint`: RGB multiplier for those lenses (0.5 = smoked grey glass). Default 1.0.
