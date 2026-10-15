Per-character render overrides for `tools/bdsp_vs.py`: `<model id>.json`, e.g. `tr1037_00.json`.

Keys (all optional):
- `ball`: `"RItem1"` / `"LItem1"` (Poké Ball in that hand) or `null` (no ball).
- `ballscale`: ball scale (default 1).
- `fx`: `"police"` or `"psychic"` (light-effect sprites), `null` for none.
- `env`: lighting parameter overrides on top of the `bdspVs` preset (merged over `tools/bdsp_vs_lighting.json`).
