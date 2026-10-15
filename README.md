# pokemon-model-ripping

A pipeline that converts ripped Pokémon game models into one common format (rigged glTF with every animation as a named clip, plus a `manifest.json`), the Blender-community formats derived from it (FBX, OBJ+MTL, PNG textures), and a web viewer that reads the common format. Each game has its own importer under `pipeline/importers/`; nothing downstream of an importer knows which game an asset came from.

Games so far: Pokémon Legends: Z-A, Scarlet & Violet, Brilliant Diamond / Shining Pearl, Pokémon GO (trainers). See `CLAUDE.md` for architecture and format notes.

## What is not included

No models, textures, renders or game data are in this repository. You supply your own ripped data (the source zips / extracted bundles) and the external tools: a Python 3.14 venv at `.venv`, Blender (for FBX/OBJ export, thumbnails and GO FBX reading), and Node. The test fixtures (`pipeline/tests/fixtures/`) are also ripped data and are omitted, so fixture-based tests will not run without your own copies. Default data paths are hard-coded in `pipeline/catalog/*.py`; override with `--data`.

## Commands

```bash
# Python
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m pipeline list --kind pokemon
./.venv/Scripts/python.exe -m pipeline convert --id pm0001_00_00 --id tr0018_00     # -> out/za/<id>/{model.glb,manifest.json}
./.venv/Scripts/python.exe -m pipeline convert --all --kind pokemon --resume
./.venv/Scripts/python.exe -m pipeline export --id pm0001_00_00                      # FBX/OBJ/MTL/PNG via Blender
./.venv/Scripts/python.exe -m pipeline thumbs                                        # thumbnails via Blender, then reindex
./.venv/Scripts/python.exe -m pipeline index                                         # rebuild out/index.json
./.venv/Scripts/python.exe -m pipeline --game sv list --kind pokemon                 # other games: --game sv|go|bdsp

# Viewer (viewer/): npm install; npm run dev (http://localhost:5173, serves ../out at /models); npm run build
```

`out/` (git-ignored) holds the converted output and can reach tens of GB.
