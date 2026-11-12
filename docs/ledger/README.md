# Unread-value ledger

Principle: every number in the source data has a purpose until shown otherwise. This folder lists the values the
pipeline does not read (or reads without knowing what they mean), one JSON file per game, so missing materials,
effects and animation features can be found by data instead of by eye.

| file | source | how it is produced |
|---|---|---|
| `za.json`, `sv.json`, `la.json` | Trinity (`.trmdl/.trmsh/.trmbf/.trskl/.trmtr/.trmmt/.tranm/.tracm/.trvat`) | `audit-fields` |
| `swsh.json`, `lgpe.json` | `.gfbmdl` / `.gfbanm` (FlatBuffers, read through the same `fb.py`) | `audit-fields` |
| `bdsp.json` | Unity material properties | seeded by hand from `docs/bdsp_ignored_values.md` (no scanner for Unity yet) |
| `stadium2.json` | N64 geo layout / F3DEX2 / clip banks | seeded by hand from the importer survey (no scanner for N64 data yet) |

## Run

```bash
./.venv/Scripts/python.exe -m pipeline --game za audit-fields --sample 20            # 20 models per kind (+ their clips)
./.venv/Scripts/python.exe -m pipeline --game za audit-fields --sample 20 --id tr0566_00   # add specific models (crowd .trvat)
./.venv/Scripts/python.exe -m pipeline --game sv audit-fields --kind pokemon --sample 50 --no-write   # print only
./.venv/Scripts/python.exe -m pipeline.audit_materials --game za --kind trainer      # params-only report (old command)
```

Models are spread evenly over the sorted catalog. A 20-per-kind run takes from under a minute (SwSh, LGPE) to a
few minutes (Z-A/SV/LA, which parse every clip). Nothing is written to `out/`.

How "read" is decided (runtime coverage, never a source grep):

- **FlatBuffer fields.** `fb.recording()` patches the table accessors while the normal `cat.import_ref` runs and logs
  every `(table path, field index)` a parser asks for (`has()` counts). Afterwards every table the parsers reached is
  walked through its vtable, and a field that is present but was never asked for becomes an entry. Untouched
  sub-tables are walked heuristically (`fb.guess_field`, up to 3 levels, 4 elements per vector) and flagged
  `heuristic: true`. Their type guesses (`str`, `f32`, `vec<table>[n]`, `struct (...)`) can be wrong: a small
  integer can look like an offset.
- **Named material params** (`.trmtr` flags/colours/floats/ints, SwSh/LGPE material switches/floats/colours). The
  parsed name -> value dicts are swapped for recording dicts, so a param is read only if code looked it up (`get`,
  `[]`, `in`) during the import. Copies and iteration do not count. Neutral values (0, white, black, `False`, empty)
  are skipped. A **texture slot** is read when the importer opened its file, or (LGPE) looked the slot up.
- **`# FITTED:` / `// FITTED:` comments** in `pipeline/` and `viewer/src` become `fitted` entries. Write them as
  `# FITTED: <what> on <models>, <date>`, or as `# FITTED(za,sv): ...` to name the games. Without a tag the game
  comes from the path: `importers/za` applies to za, sv and la; `viewer/` and `common/` apply to every game.
  The comment text is stored in `comment` and refreshed on every scan. Removing the comment removes the entry,
  unless the entry has a hand-written `hypothesis` or `notes`.

"Read" means read by the import of the sampled models. A conditional read (`NumMaterialLayer` is consulted only when
a layer-5 colour exists) shows up as unread when no sampled model takes that branch. Raise `--sample` or add
`--id` before concluding.

## Read an entry

```json
{
 "key": "trmtr:/1[]/16",                 // table path + field index, or param:<fmt>:<kind>:<name>, or fitted:<file>:<name>
 "format": "trmtr",                      // file type = the parser's name without parse_ (gfbanm for SwSh clips; LGPE clips use the .tranm layout and show as tranm)
 "where": "table `/1[]` field 16",       // /1[] = elements of the vector in root field 1 (for .trmtr: the materials)
 "status": "unread",
 "values": [["0.5", 120], ["1", 3]],     // top values as count pairs ("<other>" = beyond 200 distinct)
 "distinct": 2, "instances": 123, "models": 18,
 "examples": ["pm0004_00_00: 0.5"],
 "types": {"f32": 123},                  // fields only: type guesses
 "heuristic": true,                      // only seen inside a subtree no parser opened
 "hypothesis": "", "notes": "",          // hand-written, kept on every rewrite
 "seen": true                            // false = not found by the last scan, kept because it has hand content
}
```

Path syntax: `<fmt>:` is the root table; `/i` is the table in field i; `/i[]` is any element of the vector of tables
in field i. `trmtr:/1[]/4[]` = the float-param tables of every material.

## Status

| status | meaning |
|---|---|
| `unread` | present in the data with non-trivial values, never consumed. The default for new entries. |
| `unknown_meaning` | read, or decoded and inspected, but its purpose is unknown. Set it by hand. |
| `fitted` | a constant in our code fitted to a few models (a `FITTED:` comment), not derived from data. |
| `constant` | one value across at least 10 instances and 3 models. Probably a default, but still a candidate. |
| `dummy` | a placeholder, such as a `dummy` texture name or an authoring swatch. |

The scanner chooses `unread`, `constant`, `dummy` or `fitted` itself and marks that choice with `status_auto: true`.

## Annotate

Edit the JSON directly:
- Fill in `hypothesis` (what you think the value does) and `notes` (evidence: models, screenshots, CLAUDE.md
  sections).
- Set `status` by hand, and delete `status_auto` so the scanner keeps your choice.

A rewrite keeps `hypothesis`, `notes` and hand-set statuses for existing keys and replaces the data fields. It drops
entries the scan no longer finds unless they carry hand content; those stay with `seen: false`. When a value starts
being read, add a note saying where, or let it drop out.

Code: `pipeline/importers/za/fb.py` (`recording`, `Recorder`, `field_layout`, `guess_field`),
`pipeline/audit_fields.py` (`FieldAudit`, `merge_ledger`, `scan_fitted`), `pipeline/audit_materials.py` (wrapper).
