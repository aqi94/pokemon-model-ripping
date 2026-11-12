"""Write pipeline/catalog/sv_model_dex.json: SV Pokémon model folder (pm####) -> national dex number.

SV's model folders are not dex numbers from pm0704 up (Gen 6+ use internal ids: pm0720 = Chespin #650, pm1013 =
Fuecoco #909, pm0909 = Skwovet #819). The game's own map is `romfs/pokemon/catalog/catalog/poke_resource_table.
trpmcatalog` in the full dump `SV-Everything.7z`: root[1] = entries {0 key {0 dex (u16), 1 form, 2 gender},
1 model path `pmNNNN/<id>/<id>.trmdl`, 2 trmmt, 3 trpokecfg, ...}. The key's dex is SV's INTERNAL species number,
which differs from the national dex for Gen 9 (internal 917 = Dudunsparce #982); `sv_species.national9` converts it.
The dump is base-game only: DLC folders absent from the table are handled by `SvCatalog.species_dex`.

    python -m tools.gen_sv_model_dex [--catalog <extracted .trpmcatalog>] [--archive SV-Everything.7z]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile

from pipeline.catalog.sv_species import national9
from pipeline.importers.za import fb

ARCHIVE = os.path.expanduser("~/Documents/MEGA downloads/SV Every File/SV-Everything.7z")
MEMBER = "romfs/pokemon/catalog/catalog/poke_resource_table.trpmcatalog"
OUT = os.path.join(os.path.dirname(__file__), "..", "pipeline", "catalog", "sv_model_dex.json")


def model_dex(data: bytes) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in fb.root(data).tables(1):
        m = re.match(r"(pm\d{4})/", t.string(1))
        key = t.table(0)
        if not m or key is None:
            continue
        dex = national9(key.u16(0))
        if out.setdefault(m.group(1), dex) != dex:
            raise ValueError(f"{m.group(1)}: forms map to different dex numbers ({out[m.group(1)]}, {dex})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalog")
    ap.add_argument("--archive", default=ARCHIVE)
    args = ap.parse_args()
    if args.catalog:
        with open(args.catalog, "rb") as f:
            data = f.read()
    else:
        import py7zr

        with tempfile.TemporaryDirectory() as tmp:
            with py7zr.SevenZipFile(args.archive, "r") as z:
                z.extract(path=tmp, targets=[MEMBER])
            with open(os.path.join(tmp, MEMBER), "rb") as f:
                data = f.read()
    table = model_dex(data)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(table.items())), f, indent=0)
        f.write("\n")
    moved = sum(1 for k, v in table.items() if int(k[2:]) != v)
    print(f"{len(table)} models ({moved} renumbered) -> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
