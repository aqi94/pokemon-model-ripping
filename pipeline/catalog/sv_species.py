"""SV internal species numbers -> national dex.

SV numbers Gen 9 species in its own order from 917 on (internal 917 = Dudunsparce #982); below 917 internal =
national. The game's `poke_resource_table` (tools/gen_sv_model_dex.py) stores the internal number. Deltas copied
from PKHeX `PKHeX.Core/PKM/Util/Conversion/SpeciesConverter.cs` `Table9InternalToNational` (index = internal - 917).
"""

from __future__ import annotations

FIRST_UNALIGNED_INTERNAL9 = 917

TABLE9_INTERNAL_TO_NATIONAL = (
                                       65, -1, -1,
    -1, -1, 31, 31, 47, 47, 29, 29, 53, 31,
    31, 46, 44, 30, 30, -7, -7, -7, 13, 13,
    -2, -2, 23, 23, 24, -21, -21, 27, 27, 47,
    47, 47, 26, 14, -33, -33, -33, -17, -17, 3,
    -29, 12, -12, -31, -31, -31, 3, 3, -24, -24,
    -44, -44, -30, -30, -28, -28, 23, 23, 6, 7,
    29, 8, 3, 4, 4, 20, 4, 23, 6, 3,
    3, 4, -1, 13, 9, 7, 5, 7, 9, 9,
    -43, -43, -43, -68, -68, -68, -58, -58, -25, -29,
    -31, 6, -1, 6, 0, 0, 0, 3, 3, 4,
    2, 3, 3, -5, -12, -12,
)  # fmt: skip

# Model folders of Gen 9 species: the base-game table maps pm1010..pm1114 to internal 906..1010 (folder = internal
# + 104). DLC folders (absent from the base-game table) continue at internal + 109 (pm1115..1119 do not exist):
# pm1120 = internal 1011 = Ogerpon (4 mask forms), pm1130 = Terapagos (3 forms), pm1133/1134 = Poltchageist/
# Sinistcha (2 forms each). Inferred from those form counts, not read from data (docs/ledger/sv.json).
DLC_GEN9_FIRST_FOLDER = 1115
DLC_GEN9_FOLDER_OFFSET = 109


def national9(internal: int) -> int:
    i = internal - FIRST_UNALIGNED_INTERNAL9
    return internal + TABLE9_INTERNAL_TO_NATIONAL[i] if 0 <= i < len(TABLE9_INTERNAL_TO_NATIONAL) else internal
