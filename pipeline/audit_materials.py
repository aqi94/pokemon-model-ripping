"""Which material parameters do the game files use that the importer never reads?

    python -m pipeline.audit_materials [--game za|sv|la|swsh|lgpe] [--kind trainer|pokemon|object] [--sample 30]

Thin wrapper over `pipeline.audit_fields` (params only, no ledger write): imports a sample of models with the
material param dicts swapped for recording dicts, so a name counts as read only when code really looked it up
during the import (the old source grep of importer.py missed names used in composite.py / form_layers.py).
Neutral values (0, white, black, False) are skipped. For the full ledger use `python -m pipeline audit-fields`.
"""
from __future__ import annotations

import argparse

from .audit_fields import FieldAudit, _neutral, sample_refs  # noqa: F401  (_neutral kept for old imports)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="za", choices=["za", "sv", "la", "swsh", "lgpe"])
    ap.add_argument("--kind", default="trainer", choices=["trainer", "pokemon", "object"])
    ap.add_argument("--data", default=None)
    ap.add_argument("--trainer-data", default=None)
    ap.add_argument("--sample", type=int, default=30, help="models (evenly spread; 0 = all)")
    a = ap.parse_args(argv)
    from .cli import _catalog

    cat = _catalog(argparse.Namespace(game=a.game, data=a.data, trainer_data=a.trainer_data))
    refs = sample_refs(cat, [a.kind], a.sample)
    audit = FieldAudit(a.game)
    audit.run(cat, refs, with_anim=False, log=lambda *_: None)
    n_mats = sum(st.mats for st in audit.params.values())
    print(f"{len(refs)} {a.kind} models, {n_mats} non-neutral param values\n")
    print("NOT READ by the importer (non-neutral values):")
    for key, st in sorted(audit.unread_params(), key=lambda kv: -len(kv[1].models)):
        _, _, kind, name = key.split(":", 3)
        print(f"  {kind:7} {name:34} {st.mats:4} mats {len(st.models):3} models  e.g. {'; '.join(st.examples[:3])}")


if __name__ == "__main__":
    main()
