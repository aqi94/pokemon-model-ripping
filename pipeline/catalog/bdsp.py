"""Catalog of Brilliant Diamond / Shining Pearl bundles (extracted from the BDSP-*.7z rips).

Layout under the data dir (`pokemon_raw_data/bdsp`): `trainers/Characters/persons/{battle,field}/<id>` (one
self-contained UnityFS bundle per model) and `trainers/Characters/objects/ob####_##`.
"""

from __future__ import annotations

import json
import os
import re

from ..importers.bdsp.importer import externals, import_bundle
from .za import ModelRef

DEFAULT_DATA_DIR = os.environ.get("POKEMODELS_BDSP_DATA") or os.path.join(os.path.expanduser("~"), "Documents", "pokemon_raw_data", "bdsp")
NAMES_PATH = os.path.join(os.path.dirname(__file__), "bdsp_names.json")

# (folder under trainers/Characters, id regex, kind, label)
TRAINER_DIRS = [
    ("persons/battle", re.compile(r"^(tr|pc)\d{4}_\d{2}$"), "trainer", "Battle"),
    ("persons/field", re.compile(r"^fc\d{4}_\d{2}$"), "trainer", "Field"),
]
OBJECT_DIR = ("objects", re.compile(r"^ob\d{4}_\d{2}$"))


def _names() -> dict[str, str]:
    try:
        with open(NAMES_PATH, encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return {}


class BdspCatalog:
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir
        self.root = os.path.join(data_dir, "trainers", "Characters")

    def _scan(self, sub: str, rx: re.Pattern, kind: str, label: str) -> list[ModelRef]:
        d = os.path.join(self.root, *sub.split("/"))
        names = _names()
        refs = []
        if not os.path.isdir(d):
            return refs
        for f in sorted(os.listdir(d)):
            if not rx.match(f) or not os.path.isfile(os.path.join(d, f)):
                continue
            nice = names.get(f)
            refs.append(
                ModelRef(
                    game="bdsp",
                    kind=kind,
                    id=f,
                    folder=f"trainers/Characters/{sub}/{f}",
                    display_name=f"{nice} ({label})" if nice and label else (nice or f),
                    species=f.split("_")[0],
                    form=f.split("_")[1],
                )
            )
        return refs

    def trainers(self) -> list[ModelRef]:
        return [r for sub, rx, kind, label in TRAINER_DIRS for r in self._scan(sub, rx, kind, label)]

    def objects(self) -> list[ModelRef]:
        return self._scan(OBJECT_DIR[0], OBJECT_DIR[1], "object", "")

    def pokemon(self) -> list[ModelRef]:
        return []

    def all(self) -> list[ModelRef]:
        return self.trainers() + self.objects()

    def find(self, model_id: str) -> ModelRef | None:
        return next((r for r in self.all() if r.id == model_id), None)

    def cab_index(self) -> dict[str, str]:
        """CAB name -> bundle path (relative to the data dir); built once by opening every bundle."""
        if getattr(self, "_cabs", None) is None:
            path = os.path.join(self.data_dir, "cab_index.json")
            if not os.path.isfile(path):
                import UnityPy

                cabs = {}
                for sub, *_ in TRAINER_DIRS + [(OBJECT_DIR[0],)]:
                    d = os.path.join(self.root, *sub.split("/"))
                    for f in sorted(os.listdir(d)):
                        if os.path.isfile(os.path.join(d, f)):
                            bf = next(iter(UnityPy.load(os.path.join(d, f)).files.values()))
                            for k in bf.files:
                                if k.lower().startswith("cab-") and not k.lower().endswith(".ress"):
                                    cabs[k.lower()] = f"trainers/Characters/{sub}/{f}"
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(cabs, fh, indent=0)
            with open(path, encoding="utf-8") as fh:
                self._cabs = json.load(fh)
        return self._cabs

    def import_ref(self, ref: ModelRef, with_anim: bool = True):
        path = os.path.join(self.data_dir, ref.folder)
        cabs = self.cab_index()
        deps = sorted({os.path.join(self.data_dir, cabs[c]) for c in externals(path) if c in cabs} - {path})
        return import_bundle(path, ref.id, ref.display_name, kind=ref.kind, game=ref.game, with_anim=with_anim, deps=deps)
