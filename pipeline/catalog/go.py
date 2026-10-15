"""Catalog of Pokémon GO models in the raw asset dump (FBX + PNG per folder)."""

from __future__ import annotations

import os
import re

from ..importers.go.importer import import_model as import_go_model
from .za import ModelRef

DEFAULT_DATA_DIR = os.environ.get("POKEMODELS_GO_DATA") or os.path.join(os.path.expanduser("~"), "Documents", "pokemon_raw_data", "go", "pogo_assets", "3D Assets")

# Characters/<folder>/ holds one FBX (+ its PNGs): the trainers/NPCs.
TRAINER_DIR = "Characters"


def pretty_name(folder: str) -> str:
    words = []
    for tok in folder.split("_"):
        if tok.lower() == "rig":
            continue
        tok = re.sub(r"(?i)^(.*?)v(\d)$", r"\1 V\2", tok)
        words.append(tok[:1].upper() + tok[1:])
    return " ".join(words).strip() or folder


class GoCatalog:
    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = data_dir

    def trainers(self) -> list[ModelRef]:
        root = os.path.join(self.data_dir, TRAINER_DIR)
        refs = []
        if not os.path.isdir(root):
            return refs
        for folder in sorted(os.listdir(root)):
            fdir = os.path.join(root, folder)
            fbxs = [f for f in os.listdir(fdir) if f.lower().endswith(".fbx")] if os.path.isdir(fdir) else []
            if not fbxs:
                continue
            fbx = max(fbxs, key=lambda f: os.path.getsize(os.path.join(fdir, f)))
            refs.append(
                ModelRef(
                    game="go",
                    kind="trainer",
                    id=f"go_{folder.lower()}",
                    folder=f"{TRAINER_DIR}/{folder}/{fbx}",
                    display_name=pretty_name(folder),
                    species=folder,
                    extras={"folder": folder},
                )
            )
        return refs

    def pokemon(self) -> list[ModelRef]:
        return []

    def objects(self) -> list[ModelRef]:
        return []

    def all(self) -> list[ModelRef]:
        return self.trainers()

    def find(self, model_id: str) -> ModelRef | None:
        return next((r for r in self.all() if r.id == model_id), None)

    def import_ref(self, ref: ModelRef, with_anim: bool = True):
        model = import_go_model(os.path.join(self.data_dir, ref.folder), ref.id, ref.display_name, kind=ref.kind, game=ref.game)
        if not with_anim:
            model.clips, model.default_clip = [], None
        return model
