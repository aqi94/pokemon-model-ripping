"""Write `pipeline/catalog/unite_fx_donors.json`: for each idle effect prefab (`pref_fx_*`, not attack/skill/buff) the
UNITE models whose copy of it still holds mesh geometry.

Most skins keep an idle effect only as empty emitter Nulls (`importer.particle_emitters`), but some skins' FBX kept the
same prefab with real meshes: Gastly skin 10's `pref_fx_Gastly_idle_lv3` holds the gas discs `mesh huo1` / `mesh huo2`
that skins 00/01/09/11 lack. The importer borrows such geometry by prefab name, like clips from a rig donor.

    python -m tools.gen_unite_fx_donors
"""

from __future__ import annotations

import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "pipeline", "catalog", "unite_fx_donors.json")


def main() -> None:
    from pipeline.catalog.unite import UniteCatalog
    from pipeline.importers.unite import fbx_anim
    from pipeline.importers.unite.importer import ATTACK_PREFAB

    cat = UniteCatalog()
    table: dict[str, list[dict]] = {}
    seen = set()
    for ref in cat.all():
        path = ref.folder if os.path.isabs(ref.folder) else os.path.join(cat.data_dir, ref.folder)
        if path in seen or not path.lower().endswith(".fbx") or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            scene = fbx_anim.parse(path)
        except Exception:  # noqa: BLE001
            continue
        kids: dict[int, list[int]] = {}
        for n in scene.nodes.values():
            kids.setdefault(n.parent, []).append(n.id)
        for n in scene.nodes.values():
            name = n.name
            if not name.lower().startswith("pref_fx_") or ATTACK_PREFAB.search(name):
                continue
            par = scene.nodes.get(n.parent)
            if par is not None and par.name.lower().startswith("pref_fx_"):
                continue
            meshes, nested, stack = [], [], list(kids.get(n.id, []))
            while stack:
                k = stack.pop()
                stack += kids.get(k, [])
                if scene.nodes[k].kind == "Mesh":
                    meshes.append(scene.nodes[k].name)
                elif scene.nodes[k].name.lower().startswith("pref_fx_") and scene.nodes[k].name != name:
                    nested.append(scene.nodes[k].name)
            if meshes:
                # `nested`: effect prefabs inside this one (Charmander's `pref_fx_0004_battlefire_lv3` holds the
                # particle prefab `pref_fx_0004_lobbyfire_atk` next to its mesh flame): a file that kept only such a
                # nested prefab (every lobby Charizard) borrows this prefab's meshes (importer.attach_fx_donors)
                table.setdefault(name, []).append({"model": ref.id, "fbx": os.path.relpath(path, cat.data_dir) if not os.path.isabs(ref.folder) else path,
                                                   "meshes": sorted(meshes), "nested": sorted(set(nested))})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"source": "tools/gen_unite_fx_donors.py", "prefabs": table}, f, indent=1)
    for k, v in sorted(table.items()):
        print(k, [(d["model"], d["meshes"]) for d in v][:4])
    print(f"{len(table)} prefabs with mesh donors -> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
