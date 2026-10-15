"""manifest.json: the viewer's per-model interface (game agnostic)."""

from __future__ import annotations

import json

from .model import Model

MANIFEST_VERSION = 1


def build_manifest(model: Model, glb_summary: dict | None = None) -> dict:
    groups: dict[str, dict] = {}
    for vi, v in enumerate(model.variants):
        grp = groups.setdefault(v.group, {"name": v.group, "label": v.group.replace("_", " ").title(), "options": []})
        grp["options"].append({"variant": vi, "name": v.name, "label": v.label or v.name, "default": v.default})
    for grp in groups.values():
        if not any(o["default"] for o in grp["options"]) and grp["options"]:
            grp["options"][0]["default"] = True

    clips = [
        {
            "name": c.name,
            "index": c.index,
            "duration": round(c.duration, 4),
            "frameRate": c.frame_rate,
            "frameCount": c.frame_count,
            "loop": c.loop,
            "category": c.category,
            "group": c.group,
            "default": c.name == model.default_clip,
            **({"visibility": {k: [[f, int(v)] for f, v in keys] for k, keys in c.visibility.items()}} if c.visibility else {}),
        }
        for c in model.clips
    ]
    parts = [
        {
            "node": m.name,
            "label": _part_label(m.name, model.id),
            "lod": m.lod,
            "visible": m.visible,
            **({k: m.extras[k] for k in ("group", "option") if k in m.extras}),
            **({"accessory": True} if m.extras.get("accessory") else {}),
        }
        for m in model.meshes
    ]
    # Modular models (generic NPCs): meshes tagged extras {group, option}; the
    # viewer shows exactly one option per group.
    part_groups = []
    for g in model.extras.get("part_groups", []):
        part_groups.append(
            {
                **g,
                "options": [
                    {**o, "nodes": [m.name for m in model.meshes if m.extras.get("group") == g["name"] and m.extras.get("option") == o["id"]]}
                    for o in g["options"]
                ],
            }
        )
    # Held-item attachment points: joints named *_attach; a clip "shows" the
    # item when it animates that joint's scale (the game hides props by
    # scaling the attach bone to 0.01).
    props = []
    if model.skeleton:
        for j in model.skeleton.joints:
            if j.name.endswith("_attach"):
                clips_with = [c.name for c in model.clips if any(t.joint == j.name and t.kind == "scale" for t in c.tracks)]
                props.append({"bone": j.name, "side": "left" if j.name.startswith("left") else "right" if j.name.startswith("right") else "", "clips": clips_with})
    return {
        "manifestVersion": MANIFEST_VERSION,
        "props": props,
        "game": model.game,
        "id": model.id,
        "kind": model.kind,
        "displayName": model.display_name,
        "files": {"model": "model.glb"},
        "variantGroups": list(groups.values()),
        "clips": clips,
        "defaultClip": model.default_clip,
        "parts": parts,
        **({"partGroups": part_groups} if part_groups else {}),
        "stats": {
            "joints": len(model.skeleton.joints) if model.skeleton else 0,
            "meshes": len(model.meshes),
            "triangles": int(sum(len(p.indices) // 3 for m in model.meshes if m.lod == 0 for p in m.primitives)),
            "textures": len(model.textures),
            "clips": len(model.clips),
            **({"glbBytes": glb_summary["bytes"]} if glb_summary else {}),
        },
        "source": model.source_paths,
        "extras": {k: v for k, v in model.extras.items() if k != "part_groups"},
    }


def _part_label(node: str, model_id: str) -> str:
    label = node
    if "__" in label:  # modular parts: "<part id>__<mesh>"
        label = label.split("__", 1)[1]
    if label.startswith(model_id + "_"):
        label = label[len(model_id) + 1 :]
    label = label.replace("_mesh", "")
    return label.replace("_", " ")


def write_manifest(model: Model, path: str, glb_summary: dict | None = None) -> dict:
    m = build_manifest(model, glb_summary)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1, ensure_ascii=False)
    return m
