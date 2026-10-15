"""Headless Blender: re-import an exported FBX (and OBJ) and report what came back.

Usage: blender -b --python blender/verify_export.py -- <file.fbx> <file.obj> <result.json>
"""

import json
import sys
import traceback

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
fbx, obj, result_file = argv[0], argv[1], argv[2]
result = {}
try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.import_scene.fbx(filepath=fbx)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    result["fbx"] = {
        "meshes": len(meshes),
        "vertices": sum(len(m.data.vertices) for m in meshes),
        "bones": [len(a.data.bones) for a in arms],
        "actions": len(bpy.data.actions),
        "action_names": sorted(a.name for a in bpy.data.actions)[:5],
        "skinned": sum(1 for m in meshes if any(mod.type == "ARMATURE" for mod in m.modifiers)),
        "materials": sorted(m.name for m in bpy.data.materials),
        "images": sorted(i.name for i in bpy.data.images),
    }
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.wm.obj_import(filepath=obj)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    result["obj"] = {
        "objects": sorted(o.name for o in meshes),
        "vertices": sum(len(m.data.vertices) for m in meshes),
        "materials": sorted(m.name for m in bpy.data.materials),
        "images": sorted(i.name for i in bpy.data.images),
    }
    result["ok"] = True
except Exception:
    result["ok"] = False
    result["error"] = traceback.format_exc()
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
