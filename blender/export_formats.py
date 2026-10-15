"""Headless Blender: model.glb -> FBX (Unreal-ready, with animations) + OBJ/MTL + PNG textures.

Output mirrors the community rip layout (see Adaman/):
  <name>.fbx   binary FBX 7.4, armature + all clips as actions/NLA
  <name>.obj   static mesh, every LOD as its own object named _NNN_<mesh>_shape[_lodN]
  <name>.mtl   map_Kd -> PNG next to it
  <name>_<texture>.png  one PNG per glTF image

Usage:
  blender -b --python blender/export_formats.py -- <model.glb> <out_dir> <name> <result.json>
"""

import json
import os
import re
import sys
import traceback

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
glb, out_dir, name, result_file = argv[0], argv[1], argv[2], argv[3]
os.makedirs(out_dir, exist_ok=True)
result = {"glb": glb, "out_dir": out_dir, "name": name}


def log(msg):
    result.setdefault("log", []).append(str(msg))


try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb, import_shading="NORMALS", bone_heuristic="TEMPERANCE")
    for o in list(bpy.data.objects):
        if o in before:
            bpy.data.objects.remove(o, do_unlink=True)

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    arm = arms[0] if arms else None

    # ---- textures: save every image as PNG with the rip naming ----------
    tex_files = {}
    for img in bpy.data.images:
        if not img.has_data and img.size[0] == 0:
            continue
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", img.name)
        fn = f"{stem}.png"
        path = os.path.join(out_dir, fn)
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        tex_files[img.name] = fn
    result["textures"] = sorted(tex_files.values())

    # ---- rename objects like the rip: _NNN_<mesh>_shape[_lodN] ------------
    def sort_key(o):
        lod = int(o.get("lod", 0))
        return (lod, o.name)

    ordered = sorted(meshes, key=sort_key)
    for i, o in enumerate(ordered):
        base = o.name.replace(".001", "")
        base = re.sub(r"^" + re.escape(name) + "_", "", base)
        m = re.match(r"^(.*)_lod(\d+)$", base)
        if m:
            shape = f"{m.group(1)}_shape_lod{m.group(2)}"
        else:
            shape = f"{base}_shape"
        o.name = f"_{len(ordered) - i:03d}_{shape}"
        o.data.name = o.name

    # Non-LOD0 meshes are hidden in the viewport but still exported.
    for o in ordered:
        o.hide_viewport = False
        o.hide_set(False)
        o.hide_render = False

    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.data.objects:
        o.select_set(True)

    # ---- OBJ (static, all LODs, materials with map_Kd) ----------------------
    obj_path = os.path.join(out_dir, f"{name}.obj")
    bpy.ops.wm.obj_export(
        filepath=obj_path,
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
        export_triangulated_mesh=True,
        apply_modifiers=True,
        path_mode="RELATIVE",
        forward_axis="NEGATIVE_Z",
        up_axis="Y",
        global_scale=1.0,
    )
    result["obj"] = os.path.basename(obj_path)
    # Rewrite map_* lines in the MTL to point at our PNG names (relative)
    mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
    if os.path.isfile(mtl_path):
        lines = []
        with open(mtl_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("map_"):
                    key, _, path = line.partition(" ")
                    path = path.strip().replace("\\", "/")
                    line = f"{key} {os.path.basename(path)}\n"
                lines.append(line)
        with open(mtl_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        result["mtl"] = os.path.basename(mtl_path)

    # ---- FBX (Unreal-ready, with all actions) -------------------------------
    fbx_path = os.path.join(out_dir, f"{name}.fbx")
    actions = list(bpy.data.actions)
    if arm is not None and actions:
        # The exporter's "all actions" mode writes one animation stack per
        # action that is compatible with the armature. Clear any NLA state so
        # nothing is double-counted or muted away.
        if arm.animation_data is None:
            arm.animation_data_create()
        arm.animation_data.action = None
        for track in list(arm.animation_data.nla_tracks):
            arm.animation_data.nla_tracks.remove(track)
    bpy.ops.export_scene.fbx(
        filepath=fbx_path,
        use_selection=True,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        axis_forward="-Z",
        axis_up="Y",
        object_types={"ARMATURE", "MESH"},
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        use_tspace=False,
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        armature_nodetype="NULL",
        bake_anim=bool(actions),
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=True,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
        path_mode="RELATIVE",
        embed_textures=False,
        batch_mode="OFF",
    )
    result["fbx"] = os.path.basename(fbx_path)
    result["actions"] = [a.name for a in actions]
    result["mesh_objects"] = [o.name for o in ordered]
    result["bones"] = len(arm.data.bones) if arm else 0
    result["ok"] = True
except Exception:
    result["ok"] = False
    result["error"] = traceback.format_exc()

with open(result_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
