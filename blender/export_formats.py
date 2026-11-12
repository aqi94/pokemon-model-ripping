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

    # Optional subset (pipeline export --clips / --rest-visible, see blender_runner.export_formats).
    clip_re = os.environ.get("EXPORT_CLIPS") or ""
    if clip_re:
        keep = re.compile(clip_re)
        for a in list(bpy.data.actions):
            if not keep.search(a.name):
                bpy.data.actions.remove(a)
        result["clip_filter"] = clip_re
    # --parts group=option,...: pick part options like the viewer's part picker (only the picked option's meshes of each named
    # group are visible; accessory meshes stay off unless --accessories), before the rest-visible filter below.
    parts_sel = dict(kv.split("=", 1) for kv in (os.environ.get("EXPORT_PARTS") or "").split(",") if "=" in kv)
    if parts_sel:
        with_acc = bool(os.environ.get("EXPORT_ACCESSORIES"))
        found = set()
        for o in bpy.data.objects:
            g = o.get("group")
            if o.type != "MESH" or g not in parts_sel:
                continue
            on = o.get("option") == parts_sel[g] and (with_acc or not o.get("accessory", False))
            found.add((g, o.get("option")))
            o["visible"] = bool(on)
        missing = [f"{g}={v}" for g, v in parts_sel.items() if (g, v) not in found]
        if missing:
            raise RuntimeError(f"--parts options not found on any mesh: {missing}")
        result["parts"] = parts_sel
    if os.environ.get("EXPORT_LOD0"):
        lods = [o for o in bpy.data.objects if o.type == "MESH" and int(o.get("lod", 0) or 0) > 0]
        for o in lods:
            bpy.data.objects.remove(o, do_unlink=True)
        result["dropped_lod_meshes"] = len(lods)
    if os.environ.get("EXPORT_REST_VISIBLE"):
        hidden = [o for o in bpy.data.objects if o.type == "MESH" and o.get("visible", True) in (False, 0)]
        for o in hidden:
            bpy.data.objects.remove(o, do_unlink=True)
        result["dropped_hidden_meshes"] = len(hidden)

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
    # Images referenced only from material extras (LGPE field layer recipes: extras.fieldBlend/fieldRamp/fieldWater) are not
    # loaded by Blender's glTF importer: copy them straight out of the GLB (PNG bytes) so every glTF image gets its PNG.
    import struct

    with open(glb, "rb") as f:
        glb_bytes = f.read()
    json_len = struct.unpack_from("<I", glb_bytes, 12)[0]
    gjson = json.loads(glb_bytes[20 : 20 + json_len])
    bin_start = 20 + json_len + 8
    written = {os.path.splitext(v)[0] for v in tex_files.values()}
    for gimg in gjson.get("images", []):
        if gimg.get("bufferView") is None or not gimg.get("name") or gimg.get("mimeType", "image/png") != "image/png":
            continue
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", gimg["name"])
        if stem in written:
            continue
        view = gjson["bufferViews"][gimg["bufferView"]]
        start = bin_start + view.get("byteOffset", 0)
        with open(os.path.join(out_dir, f"{stem}.png"), "wb") as f:
            f.write(glb_bytes[start : start + view["byteLength"]])
        written.add(stem)
        tex_files[gimg["name"]] = f"{stem}.png"
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
