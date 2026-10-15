"""Headless Blender check: import a GLB, report counts, render preview images.

Usage (Store Blender gives no stdout, so results go to a JSON file):
  blender-launcher.exe -b --python blender/inspect_glb.py -- <model.glb> <out_dir> [clip_name]
Writes <out_dir>/inspect.json and <out_dir>/preview_*.png
"""

import json
import math
import os
import sys
import traceback

import bpy

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
glb, out_dir = argv[0], argv[1]
clip = argv[2] if len(argv) > 2 else None
os.makedirs(out_dir, exist_ok=True)
result = {"glb": glb}

try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb)
    for o in list(bpy.data.objects):
        if o in before or o.name.startswith(("Icosphere", "Cube", "Light", "Camera")):
            bpy.data.objects.remove(o, do_unlink=True)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    result["meshes"] = [m.name for m in meshes]
    result["mesh_count"] = len(meshes)
    result["vertex_count"] = sum(len(m.data.vertices) for m in meshes)
    result["armatures"] = [(a.name, len(a.data.bones)) for a in arms]
    result["actions"] = [(a.name, int(a.frame_range[0]), int(a.frame_range[1])) for a in bpy.data.actions]
    result["materials"] = [m.name for m in bpy.data.materials]
    result["images"] = [(i.name, i.size[0], i.size[1]) for i in bpy.data.images]
    result["skinned_meshes"] = [m.name for m in meshes if any(mod.type == "ARMATURE" for mod in m.modifiers)]

    # Hide LODs > 0 for the preview
    for m in meshes:
        lod = m.get("lod", 0)
        if lod:
            m.hide_render = True
            m.hide_viewport = True

    # Bounds of LOD0
    xs, ys, zs = [], [], []
    for m in meshes:
        if m.get("lod", 0):
            continue
        for v in m.data.vertices:
            w = m.matrix_world @ v.co
            xs.append(w.x)
            ys.append(w.y)
            zs.append(w.z)
    cx, cy, cz = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2
    size = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    result["bounds"] = {"min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]}

    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except Exception:
        scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    # PNG exports always have a transparent background; the world still provides ambient light.
    scene.render.film_transparent = True
    scene.render.image_settings.color_mode = "RGBA"
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.25, 0.27, 0.3, 1)
    bg.inputs[1].default_value = 1.5
    scene.world = world

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), math.radians(20), math.radians(30))
    scene.collection.objects.link(sun)

    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    dist = size * 2.2

    def shoot(name, az_deg, el_deg=12):
        az, el = math.radians(az_deg), math.radians(el_deg)
        cam.location = (cx + dist * math.cos(el) * math.sin(az), cy - dist * math.cos(el) * math.cos(az), cz + dist * math.sin(el))
        direction = (cx - cam.location.x, cy - cam.location.y, cz - cam.location.z)
        import mathutils

        cam.rotation_euler = mathutils.Vector(direction).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = os.path.join(out_dir, f"preview_{name}.png")
        bpy.ops.render.render(write_still=True)

    scene.frame_set(0)
    shoot("front", 0)
    shoot("side", 90)
    shoot("top", 20, 35)
    shoot("back", 180)

    if clip and clip in bpy.data.actions and arms:
        arm = arms[0]
        arm.animation_data_create()
        arm.animation_data.action = bpy.data.actions[clip]
        act = bpy.data.actions[clip]
        mid = int((act.frame_range[0] + act.frame_range[1]) / 2)
        scene.frame_set(mid)
        shoot(f"anim_{clip}_f{mid}", 30)
        result["anim_preview_frame"] = mid
    result["ok"] = True
except Exception:
    result["ok"] = False
    result["error"] = traceback.format_exc()

with open(os.path.join(out_dir, "inspect.json"), "w") as f:
    json.dump(result, f, indent=1)
