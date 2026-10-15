"""Headless Blender: render close-ups of a GLB's head region at rest pose.

blender -b --python blender/closeup.py -- <model.glb> <out_dir> <result.json> [hide:<material substring>] [flipnormalg]
Renders preview_face_*.png (front, three-quarter, below) at 900x900.
"""

import json
import math
import os
import sys
import traceback

import bpy
import mathutils

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
glb, out_dir, result_file = argv[0], argv[1], argv[2]
opts = argv[3:]
os.makedirs(out_dir, exist_ok=True)
result = {"glb": glb}
try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb)
    for o in list(bpy.data.objects):
        if o in before or o.name.startswith(("Icosphere", "Cube", "Light", "Camera")):
            bpy.data.objects.remove(o, do_unlink=True)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    for m in meshes:
        if m.get("lod", 0):
            m.hide_render = True
            m.hide_viewport = True
    # optional: hide primitives using a material whose name contains X
    for opt in opts:
        if opt.startswith("hide:"):
            key = opt[5:]
            for m in meshes:
                for slot in m.material_slots:
                    if slot.material and key in slot.material.name:
                        # remove those polygons by hiding via a transparent material
                        mat = slot.material
                        mat.blend_method = "BLEND" if hasattr(mat, "blend_method") else mat.blend_method
                        mat.use_nodes = True
                        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
                        if bsdf:
                            bsdf.inputs["Alpha"].default_value = 0.0
                            for link in list(mat.node_tree.links):
                                if link.to_socket == bsdf.inputs["Alpha"]:
                                    mat.node_tree.links.remove(link)
        if opt == "noskin":
            for m in meshes:
                for mod in m.modifiers:
                    if mod.type == "ARMATURE":
                        mod.show_render = False
                        mod.show_viewport = False
        if opt.startswith("cz:"):
            os.environ["CLOSEUP_CZ"] = opt[3:]
        if opt.startswith("zoom:"):
            os.environ["CLOSEUP_ZOOM"] = opt[5:]
        if opt.startswith("cx:"):
            os.environ["CLOSEUP_CX"] = opt[3:]
        if opt.startswith("target:"):
            os.environ["CLOSEUP_TARGET"] = opt[7:]
        if opt.startswith("dist:"):
            os.environ["CLOSEUP_DIST"] = opt[5:]
        if opt == "nonormal":
            for mat in bpy.data.materials:
                if mat.use_nodes:
                    for n in list(mat.node_tree.nodes):
                        if n.type == "NORMAL_MAP":
                            mat.node_tree.nodes.remove(n)
        if opt == "flipnormalg":
            for mat in bpy.data.materials:
                if not mat.use_nodes:
                    continue
                for n in mat.node_tree.nodes:
                    if n.type == "NORMAL_MAP":
                        # insert an RGB curve inverting green between the image and the normal map node
                        src_link = next((l for l in mat.node_tree.links if l.to_socket == n.inputs["Color"]), None)
                        if not src_link:
                            continue
                        sep = mat.node_tree.nodes.new("ShaderNodeSeparateColor")
                        comb = mat.node_tree.nodes.new("ShaderNodeCombineColor")
                        inv = mat.node_tree.nodes.new("ShaderNodeMath")
                        inv.operation = "SUBTRACT"
                        inv.inputs[0].default_value = 1.0
                        mat.node_tree.links.new(src_link.from_socket, sep.inputs[0])
                        mat.node_tree.links.new(sep.outputs[1], inv.inputs[1])
                        mat.node_tree.links.new(sep.outputs[0], comb.inputs[0])
                        mat.node_tree.links.new(inv.outputs[0], comb.inputs[1])
                        mat.node_tree.links.new(sep.outputs[2], comb.inputs[2])
                        mat.node_tree.links.remove(src_link)
                        mat.node_tree.links.new(comb.outputs[0], n.inputs["Color"])
    # head region = top 35% of the LOD0 bounds
    xs, ys, zs = [], [], []
    for m in meshes:
        if m.get("lod", 0):
            continue
        for v in m.data.vertices:
            w = m.matrix_world @ v.co
            xs.append(w.x); ys.append(w.y); zs.append(w.z)
    zmin, zmax = min(zs), max(zs)
    cz = zmin + (zmax - zmin) * float(os.environ.get("CLOSEUP_CZ", "0.72"))
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    cx += (max(xs) - min(xs)) * float(os.environ.get("CLOSEUP_CX", "0"))
    size = (zmax - zmin) * 0.55 / float(os.environ.get("CLOSEUP_ZOOM", "1"))
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except Exception:
        scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = scene.render.resolution_y = 900
    scene.render.film_transparent = True  # transparent PNGs; the world still lights the model
    scene.render.image_settings.color_mode = "RGBA"
    world = bpy.data.worlds.new("W"); world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.3, 0.3, 0.33, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.2
    scene.world = world
    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN")); sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), math.radians(20), math.radians(30)); scene.collection.objects.link(sun)
    cam = bpy.data.objects.new("Cam", bpy.data.cameras.new("Cam")); scene.collection.objects.link(cam); scene.camera = cam
    cam.data.lens = 60
    dist = size * 2.6
    if os.environ.get("CLOSEUP_TARGET"):
        cx, cy, cz = [float(x) for x in os.environ["CLOSEUP_TARGET"].split(",")]
        dist = float(os.environ.get("CLOSEUP_DIST", "0.4"))
    def shoot(name, az_deg, el_deg):
        az, el = math.radians(az_deg), math.radians(el_deg)
        cam.location = (cx + dist * math.cos(el) * math.sin(az), cy - dist * math.cos(el) * math.cos(az), cz + dist * math.sin(el))
        d = mathutils.Vector((cx - cam.location.x, cy - cam.location.y, cz - cam.location.z))
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = os.path.join(out_dir, f"face_{name}.png")
        bpy.ops.render.render(write_still=True)
    scene.frame_set(0)
    shoot("front", 0, 5)
    shoot("quarter", 35, 10)
    shoot("below", 0, -25)
    result["ok"] = True
except Exception:
    result["ok"] = False
    result["error"] = traceback.format_exc()
with open(result_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
