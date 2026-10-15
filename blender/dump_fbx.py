"""Headless Blender: import an FBX and dump a neutral, game-agnostic description.

Usage (Store Blender gives no stdout, so results go to a JSON file):
  blender-launcher.exe -b --python blender/dump_fbx.py -- <model.fbx> <out_dir> <result.json>
Writes <out_dir>/dump.npz (arrays) and the metadata into <result.json>.

Everything is expressed in glTF space (Y up, metres):
  * the armature's object scale (Unity exports cm under a 0.01 parent) is folded into bone translations only, so
    joints keep unit-scale bases and every consumer sees real metres;
  * root joints carry the Z-up -> Y-up rotation;
  * non-root joint locals are parent-relative in bone space, so they are identical to what the file stores.
"""

import json
import math
import os
import sys
import traceback

import bpy
import numpy as np
from mathutils import Matrix

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
fbx, out_dir, result_file = argv[0], argv[1], argv[2]
os.makedirs(out_dir, exist_ok=True)
result = {"fbx": fbx}
CM = Matrix.Rotation(-math.pi / 2, 4, "X")  # (x, y, z) -> (x, z, -y)


def mat_np(m):
    return np.array([list(r) for r in m], dtype=np.float32)


def decompose(m):
    loc, rot, scl = m.decompose()
    return list(loc), [rot.w, rot.x, rot.y, rot.z], list(scl)


def bone_order(bones):
    """Parents before children."""
    order, seen = [], set()

    def visit(b):
        if b.name in seen:
            return
        seen.add(b.name)
        order.append(b)
        for c in b.children:
            visit(c)

    for b in bones:
        if b.parent is None:
            visit(b)
    return order


def material_info(mat):
    imgs, base = [], None
    if mat and mat.node_tree:
        for n in mat.node_tree.nodes:
            if n.type == "BSDF_PRINCIPLED":
                link = n.inputs["Base Color"].links
                if link and link[0].from_node.type == "TEX_IMAGE" and link[0].from_node.image:
                    base = os.path.basename(link[0].from_node.image.filepath.replace("\\", "/"))
        for n in mat.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image:
                imgs.append(os.path.basename(n.image.filepath.replace("\\", "/")))
    return {"name": mat.name if mat else "", "base": base, "images": imgs}


try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx)
    scene = bpy.context.scene
    fps = scene.render.fps / (scene.render.fps_base or 1.0)
    result["fps"] = fps
    arrays = {}

    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    arm = max(arms, key=lambda o: len(o.data.bones)) if arms else None
    bones, bone_idx, sinv = [], {}, Matrix.Identity(4)
    if arm:
        s = arm.matrix_world.to_scale()[0]
        sinv = Matrix.Scale(1.0 / s, 4)
        bones = bone_order(arm.data.bones)
        bone_idx = {b.name: i for i, b in enumerate(bones)}
        base = CM @ arm.matrix_world
        rest_w = [base @ b.matrix_local @ sinv for b in bones]
        joints = []
        ibm = np.zeros((len(bones), 4, 4), np.float32)
        for i, b in enumerate(bones):
            p = bone_idx[b.parent.name] if b.parent else -1
            local = rest_w[i] if p < 0 else rest_w[p].inverted() @ rest_w[i]
            loc, rot, scl = decompose(local)
            joints.append({"name": b.name, "parent": p, "t": loc, "r": rot, "s": scl})
            ibm[i] = mat_np(rest_w[i].inverted())
        result["joints"] = joints
        arrays["ibm"] = ibm

    # -- meshes ---------------------------------------------------------
    meshes = []
    materials = {}
    for oi, obj in enumerate([o for o in bpy.data.objects if o.type == "MESH" and len(o.data.vertices)]):
        me = obj.data
        me.calc_loop_triangles()
        nv, nl, nt = len(me.vertices), len(me.loops), len(me.loop_triangles)
        co = np.empty(nv * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        loop_v = np.empty(nl, np.int32)
        me.loops.foreach_get("vertex_index", loop_v)
        tris = np.empty(nt * 3, np.int32)
        me.loop_triangles.foreach_get("loops", tris)
        tri_mat = np.empty(nt, np.int32)
        me.loop_triangles.foreach_get("material_index", tri_mat)
        try:
            nrm = np.empty(nl * 3, np.float32)
            me.corner_normals.foreach_get("vector", nrm)
            nrm = nrm.reshape(-1, 3)
        except Exception:  # noqa: BLE001
            vn = np.empty(nv * 3, np.float32)
            me.vertices.foreach_get("normal", vn)
            nrm = vn.reshape(-1, 3)[loop_v]
        if me.uv_layers:
            uv = np.empty(nl * 2, np.float32)
            me.uv_layers[0].data.foreach_get("uv", uv)
            uv = uv.reshape(-1, 2)
        else:
            uv = np.zeros((nl, 2), np.float32)

        m4 = CM @ obj.matrix_world
        m3 = np.array(m4.to_3x3(), np.float32)
        pos = co @ m3.T + np.array(m4.translation, np.float32)
        nrm_m = np.array(m4.to_3x3().inverted().transposed(), np.float32)
        nrm = nrm @ nrm_m.T
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-8)
        flip = m4.to_3x3().determinant() < 0

        # skin weights: top 4 per vertex
        vj = np.zeros((nv, 4), np.uint16)
        vw = np.zeros((nv, 4), np.float32)
        skinned = bool(arm) and any(md.type == "ARMATURE" for md in obj.modifiers)
        if skinned:
            gname = [g.name for g in obj.vertex_groups]
            gmap = [bone_idx.get(n, -1) for n in gname]
            for vi, v in enumerate(me.vertices):
                ws = sorted(((g.weight, gmap[g.group]) for g in v.groups if gmap[g.group] >= 0 and g.weight > 1e-4), reverse=True)[:4]
                tot = sum(w for w, _ in ws)
                if tot <= 0:
                    vw[vi, 0] = 1.0
                    continue
                for k, (w, j) in enumerate(ws):
                    vj[vi, k] = j
                    vw[vi, k] = w / tot
        elif arm and obj.parent_type == "BONE" and obj.parent_bone in bone_idx:
            vj[:, 0] = bone_idx[obj.parent_bone]
            vw[:, 0] = 1.0
            skinned = True

        slots = []
        for k in range(max(1, len(me.materials))):
            sel = tri_mat == k if len(me.materials) else np.ones(nt, bool)
            if not sel.any():
                continue
            loops = tris.reshape(-1, 3)[sel].ravel()
            key = np.concatenate([loop_v[loops, None].astype(np.float32), uv[loops], nrm[loops]], axis=1)
            _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
            lu = loops[first]
            idx = inv.astype(np.uint32).reshape(-1, 3)
            if flip:
                idx = idx[:, ::-1]
            pre = f"m{oi}_s{k}_"
            arrays[pre + "pos"] = pos[loop_v[lu]]
            arrays[pre + "nrm"] = nrm[lu]
            arrays[pre + "uv"] = uv[lu]
            arrays[pre + "idx"] = idx.ravel()
            if skinned:
                arrays[pre + "joints"] = vj[loop_v[lu]]
                arrays[pre + "weights"] = vw[loop_v[lu]]
            mat = me.materials[k] if k < len(me.materials) else None
            mname = mat.name if mat else f"{obj.name}_mat"
            if mname not in materials:
                info = material_info(mat)
                info["name"] = mname
                materials[mname] = info
            slots.append({"key": pre, "material": mname, "skinned": skinned})
        meshes.append({"name": obj.name, "slots": slots})
    result["meshes"] = meshes
    result["materials"] = list(materials.values())

    # -- animation ------------------------------------------------------
    clips = []
    if arm:
        ad = arm.animation_data_create()
        acts = [a for a in bpy.data.actions if a.name.startswith(arm.name + "|")]
        base = CM @ arm.matrix_world
        for ai, act in enumerate(acts):
            for pb in arm.pose.bones:
                pb.matrix_basis = Matrix.Identity(4)
            ad.action = act
            if hasattr(ad, "action_slot") and getattr(act, "slots", None):
                ad.action_slot = act.slots[0]
            f0, f1 = int(round(act.frame_range[0])), int(round(act.frame_range[1]))
            nf = max(1, f1 - f0 + 1)
            loc = np.zeros((nf, len(bones), 3), np.float32)
            rot = np.zeros((nf, len(bones), 4), np.float32)
            scl = np.ones((nf, len(bones), 3), np.float32)
            for fi in range(nf):
                scene.frame_set(f0 + fi)
                bpy.context.view_layer.update()
                ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
                world = []
                for i, b in enumerate(bones):
                    world.append(base @ ev.pose.bones[b.name].matrix @ sinv)
                for i, b in enumerate(bones):
                    p = bone_idx[b.parent.name] if b.parent else -1
                    local = world[i] if p < 0 else world[p].inverted() @ world[i]
                    l, r, sc = decompose(local)
                    loc[fi, i], rot[fi, i], scl[fi, i] = l, r, sc
            name = act.name.split("|")[1] if act.name.count("|") >= 1 else act.name
            arrays[f"a{ai}_loc"], arrays[f"a{ai}_rot"], arrays[f"a{ai}_scl"] = loc, rot, scl
            clips.append({"name": name, "key": f"a{ai}_", "frames": nf, "start": f0})
        ad.action = None
    result["clips"] = clips

    np.savez_compressed(os.path.join(out_dir, "dump.npz"), **arrays)
    result["ok"] = True
except Exception:  # noqa: BLE001
    result["ok"] = False
    result["error"] = traceback.format_exc()

with open(result_file, "w", encoding="utf-8") as f:
    json.dump(result, f)
