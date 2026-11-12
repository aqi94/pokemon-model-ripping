"""Run Blender headless and read back a JSON result file.

The Microsoft Store build of Blender is only reachable through its launcher
alias, which gives no stdout; scripts therefore write a result JSON that we
read after the process exits. A conventional blender.exe on PATH also works.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

BLENDER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "blender")


def find_blender() -> str:
    env = os.environ.get("BLENDER")
    if env and os.path.isfile(env):
        return env
    for cand in ("blender", "blender-launcher"):
        p = shutil.which(cand)
        if p:
            return p
    alias = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps", "blender-launcher.exe")
    if os.path.isfile(alias):
        return alias
    raise FileNotFoundError("Blender not found: set BLENDER=<path to blender.exe or blender-launcher.exe>")


def run_script(script: str, args: list[str], result_file: str, timeout: int = 1800, env: dict | None = None) -> dict:
    exe = find_blender()
    if os.path.isfile(result_file):
        os.remove(result_file)
    cmd = [exe, "-b", "--python", script, "--", *args, result_file]
    subprocess.run(cmd, check=False, timeout=timeout, capture_output=True, env={**os.environ, **env} if env else None)
    if not os.path.isfile(result_file):
        raise RuntimeError(f"Blender produced no result file for {script} (cmd: {cmd})")
    with open(result_file, encoding="utf-8") as f:
        return json.load(f)


def export_formats(glb: str, out_dir: str, model_name: str | None = None, thumbnail: str | None = None,
                   clips: str | None = None, rest_visible: bool = False, parts: str | None = None,
                   accessories: bool = False, lod0: bool = False) -> dict:
    """GLB -> <out_dir>/<name>.fbx, .obj, .mtl, textures as PNG (Adaman layout).

    Subset exports (`pipeline export --clips/--rest-visible`, e.g. a game engine's retarget source): `clips` = regex, only
    the actions whose name it matches (re.search) are written; `rest_visible` drops meshes hidden at rest (glTF node
    extras `visible: false`: alternate part options, colourways, accessories). `parts` = "group=option,..." re-picks options
    of modular models first (node extras group/option, as the viewer's part picker does; `accessories` also shows the
    picked options' accessory meshes); `lod0` drops meshes with node extras lod > 0."""
    os.makedirs(out_dir, exist_ok=True)
    result_file = os.path.join(tempfile.gettempdir(), f"blender_export_{os.getpid()}.json")
    args = [os.path.abspath(glb), os.path.abspath(out_dir), model_name or os.path.basename(os.path.dirname(os.path.abspath(glb)))]
    env = {"EXPORT_CLIPS": clips or "", "EXPORT_REST_VISIBLE": "1" if rest_visible else "", "EXPORT_PARTS": parts or "",
           "EXPORT_ACCESSORIES": "1" if accessories else "", "EXPORT_LOD0": "1" if lod0 else ""}
    return run_script(os.path.join(BLENDER_DIR, "export_formats.py"), args, result_file, env=env)


def inspect(glb: str, out_dir: str, clip: str | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    result_file = os.path.join(out_dir, "inspect.json")
    args = [os.path.abspath(glb), os.path.abspath(out_dir)] + ([clip] if clip else [])
    return run_script(os.path.join(BLENDER_DIR, "inspect_glb.py"), args, result_file)
