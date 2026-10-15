"""Command line entry point.

  python -m pipeline list    --game za|sv|go|bdsp [--kind pokemon|trainer]
  python -m pipeline convert --game za --id pm0001_00_00 [--id ...] [--out out] [--no-anim]
  python -m pipeline convert --game za --all [--kind pokemon] [--limit N] [--resume]
  python -m pipeline index   [--out out]           # rebuild out/index.json from manifests
  python -m pipeline export  --game za --id pm0001_00_00   # FBX/OBJ via Blender
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

from .catalog.bdsp import DEFAULT_DATA_DIR as BDSP_DATA_DIR
from .catalog.bdsp import BdspCatalog
from .catalog.swsh import SwshCatalog
from .catalog.go import DEFAULT_DATA_DIR as GO_DATA_DIR
from .catalog.go import GoCatalog
from .catalog.la import LaCatalog
from .catalog.lgpe import DEFAULT_TRAINER_DIR as LGPE_TRAINER_DIR
from .catalog.lgpe import LgpeCatalog
from .catalog.sv import SvCatalog
from .catalog.za import DEFAULT_DATA_DIR, ModelRef, ZaCatalog
from .common.gltf_writer import write_glb
from .common.manifest import write_manifest


def _catalog(args) -> ZaCatalog | SvCatalog | GoCatalog | BdspCatalog:
    if args.game == "za":
        return ZaCatalog(args.data or DEFAULT_DATA_DIR)
    if args.game == "sv":
        return SvCatalog(args.data or DEFAULT_DATA_DIR)
    if args.game == "la":
        return LaCatalog(args.data or DEFAULT_DATA_DIR)
    if args.game == "go":
        return GoCatalog(args.data or GO_DATA_DIR)
    if args.game == "swsh":
        return SwshCatalog(args.data or DEFAULT_DATA_DIR)
    if args.game == "bdsp":
        return BdspCatalog(args.data or BDSP_DATA_DIR)
    if args.game == "lgpe":
        return LgpeCatalog(args.data or DEFAULT_DATA_DIR, trainer_dir=args.trainer_data or LGPE_TRAINER_DIR)
    sys.exit(f"game '{args.game}' has no importer yet")


def cmd_list(args) -> None:
    cat = _catalog(args)
    refs = {"pokemon": cat.pokemon, "trainer": cat.trainers, "object": cat.objects}.get(args.kind, cat.all)()
    for r in refs:
        print(f"{r.kind:8s} {r.id:16s} {r.display_name}")
    print(f"{len(refs)} models", file=sys.stderr)


def convert_one(cat: ZaCatalog | SvCatalog | GoCatalog | BdspCatalog, ref: ModelRef, out_root: str, with_anim: bool = True) -> dict:
    t0 = time.time()
    model = cat.import_ref(ref, with_anim)
    model.extras.update({k: v for k, v in ref.extras.items() if k != "folder"})
    model.extras["species"] = ref.species
    model.extras["form"] = ref.form
    model.source_paths = [ref.folder]
    out_dir = os.path.join(out_root, ref.game, ref.id)
    os.makedirs(out_dir, exist_ok=True)
    summary = write_glb(model, os.path.join(out_dir, "model.glb"))
    manifest = write_manifest(model, os.path.join(out_dir, "manifest.json"), summary)
    summary["seconds"] = round(time.time() - t0, 2)
    summary["clips"] = len(model.clips)
    return {"ok": True, "id": ref.id, "manifest": manifest, "summary": summary}


def cmd_convert(args) -> None:
    cat = _catalog(args)
    if args.all:
        refs = {"pokemon": cat.pokemon, "trainer": cat.trainers, "object": cat.objects}.get(args.kind, cat.all)()
        if args.limit:
            refs = refs[: args.limit]
    else:
        refs = []
        for mid in args.id:
            r = cat.find(mid)
            if r is None:
                sys.exit(f"unknown model id {mid}")
            refs.append(r)
    log_path = os.path.join(args.out, "convert_log.jsonl")
    os.makedirs(args.out, exist_ok=True)
    done = set()
    if args.resume and os.path.isfile(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("ok"):
                        done.add(rec["id"])
                except json.JSONDecodeError:
                    pass
    ok = fail = 0
    with open(log_path, "a", encoding="utf-8") as log:
        for i, ref in enumerate(refs):
            if ref.id in done:
                continue
            try:
                res = convert_one(cat, ref, args.out, with_anim=not args.no_anim)
                s = res["summary"]
                print(f"[{i + 1}/{len(refs)}] {ref.id:16s} {ref.display_name:24s} {s['clips']:3d} clips {s['bytes'] / 1e6:6.1f} MB {s['seconds']:5.1f}s")
                log.write(json.dumps({"ok": True, "id": ref.id, "summary": s}) + "\n")
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"[{i + 1}/{len(refs)}] {ref.id:16s} FAILED: {e}")
                log.write(json.dumps({"ok": False, "id": ref.id, "error": str(e), "trace": traceback.format_exc()}) + "\n")
                fail += 1
            log.flush()
    print(f"done: {ok} ok, {fail} failed")
    build_index(args.out)


def build_index(out_root: str) -> dict:
    models = []
    for game in sorted(os.listdir(out_root)):
        gdir = os.path.join(out_root, game)
        if not os.path.isdir(gdir):
            continue
        for mid in sorted(os.listdir(gdir)):
            mp = os.path.join(gdir, mid, "manifest.json")
            if not os.path.isfile(mp):
                continue
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            models.append(
                {
                    "game": m["game"],
                    "id": m["id"],
                    "kind": m["kind"],
                    "displayName": m["displayName"],
                    "clips": m["stats"]["clips"],
                    "variantGroups": [g["name"] for g in m["variantGroups"]],
                    "path": f"{game}/{mid}/",
                    "dex": m.get("extras", {}).get("dex"),
                    "form": m.get("extras", {}).get("form"),
                    "item": m.get("extras", {}).get("item"),
                    "props": m.get("props"),
                    "thumbnail": f"{game}/{mid}/thumb.webp" if os.path.isfile(os.path.join(gdir, mid, "thumb.webp")) else None,
                }
            )
    names = {"za": "Pokémon Legends: Z-A", "sv": "Pokémon Scarlet & Violet", "go": "Pokémon GO", "bdsp": "Pokémon Brilliant Diamond & Shining Pearl", "swsh": "Pokémon Sword & Shield", "la": "Pokémon Legends: Arceus", "lgpe": "Pokémon: Let's Go, Pikachu! & Eevee!"}
    games = {g: names.get(g, g) for g in names if any(m["game"] == g for m in models)}
    index = {"indexVersion": 1, "games": games, "models": models}
    with open(os.path.join(out_root, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    return index


def cmd_index(args) -> None:
    idx = build_index(args.out)
    print(f"{len(idx['models'])} models indexed")


def cmd_export(args) -> None:
    from .blender_runner import export_formats

    for mid in args.id:
        glb = os.path.join(args.out, args.game, mid, "model.glb")
        if not os.path.isfile(glb):
            sys.exit(f"{glb} not found; run convert first")
        res = export_formats(glb, os.path.join(args.out, "blender", args.game, mid))
        print(json.dumps(res, indent=1))


def cmd_thumbs(args) -> None:
    """Render thumb.webp for converted models (Blender headless, ~4 s each)."""
    import shutil
    import tempfile

    from PIL import Image

    from .blender_runner import inspect

    root = os.path.join(args.out, args.game)
    ids = args.id or sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "model.glb")))
    tmp = tempfile.mkdtemp(prefix="thumbs_")
    n = 0
    for mid in ids:
        dst = os.path.join(root, mid, "thumb.webp")
        if os.path.isfile(dst) and not args.force:
            continue
        glb = os.path.join(root, mid, "model.glb")
        try:
            r = inspect(glb, os.path.join(tmp, mid))
            src = os.path.join(tmp, mid, "preview_front.png")
            if not r.get("ok") or not os.path.isfile(src):
                print(f"{mid}: render failed: {r.get('error', '')[:200]}")
                continue
            im = Image.open(src).convert("RGB")
            im = im.resize((args.size, args.size), Image.LANCZOS)
            im.save(dst, "WEBP", quality=82)
            n += 1
            print(f"{mid}: thumb.webp")
        except Exception as e:  # noqa: BLE001
            print(f"{mid}: {e}")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{n} thumbnails written")
    build_index(args.out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pipeline")
    ap.add_argument("--game", default="za")
    ap.add_argument("--data", default=None, help="folder with the ripped zips (za/sv) / the 3D Assets folder (go)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--trainer-data", default=None, help="lgpe: folder with the extracted LGPE-Trainers.7z")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list")
    p.add_argument("--kind", choices=["pokemon", "trainer", "object"])
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("convert")
    p.add_argument("--id", action="append", default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--kind", choices=["pokemon", "trainer", "object"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-anim", action="store_true")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("index")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("export")
    p.add_argument("--id", action="append", required=True)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("thumbs")
    p.add_argument("--id", action="append", default=[])
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_thumbs)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
