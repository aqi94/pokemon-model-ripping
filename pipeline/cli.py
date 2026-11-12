"""Command line entry point.

  python -m pipeline list    --game za|sv|go|bdsp|swsh|la|lgpe|home|unite [--kind pokemon|trainer]
  python -m pipeline convert --game za --id pm0001_00_00 [--id ...] [--out out] [--no-anim]
  python -m pipeline convert --game za --all [--kind pokemon] [--limit N] [--resume]
  python -m pipeline index   [--out out]           # rebuild out/index.json from manifests
  python -m pipeline relabel --game sv              # refresh names/dex of converted manifests from the catalog, reindex
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
from .catalog.home import DEFAULT_DATA_DIR as HOME_DATA_DIR
from .catalog.home import HomeCatalog
from .catalog.la import LaCatalog
from .catalog.lgpe import DEFAULT_TRAINER_DIR as LGPE_TRAINER_DIR
from .catalog.lgpe import LgpeCatalog
from .catalog.sv import SvCatalog
from .catalog.unite import DEFAULT_DATA_DIR as UNITE_DATA_DIR
from .catalog.unite import UniteCatalog
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
    if args.game == "home":
        return HomeCatalog(args.data or HOME_DATA_DIR)
    if args.game == "lgpe":
        return LgpeCatalog(args.data or DEFAULT_DATA_DIR, trainer_dir=args.trainer_data or LGPE_TRAINER_DIR)
        return UniteCatalog(args.data or UNITE_DATA_DIR)
    sys.exit(f"game '{args.game}' has no importer yet")


KINDS = ["pokemon", "trainer", "object", "map", "prop"]


def _refs_by_kind(cat, kind: str | None) -> list[ModelRef]:
    if hasattr(cat, "by_kind"):  # catalogs with more kinds than pokemon/trainer/object (LGPE field maps)
        return cat.by_kind(kind)
    return {"pokemon": cat.pokemon, "trainer": cat.trainers, "object": cat.objects}.get(kind, cat.all)() if kind in (None, "pokemon", "trainer", "object") else []


def cmd_list(args) -> None:
    cat = _catalog(args)
    refs = _refs_by_kind(cat, args.kind)
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
        refs = _refs_by_kind(cat, args.kind)
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
    # National dex order (folder ids are internal numbers in SV/Z-A/LA above Gen 5); models without one keep id order.
    models.sort(key=lambda m: (m["game"], m["dex"] is None, m["dex"] or 0, m["id"]))
    names = {"za": "Pokémon Legends: Z-A", "sv": "Pokémon Scarlet & Violet", "go": "Pokémon GO", "bdsp": "Pokémon Brilliant Diamond & Shining Pearl", "swsh": "Pokémon Sword & Shield", "la": "Pokémon Legends: Arceus", "lgpe": "Pokémon: Let's Go, Pikachu! & Eevee!", "home": "Pokémon HOME", "unite": "Pokémon UNITE"}
    games = {g: names.get(g, g) for g in names if any(m["game"] == g for m in models)}
    index = {"indexVersion": 1, "games": games, "models": models}
    with open(os.path.join(out_root, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    # Per-game indexes + a tiny games list: the viewer loads only the game (domain) being viewed.
    summary = {}
    for g, name in games.items():
        gm = [m for m in models if m["game"] == g]
        with open(os.path.join(out_root, g, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"indexVersion": 1, "game": g, "name": name, "models": gm}, f, ensure_ascii=False)
        kinds = {}
        for m in gm:
            kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
        summary[g] = {"name": name, "count": len(gm), "kinds": kinds}
    with open(os.path.join(out_root, "games.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False)
    return index


def cmd_relabel(args) -> None:
    """Rewrite displayName + extras.dex of already converted models from the catalog (no reconversion)."""
    cat = _catalog(args)
    changed = 0
    for ref in cat.all():
        mp = os.path.join(args.out, ref.game, ref.id, "manifest.json")
        if not os.path.isfile(mp):
            continue
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        extras = m.setdefault("extras", {})
        if m.get("displayName") == ref.display_name and extras.get("dex") == ref.extras.get("dex"):
            continue
        print(f"{ref.id:16s} {m.get('displayName')!s:28s} -> {ref.display_name}")
        m["displayName"] = ref.display_name
        if "dex" in ref.extras:
            extras["dex"] = ref.extras["dex"]
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
        changed += 1
    print(f"{changed} manifests relabelled")
    build_index(args.out)


def cmd_index(args) -> None:
    idx = build_index(args.out)
    print(f"{len(idx['models'])} models indexed")


def cmd_export(args) -> None:
    from .blender_runner import export_formats

    for mid in args.id:
        glb = os.path.join(args.out, args.game, mid, "model.glb")
        if not os.path.isfile(glb):
            sys.exit(f"{glb} not found; run convert first")
        out_dir = os.path.join(args.out, "blender", args.game, mid)
        if args.subset:  # a named subset goes to its own folder so the full export stays intact
            out_dir = os.path.join(out_dir, args.subset)
        res = export_formats(glb, out_dir, mid, clips=args.clips, rest_visible=args.rest_visible, parts=args.parts,
                             accessories=args.accessories, lod0=args.lod0)
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
    if getattr(args, "renderer", "blender") == "viewer":
        # The real viewer (three.js through tools/render_models.mjs; dev server must run). Blender cannot hold bone
        # scale: its glTF importer moves vertices bound to a non-uniformly scaled joint (UNITE: 42% of models, e.g.
        # Charmander's (0.01, 1, 1) mouth bone put 174 vertices 35 m away and framed the model as a speck).
        import subprocess

        todo = [m for m in ids if args.force or not os.path.isfile(os.path.join(root, m, "thumb.webp"))]
        for i in range(0, len(todo), 40):
            batch = todo[i:i + 40]
            subprocess.run(["node", os.path.join("tools", "render_models.mjs"), "--out", tmp, "--size", "640x640",
                            "--framing", "fit", "--bg", "transparent", *[f"{args.game}/{m}" for m in batch]], check=False)
            for mid in batch:
                src = os.path.join(tmp, f"{args.game}_{mid}_full.png")
                if not os.path.isfile(src):
                    print(f"{mid}: render failed")
                    continue
                im = Image.open(src).convert("RGBA")
                bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
                bg.alpha_composite(im)
                bg.convert("RGB").resize((args.size, args.size), Image.LANCZOS).save(os.path.join(root, mid, "thumb.webp"), "WEBP", quality=82)
                n += 1
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"{n} thumbnails written")
        build_index(args.out)
        return
    for mid in ids:
        dst = os.path.join(root, mid, "thumb.webp")
        if os.path.isfile(dst) and not args.force:
            continue
        glb = os.path.join(root, mid, "model.glb")
        try:
            r = inspect(glb, os.path.join(tmp, mid))
            kind = ""
            try:
                with open(os.path.join(root, mid, "manifest.json"), encoding="utf-8") as f:
                    kind = json.load(f).get("kind", "")
            except (OSError, ValueError):
                pass
            # maps are wide and flat: the 35 deg "top" shot shows the layout, the 12 deg front shot only the near edge
            src = os.path.join(tmp, mid, "preview_top.png" if kind == "map" else "preview_front.png")
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


def cmd_lights(args) -> None:
    """Scene light rigs (`.trlgt`): list / dump / export / presets. SV only for now (see pipeline/lighting)."""
    from .lighting.catalog import SvLightCatalog, merge
    from .importers.za.trlgt import dump_rig

    if args.game != "sv":
        sys.exit("lights: only --game sv has an extracted light-rig tree (POKEMODELS_SV_ENV)")
    cat = SvLightCatalog(args.env_dir) if args.env_dir else SvLightCatalog()
    if not cat.available():
        sys.exit(f"lights: no rigs under {cat.romfs}; extract SV-Everything.7z romfs/env there first (POKEMODELS_SV_ENV)")
    if args.what == "list":
        for r in cat.rigs():
            if args.group and r.group != args.group:
                continue
            print(f"{r.group:10} {r.name}")
        return
    if args.what == "dump":
        rig = cat.load(args.rig)
        if args.merge:
            rig = merge(cat.master(), rig)
        print(dump_rig(rig))
        return
    if args.what == "export":
        from .lighting.export import export_rigs
        names = [r.name for r in cat.rigs()] if args.all else list(args.rig_list)
        out_dir = os.path.join(args.out, "sv", "lights")
        n = export_rigs(cat, names, out_dir, probes=not args.no_probes)
        print(f"wrote {n} rigs to {out_dir}")
        return
    if args.what == "presets":
        from .lighting.presets import write_sv_presets
        path = write_sv_presets(cat, args.ts_out, category=args.category)
        print(f"wrote {path}")
        return


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="pipeline")
    ap.add_argument("--game", default="za")
    ap.add_argument("--data", default=None, help="folder with the ripped zips (za/sv) / the 3D Assets folder (go)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--trainer-data", default=None, help="lgpe: folder with the extracted LGPE-Trainers.7z")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list")
    p.add_argument("--kind", choices=KINDS)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("convert")
    p.add_argument("--id", action="append", default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-anim", action="store_true")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("index")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("relabel", help="refresh displayName/dex of converted manifests from the catalog, then reindex")
    p.set_defaults(fn=cmd_relabel)

    p = sub.add_parser("export")
    p.add_argument("--id", action="append", required=True)
    p.add_argument("--clips", default=None, help="regex: export only the clips whose name matches (re.search)")
    p.add_argument("--rest-visible", action="store_true", help="drop meshes hidden at rest (other part options, colourways)")
    p.add_argument("--subset", default=None, help="write to out/blender/<game>/<id>/<subset>/ instead of the full export folder")
    p.add_argument("--parts", default=None,
                   help="modular models: 'group=option,...' (manifest partGroups) shown instead of the default options; "
                        "combine with --rest-visible to drop the other options")
    p.add_argument("--accessories", action="store_true", help="with --parts: also show the selected options' accessory meshes")
    p.add_argument("--lod0", action="store_true", help="drop LOD1+ meshes (node extras lod > 0), e.g. for engines that build their own LODs")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("lights", help="scene light rigs (.trlgt): list | dump <rig> | export | presets")
    p.add_argument("what", choices=["list", "dump", "export", "presets"])
    p.add_argument("rig", nargs="?", help="rig name for dump (e.g. master, fel_interior, weather_rain)")
    p.add_argument("--group", help="list: master|weather|area|cutscene|filter|terrain|shieldmap")
    p.add_argument("--merge", action="store_true", help="dump: merge the rig over master first")
    p.add_argument("--all", action="store_true", help="export: every rig")
    p.add_argument("--rig-list", nargs="*", default=[], help="export: rig names")
    p.add_argument("--no-probes", action="store_true", help="export: skip probe cube PNGs")
    p.add_argument("--env-dir", default=None, help="extracted SV env tree (default POKEMODELS_SV_ENV)")
    p.add_argument("--category", type=int, default=5, help="presets: material CategoryLabel (5 trainer skin, 6 Pokemon)")
    p.add_argument("--ts-out", default=os.path.join("viewer", "src", "scene", "svPresets.ts"))
    p.set_defaults(fn=cmd_lights)

    p = sub.add_parser("thumbs")
    p.add_argument("--id", action="append", default=[])
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--force", action="store_true")
    p.add_argument("--renderer", choices=["blender", "viewer"], default="blender",
                   help="viewer = tools/render_models.mjs through the running dev server (exact glTF skinning)")
    p.set_defaults(fn=cmd_thumbs)

    p = sub.add_parser("audit-fields", help="unread-value ledger: source fields/params the import never reads -> docs/ledger/<game>.json")
    p.add_argument("--sample", type=int, default=20, help="models per kind (evenly spread; 0 = all)")
    p.add_argument("--kind", action="append", choices=KINDS, help="default: pokemon, trainer, object")
    p.add_argument("--id", action="append", default=[], help="extra model ids to include")
    p.add_argument("--no-anim", action="store_true", help="skip clips (much faster, no .tranm/.tracm coverage)")
    p.add_argument("--no-write", action="store_true", help="print only, do not rewrite the ledger")
    p.add_argument("--ledger-dir", default=None, help="default docs/ledger")
    p.add_argument("--top", type=int, default=8, help="values kept per entry")
    p.add_argument("--show", type=int, default=40, help="entries printed")
    p.set_defaults(fn=lambda a: __import__("pipeline.audit_fields", fromlist=["x"]).cmd_audit_fields(a, _catalog))

    p = sub.add_parser("rigs", help="rig signatures: families (shared skeletons, candidate clip borrowers) | compare <idA> <idB>")
    p.add_argument("what", choices=["families", "compare"])
    p.add_argument("ids", nargs="*", help="compare: two model ids, optionally [game:]id (e.g. la:tr0003_00 sv:tr0350_00)")
    p.add_argument("--kind", action="append", choices=["pokemon", "trainer", "object"], help="families: default all kinds")
    p.add_argument("--tol", type=float, default=None, help="bind-local translation tolerance in metres (default 0.002)")
    p.add_argument("--all", action="store_true", help="families: also print single-member families")
    p.add_argument("--sample", type=int, default=12, help="clips sampled per donor for coverage")
    p.add_argument("--json", default=None, help="families: write the result as JSON")
    p.set_defaults(fn=lambda a: __import__("pipeline.rigs", fromlist=["x"]).cmd_rigs(a, _catalog))

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
