"""Scout YouTube playthroughs and grab exact gameplay frames as lighting references.

  python -m tools.lightrefs.fetch_frames doctor
  python -m tools.lightrefs.fetch_frames search  --scene <id> [--query "..."] [--channel @Mixeli] [--n 10]
  python -m tools.lightrefs.fetch_frames info    --video <id|url> [--scene <id>]
  python -m tools.lightrefs.fetch_frames scout   --scene <id> --video <id> --from mm:ss --to mm:ss [--every 20] [--tile 384]
  python -m tools.lightrefs.fetch_frames around  --scene <id> --video <id> --at mm:ss [--span 2] [--step 0.25]
  python -m tools.lightrefs.fetch_frames grab    --scene <id> --video <id> --at mm:ss.f --slug <slug> --model <id> --name "<name>"
                                                 [--clip <clip>] [--time day|dusk|night|any] [--weather sunny|overcast|rain|none]
                                                 [--location "..."] [--light ...] [--indoor] [--context cutscene|field|battle|vs]
                                                 [--confidence high|medium|low] [--ui] [--dof none|background|subject] [--notes ...] [--force]
  python -m tools.lightrefs.fetch_frames regrab  (--scene <id> | --all)

Everything lands in refs/<game>/<scene>/ (frames are git-ignored PNGs; the JSON sidecars make them re-fetchable).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import ytdl
from .scenes import SCENES, get_scene
from .sidecar import load_json, make_sidecar, parse_timecode, save_json, validate_sidecar

ROOT = ytdl.ROOT


def scene_dir(game: str, scene: str) -> str:
    return os.path.join(ROOT, "refs", game, scene)


def _cleanup(tiles) -> None:
    import shutil
    if tiles:
        shutil.rmtree(os.path.dirname(tiles[0][1]), ignore_errors=True)


def cmd_doctor(a) -> None:
    import yt_dlp
    print("yt-dlp", yt_dlp.version.__version__)
    print("ffmpeg", ytdl.ffmpeg_exe())
    import shutil
    print("node", shutil.which("node"))
    r = ytdl.search("Pokemon Legends Z-A gameplay 4K no commentary", 3)
    for e in r:
        print("  ", e["id"], e["platform"], "|", (e["title"] or "")[:70], "|", e["channel"], "|", e["duration"])
    print("search OK" if r else "search returned nothing")


def cmd_search(a) -> None:
    sc = get_scene(a.game, a.scene)
    queries = [a.query] if a.query else [f"Pokemon Legends Z-A {q} 4K no commentary" for q in sc.queries[:2]]
    seen = {}
    if a.channel:
        for e in ytdl.channel_videos(a.channel, 60, match=r"Z-?A|Legends"):
            seen[e["id"]] = e
    for q in queries:
        for e in ytdl.search(q, a.n):
            seen.setdefault(e["id"], e)
    d = os.path.join(scene_dir(a.game, a.scene), "scout")
    os.makedirs(d, exist_ok=True)
    rows = sorted(seen.values(), key=lambda e: (e["platform"] != "switch2", -(e["duration"] or 0)))
    for e in rows:
        print(f"{e['id']}  {e['platform']:8} {str(e['duration'] or '').rjust(6)}s  {e['channel'] or '?':24.24}  {(e['title'] or '')[:80]}")
    save_json(os.path.join(d, "search.json"), rows)
    print(f"{len(rows)} videos -> {os.path.relpath(os.path.join(d, 'search.json'), ROOT)}; run `info --video <id>` for chapters")


def cmd_info(a) -> None:
    info = ytdl.video_info(a.video)
    print(info["id"], info["platform"], info["title"], "|", info["channel"], "|", info["duration"], "s", "| formats", info["formats"])
    for c in info["chapters"]:
        print(f"  {int(c['start'] // 3600):02d}:{int(c['start'] % 3600 // 60):02d}:{int(c['start'] % 60):02d}  {c['title']}")
    if not info["chapters"]:
        print("  (no chapters; description head:)")
        print("  " + info["description"][:1500].replace("\n", "\n  "))
    if a.scene:
        d = os.path.join(scene_dir(a.game, a.scene), "scout")
        save_json(os.path.join(d, f"{info['id']}.json"), info)
        print(f"-> {os.path.relpath(os.path.join(d, info['id'] + '.json'), ROOT)}")


def cmd_scout(a) -> None:
    get_scene(a.game, a.scene)
    vid = ytdl.video_id(a.video)
    start, end = parse_timecode(a.frm), parse_timecode(a.to)
    if end <= start:
        sys.exit("--to must be after --from")
    d = os.path.join(scene_dir(a.game, a.scene), "scout")
    tiles = ytdl.tiles(vid, start, end, a.every, a.tile, os.path.join(ytdl.CACHE_DIR, "tiles_" + vid))
    outs = ytdl.contact_sheets(tiles, os.path.join(d, f"{vid}_{int(start)}-{int(end)}"), label=vid)
    _cleanup(tiles)
    for o in outs:
        print("sheet", os.path.relpath(o, ROOT))
    print(f"{len(tiles)} tiles every {a.every}s; Read the sheets and note timecodes")


def cmd_around(a) -> None:
    get_scene(a.game, a.scene)
    vid = ytdl.video_id(a.video)
    t = parse_timecode(a.at)
    d = os.path.join(scene_dir(a.game, a.scene), "scout")
    tiles = ytdl.tiles(vid, max(0.0, t - a.span), t + a.span, a.step, a.tile, os.path.join(ytdl.CACHE_DIR, "around_" + vid),
                       kind="best", fmt=ytdl.FORMAT_BEST)
    outs = ytdl.contact_sheets(tiles, os.path.join(d, f"{vid}_around_{int(t)}"), cols=4, rows=4, label=vid)
    _cleanup(tiles)
    for o in outs:
        print("sheet", os.path.relpath(o, ROOT))


def cmd_grab(a) -> None:
    sc = get_scene(a.game, a.scene)
    vid = ytdl.video_id(a.video)
    t = parse_timecode(a.at)
    d = scene_dir(a.game, a.scene)
    png = os.path.join(d, a.slug + ".png")
    side = os.path.join(d, a.slug + ".json")
    if os.path.exists(side) and not a.force:
        sys.exit(f"{os.path.relpath(side, ROOT)} exists; use --force")
    info = ytdl.video_info(vid)
    s = ytdl.grab_frame(vid, t, png, a.width)
    from PIL import Image
    w, h = Image.open(png).size
    sidecar = make_sidecar(
        slug=a.slug, scene=sc.id, game=a.game,
        video={"id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "title": info["title"], "channel": info["channel"],
               "platform": info["platform"], "formatId": s.format_id, "width": s.width, "height": s.height, "fps": s.fps, "vcodec": s.vcodec},
        timestamp=t, image={"file": a.slug + ".png", "width": w, "height": h, "scaledFrom": [s.width, s.height]},
        tags={"timeOfDay": a.time or sc.time_of_day, "weather": a.weather or sc.weather, "location": a.location or "",
              "indoor": bool(a.indoor) if a.indoor is not None else sc.indoor, "lightType": a.light or sc.light_type},
        subject={"kind": "pokemon" if a.model.startswith("pm") else "trainer", "model": a.model, "name": a.name, "confidence": a.confidence},
        clip_guess=a.clip, context=a.context,
        quality={"ui": bool(a.ui), "motionBlur": False, "dof": a.dof, "fade": False}, notes=a.notes or "",
    )
    errs = validate_sidecar(sidecar)
    if errs:
        os.remove(png)
        sys.exit("invalid sidecar: " + "; ".join(errs))
    save_json(side, sidecar)
    print(f"wrote {os.path.relpath(png, ROOT)} ({w}x{h} from {s.width}x{s.height} {s.vcodec}) + sidecar")


def cmd_regrab(a) -> None:
    scenes = [a.scene] if a.scene else sorted(SCENES[a.game])
    n = 0
    for scn in scenes:
        d = scene_dir(a.game, scn)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json") or f.endswith("_regions.json") or f in ("fit.json",):
                continue
            sc = load_json(os.path.join(d, f))
            if sc.get("schema") != 1 or not sc.get("video"):
                continue
            png = os.path.join(d, sc["image"]["file"])
            if os.path.exists(png) and not a.force:
                continue
            ytdl.grab_frame(sc["video"]["id"], sc["timestamp"], png, sc["image"]["width"])
            print("regrabbed", os.path.relpath(png, ROOT))
            n += 1
    print(f"{n} frames")


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):  # video titles are often non-Latin; never crash on the console encoding
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="fetch_frames")
    ap.add_argument("--game", default="za")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    p = sub.add_parser("search"); p.add_argument("--scene", required=True); p.add_argument("--query"); p.add_argument("--channel"); p.add_argument("--n", type=int, default=10); p.set_defaults(fn=cmd_search)
    p = sub.add_parser("info"); p.add_argument("--video", required=True); p.add_argument("--scene"); p.set_defaults(fn=cmd_info)
    p = sub.add_parser("scout"); p.add_argument("--scene", required=True); p.add_argument("--video", required=True); p.add_argument("--from", dest="frm", required=True); p.add_argument("--to", required=True); p.add_argument("--every", type=float, default=20); p.add_argument("--tile", type=int, default=384); p.set_defaults(fn=cmd_scout)
    p = sub.add_parser("around"); p.add_argument("--scene", required=True); p.add_argument("--video", required=True); p.add_argument("--at", required=True); p.add_argument("--span", type=float, default=2); p.add_argument("--step", type=float, default=0.25); p.add_argument("--tile", type=int, default=640); p.set_defaults(fn=cmd_around)
    p = sub.add_parser("grab")
    for k, kw in [("--scene", {"required": True}), ("--video", {"required": True}), ("--at", {"required": True}), ("--slug", {"required": True}),
                  ("--model", {"required": True}), ("--name", {"required": True}), ("--clip", {}), ("--time", {}), ("--weather", {}),
                  ("--location", {}), ("--light", {}), ("--context", {"default": "field"}), ("--confidence", {"default": "high"}),
                  ("--dof", {"default": "none"}), ("--notes", {}), ("--width", {"type": int, "default": 1920})]:
        p.add_argument(k, **kw)
    p.add_argument("--indoor", action="store_true", default=None)
    p.add_argument("--ui", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_grab)
    p = sub.add_parser("regrab"); p.add_argument("--scene"); p.add_argument("--all", action="store_true"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_regrab)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
