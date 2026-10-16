"""yt-dlp + bundled ffmpeg (imageio-ffmpeg) helpers: search, metadata, stream URL cache, exact frame grabs, contact sheets.

Only DASH `https` video-only formats are used: the HLS (`m3u8`) formats yt-dlp offers first cannot be seeked by
ffmpeg without walking the whole playlist (a 7-hour playthrough hung for minutes); with a DASH URL
`ffmpeg -ss T -i <url> -frames:v 1` fetches a few MB and is frame accurate (seek to the keyframe, decode to T).
Stream URLs expire after a few hours, so they are cached with their expiry under refs/_cache/.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(ROOT, "refs", "_cache")
FORMAT_BEST = "bv*[height<=2160][protocol=https][vcodec^=vp09]/bv*[height<=2160][protocol=https]/bv*[protocol=https]/bv*"
FORMAT_SCOUT = "bv*[height<=480][protocol=https]/bv*[height<=720][protocol=https]/bv*[protocol=https]"
STREAM_TTL = 4 * 3600


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ydl(**extra):
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "js_runtimes": {"node": {}}, "noplaylist": True}
    opts.update(extra)
    return yt_dlp.YoutubeDL(opts)


def video_id(s: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", s)
    return m.group(1) if m else s.strip()


def platform_guess(title: str, description: str = "") -> str:
    t = f"{title} {description}".lower()
    if re.search(r"switch\s*2|4k|2160", t):
        return "switch2"
    return "switch1"


def search(query: str, n: int = 10) -> list[dict]:
    with _ydl(extract_flat=True) as y:
        r = y.extract_info(f"ytsearch{n}:{query}", download=False)
    out = []
    for e in r.get("entries") or []:
        out.append({"id": e.get("id"), "title": e.get("title"), "channel": e.get("channel") or e.get("uploader"),
                    "duration": e.get("duration"), "url": e.get("url") or f"https://www.youtube.com/watch?v={e.get('id')}",
                    "platform": platform_guess(e.get("title") or "")})
    return out


def channel_videos(channel: str, n: int = 50, match: str | None = None) -> list[dict]:
    """`@Handle` -> recent uploads (flat), optionally filtered by a regex on the title."""
    url = channel if channel.startswith("http") else f"https://www.youtube.com/{channel}/videos"
    with _ydl(extract_flat=True, playlistend=n) as y:
        r = y.extract_info(url, download=False)
    out = []
    for e in r.get("entries") or []:
        t = e.get("title") or ""
        if match and not re.search(match, t, re.I):
            continue
        out.append({"id": e.get("id"), "title": t, "channel": r.get("channel") or r.get("uploader") or channel,
                    "duration": e.get("duration"), "url": f"https://www.youtube.com/watch?v={e.get('id')}", "platform": platform_guess(t)})
    return out


def video_info(vid: str) -> dict:
    """Full metadata (title, channel, duration, chapters, description) without resolving a stream."""
    with _ydl(skip_download=True) as y:
        info = y.extract_info(f"https://www.youtube.com/watch?v={video_id(vid)}", download=False)
    return {
        "id": info["id"], "title": info.get("title"), "channel": info.get("channel") or info.get("uploader"),
        "duration": info.get("duration"), "upload_date": info.get("upload_date"),
        "platform": platform_guess(info.get("title") or "", info.get("description") or ""),
        "chapters": [{"start": c.get("start_time"), "end": c.get("end_time"), "title": c.get("title")} for c in (info.get("chapters") or [])],
        "description": (info.get("description") or "")[:4000],
        "formats": sorted({(f.get("height") or 0, f.get("fps") or 0) for f in info.get("formats") or [] if f.get("vcodec") not in (None, "none")}, reverse=True)[:6],
    }


@dataclass
class Stream:
    url: str
    format_id: str
    width: int
    height: int
    fps: float
    vcodec: str
    expires: float


def resolve_stream(vid: str, fmt: str = FORMAT_BEST, kind: str = "best") -> Stream:
    vid = video_id(vid)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{vid}_{kind}.json")
    if os.path.isfile(cache):
        try:
            with open(cache, encoding="utf-8") as f:
                d = json.load(f)
            if d.get("expires", 0) > time.time() + 60:
                return Stream(**d)
        except (OSError, ValueError, TypeError):
            pass
    with _ydl(format=fmt) as y:
        info = y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
    if "url" not in info:
        raise RuntimeError(f"no stream URL for {vid}")
    m = re.search(r"[?&]expire=(\d+)", info["url"])
    s = Stream(info["url"], str(info.get("format_id")), int(info.get("width") or 0), int(info.get("height") or 0),
               float(info.get("fps") or 0), str(info.get("vcodec")), float(m.group(1)) if m else time.time() + STREAM_TTL)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(s.__dict__, f)
    return s


def _run(cmd: list[str], timeout: int) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({r.returncode}): {r.stderr[-600:]}")


def _forget(vid: str, kind: str) -> None:
    p = os.path.join(CACHE_DIR, f"{video_id(vid)}_{kind}.json")
    if os.path.exists(p):
        os.remove(p)


def _frame_cmd(url: str, t: float, out_png: str, width: int) -> list[str]:
    return [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", url, "-frames:v", "1",
            "-vf", f"scale={width}:-2:flags=lanczos", "-update", "1", "-y", out_png]


def _with_retry(vid: str, kind: str, fmt: str, fn, timeout: int):
    """Run fn(stream); on a 403 (stale/IP-bound URL) resolve the stream again once and retry."""
    s = resolve_stream(vid, fmt, kind)
    try:
        fn(s)
    except RuntimeError as e:
        if "403" not in str(e) and "Forbidden" not in str(e):
            raise
        _forget(vid, kind)
        s = resolve_stream(vid, fmt, kind)
        fn(s)
    return s


def grab_frame(vid: str, t: float, out_png: str, width: int = 1920, timeout: int = 180) -> Stream:
    """Frame-accurate single frame at `t` seconds from the best stream, downscaled to `width` px wide."""
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    return _with_retry(vid, "best", FORMAT_BEST, lambda s: _run(_frame_cmd(s.url, t, out_png, width), timeout), timeout)


def tiles(vid: str, start: float, end: float, every: float, tile_w: int, out_dir: str, kind: str = "scout",
          fmt: str = FORMAT_SCOUT, timeout: int = 120, workers: int = 4) -> list[tuple[float, str]]:
    """One frame every `every` seconds in [start, end) -> [(t, png)]. Each tile is its own seek (a few MB each,
    run `workers` at a time); decoding the whole window with an fps filter took 3+ minutes per 10 minutes of video."""
    from concurrent.futures import ThreadPoolExecutor
    import tempfile
    # a private folder per call: several agents scout the same video at once, a shared folder lost their tiles
    os.makedirs(out_dir, exist_ok=True)
    out_dir = tempfile.mkdtemp(prefix="t_", dir=out_dir)
    times = []
    t = start
    while t < end:
        times.append(round(t, 3))
        t += every
    s = resolve_stream(vid, fmt, kind)

    def one(i_t):
        i, tt = i_t
        png = os.path.join(out_dir, f"tile_{i:04d}.png")
        try:
            _run(_frame_cmd(s.url, tt, png, tile_w), timeout)
        except RuntimeError as e:
            if "403" in str(e) or "Forbidden" in str(e):
                _forget(vid, kind)
                _run(_frame_cmd(resolve_stream(vid, fmt, kind).url, tt, png, tile_w), timeout)
            else:
                raise
        return tt, png

    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(one, enumerate(times)))
    return out


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def contact_sheets(frames: list[tuple[float, str]], out_prefix: str, cols: int = 5, rows: int = 5, label: str = "") -> list[str]:
    """Label every tile with its timecode and pack cols x rows per sheet -> [<out_prefix>_sNN.png]."""
    from .sidecar import format_timecode
    if not frames:
        return []
    per = cols * rows
    outs = []
    font = _font(16)
    for si in range(0, len(frames), per):
        chunk = frames[si:si + per]
        w, h = Image.open(chunk[0][1]).size
        sheet = Image.new("RGB", (cols * w, rows * (h + 22)), "#202020")
        d = ImageDraw.Draw(sheet)
        for i, (t, p) in enumerate(chunk):
            x, y = (i % cols) * w, (i // cols) * (h + 22)
            sheet.paste(Image.open(p).convert("RGB"), (x, y + 22))
            d.text((x + 4, y + 3), f"{format_timecode(t)[:-4]}  {label}", fill="white", font=font)
        out = f"{out_prefix}_s{si // per:02d}.png"
        sheet.save(out)
        outs.append(out)
    return outs
