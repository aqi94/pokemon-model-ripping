"""Sidecar / regions JSON schemas and validation for lighting references (no network, no viewer)."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from .scenes import SCENES

SCHEMA = 1
MODEL_RE = re.compile(r"^(tr\d{4}_\d{2}|pm\d{4}_\d{2}_\d{2}|cc_\w+|ob\d{4}_\d{2})$")
TIMES = {"day", "dusk", "night", "any"}
WEATHERS = {"sunny", "overcast", "rain", "none"}
CONTEXTS = {"cutscene", "field", "battle", "vs"}
CONFIDENCE = {"high", "medium", "low"}
SLUG_RE = re.compile(r"^[a-z0-9_]+$")


def parse_timecode(s: str | float | int) -> float:
    """'83', '1:23', '1:23.5', '01:02:03.25' -> seconds."""
    if isinstance(s, (int, float)):
        return float(s)
    parts = s.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(p.strip() for p in parts):
        raise ValueError(f"bad timecode {s!r}")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return secs


def format_timecode(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_sidecar(*, slug: str, scene: str, game: str, video: dict | None, timestamp: float, image: dict,
                 tags: dict, subject: dict, clip_guess: str | None, context: str, quality: dict | None = None,
                 notes: str = "", source: str = "youtube") -> dict:
    return {
        "schema": SCHEMA, "slug": slug, "scene": scene, "game": game, "source": source, "video": video,
        "timestamp": round(timestamp, 3), "timecode": format_timecode(timestamp), "grabbedAt": now_iso(),
        "image": image, "tags": tags, "subject": subject, "clipGuess": clip_guess, "context": context,
        "quality": quality or {"ui": False, "motionBlur": False, "dof": "none", "fade": False}, "notes": notes,
    }


def validate_sidecar(d: dict) -> list[str]:
    errs = []
    if d.get("schema") != SCHEMA:
        errs.append("schema must be 1")
    game = d.get("game")
    if game not in SCENES:
        errs.append(f"unknown game {game!r}")
    elif d.get("scene") not in SCENES[game]:
        errs.append(f"unknown scene {d.get('scene')!r} for {game}")
    slug = d.get("slug") or ""
    if not SLUG_RE.match(slug):
        errs.append(f"slug {slug!r} must be [a-z0-9_]+")
    img = d.get("image") or {}
    if img.get("file") != slug + ".png":
        errs.append("image.file must be <slug>.png")
    for k in ("width", "height"):
        if not isinstance(img.get(k), int) or img[k] <= 0:
            errs.append(f"image.{k} must be a positive int")
    tags = d.get("tags") or {}
    if tags.get("timeOfDay") not in TIMES:
        errs.append(f"tags.timeOfDay {tags.get('timeOfDay')!r} not in {sorted(TIMES)}")
    if tags.get("weather") not in WEATHERS:
        errs.append(f"tags.weather {tags.get('weather')!r} not in {sorted(WEATHERS)}")
    if not isinstance(tags.get("indoor"), bool):
        errs.append("tags.indoor must be a bool")
    subj = d.get("subject") or {}
    if not MODEL_RE.match(subj.get("model") or ""):
        errs.append(f"subject.model {subj.get('model')!r} is not a model id")
    if subj.get("confidence") not in CONFIDENCE:
        errs.append("subject.confidence must be high|medium|low")
    if d.get("context") not in CONTEXTS:
        errs.append(f"context must be one of {sorted(CONTEXTS)}")
    if not isinstance(d.get("timestamp"), (int, float)):
        errs.append("timestamp must be a number (seconds)")
    return errs


def validate_regions(r: dict, sidecar: dict | None = None) -> list[str]:
    errs = []
    for k in ("model", "clip", "frame", "screenshot", "width", "height", "camera", "regions"):
        if k not in r:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if not re.match(r"^[a-z]+/[A-Za-z0-9_]+$", r["model"]):
        errs.append("model must be '<game>/<id>'")
    cam = r["camera"]
    if not (isinstance(cam.get("fov"), (int, float)) and len(cam.get("position", [])) == 3 and len(cam.get("target", [])) == 3):
        errs.append("camera needs fov, position[3], target[3]")
    if not isinstance(r["frame"], int) or r["frame"] < 0:
        errs.append("frame must be a non-negative int")
    if not r["regions"]:
        errs.append("no regions")
    for i, rg in enumerate(r["regions"]):
        for key in ("box",) + (("renderBox",) if "renderBox" in rg else ()):
            b = rg.get(key)
            if not (isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
                errs.append(f"region {i} {key} must be [x0,y0,x1,y1]")
                continue
            if not (0 <= b[0] < b[2] <= r["width"] and 0 <= b[1] < b[3] <= r["height"]):
                errs.append(f"region {i} {key} {b} outside {r['width']}x{r['height']} or inverted")
        if not rg.get("name"):
            errs.append(f"region {i} has no name")
    if sidecar is not None:
        if r["screenshot"] != sidecar["image"]["file"]:
            errs.append("screenshot differs from the sidecar image")
        if (r["width"], r["height"]) != (sidecar["image"]["width"], sidecar["image"]["height"]):
            errs.append("width/height differ from the sidecar image")
        if r["model"].split("/", 1)[1] != sidecar["subject"]["model"]:
            errs.append("model differs from the sidecar subject")
    return errs


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: dict | list) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        f.write("\n")
