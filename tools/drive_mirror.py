"""Mirror a public Google Drive folder file by file (no zips), resumable, with a completeness check.

Drive's folder "Download" packs the folder into several zip parts; the UNITE rip downloaded that way lost most files
(2026-11-01: local Bulbasaur battle folder had 1 of its 8 files, 852 of 878 battle folders). This walks the tree through
Drive's embedded folder view (lists every entry; gdown's folder listing stops at 50) and downloads each file by id from
`drive.usercontent.google.com/download` (gdown 6.5.1 could not resolve these links: "Cannot retrieve the public link",
while the direct endpoint serves them anonymously). An HTML answer (quota / virus-scan page) counts as a failure.

  python -m tools.drive_mirror list   <folder_id> <dest>          # walk + write <dest>/_drive_manifest.json
  python -m tools.drive_mirror fetch  <dest> [--workers 4]        # download missing files from the manifest
  python -m tools.drive_mirror verify <dest>                      # every manifest file present and non-empty
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

VIEW = "https://drive.google.com/embeddedfolderview?id={}"
DOWNLOAD = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
ENTRY = re.compile(r'<div class="flip-entry" id="entry-([A-Za-z0-9_-]+)".*?<a href="([^"]+)".*?flip-entry-title">([^<]*)<', re.S)
MANIFEST = "_drive_manifest.json"


def _get(url: str, tries: int = 4) -> str:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    return ""


def list_folder(folder_id: str) -> list[dict]:
    out = []
    for eid, href, name in ENTRY.findall(_get(VIEW.format(folder_id))):
        out.append({"id": eid, "name": html.unescape(name).strip(), "folder": "/folders/" in href})
    return out


def walk(folder_id: str, rel: str = "", files: list | None = None, workers: int = 8) -> list[dict]:
    """All files under `folder_id` as {id, path}. Subfolders are listed in parallel, one level at a time."""
    files = [] if files is None else files
    level = [(folder_id, rel)]
    while level:
        nxt = []
        with ThreadPoolExecutor(workers) as ex:
            futs = {ex.submit(list_folder, fid): r for fid, r in level}
            for f in as_completed(futs):
                r = futs[f]
                for e in f.result():
                    p = f"{r}/{e['name']}" if r else e["name"]
                    (nxt.append((e["id"], p)) if e["folder"] else files.append({"id": e["id"], "path": p}))
        print(f"  listed {len(level)} folders, {len(files)} files so far", file=sys.stderr)
        level = nxt
    return files


def download(file_id: str, out: str) -> bool:
    req = urllib.request.Request(DOWNLOAD.format(file_id), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        if "text/html" in (r.headers.get("Content-Type") or ""):
            return False
        tmp = out + ".part"
        with open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    if os.path.getsize(tmp) == 0:
        return False
    os.replace(tmp, out)
    return True


def fetch(dest: str, workers: int = 4) -> list[str]:
    files = json.load(open(os.path.join(dest, MANIFEST), encoding="utf-8"))["files"]
    todo = [f for f in files if not (os.path.isfile(os.path.join(dest, f["path"])) and os.path.getsize(os.path.join(dest, f["path"])) > 0)]
    print(f"{len(files)} files in manifest, {len(todo)} to download", file=sys.stderr)
    failed: list[str] = []

    def one(f: dict) -> str | None:
        out = os.path.join(dest, f["path"])
        os.makedirs(os.path.dirname(out), exist_ok=True)
        for k in range(4):
            try:
                if download(f["id"], out):
                    return None
            except Exception:  # noqa: BLE001
                pass
            time.sleep(3 * (k + 1))
        return f["path"]

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for r in as_completed([ex.submit(one, f) for f in todo]):
            done += 1
            if r := r.result():
                failed.append(r)
            if done % 100 == 0:
                print(f"  {done}/{len(todo)} ({len(failed)} failed)", file=sys.stderr)
    return failed


def verify(dest: str) -> list[str]:
    files = json.load(open(os.path.join(dest, MANIFEST), encoding="utf-8"))["files"]
    return [f["path"] for f in files if not (os.path.isfile(os.path.join(dest, f["path"])) and os.path.getsize(os.path.join(dest, f["path"])) > 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("folder_id"); p.add_argument("dest")
    p = sub.add_parser("fetch"); p.add_argument("dest"); p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("verify"); p.add_argument("dest")
    a = ap.parse_args()
    if a.cmd == "list":
        os.makedirs(a.dest, exist_ok=True)
        files = walk(a.folder_id)
        json.dump({"folder": a.folder_id, "listed": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": files},
                  open(os.path.join(a.dest, MANIFEST), "w", encoding="utf-8"), indent=0)
        print(f"{len(files)} files listed")
    elif a.cmd == "fetch":
        failed = fetch(a.dest, a.workers)
        print(f"failed: {len(failed)}", *failed[:20], sep="\n")
    else:
        missing = verify(a.dest)
        print(f"missing or empty: {len(missing)}", *missing[:20], sep="\n")
        sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
