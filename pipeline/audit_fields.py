"""Unread-value ledger: which source-data values does the pipeline never look at?

    python -m pipeline --game za audit-fields [--sample 20] [--kind pokemon] [--id tr0566_00] [--no-write]

Imports a sample of models through the game's normal catalog (`cat.import_ref`) with three recorders on:

1. FlatBuffer fields (`fb.recording()`): every field accessor of every table the parsers reach logs
   (table path, field index). Afterwards every table the parsers reached is walked through its vtable; a field that
   is present in the file but was never asked for is "unread". Untouched sub-tables are walked heuristically
   (`fb.guess_field`) a few levels deep so their contents get entries too (`heuristic: true`).
2. Named material params (`.trmtr` flags/colours/floats/ints, SwSh/LGPE material switches/floats/colours): the
   parsed name -> value dicts are swapped for recording dicts, so a param counts as read only when some code
   actually looked it up during the import (no source grep: names used only in composite.py / the viewer are
   judged by what really happens). Neutral values (0, white, black, False) are skipped like `audit_materials` did.
   Texture slots count as read when the importer opened the texture file (or looked the slot up by name).
3. `# FITTED: ...` / `// FITTED: ...` comments in pipeline/ and viewer/src (constants fitted to a few models).

Results go to `docs/ledger/<game>.json`; rewriting keeps hand-written `hypothesis` / `notes` / `status` of existing
keys (see docs/ledger/README.md).
"""
from __future__ import annotations

import builtins
import collections
import datetime as _dt
import json
import os
import re
import sys
import time
import zipfile

from .importers.za import fb

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER_DIR = os.path.join(REPO, "docs", "ledger")
STATUSES = ("unread", "unknown_meaning", "fitted", "constant", "dummy")
HAND_FIELDS = ("hypothesis", "notes")
MAX_DISTINCT = 200  # distinct values tracked per key (the rest counted as "<other>")
GUESSES_PER_MODEL = 64  # value guesses per field per model (instances are always counted)
HEUR_DEPTH = 3  # levels of untouched sub-tables walked heuristically
HEUR_ITEMS = 4  # elements of an untouched vector of tables walked


# --------------------------------------------------------------------------- neutral values

def _neutral(kind: str, value) -> bool:
    """Values that mean "feature off / no effect" for a named param of `kind`."""
    if kind in ("flag", "switch"):
        return value in (False, "False", "false", "0", "", 0, None)
    if kind == "color":
        v = tuple(value)
        return all(abs(x - 1.0) < 1e-3 for x in v[:3]) or all(abs(x) < 1e-3 for x in v[:4])
    if kind in ("float", "int"):
        return value == 0
    if kind == "texture":
        return not value
    return False


def _fmt_value(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_fmt_value(x) for x in v) + ")"
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


# --------------------------------------------------------------------------- stats

class _Stat:
    __slots__ = ("instances", "models", "values", "examples", "types", "fmt", "where", "kind", "heuristic", "mats")

    def __init__(self, fmt: str, where: str, kind: str):
        self.instances = 0
        self.mats = 0
        self.models: set[str] = set()
        self.values: collections.Counter = collections.Counter()
        self.examples: list[str] = []
        self.types: collections.Counter = collections.Counter()
        self.fmt, self.where, self.kind = fmt, where, kind
        self.heuristic = True

    def add_value(self, model: str, value: str, example: str | None = None) -> None:
        if value in self.values or len(self.values) < MAX_DISTINCT:
            self.values[value] += 1
        else:
            self.values["<other>"] += 1
        if example and len(self.examples) < 5 and not any(e.startswith(model + ":") or e.startswith(model + "/") for e in self.examples):
            self.examples.append(example)


# --------------------------------------------------------------------------- recording dict for named params

class _RecDict(dict):
    """A dict that logs which keys were looked up (get / [] / in). Iteration and copies are not logged."""

    __slots__ = ("_tag",)
    reads: set = set()

    def __init__(self, data, tag):
        super().__init__(data)
        self._tag = tag

    def __getitem__(self, k):
        _RecDict.reads.add((self._tag, k))
        return super().__getitem__(k)

    def get(self, k, default=None):
        _RecDict.reads.add((self._tag, k))
        return super().get(k, default)

    def __contains__(self, k):
        _RecDict.reads.add((self._tag, k))
        return super().__contains__(k)


# --------------------------------------------------------------------------- audit

class FieldAudit:
    def __init__(self, game: str):
        self.game = game
        self.rec = fb.Recorder()
        self.fields: dict[tuple[str, int], _Stat] = {}
        self.params: dict[tuple[str, str, str], _Stat] = {}  # (fmt, kind, name)
        self.files_read: set[str] = set()  # lower-case stems of every file opened during imports
        self.tex_slots_read: set[tuple[str, str]] = set()
        self.model = "?"
        self.models_done: list[str] = []
        self.failures: list[tuple[str, str]] = []
        self._patches: list[tuple[object, str, object]] = []

    # -- named params ----------------------------------------------------
    def _param(self, fmt: str, kind: str, name: str, value, mat: str) -> None:
        if _neutral(kind, value):
            return
        st = self.params.get((fmt, kind, name))
        if st is None:
            st = self.params[(fmt, kind, name)] = _Stat(fmt, f"material {kind} param `{name}`", "param")
            st.heuristic = False
        st.instances += 1
        st.mats += 1
        st.models.add(self.model)
        v = _fmt_value(tuple(round(float(x), 3) for x in value) if isinstance(value, tuple) else value)
        st.add_value(self.model, v, f"{self.model}/{mat}={v}")

    def _wrap_trmtr(self, orig):
        def parse_trmtr(data):
            mats = orig(data)
            for tm in mats:
                for kind, attr in (("flag", "shader_flags"), ("color", "colors"), ("float", "floats"), ("int", "ints")):
                    d = getattr(tm, attr)
                    for k, v in d.items():
                        self._param("trmtr", kind, k, v, tm.name)
                    setattr(tm, attr, _RecDict(d, ("trmtr", kind)))
                for t in tm.textures:
                    self._param("trmtr", "texture", t.slot, t.file.replace("\\", "/").split("/")[-1], tm.name)
            return mats
        return parse_trmtr

    def _wrap_gfbmdl(self, orig, fmt: str):
        def parse_gfbmdl(data):
            g = orig(data)
            for m in g.materials:
                for kind, attr in (("switch", "flags"), ("switch", "switches"), ("float", "values"), ("float", "floats"), ("color", "colors")):
                    d = getattr(m, attr, None)
                    if not isinstance(d, dict):
                        continue
                    for k, v in d.items():
                        self._param(fmt, kind, k, v, m.name)
                    setattr(m, attr, _RecDict(d, (fmt, kind)))
                tex = m.textures
                if isinstance(tex, dict):  # LGPE: slot -> texture name
                    for slot, f in tex.items():
                        self._param(fmt, "texture", slot, f, m.name)
                    m.textures = _RecDict(tex, (fmt, "texture"))
                else:  # SwSh: list of TexSlot
                    for t in tex:
                        self._param(fmt, "texture", t.sampler, t.file, m.name)
            return g
        return parse_gfbmdl

    def _patch_function(self, orig, replacement) -> None:
        for mod in list(sys.modules.values()):
            if mod is None or not getattr(mod, "__name__", "").startswith("pipeline"):
                continue
            for attr, val in list(vars(mod).items()):
                if val is orig:
                    self._patches.append((mod, attr, orig))
                    setattr(mod, attr, replacement)

    def _install(self) -> None:
        # import every parser module first so `from x import parse_y` copies exist to patch
        from .importers.lgpe import gfbmdl as lgpe_gfbmdl
        from .importers.lgpe import importer as _li  # noqa: F401
        from .importers.lgpe import field as _lf  # noqa: F401  (field maps/props call parse_gfbmdl too)
        from .importers.swsh import gfbmdl as swsh_gfbmdl
        from .importers.swsh import importer as _si  # noqa: F401
        from .importers.za import importer as _zi  # noqa: F401
        from .importers.za import trinity

        self._patch_function(trinity.parse_trmtr, self._wrap_trmtr(trinity.parse_trmtr))
        self._patch_function(swsh_gfbmdl.parse_gfbmdl, self._wrap_gfbmdl(swsh_gfbmdl.parse_gfbmdl, "gfbmdl"))
        self._patch_function(lgpe_gfbmdl.parse_gfbmdl, self._wrap_gfbmdl(lgpe_gfbmdl.parse_gfbmdl, "gfbmdl"))
        files = self.files_read
        orig_open, orig_zopen = builtins.open, zipfile.ZipFile.open

        def _stem(name) -> str:
            n = str(getattr(name, "filename", name)).replace("\\", "/").split("/")[-1].lower()
            return n.rsplit(".", 1)[0] if "." in n else n

        def rec_open(file, mode="r", *a, **k):
            if isinstance(file, (str, os.PathLike)) and "r" in str(mode):
                files.add(_stem(os.fspath(file)))
            return orig_open(file, mode, *a, **k)

        def rec_zopen(zself, name, mode="r", *a, **k):
            if mode == "r":
                files.add(_stem(name))
            return orig_zopen(zself, name, mode, *a, **k)

        self._patches += [(builtins, "open", orig_open), (zipfile.ZipFile, "open", orig_zopen)]
        builtins.open = rec_open
        zipfile.ZipFile.open = rec_zopen

    def _uninstall(self) -> None:
        for obj, attr, orig in reversed(self._patches):
            setattr(obj, attr, orig)
        self._patches = []

    # -- fields ----------------------------------------------------------
    def _field_stat(self, fmt: str, path: str, idx: int, heuristic: bool) -> _Stat:
        st = self.fields.get((path, idx))
        if st is None:
            st = self.fields[(path, idx)] = _Stat(fmt, f"table `{path.split(':', 1)[1] or '/'}` field {idx}", "field")
        if not heuristic:
            st.heuristic = False
        return st

    def _walk_table(self, buf, pos: int, path: str, fmt: str, heuristic: bool, depth: int, per_model: collections.Counter, seen: set) -> None:
        if (pos, path) in seen:
            return
        seen.add((pos, path))
        try:
            layout = fb.field_layout(buf, pos)
        except Exception:
            return
        for idx, off, size in layout:
            touched = (path, idx) in self.rec.touched
            if touched and not heuristic:
                continue  # read by a parser: nothing to report, no need to describe it
            st = self._field_stat(fmt, path, idx, heuristic)
            st.instances += 1
            st.models.add(self.model)
            per_model[(path, idx)] += 1
            if per_model[(path, idx)] > GUESSES_PER_MODEL:
                continue
            try:
                typ, val = fb.guess_field(buf, off, size)
            except Exception:
                typ, val = "?", None
            st.types[typ] += 1
            if typ == "table":
                shown = "table"
            elif typ == "vec<table>":
                shown = f"vec<table>[{val[0]}]"
            elif typ == "vec":
                shown = f"vec[{val[0]}] {val[1]}"
            else:
                shown = _fmt_value(val)
            st.add_value(self.model, shown, f"{self.model}: {shown}")
            if touched or depth >= HEUR_DEPTH:
                continue
            if typ == "table":
                self._walk_table(buf, val, f"{path}/{idx}", fmt, True, depth + 1, per_model, seen)
            elif typ == "vec<table>":
                n, p0 = val
                for k in range(min(n, HEUR_ITEMS)):
                    p = p0 + 4 * k
                    tpos = p + fb._u32(buf, p)
                    if fb.looks_like_table(buf, tpos):
                        self._walk_table(buf, tpos, f"{path}/{idx}[]", fmt, True, depth + 1, per_model, seen)

    def _collect(self) -> None:
        groups: dict[tuple, tuple[object, str, dict[int, str]]] = {}
        for info in self.rec.take():
            if info.fmt == "?":
                continue
            g = groups.get(info.key)
            if g is None:
                groups[info.key] = (info.buf, info.fmt, dict(info.tables))
            else:
                for pos, p in info.tables.items():
                    g[2].setdefault(pos, p)
        per_model: collections.Counter = collections.Counter()
        for buf, fmt, tables in groups.values():
            seen: set = set()
            for pos, path in tables.items():
                if path.endswith(":?") or ":?" in path:
                    continue
                self._walk_table(buf, pos, path, fmt, False, 0, per_model, seen)

    # -- driver ----------------------------------------------------------
    def run(self, cat, refs, with_anim: bool = True, log=print) -> None:
        _RecDict.reads = set()
        self._install()
        try:
            with fb.recording(self.rec):
                for i, ref in enumerate(refs):
                    self.model = ref.id
                    t0 = time.time()
                    try:
                        cat.import_ref(ref, with_anim)
                        self.models_done.append(ref.id)
                    except Exception as e:  # noqa: BLE001
                        self.failures.append((ref.id, f"{type(e).__name__}: {e}"))
                    self._collect()
                    log(f"[{i + 1}/{len(refs)}] {ref.kind:8s} {ref.id:18s} {time.time() - t0:5.1f}s")
        finally:
            self._uninstall()
        self.param_reads = set(_RecDict.reads)

    # -- results ---------------------------------------------------------
    def unread_fields(self) -> list[tuple[str, _Stat]]:
        out = []
        for (path, idx), st in self.fields.items():
            if (path, idx) in self.rec.touched:
                continue
            if st.heuristic:
                # sub-field of an untouched field: drop it if that parent field is read after all
                head, _, last = path.rpartition("/")
                try:
                    pidx = int(last.rstrip("[]"))
                except ValueError:
                    pidx = None
                if pidx is not None and (head, pidx) in self.rec.touched:
                    continue
            out.append((f"{path}/{idx}", st))
        return out

    def unread_params(self) -> list[tuple[str, _Stat]]:
        out = []
        for (fmt, kind, name), st in self.params.items():
            if kind == "texture":
                if ((fmt, "texture"), name) in self.param_reads:
                    continue
                vals = [v for v in st.values if v != "<other>"]
                stems = {v.lower().rsplit(".", 1)[0] for v in vals}
                if stems & self.files_read:
                    continue
            elif ((fmt, kind), name) in self.param_reads:
                continue
            out.append((f"param:{fmt}:{kind}:{name}", st))
        return out


# --------------------------------------------------------------------------- FITTED comments

FITTED_RE = re.compile(r"(?:#|//)\s*FITTED(?:\(([\w, ]+)\))?:\s*(.+)$")
GAMES = ("za", "sv", "la", "swsh", "lgpe", "bdsp", "go", "unite")


def _games_for_path(rel: str) -> tuple[str, ...]:
    m = re.match(r"pipeline/(?:importers|catalog)/(\w+?)(?:/|\.py|_)", rel)
    if m and m.group(1) in GAMES:
        return ("za", "sv", "la") if m.group(1) == "za" else (m.group(1),)
    return GAMES


def scan_fitted(root: str = REPO) -> list[dict]:
    out = []
    for base, exts in (("pipeline", (".py",)), (os.path.join("viewer", "src"), (".ts", ".tsx"))):
        top = os.path.join(root, base)
        for d, _, files in os.walk(top):
            if "node_modules" in d or "__pycache__" in d or os.sep + "tests" in d:
                continue
            for fn in files:
                if not fn.endswith(exts) or fn == "audit_fields.py":
                    continue
                p = os.path.join(d, fn)
                rel = os.path.relpath(p, root).replace("\\", "/")
                try:
                    lines = open(p, encoding="utf-8").read().splitlines()
                except (OSError, UnicodeDecodeError):
                    continue
                for ln, line in enumerate(lines, 1):
                    m = FITTED_RE.search(line)
                    if not m:
                        continue
                    games = tuple(g.strip() for g in m.group(1).split(",")) if m.group(1) else _games_for_path(rel)
                    text = m.group(2).strip()
                    code = line[: m.start()].strip()
                    out.append({
                        "key": f"fitted:{rel}:{re.sub(r'[^A-Za-z0-9_]+', '_', (code or text)[:60]).strip('_')}",
                        "games": games,
                        "format": "code",
                        "where": f"{rel}:{ln}",
                        "text": text,
                        "code": code[:120],
                    })
    return out


# --------------------------------------------------------------------------- ledger

def _auto_status(st: _Stat) -> str:
    vals = [v for v in st.values if v != "<other>"]
    if vals and all("dummy" in v.lower() for v in vals):
        return "dummy"
    if len(vals) == 1 and st.instances >= 10 and len(st.models) >= 3 and "<other>" not in st.values:
        return "constant"
    return "unread"


def entries_from_audit(audit: FieldAudit, top: int = 8) -> list[dict]:
    out = []
    for key, st in audit.unread_fields() + audit.unread_params():
        e = {
            "key": key,
            "format": st.fmt,
            "where": st.where,
            "status": _auto_status(st),
            "status_auto": True,
            "values": [[v, n] for v, n in st.values.most_common(top)],
            "distinct": len(st.values),
            "instances": st.instances,
            "models": len(st.models),
            "examples": st.examples[:5],
        }
        if st.kind == "field":
            e["types"] = dict(st.types.most_common(4))
            if st.heuristic:
                e["heuristic"] = True
        out.append(e)
    return out


def merge_ledger(old: dict | None, new_entries: list[dict], meta: dict, fitted: list[dict] | None = None) -> dict:
    """Merge scanned entries into an existing ledger. Data fields come from the scan; `hypothesis`, `notes` and a
    hand-set `status` (one without `status_auto`) survive. Keys missing from the scan are kept (marked
    `seen: false`) when they carry hand-written content, else dropped."""
    old_entries = {e["key"]: e for e in (old or {}).get("entries", [])}
    merged: dict[str, dict] = {}
    for e in new_entries + [_fitted_entry(f) for f in (fitted or [])]:
        prev = old_entries.get(e["key"])
        e = dict(e)
        e.setdefault("hypothesis", "")
        e.setdefault("notes", "")
        e["seen"] = True
        if prev:
            for f in HAND_FIELDS:
                if prev.get(f):
                    e[f] = prev[f]
            if prev.get("status") and not prev.get("status_auto"):
                e["status"] = prev["status"]
                e.pop("status_auto", None)
        merged[e["key"]] = e
    for k, prev in old_entries.items():
        if k in merged:
            continue
        hand = any(prev.get(f) for f in HAND_FIELDS) or (prev.get("status") and not prev.get("status_auto"))
        if hand:
            prev = dict(prev)
            prev["seen"] = False
            merged[k] = prev
    ordered = sorted(merged.values(), key=lambda e: (e.get("status") == "constant", -int(e.get("models") or 0), e["key"]))
    return {"game": meta.get("game"), "meta": meta, "entries": ordered}


def _fitted_entry(f: dict) -> dict:
    return {
        "key": f["key"],
        "format": "code",
        "where": f["where"],
        "status": "fitted",
        "status_auto": True,
        "values": [[f["code"], 1]] if f.get("code") else [],
        "models": 0,
        "examples": [],
        "comment": f["text"],  # rewritten from the source on every scan (not hand content)
        "hypothesis": "",
        "notes": "",
    }


def load_ledger(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_ledger(path: str, ledger: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- sampling + CLI glue

def sample_refs(cat, kinds: list[str], n: int, ids: list[str] | None = None) -> list:
    """`n` models per kind, evenly spread over the sorted catalog (covers early and late dex numbers / classes)."""
    out, seen = [], set()
    for kind in kinds:
        refs = cat.by_kind(kind) if hasattr(cat, "by_kind") else {"pokemon": cat.pokemon, "trainer": cat.trainers, "object": cat.objects}[kind]()
        if not refs:
            continue
        refs = sorted(refs, key=lambda r: r.id)
        if n and len(refs) > n:
            step = len(refs) / n
            refs = [refs[int(i * step)] for i in range(n)]
        for r in refs:
            if r.id not in seen:
                seen.add(r.id)
                out.append(r)
    for mid in ids or []:
        r = cat.find(mid)
        if r is None:
            print(f"audit-fields: unknown id {mid}", file=sys.stderr)
        elif r.id not in seen:
            seen.add(r.id)
            out.append(r)
    return out


def run_audit(game: str, cat, kinds: list[str], sample: int, ids: list[str] | None = None, with_anim: bool = True,
              ledger_dir: str = LEDGER_DIR, write: bool = True, top: int = 8, log=print) -> tuple[FieldAudit, dict]:
    refs = sample_refs(cat, kinds, sample, ids)
    audit = FieldAudit(game)
    t0 = time.time()
    audit.run(cat, refs, with_anim=with_anim, log=log)
    meta = {
        "game": game,
        "scanned": _dt.date.today().isoformat(),
        "sample_per_kind": sample,
        "kinds": kinds,
        "models": audit.models_done,
        "failures": [f"{m}: {e}" for m, e in audit.failures],
        "formats": sorted(audit.rec.formats),
        "seconds": round(time.time() - t0, 1),
        "tool": "python -m pipeline --game %s audit-fields --sample %d" % (game, sample),
    }
    fitted = [f for f in scan_fitted() if game in f["games"]]
    path = os.path.join(ledger_dir, f"{game}.json")
    ledger = merge_ledger(load_ledger(path), entries_from_audit(audit, top), meta, fitted)
    if write:
        write_ledger(path, ledger)
    return audit, ledger


def print_summary(ledger: dict, limit: int = 40) -> None:
    ents = [e for e in ledger["entries"] if e.get("seen", True)]
    by_status = collections.Counter(e["status"] for e in ents)
    print(f"\n{ledger['game']}: {len(ents)} entries {dict(by_status)}; models {len(ledger['meta']['models'])}, "
          f"failures {len(ledger['meta']['failures'])}, {ledger['meta']['seconds']} s")
    shown = 0
    for e in ents:
        if e["status"] not in ("unread", "unknown_meaning"):
            continue
        vals = "; ".join(f"{v} x{n}" for v, n in e["values"][:4])
        print(f"  {e['models']:3d} models {e['key'][:70]:70s} {vals[:90]}")
        shown += 1
        if shown >= limit:
            break


def cmd_audit_fields(args, catalog_factory) -> None:
    kinds = args.kind or ["pokemon", "trainer", "object"]
    cat = catalog_factory(args)
    _, ledger = run_audit(args.game, cat, kinds, args.sample, args.id, with_anim=not args.no_anim,
                          ledger_dir=args.ledger_dir or LEDGER_DIR, write=not args.no_write, top=args.top)
    print_summary(ledger, args.show)
