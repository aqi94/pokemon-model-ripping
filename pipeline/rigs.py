"""Rig signatures and rig families: which models share a skeleton, and whether borrowed clips bind.

Generic NPCs in Z-A / SV animate from abstract base rigs (`cc_base`, `motion_cc_base` personality sets), and the
catalogs borrow clips between models in several ways. This module makes the compatibility measurable instead of
assumed:

- `rig_signature(skel)`: joint names, parent names, bind-local translation / rotation (rounded), bind world
  positions and the palette (joint names in vertex-index order).
- `compare(a, b)`: shared / missing joints, parent mismatches, max bind deltas, joint-order / palette equality,
  subset relations and a median length ratio (proportions: Kid vs Adult).
- `clip_coverage(target, clips)`: the fraction of clip tracks whose bone exists in a rig (what the glTF writer
  would keep; it drops the rest, see `gltf_writer.dropped_tracks`).
- `families(cat)`: clusters a catalog's models by signature and reports clip holders, 0-clip members of a family
  with clips (candidate borrowers) and subset / same-topology-different-proportion relations between families.

`python -m pipeline --game za|sv|la|swsh|lgpe rigs families|compare` is the CLI (`pipeline/cli.py`).
Skeletons are read cheaply (only `.trmdl` + `.trskl`, or the bone table of a `.gfbmdl`), never whole models.
"""

from __future__ import annotations

import bisect
import math
import os
import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .common.model import Joint, Skeleton

# A joint whose bind-local translation differs by more than this (metres) means different proportions.
DEFAULT_TOL = 0.002
ROT_TOL_DEG = 1.0
JACCARD_MIN = 0.9  # joint-name overlap for "same rig modulo extras"
MIN_RELATION_JOINTS = 10  # tiny rigs (1-3 locator joints) are a subset of everything


# --------------------------------------------------------------------------- signature


@dataclass
class RigSignature:
    names: list[str]
    parents: list[str]  # parent joint name, "" for roots
    t: np.ndarray  # (N, 3) bind-local translation, rounded
    r: np.ndarray  # (N, 4) bind-local rotation (x, y, z, w), w >= 0 (exact; rounded only in key())
    world: np.ndarray  # (N, 3) bind world position
    palette: list[str]  # joint names in vertex joint-index order
    skinned: frozenset[str]
    index: dict[str, int] = field(default_factory=dict)

    @property
    def name_set(self) -> frozenset[str]:
        return frozenset(self.index)

    def __len__(self) -> int:
        return len(self.names)

    def key(self) -> tuple:
        """Exact (rounded) identity: equal keys = interchangeable rigs."""
        return tuple(zip(self.names, self.parents, map(tuple, self.t.tolist()), map(tuple, (np.round(self.r, 4) + 0.0).tolist())))


def _world_positions(joints: list[Joint]) -> np.ndarray:
    from .importers.za.composite import _world_matrices

    if not joints:
        return np.zeros((0, 3))
    w = _world_matrices(joints)
    return np.array([w[i][:3, 3] for i in range(len(joints))], dtype=np.float64)


def rig_signature(skel: Skeleton, decimals: int = 4) -> RigSignature:
    names = [j.name for j in skel.joints]
    parents = [skel.joints[j.parent].name if 0 <= j.parent < len(skel.joints) else "" for j in skel.joints]
    t = np.round(np.array([j.translation for j in skel.joints], dtype=np.float64).reshape(-1, 3), decimals)
    r = np.array([j.rotation for j in skel.joints], dtype=np.float64).reshape(-1, 4)
    r[r[:, 3] < 0] *= -1.0
    index: dict[str, int] = {}
    for i, n in enumerate(names):
        index.setdefault(n, i)
    palette = [names[p] if 0 <= p < len(names) else "" for p in skel.palette]
    return RigSignature(
        names=names, parents=parents, t=t + 0.0, r=r, world=_world_positions(skel.joints), palette=palette,
        skinned=frozenset(j.name for j in skel.joints if j.skinning), index=index,
    )


def _as_signature(x) -> RigSignature:
    return x if isinstance(x, RigSignature) else rig_signature(x)


@dataclass
class RigComparison:
    a_joints: int
    b_joints: int
    shared: list[str]
    only_a: list[str]
    only_b: list[str]
    parent_mismatch: list[str]  # shared joints whose parent name differs
    max_local_t: float  # metres, over shared joints with the same parent
    max_local_t_joint: str
    max_rot_deg: float
    max_rot_joint: str
    max_world: float  # metres, bind world position over shared joints
    max_world_joint: str
    length_ratio: float  # median |t_b| / |t_a| over shared bones longer than 1 cm (1.0 = same proportions)
    same_order: bool  # identical joint lists (node indices interchangeable)
    same_palette: bool  # identical vertex joint-index -> name tables
    palette_prefix: bool  # one palette starts with the other (a base rig inside a composite)

    @property
    def jaccard(self) -> float:
        u = len(self.shared) + len(self.only_a) + len(self.only_b)
        return len(self.shared) / u if u else 1.0

    @property
    def a_subset_b(self) -> bool:
        return not self.only_a

    @property
    def b_subset_a(self) -> bool:
        return not self.only_b

    def same_bind(self, tol: float = DEFAULT_TOL, rot_tol_deg: float = ROT_TOL_DEG) -> bool:
        return not self.parent_mismatch and self.max_local_t <= tol and self.max_rot_deg <= rot_tol_deg

    def compatible(self, tol: float = DEFAULT_TOL, rot_tol_deg: float = ROT_TOL_DEG) -> bool:
        """Clips of either play on the other: same rig modulo extra joints, same bind pose on the shared ones."""
        return (self.jaccard >= JACCARD_MIN or self.a_subset_b or self.b_subset_a) and self.same_bind(tol, rot_tol_deg)

    def as_dict(self) -> dict:
        return {
            "joints": [self.a_joints, self.b_joints], "shared": len(self.shared), "onlyA": self.only_a, "onlyB": self.only_b,
            "jaccard": round(self.jaccard, 4), "parentMismatch": self.parent_mismatch,
            "maxLocalT": [round(self.max_local_t, 5), self.max_local_t_joint], "maxRotDeg": [round(self.max_rot_deg, 3), self.max_rot_joint],
            "maxWorld": [round(self.max_world, 5), self.max_world_joint], "lengthRatio": round(self.length_ratio, 4),
            "sameOrder": self.same_order, "samePalette": self.same_palette, "palettePrefix": self.palette_prefix,
        }


def compare(a, b) -> RigComparison:
    """`a`, `b`: Skeleton or RigSignature."""
    a, b = _as_signature(a), _as_signature(b)
    shared = [n for n in a.index if n in b.index]
    ia = np.array([a.index[n] for n in shared], dtype=int)
    ib = np.array([b.index[n] for n in shared], dtype=int)
    pm = [n for n in shared if a.parents[a.index[n]] != b.parents[b.index[n]]]
    same_parent = np.array([n not in set(pm) for n in shared], dtype=bool) if shared else np.zeros(0, bool)

    def worst(d: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, str]:
        if not len(d):
            return 0.0, ""
        d = np.where(mask, d, -1.0) if mask is not None else d
        k = int(np.argmax(d))
        return (float(d[k]), shared[k]) if d[k] >= 0 else (0.0, "")

    if shared:
        dt = np.linalg.norm(a.t[ia] - b.t[ib], axis=1)
        dots = np.clip(np.abs(np.sum(a.r[ia] * b.r[ib], axis=1)), 0.0, 1.0)
        dr = np.degrees(2.0 * np.arccos(dots))
        dw = np.linalg.norm(a.world[ia] - b.world[ib], axis=1)
        la, lb = np.linalg.norm(a.t[ia], axis=1), np.linalg.norm(b.t[ib], axis=1)
        long = (la > 0.01) & same_parent
        ratio = float(np.median(lb[long] / la[long])) if long.any() else 1.0
    else:
        dt = dr = dw = np.zeros(0)
        ratio = 1.0
    t_max, t_j = worst(dt, same_parent)
    r_max, r_j = worst(dr, same_parent)
    w_max, w_j = worst(dw)
    pa, pb = a.palette, b.palette
    short, long_ = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
    return RigComparison(
        a_joints=len(a), b_joints=len(b), shared=shared,
        only_a=[n for n in a.index if n not in b.index], only_b=[n for n in b.index if n not in a.index],
        parent_mismatch=pm, max_local_t=t_max, max_local_t_joint=t_j, max_rot_deg=r_max, max_rot_joint=r_j,
        max_world=w_max, max_world_joint=w_j, length_ratio=ratio,
        same_order=a.names == b.names and a.parents == b.parents, same_palette=pa == pb,
        palette_prefix=bool(short) and long_[: len(short)] == short,
    )


# --------------------------------------------------------------------------- clip coverage


@dataclass
class Coverage:
    tracks: int
    bound: int
    missing: Counter  # bone name -> tracks lost
    clips: int = 0

    @property
    def fraction(self) -> float:
        return self.bound / self.tracks if self.tracks else 1.0

    def __str__(self) -> str:
        miss = ", ".join(f"{n}" for n, _ in self.missing.most_common(6))
        return f"{100 * self.fraction:.1f}% ({self.bound}/{self.tracks} tracks, {self.clips} clips)" + (f" missing: {miss}" if miss else "")


def _track_bones(clip) -> list[str]:
    tracks = getattr(clip, "tracks", None)
    if tracks is None:
        return list(clip)  # already a list of bone names
    return [getattr(t, "joint", None) or getattr(t, "bone", "") for t in tracks]


def clip_coverage(target, clips: Iterable) -> Coverage:
    """Fraction of tracks whose bone exists in `target` (Skeleton, RigSignature or a set of joint names).

    `clips`: IR `Clip`s (tracks with `.joint`; one entry per channel), parsed `.tranm` / `.gfbanm` (tracks with
    `.bone`; one entry per bone) or plain lists of bone names."""
    if isinstance(target, Skeleton):
        names = {j.name for j in target.joints}
    elif isinstance(target, RigSignature):
        names = set(target.index)
    else:
        names = set(target)
    cov = Coverage(0, 0, Counter())
    for c in clips:
        cov.clips += 1
        for bone in _track_bones(c):
            cov.tracks += 1
            if bone in names:
                cov.bound += 1
            else:
                cov.missing[bone] += 1
    return cov


def clip_bind_agreement(target, tranms: Iterable, tol: float = 1e-3, skip: Iterable[str] = ("origin",)) -> tuple[int, int]:
    """(agreeing, total) translation tracks of parsed `.tranm` clips whose first key equals the rig's bind-local
    translation within `tol` metres. Animators key most bones at their bind offset, so clips authored on a rig agree
    with it almost everywhere: this checks a clip set against a skeleton it was not shipped with (CC_SKELETON_DONORS:
    the female cc bases have clips but no skeleton in the rip). Root-motion bones (`skip`) are left out."""
    sig = _as_signature(target)
    skip = set(skip)
    ok = total = 0
    for tr in tranms:
        for bt in tr.tracks:
            ch = bt.translation
            if ch is None or bt.bone in skip or bt.bone not in sig.index or not len(ch.values):
                continue
            total += 1
            ok += float(np.linalg.norm(np.asarray(ch.values[0], np.float64) - sig.t[sig.index[bt.bone]])) <= tol
    return ok, total


# --------------------------------------------------------------------------- skeleton loading


def trinity_skeleton(read: Callable[[str], bytes], exists: Callable[[str], bool], stem: str) -> Skeleton | None:
    """`<stem>.trmdl` -> its `.trskl` -> Skeleton (names resolved relative to the model folder)."""
    from .importers.za import trinity as T
    from .importers.za.importer import _build_skeleton

    if not exists(f"{stem}.trmdl"):
        return None
    trmdl = T.parse_trmdl(read(f"{stem}.trmdl"))
    if not trmdl.skeleton or not exists(trmdl.skeleton):
        return None
    return _build_skeleton(T.parse_trskl(read(trmdl.skeleton)))


class _ZipIndex:
    """name -> zip over several zips (later win), with sorted prefix listing; built once per zip list."""

    def __init__(self, zips):
        self.map = {}
        for z in zips:
            for n in z.namelist():
                if not n.endswith("/"):
                    self.map[n] = z
        self.sorted = sorted(self.map)

    def list(self, prefix: str) -> list[str]:
        i = bisect.bisect_left(self.sorted, prefix)
        out = []
        while i < len(self.sorted) and self.sorted[i].startswith(prefix):
            out.append(self.sorted[i])
            i += 1
        return out

    def read(self, name: str) -> bytes:
        return self.map[name].read(name)


@dataclass
class RigEntry:
    ref: object  # ModelRef
    sig: RigSignature
    own: list  # clip keys of the model's own animation folder
    inherited: list  # clip keys the catalog borrows for it (cc sets, sv motion plans, cc_ir donors)

    @property
    def clips(self) -> int:
        return len(self.own) + len(self.inherited)

    @property
    def id(self) -> str:
        return self.ref.id


class _Rigs:
    """Per-game adapter: refs, cheap skeletons, clip keys and their bone names."""

    def __init__(self, cat):
        self.cat = cat
        self._bones: dict = {}

    def refs(self, kinds: list[str] | None = None) -> list:
        kinds = kinds or ["pokemon", "trainer", "object"]
        out = []
        for k in kinds:
            out += {"pokemon": self.cat.pokemon, "trainer": self.cat.trainers, "object": self.cat.objects}[k]()
        if "trainer" in kinds:
            out += self.base_refs()
        return out

    def base_refs(self) -> list:
        """Abstract base rigs the catalog does not list as models (kind "base", id `base:<stem>`)."""
        return []

    def find(self, model_id: str):
        if model_id.startswith("base:"):
            return next((r for r in self.base_refs() if r.id == model_id), None)
        return self.cat.find(model_id)

    def skeleton(self, ref) -> Skeleton | None:
        raise NotImplementedError

    def own_clips(self, ref) -> list:
        return []

    def inherited_clips(self, ref) -> list:
        return []

    def _read_bones(self, key) -> list[str]:
        raise NotImplementedError

    def clip_bones(self, key) -> list[str]:
        if key not in self._bones:
            try:
                self._bones[key] = self._read_bones(key)
            except Exception:  # noqa: BLE001 - material-only / broken clips bind nothing
                self._bones[key] = []
        return self._bones[key]

    def entry(self, ref) -> RigEntry | None:
        try:
            sk = self.skeleton(ref)
        except Exception:  # noqa: BLE001 - one unreadable rig must not stop a survey
            return None
        if sk is None or not sk.joints:
            return None
        return RigEntry(ref, rig_signature(sk), self.own_clips(ref), self.inherited_clips(ref))


class _TrinityRigs(_Rigs):
    """za / sv / la (ZaCatalog and subclasses)."""

    def __init__(self, cat):
        super().__init__(cat)
        self._idx: dict[tuple, _ZipIndex] = {}

    def index(self, zip_names: list[str]) -> _ZipIndex:
        key = tuple(zip_names)
        if key not in self._idx:
            self._idx[key] = _ZipIndex(self.cat.zips(list(zip_names)))
        return self._idx[key]

    BASE_RIG_RE = re.compile(r"(?:^|/)model_cc_base/(?:[^/]+/)*([^/]+)\.trmdl$")
    MOTION_SET_RE = re.compile(r"cc\d{4}_[a-z_]+$")

    def base_refs(self) -> list:
        """`model_cc_base` skeletons without a catalog model of their own (SV `cc_base`, LA `cc_base0001`); Z-A lists
        its bases as the `cc_<base>` composites. Their clips are the shared `cc####_*` motion sets."""
        from .catalog.za import ModelRef

        c = self.cat
        if getattr(c, "CC_MODEL_ZIPS", None):
            return []
        if hasattr(self, "_base_refs"):
            return self._base_refs
        found = []
        for n in self.index(list(c.TRAINER_MODEL_ZIPS)).sorted:
            m = self.BASE_RIG_RE.search(n)
            if m and not m.group(1).endswith("_loc"):
                found.append((n[: n.rindex("/") + 1], m.group(1)))
        extract = getattr(c, "extract_dir", None)
        if extract:
            root = os.path.join(extract, "trainers", "chara", "data", "cc", "base")
            for d in sorted(os.listdir(root)) if os.path.isdir(root) else []:
                if os.path.isfile(os.path.join(root, d, d + ".trmdl")):
                    found.append((os.path.join(root, d), d))
        self._base_refs = [
            ModelRef(game=c.GAME, kind="base", id=f"base:{stem}", folder=folder, display_name=f"{stem} (abstract base rig)", extras={"stem": stem, "base_rig": True})
            for folder, stem in found
        ]
        return self._base_refs

    def skeleton(self, ref) -> Skeleton | None:
        c = self.cat
        stem = ref.extras.get("stem", ref.id)
        if os.path.isdir(ref.folder):  # LA trainers/items: extracted folders
            folder = ref.folder
            return trinity_skeleton(lambda n: open(os.path.join(folder, n), "rb").read(), lambda n: os.path.isfile(os.path.join(folder, n)), stem)
        if ref.extras.get("cc_base"):
            zips, stem = c.CC_MODEL_ZIPS, f"{ref.extras['cc_skeleton']}_base"
        elif ref.extras.get("cc_ir"):
            zips = c.CC_MODEL_ZIPS
        elif ref.kind == "pokemon":
            zips = c.POKE_MODEL_ZIPS
        else:
            zips = c.TRAINER_MODEL_ZIPS
        idx = self.index(zips)
        prefix = ref.folder.rstrip("/") + "/"

        def full(n: str) -> str:
            return posixpath.normpath(prefix + n.replace("\\", "/"))

        return trinity_skeleton(lambda n: idx.read(full(n)), lambda n: full(n) in idx.map, stem)

    def _anim_index(self, ref) -> _ZipIndex:
        c = self.cat
        return self.index(c.POKE_ANIM_ZIPS if ref.kind == "pokemon" else c.TRAINER_ANIM_ZIPS)

    def own_clips(self, ref) -> list:
        from .importers.za.importer import clip_name_from_file

        if ref.extras.get("base_rig"):
            if not self.cat.TRAINER_ANIM_ZIPS:
                return []
            idx = self.index(self.cat.TRAINER_ANIM_ZIPS)
            root = getattr(self.cat, "CC_ANIM_ROOT", "").rstrip("/") + "/"
            if self.cat.GAME == "sv":
                root = "chara/motion_cc_base/"
            out = []
            for n in idx.list(root):
                parts = n[len(root):].split("/")
                if len(parts) == 2 and self.MOTION_SET_RE.match(parts[0]) and parts[1].endswith(".tranm"):
                    out.append((idx, n))
            return out
        if not ref.anim_folder:
            return []
        idx = self._anim_index(ref)
        prefix = ref.anim_folder.rstrip("/") + "/"
        out = []
        for n in idx.list(prefix):
            rel = n[len(prefix):]
            if not rel.endswith(".tranm"):
                continue
            if ref.extras.get("cc_base"):
                if "/" not in rel:
                    continue  # cc bodies: one sub-folder per motion set
            elif "/" in rel:
                continue
            base = rel.rsplit("/", 1)[-1]
            if ref.kind == "object" and not base.startswith(ref.extras.get("clip_id", ref.id)):
                continue
            if self.cat.GAME == "la" and clip_name_from_file(base) is None:
                continue  # `anm/<model>.tranm` template
            out.append((idx, n))
        return out

    def inherited_clips(self, ref) -> list:
        c = self.cat
        out = []
        idx = self.index(c.TRAINER_ANIM_ZIPS) if c.TRAINER_ANIM_ZIPS else None
        if idx is None:
            return out
        if ref.extras.get("cc_ir") and hasattr(c, "cc_ir_body_donor"):
            donor = c.cc_ir_body_donor(ref)
            if donor:
                prefix = f"{c.CC_ANIM_ROOT}/{donor}/"
                out += [(idx, n) for n in idx.list(prefix) if n.endswith(".tranm") and re.match(r"cc\d{4}_", n[len(prefix):])]
        plan = ref.extras.get("sv_motion")
        if plan:
            from .catalog.sv import MOTION_CC_ROOT

            if plan.get("own"):
                prefix = f"{MOTION_CC_ROOT}/{plan['own']}/"
                out += [(idx, n) for n in idx.list(prefix) if n.endswith(".tranm") and "/" not in n[len(prefix):]]
            for folder in plan.get("sets", []):
                prefix = f"{MOTION_CC_ROOT}/{folder}/"
                out += [(idx, n) for n in idx.list(prefix) if n.endswith(".tranm") and "/" not in n[len(prefix):]]
            for clip in plan.get("clips", []):
                n = f"{MOTION_CC_ROOT}/{clip}.tranm"
                if n in idx.map:
                    out.append((idx, n))
        return out

    def _read_bones(self, key) -> list[str]:
        from .importers.za import trinity as T

        idx, n = key
        return [t.bone for t in T.parse_tranm(idx.read(n)).tracks]


class _SwshRigs(_Rigs):
    def skeleton(self, ref) -> Skeleton | None:
        from .importers.swsh import gfbmdl as G
        from .importers.swsh.importer import build_skeleton

        if ref.kind == "pokemon":
            hit = self.cat._model_members().get(f"{ref.folder}mdl/{ref.id}.gfbmdl")
            if hit is None:
                return None
            data = hit[0].read(hit[1])
        else:
            with open(os.path.join(ref.folder, ref.extras["model"]), "rb") as f:
                data = f.read()
        bones = _gfb_bones(data)
        return build_skeleton(G.GfbModel(texture_files=[], materials=[], meshes=[], shapes=[], bones=[
            G.Bone(name=b["name"], type=b["type"], parent=b["parent"], visible=True, scale=b["scale"], rotation=b["euler"], translation=b["t"], rigged=b["rigged"])
            for b in bones
        ])) if bones else None

    def own_clips(self, ref) -> list:
        return [(z, m) for z, m, _ in self.cat._anim_index().get(ref.id, []) if not m.endswith(f"/{ref.id}.gfbanm")]

    def _read_bones(self, key) -> list[str]:
        from .importers.swsh.gfbanm import parse_gfbanm

        z, m = key
        return [t.bone for t in parse_gfbanm(z.read(m)).tracks]


class _LgpeRigs(_Rigs):
    def skeleton(self, ref) -> Skeleton | None:
        from .importers.lgpe import gfbmdl as G
        from .importers.lgpe.importer import build_skeleton

        if ref.kind == "trainer":
            with open(os.path.join(ref.folder, f"{ref.extras['folder']}.gfbmdl"), "rb") as f:
                data = f.read()
        else:
            data = self.cat.zip("LGPE-Poke.zip").read(ref.folder + f"{ref.id}.gfbmdl")
        bones = _gfb_bones(data, lgpe=True)
        return build_skeleton(G.Gfbmdl(version=0, bounds=(), textures=[], materials=[], meshes=[], bones=[
            G.GfBone(name=b["name"], parent=b["parent"], scale=b["scale"], euler=b["euler"], translation=b["t"]) for b in bones
        ])) if bones else None

    def own_clips(self, ref) -> list:
        from .catalog.lgpe import ANIM_ZIP

        if not ref.anim_folder:
            return []
        z = self.cat.zip(ANIM_ZIP)
        return [(z, n) for n in sorted(self.cat.names_in(ANIM_ZIP)) if n.startswith(ref.anim_folder) and n.endswith(".gfbanm") and "/" not in n[len(ref.anim_folder):]]

    def _read_bones(self, key) -> list[str]:
        from .importers.za import trinity as T

        z, n = key
        return [t.bone for t in T.parse_tranm(z.read(n)).tracks]


def _gfb_bones(data: bytes, lgpe: bool = False) -> list[dict]:
    """Bone table (field 9) of a SwSh / LGPE `.gfbmdl` without decoding meshes or materials."""
    from .importers.za import fb

    r = fb.root(data)
    out = []
    for b in r.tables(9):
        out.append({
            "name": b.string(0),
            "type": b.u32(1),
            "parent": (b.i32(2, -1) if b.has(2) else 0) if lgpe else b.i32(2),
            "scale": b.floats(5, 3) or (1.0, 1.0, 1.0),
            "euler": b.floats(6, 3) or (0.0, 0.0, 0.0),
            "t": b.floats(7, 3) or (0.0, 0.0, 0.0),
            "rigged": b.bool(10, True),
        })
    return out


def rig_source(cat) -> _Rigs:
    game = getattr(cat, "GAME", "")
    if game in ("za", "sv", "la"):
        return _TrinityRigs(cat)
    if game == "swsh":
        return _SwshRigs(cat)
    if game == "lgpe":
        return _LgpeRigs(cat)
    raise ValueError(f"rigs: no skeleton reader for game '{game}'")


# --------------------------------------------------------------------------- families


@dataclass
class Family:
    rep: RigEntry  # first member (a clip holder when the family has one)
    members: list[RigEntry]
    deltas: dict[str, RigComparison] = field(default_factory=dict)  # member id -> compare(rep, member)

    @property
    def holders(self) -> list[RigEntry]:
        return [m for m in self.members if m.clips]

    @property
    def borrowers(self) -> list[RigEntry]:
        return [m for m in self.members if not m.clips] if self.holders else []


@dataclass
class Relation:
    kind: str  # "subset" (a's joints all in b, same bind) | "proportions" (same topology, bind differs)
    a: int  # family indexes
    b: int
    cmp: RigComparison


def families(
    cat, kinds: list[str] | None = None, tol: float = DEFAULT_TOL, progress: Callable[[int, int], None] | None = None,
    rigs: _Rigs | None = None,
) -> tuple[list[Family], list[Relation], _Rigs]:
    """Cluster the catalog's models by rig signature (same joint names modulo extras, same bind pose)."""
    rigs = rigs or rig_source(cat)
    refs = rigs.refs(kinds)
    entries = []
    for i, ref in enumerate(refs):
        e = rigs.entry(ref)
        if e is not None:
            entries.append(e)
        if progress:
            progress(i + 1, len(refs))
    rigs.skipped = [r.id for r in refs if not any(e.ref is r for e in entries)]  # no skeleton (static props) or unreadable
    # clip holders first (they become representatives), larger rigs first
    entries.sort(key=lambda e: (not e.clips, -len(e.sig), e.id))
    fams: list[Family] = []
    by_size: dict[int, list[int]] = {}
    for e in entries:
        n = len(e.sig)
        placed = False
        lo, hi = math.floor(n * JACCARD_MIN), math.ceil(n / JACCARD_MIN)
        for size in range(lo, hi + 1):
            for fi in by_size.get(size, []):
                f = fams[fi]
                rs = f.rep.sig
                inter = len(rs.name_set & e.sig.name_set)
                if inter / (len(rs) + n - inter) < JACCARD_MIN:
                    continue
                c = compare(rs, e.sig)
                if c.same_bind(tol):
                    f.members.append(e)
                    f.deltas[e.id] = c
                    placed = True
                    break
            if placed:
                break
        if not placed:
            fams.append(Family(rep=e, members=[e]))
            by_size.setdefault(n, []).append(len(fams) - 1)
    rels: list[Relation] = []
    big = [i for i, f in enumerate(fams) if len(f.rep.sig) >= MIN_RELATION_JOINTS]
    for i in big:
        a = fams[i].rep.sig
        for j in big:
            if i == j:
                continue
            b = fams[j].rep.sig
            if len(a) < len(b) and a.name_set <= b.name_set:
                c = compare(a, b)
                rels.append(Relation("subset" if c.same_bind(tol) else "subset-proportions", i, j, c))
            elif i < j and min(len(a), len(b)) / max(len(a), len(b)) >= JACCARD_MIN:
                inter = len(a.name_set & b.name_set)
                if inter / (len(a) + len(b) - inter) >= JACCARD_MIN:
                    rels.append(Relation("proportions", i, j, compare(a, b)))
    return fams, rels, rigs


def donor_coverage(rigs: _Rigs, donor: RigEntry, target: RigSignature, sample: int = 12, donor_bound_only: bool = True) -> Coverage:
    """Coverage of (a sample of) the donor's clips on `target`, spread evenly over the donor's clip list.

    `donor_bound_only`: count only tracks that bind on the donor's own rig. Clips also key helper bones that no
    skeleton has (`exp_op_*` expression drivers, `end_*` leaf ends, `chara_root_exp`); those are dropped on the donor
    too, so they say nothing about the target."""
    keys = donor.own + donor.inherited
    if len(keys) > sample:
        step = len(keys) / sample
        keys = [keys[int(k * step)] for k in range(sample)]
    own = donor.sig.index
    return clip_coverage(target, ([b for b in rigs.clip_bones(k) if not donor_bound_only or b in own] for k in keys))


# --------------------------------------------------------------------------- CLI (`python -m pipeline rigs ...`)


def _entry_line(e: RigEntry, c: RigComparison | None = None) -> str:
    own, inh = len(e.own), len(e.inherited)
    clips = f"{own}" + (f"+{inh} inherited" if inh else "")
    d = f"  world {1000 * c.max_world:.1f} mm, lengths x{c.length_ratio:.3f}, +{len(c.only_b)}/-{len(c.only_a)} joints" if c else ""
    return f"    {e.id:18s} {e.ref.kind:8s} {len(e.sig):4d} joints  clips {clips:18s}{d}  {e.ref.display_name}"


def cmd_rigs(args, make_catalog) -> None:
    import argparse
    import json
    import sys

    tol = args.tol if args.tol is not None else DEFAULT_TOL
    if args.what == "compare":
        if len(args.ids) != 2:
            sys.exit("rigs compare needs two ids ([game:]id)")
        found = []
        for spec in args.ids:
            game, mid = spec.split(":", 1) if ":" in spec else (args.game, spec)
            cat = make_catalog(argparse.Namespace(**{**vars(args), "game": game}))
            rigs = rig_source(cat)
            ref = rigs.find(mid)
            if ref is None:
                sys.exit(f"unknown model id {spec}")
            e = rigs.entry(ref)
            if e is None:
                sys.exit(f"{spec}: no skeleton")
            found.append((spec, rigs, e))
        (na, ra, a), (nb, rb, b) = found
        c = compare(a.sig, b.sig)
        print(f"A {na}: {len(a.sig)} joints, {len(a.own)} own + {len(a.inherited)} inherited clips")
        print(f"B {nb}: {len(b.sig)} joints, {len(b.own)} own + {len(b.inherited)} inherited clips")
        print(f"shared {len(c.shared)}  jaccard {c.jaccard:.3f}  A subset of B: {c.a_subset_b}  B subset of A: {c.b_subset_a}")
        print(f"only A ({len(c.only_a)}): {', '.join(c.only_a[:30])}{' ...' if len(c.only_a) > 30 else ''}")
        print(f"only B ({len(c.only_b)}): {', '.join(c.only_b[:30])}{' ...' if len(c.only_b) > 30 else ''}")
        print(f"parent mismatches ({len(c.parent_mismatch)}): {', '.join(c.parent_mismatch[:20])}")
        print(f"max bind-local translation {1000 * c.max_local_t:.2f} mm ({c.max_local_t_joint}), rotation {c.max_rot_deg:.2f} deg ({c.max_rot_joint})")
        print(f"max bind world position {1000 * c.max_world:.2f} mm ({c.max_world_joint}), median bone length ratio B/A {c.length_ratio:.4f}")
        print(f"same joint order {c.same_order}, same palette {c.same_palette}, palette prefix {c.palette_prefix}")
        print(f"same bind (tol {1000 * tol:.1f} mm): {c.same_bind(tol)}, compatible: {c.compatible(tol)}")
        for (nd, rd, d), (nt, t) in (((na, ra, a), (nb, b)), ((nb, rb, b), (na, a))):
            if d.clips:
                print(f"{nd}'s clips on {nt}: {donor_coverage(rd, d, t.sig, args.sample)} of tracks that bind on {nd}; "
                      f"raw {donor_coverage(rd, d, t.sig, args.sample, donor_bound_only=False)}")
        return

    cat = make_catalog(args)

    def progress(i, n):
        if i % 200 == 0 or i == n:
            print(f"  loading rigs {i}/{n}", file=sys.stderr)

    fams, rels, rigs = families(cat, args.kind, tol, progress)
    out_json = {"game": cat.GAME, "tol": tol, "families": [], "relations": [], "candidates": []}
    rel_of: dict[int, list[Relation]] = {}
    for r in rels:
        rel_of.setdefault(r.a, []).append(r)
        rel_of.setdefault(r.b, []).append(r)
        out_json["relations"].append({"kind": r.kind, "a": r.a, "b": r.b, "aRep": fams[r.a].rep.id, "bRep": fams[r.b].rep.id, **r.cmp.as_dict()})

    def candidate(m: RigEntry, donor: RigEntry, via: str, cmp: RigComparison | None) -> str:
        cov = donor_coverage(rigs, donor, m.sig, args.sample)
        out_json["candidates"].append({
            "id": m.id, "donor": donor.id, "via": via, "coverage": round(cov.fraction, 4), "tracks": cov.tracks,
            **({"maxWorld": round(cmp.max_world, 5), "lengthRatio": round(cmp.length_ratio, 4)} if cmp else {}),
        })
        return f"    -> candidate borrower {m.id}: donor {donor.id} ({via}), coverage {cov}"

    shown = 0
    for fi, f in enumerate(fams):
        out_json["families"].append({"family": fi, "joints": len(f.rep.sig), "rep": f.rep.id, "members": [
            {"id": m.id, "kind": m.ref.kind, "own": len(m.own), "inherited": len(m.inherited),
             **({"maxWorld": round(f.deltas[m.id].max_world, 5), "lengthRatio": round(f.deltas[m.id].length_ratio, 4)} if m.id in f.deltas else {})}
            for m in f.members]})
        # relations with a family on the other side that holds clips (or any, with --all)
        # (same-bind subsets always; different-proportion relations only when this family has no clips of its own)
        links = [r for r in rel_of.get(fi, []) if args.all or (fams[r.b if r.a == fi else r.a].holders and (r.kind == "subset" or not f.holders))]
        if len(f.members) < 2 and not links and not args.all:
            continue
        shown += 1
        print(f"family {fi}: {len(f.rep.sig)} joints, {len(f.members)} members, {len(f.holders)} with clips (rep {f.rep.id})")
        for m in f.members:
            print(_entry_line(m, f.deltas.get(m.id)))
        for r in links:
            other = r.b if r.a == fi else r.a
            o = fams[other]
            if r.kind.startswith("subset"):
                rel = f"rig contains family {other}'s" if r.b == fi else f"rig lies inside family {other}'s"
            else:
                rel = f"same joint names (jaccard {r.cmp.jaccard:.2f}) as family {other}, different bind"
            print(f"    {rel} ({o.rep.id}, {len(o.rep.sig)}j, {len(o.holders)} with clips): world {1000 * r.cmp.max_world:.1f} mm, "
                  f"lengths x{r.cmp.length_ratio:.3f}, {'same bind' if r.kind == 'subset' else 'bind differs'}")
        for m in f.borrowers:
            print(candidate(m, f.rep, "family", f.deltas.get(m.id)))
        if not f.holders:
            for r in rel_of.get(fi, []):
                o = fams[r.b if r.a == fi else r.a]
                if o.holders:
                    for m in f.members:
                        print(candidate(m, o.rep, r.kind, r.cmp))
    n_models = sum(len(f.members) for f in fams)
    print(f"\n{n_models} rigs, {len(fams)} families ({shown} shown), {sum(1 for f in fams if f.holders)} with clips, "
          f"{len(out_json['candidates'])} candidate borrowers, {len(rels)} relations; {len(rigs.skipped)} models without a skeleton")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out_json, fh, indent=1)
