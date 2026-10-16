"""Cutscene ("demo") bone clips for Z-A trainers.

`ZA-TrainersAnim*.zip ik_chara/demo_*` hold ordinary bone `.tranm` files (same format as the
gameplay clips, 60 fps) named `d004_tr0057_anim_c04.tranm`: `d004` = the cutscene (ZA-Demo.zip
`ik_demo/ev/d004`), `tr0057` = the character, `cNN` = the cut number. They have no 5-digit
motion index, so they get the synthetic index 200000 + NN and the clip group `demo <cutscene>`.
"""

from __future__ import annotations

import os
import re

from ..importers.za.importer import AnimationFile, ZipSource

DEMO_CLIP_RE = re.compile(r"^(?P<ev>d\d+[ab]?)_(?P<who>[A-Za-z0-9]+?(?:_\d+)?)_anim_(?P<rest>.+)\.tranm$")
DEMO_INDEX_BASE = 200000


def demo_animations(src: ZipSource, tag: str = "") -> list[AnimationFile]:
    """Clips of every `<sub>/*.tranm` (or top-level) cutscene file in `src`.

    The clip name is `<ev>_c<cut>` (`d004_c04`); `tag` is appended to the group when one
    folder serves several characters.
    """
    out: list[AnimationFile] = []
    for n in src.list():
        if not n.endswith(".tranm"):
            continue
        m = DEMO_CLIP_RE.match(os.path.basename(n))
        if not m:
            continue
        cfg = n[: -len(".tranm")] + ".tracm"
        ev, cut = m["ev"], m["rest"]
        out.append(
            AnimationFile(
                name=f"{ev}_{m['who']}_{cut}",
                index=DEMO_INDEX_BASE + int(re.sub(r"\D", "", ev) or 0) * 100 + int((re.search(r"\d+", cut) or [0])[0]),
                data=src.read(n),
                config=src.read(cfg) if src.exists(cfg) else None,
                group=f"demo {ev}" + (f" {tag}" if tag else ""),
            )
        )
    return out
