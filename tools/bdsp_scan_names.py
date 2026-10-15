"""One-off: derive a display name per BDSP trainer/object bundle from its texture names; writes pipeline/catalog/bdsp_names.json."""
import json, os, re, sys

import UnityPy

root = sys.argv[1]
out = sys.argv[2]
names = {}
for sub in ("persons/battle", "persons/field", "objects"):
    d = os.path.join(root, "Characters", *sub.split("/"))
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if not os.path.isfile(p):
            continue
        env = UnityPy.load(p)
        toks = {}
        for o in env.objects:
            if o.type.name == "Texture2D":
                m = re.match(re.escape(f) + r"_(.+?)_(?:[a-z]+_)?(?:col|nor|msk|uti)$", o.read().m_Name)
                if m:
                    toks[m.group(1)] = toks.get(m.group(1), 0) + 1
        if toks:
            names[f] = max(toks, key=toks.get).replace("_", " ").title()
        print(f, names.get(f), flush=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(names, fh, indent=1, ensure_ascii=False)
