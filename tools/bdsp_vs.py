"""Build the BDSP "VS model" set: one 512x512 transparent render per Trainer class / named character -> out/bdsp_vs/.

    ./.venv/Scripts/python.exe tools/bdsp_vs.py [--size 512] [--keep-raw]      (viewer dev server must be running)

Pose = advent02_b frame 0 (falls back to advent_b / wait_b), lighting preset `bdspVs`, front camera, cropped to the alpha box
and fitted into the square. The class table (which battle model each Bulbapedia class page uses, which hand holds the Poké Ball,
which light effect applies) was matched by eye against the VS images on Bulbapedia; `models` with two ids are composites
(the game has no pair model for Double Team / Interviewers / Ranchers / Twins / Young Couple).
"""
import argparse
import json
import os
import subprocess
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "bdsp_vs")
RAW = os.path.join(ROOT, "out", "bdsp_vs_raw")

# label -> model id(s)
ENTRIES = {
    "Ace Trainer (M)": "tr1024_00", "Ace Trainer (M, 2)": "tr1053_00", "Ace Trainer (F)": "tr1025_00", "Ace Trainer (F, 2)": "tr1054_00",
    "Aroma Lady": "tr1007_00", "Artist": "tr0044_00", "Battle Girl": "tr1010_00", "Beauty": "tr0051_00", "Bird Keeper": "tr1030_00",
    "Black Belt": "tr0050_00", "Bug Catcher": "tr1006_00", "Cameraman": "tr0062_00", "Camper": "tr1004_00", "Clown": "tr1058_00",
    "Collector": "tr1037_00", "Cowgirl": "tr0018_00", "Cyclist (F)": "tr1013_00", "Cyclist (M)": "tr1012_00",
    "Double Team": ["tr1024_00", "tr1025_00"], "Dragon Tamer": "tr1029_00", "Fisher": "tr0048_00", "Gentleman": "tr0046_00",
    "Guitarist": "tr1052_00", "Hiker": "tr0049_00", "Idol": "tr1085_00", "Interviewers": ["tr0062_00", "tr0061_00"], "Jogger": "tr1019_00",
    "Lady": "tr1033_00", "Lass": "tr0029_00", "Madame": "tr0047_00", "Ninja Boy": "tr1028_00", "PI": "tr1051_00", "Parasol Lady": "tr1080_00",
    "Picnicker": "tr1005_00", "Poké Fan (F)": "tr1021_00", "Poké Fan (M)": "tr1020_00", "Poké Kid": "tr0027_00",
    "Pokémon Breeder (F)": "tr0043_00", "Pokémon Breeder (M)": "tr0042_00", "Pokémon Ranger (F)": "tr1040_00", "Pokémon Ranger (M)": "tr1039_00",
    "Police Officer": "tr0039_00", "Psychic (F)": "tr1050_00", "Psychic (M)": "tr1049_00", "Rancher": "tr1071_00",
    "Ranchers": ["tr1071_00", "tr0018_00"], "Reporter": "tr0061_00", "Rich Boy": "tr1032_00", "Roughneck": "tr1057_00", "Ruin Maniac": "tr1048_00",
    "Sailor": "tr1046_00", "School Kid (F)": "tr0031_00", "School Kid (M)": "tr0030_00", "Scientist": "tr1041_00", "Skier (F)": "tr1056_00",
    "Skier (M)": "tr1055_00", "Swimmer (F)": "tr0054_00", "Swimmer (M)": "tr0053_00", "Team Galactic Grunt (F)": "tr1089_00",
    "Team Galactic Grunt (M)": "tr1073_00", "Tower Tycoon (Palmer)": "tr1097_00", "Tuber (F)": "tr1045_00", "Tuber (M)": "tr1044_00",
    "Twins": ["tr1008_00", "tr1008_01"], "Veteran": "tr1027_00", "Waiter": "tr1081_00", "Waitress": "tr1026_00", "Worker": "tr0040_00",
    "Young Couple": ["tr1023_00", "tr1023_01"], "Youngster": "tr0028_00",
    # named characters (internal numbering matched to the Bulbapedia VS images)
    "Roark (Gym 1)": "tr1062_00", "Gardenia (Gym 2)": "tr1074_00", "Maylene (Gym 3)": "tr1076_00", "Crasher Wake (Gym 4)": "tr1075_00",
    "Fantina (Gym 5)": "tr1077_00", "Byron (Gym 6)": "tr1064_00", "Candice (Gym 7)": "tr1078_00", "Volkner (Gym 8)": "tr1079_00",
    "Aaron (Elite Four)": "tr1065_00", "Bertha (Elite Four)": "tr1066_00", "Flint (Elite Four)": "tr1067_00", "Lucian (Elite Four)": "tr1068_00",
    "Cynthia (Champion)": "tr0001_00", "Cyrus (Galactic boss)": "tr1086_00", "Mars (Galactic commander)": "tr1072_00",
    "Jupiter (Galactic commander)": "tr1087_00", "Saturn (Galactic commander)": "tr1088_00", "Prof. Rowan (Doctor01)": "tr2003_00",
    "Rowan, no coat (Bag)": "tr2003_01", "Lucas (player)": "pc0001_00", "Dawn (player)": "pc0002_00", "Barry (Friend Body)": "tr0002_00",
    "Body / tr2007": "tr2007_00", "Tag02 / tr2008": "tr2008_00", "Tag03 / tr2009": "tr2009_00", "Tag04 / tr2010": "tr2010_00",
    "Tag05 / tr2011": "tr2011_00", "Jasmine / tr2018": "tr2018_00", "Mother02 / tr2036": "tr2036_00",
}
# Poké Ball in hand, matched against the VS images: attach bone per model (R = RItem1, L = LItem1)
BALL = {**{m: "RItem1" for m in "tr1024_00 tr1053_00 tr1054_00 tr1033_00 tr1028_00 tr1041_00 tr0054_00 tr1073_00".split()},
        **{m: "LItem1" for m in "tr1025_00 tr1004_00 tr0018_00 tr1013_00 tr1012_00 tr1085_00 tr0029_00 tr1040_00 tr1032_00 tr1089_00 tr1027_00".split()}}
FX = {"tr0039_00": ("police", None, 1), "tr1049_00": ("psychic", "RItem1", 2.6), "tr1050_00": ("psychic", "RItem1", 2.6)}


def render(ids, extra, size):
    cmd = ["node", os.path.join(ROOT, "tools", "render_bdsp_vs.mjs"), "--out", RAW, "--size", str(size), "--frames", "0", *extra, *[f"bdsp/{i}" for i in ids]]
    subprocess.run(cmd, check=True, cwd=ROOT)


def fit(im, size, margin=0.04):
    im = im.crop(im.getbbox())
    s = round(size * (1 - 2 * margin)) / max(im.size)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    o = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    o.paste(im, ((size - im.width) // 2, size - im.height - round(size * margin)), im)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--render-size", type=int, default=1024)
    ap.add_argument("--skip-render", action="store_true", help="reuse out/bdsp_vs_raw (just re-fit / re-compose)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ids = sorted({i for v in ENTRIES.values() for i in (v if isinstance(v, list) else [v])})
    plain = [i for i in ids if i not in BALL and i not in FX]
    if not a.skip_render:
        render(plain, [], a.render_size)
        for bone in ("RItem1", "LItem1"):
            render([i for i in ids if BALL.get(i) == bone], ["--ball", bone], a.render_size)
        for i, (fx, bone, scale) in FX.items():
            render([i], ["--fx", fx] + (["--ball", bone, "--ballscale", str(scale)] if bone else []), a.render_size)
    index = {}
    for label, v in ENTRIES.items():
        ms = v if isinstance(v, list) else [v]
        ims = [Image.open(os.path.join(RAW, f"bdsp_{m}_f0.png")).convert("RGBA") for m in ms]
        ims = [im.crop(im.getbbox()) for im in ims]
        h = max(im.height for im in ims)
        ims = [im.resize((round(im.width * h / im.height), h), Image.LANCZOS) for im in ims]
        c = Image.new("RGBA", (sum(im.width for im in ims), h), (0, 0, 0, 0))
        x = 0
        for im in ims:
            c.paste(im, (x, 0), im)
            x += im.width
        name = label.replace(" ", "_").replace("/", "-") + ".png"
        fit(c, a.size).save(os.path.join(OUT, name))
        index[label] = {"file": name, "models": ms}
    with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    print(len(index), "images ->", OUT)


if __name__ == "__main__":
    sys.exit(main())
