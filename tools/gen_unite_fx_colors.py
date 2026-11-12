"""Write `pipeline/catalog/unite_home_fx.json`: Pokémon HOME's flame/smoke effect colours for every species in the UNITE catalog.

UNITE's idle particle effects (Charizard's tail flame, Gastly's gas) are Unity ParticleSystems that the rip does not contain (only
their empty emitter Nulls survive, see `importers/unite/importer.py particle_emitters`). Their colours come from the same species'
HOME effect materials (user decision 2026-11-01: flame colours are the same across media): the core card (`_Blend0Tex` present:
`FireCore` / `SmokeCore`) gives `_BaseColor`, `_LayerColor` and the `_LerpTex` gradient that blends between them.

    python -m tools.gen_unite_fx_colors [--home <dir>]

Rule over data: same national dex, HOME base form `pm<dex>_00_00`. A species with no HOME effect material gets no entry.

Motion (2026-11-10): HOME's looping idle clip scrolls the core's two noise layers (`noise[].speed`, UV/s, from
`home.importer.uv_motion`); the layers are written as grey PNGs to `pipeline/catalog/unite_home_fx/`. `asFire` lists
painted flame materials (`_AsFire` 1: Skeledirge `BodyC`) with their colour/displacement scroll speeds.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "pipeline", "catalog", "unite_home_fx.json")
# HOME's flame noise layers (`_Blend0Tex`/`_Blend1Tex`, 128x128 grey), shipped beside the table for the viewer's animated fire
NOISE_DIR = os.path.join(os.path.dirname(OUT), "unite_home_fx")
KIND_RE = re.compile(r"(fire|smoke)", re.I)


def lerp_profile(lt: np.ndarray, n: int = 8) -> list[float]:
    """The `_LerpTex` gradient (R, or A when R is flat, as the HOME importer reads it) averaged across U, sampled at n points
    from the texture's top row (V = 1 in Unity) to its bottom row."""
    ch = lt[..., 0] if np.ptp(lt[..., 0]) > 8 else lt[..., 3]
    rows = ch.astype(np.float32).mean(axis=1) / 255.0
    idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
    return [round(float(rows[i]), 3) for i in idx]


def save_noise(ptr) -> str:
    """Write a HOME noise layer's R channel as a grey PNG under NOISE_DIR; returns the file name."""
    t = ptr.deref().read()
    name = f"{t.m_Name}.png"
    os.makedirs(NOISE_DIR, exist_ok=True)
    dst = os.path.join(NOISE_DIR, name)
    if not os.path.exists(dst):
        from PIL import Image
        Image.fromarray(np.asarray(t.image.convert("RGBA"))[..., 0]).save(dst)
    return name


def home_effects(path: str, data_dir: str) -> tuple[list[dict], list[dict]]:
    """(fire/smoke core cards, `_AsFire` painted materials) of one HOME prefab, with the loop clip's UV scroll speeds."""
    from pipeline.importers.home.importer import _env, dependency_paths, uv_motion

    env, sf = _env(path, dependency_paths(path, data_dir))
    motion = uv_motion(env, sf)
    mm = motion.get("materials", {})
    out, as_fire = [], []
    for o in sf.objects.values():
        if o.type.name != "Material":
            continue
        m = o.read()
        props = m.m_SavedProperties
        tex = {n: v.m_Texture for n, v in props.m_TexEnvs if v.m_Texture.path_id}
        fl0 = {n: float(v) for n, v in props.m_Floats}
        if fl0.get("_AsFire") and "_Col0Tex" in tex:
            # painted flames (Skeledirge `BodyC`): the albedo scrolls (`ColorUVTranslate`); the displacement panner's
            # operator (`_DISPOPE_SIN`) is not in the rip, so only its speed and height are recorded
            col = dict(props.m_TexEnvs)["_Col0Tex"]
            as_fire.append({"material": m.m_Name, "colScale": [float(col.m_Scale.x), float(col.m_Scale.y)],
                            "displaceHeight": fl0.get("_DisplaceHeight"), "displaceUvIndex": fl0.get("_DisplaceUvIndex"),
                            "motion": {"loop": motion.get("loop"), **mm.get(m.m_Name, {})}})
            continue
        if "_Col0Tex" in tex or "_Blend0Tex" not in tex:
            continue
        kind = KIND_RE.search(m.m_Name)
        cols = {n: [round(float(c.r), 4), round(float(c.g), 4), round(float(c.b), 4), round(float(c.a), 4)] for n, c in props.m_Colors}
        fl = {n: float(v) for n, v in props.m_Floats}
        e = {
            "material": m.m_Name,
            "kind": kind.group(1).lower() if kind else None,
            "baseColor": cols.get("_BaseColor", [1, 1, 1, 1]),
            "layerColor": cols.get("_LayerColor", cols.get("_BaseColor", [1, 1, 1, 1])),
            "zWrite": fl.get("_ZWrite"),
            "blendUVScale": [[fl.get(f"_Blend{i}UVScaleU", 1.0), fl.get(f"_Blend{i}UVScaleV", 1.0)] for i in (0, 1)],
        }
        if "_LerpTex" in tex:
            img = tex["_LerpTex"].deref().read().image.convert("RGBA")
            e["lerpProfile"] = lerp_profile(np.asarray(img))
        # animated layers: the two noise textures and the loop clip's speeds (Unity UV/s, V up)
        sp = mm.get(m.m_Name, {})
        e["noise"] = [{"file": save_noise(tex[f"_Blend{i}Tex"]), "scale": e["blendUVScale"][i],
                       "speed": sp.get(f"Blend{i}UVTranslate", [0.0, 0.0])} for i in (0, 1) if f"_Blend{i}Tex" in tex]
        if len(e["noise"]) == 2:
            # the viewer sums the two layers and renormalises to the 99th percentile (as HOME's static bake renormalises
            # its layers); `noiseMean` = the mean of that, so the noise moves the colour ramp without shifting its average
            # (user decision 2026-11-10; the product left the core showing on 4% of Charmander's card, all red)
            from PIL import Image
            a, b = (np.asarray(Image.open(os.path.join(NOISE_DIR, n["file"])), np.float32) / 255.0 for n in e["noise"])
            # the two layers tile differently (`scale`); their sum's statistics over one period of each are the same
            # as over the overlapping grid, so the plain sum of the images is used
            ssum = a + b
            norm = float(np.percentile(ssum, 99))
            e["noiseNorm"] = round(norm, 4)
            e["noiseMean"] = round(float(np.clip(ssum / max(norm, 1e-3), 0, 1).mean()), 4)
        e["loop"] = motion.get("loop")
        out.append(e)
    return out, as_fire


def main() -> None:
    from pipeline.catalog.home import DEFAULT_DATA_DIR
    from pipeline.catalog.unite import UniteCatalog

    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=DEFAULT_DATA_DIR)
    a = ap.parse_args()
    dexes = sorted({int(r.extras["dex"]) for r in UniteCatalog().all() if r.extras.get("dex")})
    table = {}
    for dex in dexes:
        # the base form keys the species (`6`); other forms key `<dex>_<form>` (`6_51` = Mega Charizard X, which UNITE's
        # `0006_51` folder matches by number)
        forms = sorted(f[7:9] for f in os.listdir(os.path.join(a.home, "pokemons"))
                       if re.fullmatch(rf"pm{dex:04d}_\d\d_00\.unity3d", f))
        for form in forms:
            home = f"pm{dex:04d}_{form}_00"
            try:
                fx, as_fire = home_effects(os.path.join(a.home, "pokemons", home + ".unity3d"), a.home)
            except Exception as e:  # noqa: BLE001
                print(f"{home}: {e}")
                continue
            if fx or as_fire:
                table[str(dex) if form == "00" else f"{dex}_{form}"] = {"home": home, "effects": fx, **({"asFire": as_fire} if as_fire else {})}
                print(f"{home}: " + ", ".join([f"{e['material']} ({e['kind']})" for e in fx] + [f"{e['material']} (asFire)" for e in as_fire]))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"source": "Pokémon HOME effect core materials (tools/gen_unite_fx_colors.py)", "species": table}, f, indent=1)
    print(f"{len(table)} species -> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
