# battle_auroma (unite_battle_auroma): NO FIT

Status: no fit.json written. Footage found does not support a reliable fit; stopped after ~6 videos per the task rules.

Videos checked (all via `--game unite`):
- hq1oJzgBGdI (Marcus Alford, Auroma Park quick battle, Pikachu): correct map, bright sunny day, but source is 720p60.
  Pikachu is ~40-50 px tall in the 1920-wide frame (blurred upscale), so ears/cheeks/body would be 3-5 px windows. Frames grabbed
  as battle_auroma_pikachu_01 (2:16.0) and _02 (2:43.5) (sidecars kept, model set to unite/pm0025_00 by hand because
  tools/lightrefs/sidecar.py MODEL_RE rejects UNITE ids; clip `run`). Pikachu is running, not standing.
- R3OqBR7Iew8 (Shekhu Baba, mobile 1080p30, Sylveon, Auroma Park): Sylveon ~35-40 px tall, moving, sparkle/trail, 30 fps blur. Not usable.
- QQe13viKubg (GamerJGB Charizard 1080): that is Shivre City (snow), not Auroma.
- 4K5MdqOxb44 (trExUs Switch 2 4K60): that is Theia Sky Ruins, not Auroma.
- msXNZLSmYXE, hO3xFVUCLOE, vyevQH8ggAM, k1Psimcobg8, rEBLX-yd_ak: Auroma videos but only 720p (or 864p); same tiny subject. Not grabbed.
- Untried: P2mxTixDsZk (GoalPulse, Auroma quick match, 1080p60 mobile) and vn0shdP-OhY / ncUM146tr7c (1080p), likely the same camera scale.

Blocker: the Auroma/UNITE battle camera is high top-down, so the Pokemon is ~40-50 px tall even at 1080p; no 4K Switch 2 upload of Auroma Park
was found in search (the 4K uploads are Theia Sky Ruins). The 3 mid-tone materials per Pokemon cannot be isolated at that size.
Suggestion: use the grass/ground colours (large, flat) instead, or find a 4K Auroma Park video / spawn-area replay; otherwise reuse unite_battle_theia-like `outdoor` seeds (`day`).
Auroma look for reference: warm saturated green grass, pink-beige paving, lavender paths, sun from upper-left-ish, soft shadows.
