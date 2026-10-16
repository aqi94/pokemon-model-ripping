# lumiose_dusk: calibration notes

Status: NOT acceptable by the brief's threshold. Only one usable dusk frame was found. Final mean dE 14.52 (> 12).

## Videos used
- 4spEtYJBlVE (Shirrako, Switch 2, Pokemon Legends Z-A part 1): scouted 0:05-0:25 (every 30 s), 0:30-1:30, 1:30-3:40 (every 60-90 s).
  Almost all daylight. Night/Battle Zone at ~01:17 (battle, blue night), ~01:21 (blue street), ~03:07 (night cutscene). No warm-sun dusk found.
- 6qiJ2FnPSFM (Shirrako, Switch 2, full game 4K60): scouted hour windows 4, 6, 8, 10, 12, 14, 16 (20 min each, every 60 s).
  Dusk frames found at 12:04:56-12:05:03 (cutscene, Vert/Lumiose plaza, warm orange sky, low sun).
  Other candidates rejected: 12:17:54-12:18:04 (warm street, player only, not a listed model); 10:17-10:18 and 08:02 (night-blue).

## Shots
- lumiose_dusk_taunie_01 (Taunie, tr0001_00): video 6qiJ2FnPSFM at 12:04:57.500. Clip guess defaultwait01_loop frame 0.
  Why: sharp, face toward camera, eyes open, clear dusk light. The frame at 12:05:00 is in dialogue, 12:05:02 has eyes shut.
  Camera (fitted by hand from frame geometry): fov 35, position [-0.38, 1.4, 1.56], target [-0.38, 1.21, 0]. Framing matches within a few px.
  Pose mismatch: the shot has hands on hips; the default clip has arms down. This limits the fit.
  Start dE: 20.58 (preset sunset, boxes and camera set). After the joint fit: 14.52.

## Fit
- Joint fit (seeds sunset, day; outdoor; 3 passes): mean dE 14.59. Unlocked knobs: elevation 25, keyTemp 5800, exposure 2 (not dusk-like).
- Refit with --lock elevation=15 --lock keyTemp=3800: mean dE 14.52. The keyTemp lock did not take effect (fit.json shows keyTemp 5967).
- fit.json is the lock-refit result. Knobs: elevation 15, keyTemp 5967, exposure 1.93, key 1.6, ambient 0.7.

## Doubts
- Single shot: one frame is not enough to pin down a dusk lighting preset. Treat fit.json as provisional.
- Remaining error is mostly material colour, not light: the magenta top (dE 28.5) and the lilac jacket (dE 23.7) render as tan/red; the skin cheek is 18.2. These come from the model's colour bake, not from the scene lights.
- Pose: hands-on-hips in the shot, arms down in the render. The boxes on the arms/hips were avoided.
- Subject identity: Taunie is clearly recognisable (pink hair, bun, magenta top). The companion in the frame is the player character (not a listed model), so he is not used.
- Weather: the sky is orange-purple, which fits sunny dusk. Not confirmed as "sunny" beyond the sky.
- Missing: a second frame (another character or a Pokemon at dusk) and a time-of-day check. More scouting of 6qiJ2FnPSFM 12:00-12:30 and hours 18-20 could give one.
