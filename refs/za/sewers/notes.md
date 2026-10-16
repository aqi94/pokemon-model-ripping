# sewers (Lumiose Sewers)

## Videos used
- `6qiJ2FnPSFM`, Shirrako, Switch 2, 4K60, no commentary, full game (18.6 h). Only source used for frames.
- `search --scene sewers` also surfaced RandomRam `9oGagYBl8Is` (Switch 1, "Main Mission #19 - Unlocking The Sewers"), not used for frames.

## Timecodes (Shirrako full game)
- 08:24:30 to 08:24:57: Ivor and Emma at the Clawitzer fountain: "Quick, then! Let's search the sewers!"
- 08:25:00 mission text: "Reaching Rank D: Enter the sewers from the door near the Clawitzer fountain".
- 08:25:21 to 08:25:57: inside the sewer tunnels with Ivor and Emma. Mission text "Make your way deeper into the sewers in search of Gwynn".
- 08:26:03: first wild Pokemon in the sewers (Noibat), not a sewer-specific species.
- Scouted windows that did not show the sewers (for reference): 0:40-1:20 (early city), 1:20-3:40, 4:00-5:30, 6:00-8:30 (6:31 Tarragon warehouse is not the sewers).

## Frames
- `sewers_ivor_01.png` (08:25:36, `defaultwait01_loop` frame 0, model `za/tr0017_00`, Ivor). Full body, no dialogue box, lamp-lit tunnel. Night, indoor.
  - Only skin and hair boxes kept: Ivor's outfit in the stream (red sash and gloves) differs from the viewer default (orange trim). Hand-placed boxes: hair_crown, hair_fringe, skin_forehead_cheek, skin_chin.
  - Auto boxes were rejected: auto regions on this model landed on outfit and the forearm box was on shadow, not skin.
  - Alignment needed a closer start camera (fov 30, position [0, 0.95, 3.3]) and a subjectBox that does not touch the bottom edge, otherwise the aligner diverged.

## Fit
- Seeds: indoorCool (start dE 27.4 -> 7.9 unlocked) and indoorWarm (13.0 -> 5.9), plus restart 1 (9.98 -> 5.28).
- Final, `--lock exposure=2.2` (exposure was 2.35 unlocked, just outside the 0.8-2.2 range): mean dE 4.49 (indoorWarm seed 5.62, restart 4.49).
- Starting dE for the seed: 27.4 (indoorCool) / 13.0 (indoorWarm) on the unlocked run; the locked run's starting point was 32.7 / 17.5.
- Final knobs: key 0.533, ambient 0.533, back 0.467, backTemp 12000 K, keyTemp 2900, exposure 2.2, agx, rim 0.

## Doubts
- Only ONE shot. A second character (e.g. Taunie or Urbain) or a Pokemon (Golbat) in the sewers was not found in the scouted windows; the brief's 2-4 frame target is not met.
- Subject identity: Ivor is identified by the name box at 08:24:39 and the sewer scene; his outfit colourway differs from the viewer default, so only skin/hair were used.
- Lighting: backTemp 12000 K is high for a rim/back light (the fit pushed it to the cool side). Exposure locked by hand, so the fit is less constrained.
- Shot has no DOF or UI problems in the crop, but the tunnel has a warm bounce from the brick walls that the model may not reproduce.
