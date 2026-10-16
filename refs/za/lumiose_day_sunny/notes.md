# lumiose_day_sunny (Lumiose, sunny day)

## Videos used
- 4spEtYJBlVE: Shirrako, "POKEMON LEGENDS Z-A Gameplay Walkthrough Part 1 (4K 60FPS) No Commentary", switch2, 27865 s.
  Scouted 00:30-00:40, 00:45-00:55, 00:44:58-00:45:01, 00:47:58-00:48:01, 00:55-01:05 (scout 3300-3900 sheet),
  00:55:58-00:56:01 (around 3360).

## Frames
- lumiose_day_sunny_taunie_01.png, grabbed at 00:44:59.500. Taunie (tr0001_00) and the player on a Lumiose street in hard
  sun, blue car, grey asphalt. Dialogue box at the bottom (below the regions used). Clip defaultwait01_loop frame 0.
  Taunie is turned 3/4 toward screen-right with eyes closed, so the face is not used. subjectBox [290,70,490,950], yaw 25.
  Regions (5): jacket_lit, shorts, skin_leg (manual boxes) plus body_1 (shoe) and body_2 (jacket) auto boxes.
  Debug check: each box sits on the same part in both images. The auto aligner added only 2 windows (17 rejected for colour).
- Second frame: NOT obtained. The yellow Pokemon at 00:56:00 (red horn and tie, not Pikachu) could not be identified, so it
  has no model id to grab. Naveen (00:56:30) is not catalogued. Interiors (Hotel Z, cafe 01:00-01:04) and the Z-A Royale
  screen are not this scene.

## Fit
- Single shot, unlocked (`--seeds studio,day --profile outdoor --fit --passes 3 --restarts 1`).
  studio 25.94 -> 7.03, day 26.90 -> 7.63, restart 6.95 -> 6.71. Final mean dE 6.71 (best seed restart 1).
- Per region: jacket_lit 5.9, shorts 10.4, skin_leg 6.5, body_1 (shoe) 6.6, body_2 (jacket) 4.2.
- Fitted env (fit.json): elevation 53.8, azimuth 83.4, key 0.567 at 4500 K, ambient 2.2, back 0 (back at 6500 K, elev 25),
  exposure 1.4, agx, shadow 0.65, gi 2, saturation 0.867.
- Sanity: elevation 53.8 (OK), exposure 1.4 (OK), keyTemp 4500 (at the floor, OK). FAIL on key >= 0.4 x ambient:
  0.567 < 0.88. The outdoor profile's key >= 0.5 bound is met, but the ratio is not.

## Doubts
- One shot: the fit is overfit. Mean 6.71 says little about generalisation, and the preset is provisional.
- Key/ambient ratio fails the daytime sanity check (above); the key light is weak for hard sun.
- shorts dE 10.4 (white shorts under sun) is the worst region.
- Taunie's face and identity are not confirmed on screen (eyes closed, 3/4 turn); the name is from the dialogue box.
- Weather "sunny" comes from the scene brief and the bright street, not a sky region.
