# wild_zone_day: notes

Status: one shot only (Taunie). The brief asked for 2-4 shots; a second usable frame was not found in the time spent, so the fit rests on a single character.

## Video used
- 4spEtYJBlVE: Shirrako part 1 (Switch 2, 4K60, no commentary).

## Shot 1: wild_zone_day_taunie_01 (tr0001_00 Taunie)
- Timecode 00:47:28.000 (2848 s). Lumiose street next to the rocks, bright sunny daylight, no dialogue box over her, full body visible, facing the camera, feet visible.
- Clip defaultwait01_loop frame 0, context field, camera re-aligned by fit_scene (`--align`).
- Boxes: 6 manual (face_cheek, jacket_front, top_magenta, shorts_white, leg_skin, shoe_navy) + 2 auto (body_2, face_skin_1). The auto `body_1` box was removed: it sat on the background/arm edge (dE 30.9).
- dE vs seed (first fit, with body_1): studio 18.56 -> 11.08; day 14.39 -> 11.42; best restart 15.63 -> 10.91.
- dE vs seed (final fit, body_1 removed): studio 16.83 -> 9.70; day 11.87 -> 9.29; best restart 14.90 -> 8.98.

## Fit
- Command: `fit_scene --scene refs/za/wild_zone_day --seeds studio,day --profile outdoor --fit --passes 3 --restarts 1`.
- Final mean dE 8.98 (just above the <8 target, inside the 8-12 band; accept with note). Earlier fit with body_1 was 10.91.
- Result: ambient 2.2, key 1.233, azimuth 11.7, elevation 25, keyTemp 8000 K, back 0, skyTemp 12000 K, exposure 1.55, agx, contrast 1.375, saturation 0.867, ground #ffb059.

## Doubts
- keyTemp 8000 K sits at its profile bound. Key moved off its bound after body_1 was removed (1.23 now) but with one subject the scene lighting is still poorly constrained.
- PROVISIONAL preset: the frame is a Lumiose street beside the Wild Zone rocks, not inside a Wild Zone park, so this preset is provisional.
- One character only; no Pokemon shot. Candidates checked and rejected: Charizard café frame 01:47:30 (occluded by Urbain and Taunie, dialogue box, not a wild zone); 01:33:00 green-pillar wild zone frames (no named subject in clear view).
- Wild Zone green pillar frames at 01:33:01 confirm the daylight look, but no subject is usable there.
- Weather is "sunny" from the scene tag; the sky in the frame is mostly clear.
- Frame is a Lumiose street beside the Wild Zone rocks, not inside a zone (see PROVISIONAL above).
