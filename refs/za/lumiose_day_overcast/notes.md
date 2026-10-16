# lumiose_day_overcast: notes

## Videos used
- 4spEtYJBlVE (Shirrako part 1, Switch 2, 4K60, no commentary). Scouted 0:30-0:50, 1:00-1:30, 1:30-2:00 (contact sheets in scout/).
- Only one frame used: 1:45:00.5, Taunie in the Lumiose plaza by the food truck, dialogue "This one's my treat, to celebrate your prom".

## Frames
- lumiose_day_overcast_taunie_01 (tr0001_00 Taunie, field/cutscene dialogue, clip defaultwait01_loop frame 0; speak01_loop frame 0 did not render in the viewer, so the default idle was used).
  - Chosen because the plaza light is flat and even with no hard shadows, and Taunie faces the camera. The sky is only partly visible at the top and is pale blue, not clearly grey. This is the main doubt.
  - Subject is partly hidden by the dialogue box and the player character in front. Subject box cut at the bottom (subjectCut bottom).
  - Auto boxes: body_1, body_2 (jacket, dE 2.6 each), face_skin_1 (dE 11.1). The auto barehands_skin_1 box landed on white shorts, so it was removed by hand.

## Fit
- Command: fit_scene.mjs --scene ... --seeds overcast,studio --profile outdoor --fit --passes 3 --restarts 1.
- Starting mean dE: overcast seed 10.39, studio seed 13.42, restart 7.39.
- Final mean dE 6.23 (single shot, 3 boxes). Fit values are in fit.json.
- Exposure 2.08 is near the upper bound of the 0.8-2.2 range; key is at its 0.5 floor; skyTemp 12000 K. Plausibility is questionable.

## Doubts
- Only one shot and only three boxes: the fit is weak and mostly constrained by the jacket.
- Sky is not clearly overcast; the lighting may be closer to a bright day.
- Subject identity: Taunie confirmed by the name tag and the thumbnail.
- speak01_loop frame 0 fails to render in the viewer (empty probe); worth checking.
- Needs at least 1-2 more frames (different characters, a grey-sky shot) before the preset can be trusted.
