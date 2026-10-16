# indoor_cafe: notes (calibration BLOCKED, not completed)

## Scouting (main story, Shirrako part 1 `4spEtYJBlVE`)
- Scanned 0:00-1:45:00 in 15-minute windows at --every 30 (sheets in scout/4spEtYJBlVE_<from>-<to>_s0N.png).
  Also 1:45-2:00 scouted (sheet 105-120 min was produced, not yet reviewed). Windows 2:00-3:00 not reviewed.
- Interior cafe/restaurant with a named subject was NOT found in 0:00-1:45. Outdoor cafes/terraces (1:45:00 truck
  café with Vinnie/Taunie, 1:14:30 street kiosk) are not interiors.
- Candidate: 1:00:56-1:01:03, Hotel Z lounge (warm wall lamps, sunlit window behind curtains, armchairs, tea-table
  dialogue "café au lait"). Taunie (tr0001_00) faces the camera; face sharp, eyes closed/open mid-speech.
  Not a café, but lamp- and window-lit interior with a named subject (fits the coordinator's exception).
- Grabbed: indoor_cafe_taunie_01.png (+ .json) at 1:01:02.500, clip guess speak01_loop, camera fov 35,
  position [0,1.45,1.7], target [0,1.35,0]. Regions: indoor_cafe_taunie_01_regions.json (7 regions, first pass).
  Dialogue box overlaps the lower chest; regions chosen above it.

## Blocker
- `node tools/fit_scene.mjs ... --probe` produced `probe modelBox none (0 px)` (model not visible in the render at
  this camera; the probe rendered no model pixels, so every region reads "render -", mean dE 50).
- The script then crashes: `ReferenceError: debugShot is not defined` at tools/fit_scene.mjs:454. The function is
  not defined anywhere in tools/ (grep for debugShot finds only the call). tools/fit_scene.mjs was modified at
  07:35 today and looks half-written. Per the rules, tools/ must not be edited, so no fit or debug PNG was produced.
- Blocked on: restoring debugShot in tools/fit_scene.mjs, and/or verifying why the probe renders no model
  (camera or clip). Nothing else in refs/za/indoor_cafe depends on it.

## Not done
- Steps 3-7 of the brief (second and third grabs, probe iteration, joint fit, fit.json, final dE) are incomplete.

## Judgement call (coordinator-approved)
- Subject frame 1:01:02.500 is a Hotel Z lounge (warm wall lamps plus a sunlit window), not a café. The coordinator
  accepted it for "window + lamp". Taunie (tr0001_00) faces the camera.

## Update: viewer render of tr0001_00 is blank (blocker, not a lighting issue)
- `fit_scene.mjs --align --auto-regions` (subjectBox [640,0,1250,1080], yaw 0) reports `model not visible` and writes
  no regions. Probe mean dE stays 50.00.
- Diagnosis in a headless browser (same chrome channel as fit_scene): tr0001_00 loads (51 skinned meshes, 135 bones,
  bone world positions correct, head at y 1.38, material visible, no shader errors). The stage is still empty at
  the default camera and at the regions camera (35 deg, [0,1.45,1.7]). Detective (tr0013_00) and Hotel Manager
  (tr0003_00) render fine in the same setup. The cause is specific to tr0001's rendering (not the camera or the clip).
- Not fixed: the cause is inside the viewer (not in refs/). Per the rules, viewer/ is not edited.
- Remaining work (blocked): align, auto-regions, joint fit, final dE. No fit.json. No dE values yet.

## Update 3: alignment and regions (not finished)
- Clip now speak02_loop frame 10 (speak01_loop collapses the skeleton in the viewer). subjectBox [600,0,1250,790],
  subjectCut bottom, yaw 0, as the coordinator asked.
- `--align --auto-regions` (width matching) picked the camera at about [0.03,1.25,1.45] but the auto regions were
  mismatched (upperbody/barehands vs the dialogue box and hair). Discarded.
- Manual camera: position [0.033,1.40,0.97], target [0.033,1.30,0.02], fov 35 (render head size ~ screenshot head
  size; the width-based align makes the render too small because Taunie's arms are outside the visible box).
- Regions: skin_forehead, sweater, jacket_lapel, jacket_right (manual boxes). Probe mean dE 30.8:
  forehead 19.3, sweater 43.2, lapel 25.0, right jacket 41.3.
- Cause (not lighting): the viewer's default Taunie outfit is a red cropped top with a tan coat. The screenshot wears a
  magenta turtleneck with a dark-blue collar under the same tan coat. The garment colour differences make every
  garment box fail, so the boxes cannot be trusted for the joint fit. Skin boxes also sit on the brow/hair edge.
- The joint fit was NOT run: the fit would be driven by an outfit mismatch, not by light.
- To finish: a Taunie render with the magenta-turtleneck colouring (a palette/variant if the viewer exposes it), or
  a skin-only region set (forehead/cheek/hands) on an un-hidden skin patch, then joint fit.

## Update 4: skin / coat / hair regions (finished as far as possible; joint fit NOT run)
- Outfit: the screenshot wears a magenta turtleneck with a dark-blue collar under a tan coat; the viewer's default
  Taunie is a red cropped top under the same coat. Only skin, coat and hair boxes are used. The sweater box is dropped.
- Regions (all "manual": true): skin_forehead (w1.5), skin_cheek (w1.5), coat_lapel, coat_chest_right, coat_left_side,
  hair_left, hair_right. Coordinates are in screenshot px; renderBox = box.
- Camera: manual, fov 35, position [0.033,1.40,0.85], target [0.033,1.30,0.02]. `--align` with headBox [700,0,1200,560]
  went wrong (it aimed at the knees because the head probe only returns body/eyebrow/hair materials). Its camera was
  discarded and the manual camera restored.
- Probe mean dE 40.6 (forehead 27.1, lapel 26.3, chest 42.3, left side 40.7, hair_left 49.6; cheek and hair_right
  render empty). The render is still not framed like the screenshot (the render's head is slightly larger and
  shifted), so the boxes compare different spots. A region-level fit on these is not meaningful.
- Joint fit NOT run. Acceptance (< 10) not met. Remaining work: a camera that frames the head like the screenshot
  (the aligner's head mode needs fixing, or a camera set by hand to match eye and chin height), then regions on
  matching spots, then the joint fit.

## Update 5: faceBox attempt (blocked)
- headBox removed; faceBox [850,270,1100,520] added (face skin, brow to chin, no hair); yaw 0.
- `--align` with faceBox drifted the camera to z 10.7 (the face probe returns a small set of face materials, so the
  face-height match is not meaningful). That camera was discarded and the manual camera restored
  (fov 35, position [0.033,1.40,0.85], target [0.033,1.30,0.02]).
- Probe with the manual camera: mean dE 40.6. skin_cheek and hair_right read "render box empty", so the rule
  "no empty render box before the joint fit" is not met. The joint fit was NOT run.

## Final (joint fit done)
- Camera: from `--align` with faceBox [850,270,1100,520] (yaw 0, excludeBoxes [600,780,1300,1080] to skip the dialogue).
  Result: position [-0.0486,1.4855,1.3469], target [-0.0486,1.3252,0.017], fov 35. The render face sits on the
  screenshot face.
- Auto regions from the aligner: body_1 (dialogue box) and body_2 (magenta sweater) were deleted. Kept face_skin_1 and
  face_skin_2 (auto, weight 1.5, flagged as manual boxes after review). Added coat_chest_left (manual; renderBox
  [820,640,860,680] read off the render).
- Probe start (indoorWindow seed): face_skin_1 16.4, face_skin_2 6.4, coat_chest_left 28.1, mean 15.6.
- Joint fit (seeds indoorWindow, indoorWarm; profile indoor; 3 passes; 1 restart): indoorWindow 15.6 -> 6.04,
  indoorWarm 17.89 -> 9.25, restart 1 9.43 -> 6.19. Final mean dE 6.04 (accepted, < 10).
  Final per region: face_skin_1 1.7, face_skin_2 0.4, coat_chest_left 21.0.
- Fitted env (fit.json): ambient 1.617, key 0.35 at azimuth 80 elevation 5, keyTemp 2817, back 0.7, skyTemp 5500,
  exposure 1.867, toneMapping agx, contrast 1.417, shadow 0.45, rim 0, detail 0.5.
- Doubts: (1) 3 regions only (2 skin, 1 coat), so the fit is driven by the face; the coat box still differs by dE 21
  (the screenshot coat is in shade, the render brighter), so the coat is not matched. (2) Outfit differs from the
  viewer's default: the screenshot is a magenta turtleneck with dark-blue collar; the viewer shows a red crop top.
  Only skin and coat were used; the sweater was dropped. (3) Subject is a Hotel Z lounge (lamp + window), not a café
  (approved). (4) The face is a close-up, so the fit is mostly skin tone; keyTemp 2817 and exposure 1.87 are
  plausible for lamps but are only weakly constrained by 3 regions.
