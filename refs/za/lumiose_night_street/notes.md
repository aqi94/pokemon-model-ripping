# lumiose_night_street: notes

Status: NOT calibrated. No shot usable for the joint fit, no fit.json, no final mean dE.

## Videos scouted
- 4spEtYJBlVE (Shirrako part 1, Switch 2, 4K60, no commentary, ends at 7.7 h). Night frames are Battle Zone (red laser walls, Z-A Royale UI). No free-roam Lumiose night street found. Daytime windows checked and rejected.
- 6qiJ2FnPSFM (Shirrako full game, Switch 2, 4K60, no commentary). Windows scouted: 08:00-08:10, 10:00-10:10, 12:00-12:10, 14:00-14:10, 16:00-16:10, plus the 16:08 around-sheet.

## Candidate frames
1. 6qiJ2FnPSFM 8:03:50 (fountain square, night, lamp-lit, blue ambient, warm windows). Best lighting match for "street lamps and shop signs", but the people in frame do not match any catalogued model I checked: the hoodie man (brown hair, grey-white hoodie, green trousers) is not Urbain (tr0002_00: pink-red hair, brown jacket, grey joggers); the beige-jacket woman in the background (dark hair, cap) is not Taunie (tr0001_00: pink bun, beige jacket, white shorts). Gwynn (tr0018_00) is not in the frame either. Grab and sidecar deleted. Re-scouting this square at 8:03:46-8:03:53 or nearby would be needed to find a catalogued NPC facing camera.
2. 6qiJ2FnPSFM 16:08:25 (Lebanne, tr0020_00, speak). Identity is solid (teal hair, blue-white maid frills, outfit colours match the tr0020_00 thumbnail). Night variant: the purple Mega-tower light cast is in this scene, so it is not plain street lamp lighting. Sidecar confidence set to high, dialogue box covers the lower body. Grab: lumiose_night_street_lebanne_01.png/.json.
   - Fit attempt: `--align --auto-regions` with clip speak01_loop placed no model in the render. The idle clip defaultwait01_loop renders, but the aligner then put the camera on her legs (camera at 0.32, 0.74, 0.88) and the boxes landed on thighs, hair and the wrong skin. Start dE 26.4. Mismatch, so the shot is marked "skip": true in lumiose_night_street_lebanne_01_regions.json.

## Doubts
- Subject identity of the fountain-square frame is unresolved (see above).
- The aligner's cut-bottom path (subjectBox scaled on width, top aligned) did not converge for tr0020_00 with a dialogue box over the lower body. This is a tool problem, not a lighting result.
- Lebanne's frame is under the purple Mega-tower light, so it is not the target "street lamps" lighting even if it fits.

## Next steps
- Find a full-body catalogued NPC under street lamps (08:03-08:10 square, later windows), or a no-dialogue frame of Lebanne/Taunie/Urbain on a plain night street.
- Then regrab, write regions, run --align --auto-regions, check the debug PNG, then the joint fit with --profile night --seeds night.

## Final attempt (Lebanne, clip defaultwait01_loop frame 0)
- Alignment: 3 rounds. yaw +20 aligned the chest, not the head, and the skin box landed on the dialogue box. yaw -30 showed her back, boxes again on the dialogue box. Start dE 17.1 (yaw +20), 19.7 (yaw -30).
- Shot is marked "skip": true. No regions, so no joint fit. fit.json was NOT written. Final mean dE: not available.
- This shot is the Mega-tower purple NIGHT VARIANT, not the lamp-lit street. The lamp-lit fountain square (8:03:50) had no identifiable subject.

## Final fit (face mode)
- Shot: lumiose_night_street_lebanne_01 (tr0020_00 Lebanne, defaultwait01_loop frame 0, yaw 35, faceBox, excludeBoxes = dialogue box). Mega-tower purple NIGHT VARIANT. Not the lamp-lit street.
- Kept regions: face_skin_1/2, hairstyle_1/2, obj_1/2 (frills). Dropped body_1/2 (both on the dialogue box). Per-region dE after fit: face 4.0/4.9, hair 2.4/13.0, frills 18.8/12.8.
- Joint fit (--profile night --seeds night, 3 passes, 1 restart): start 22.35 (night) / 18.03 (restart 1), final mean dE 8.62 (single shot). Accepted range is < 8; this is 8-12 so noted. Provisional.
- Result in fit.json (written): ambient 2.397, key 0.4, skyTemp 9500, exposure 1.387, agx, contrast 1.5, saturation 0.8, gi 1.327. Sanity: key < 1.0 ok, skyTemp > 7000 ok, exposure 0.8-2.0 ok.
- Doubts: this is a single purple-tower shot, so the fit describes that variant, not generic street lamps. The fountain square has no identified subject. Frill boxes are at 13-19 dE, hairstyle_2 at 13. fit.json is provisional.
