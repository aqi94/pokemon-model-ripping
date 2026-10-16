# indoor_lab notes

Status: one usable shot. The scene is under-supplied; see doubts.

## Videos used
- `4spEtYJBlVE` (Shirrako, Switch 2, 4K60, no commentary, Pokemon Legends Z-A part 1). Scouted 0:20:00-1:10:00 in 30 s steps.
  - Frames used: 0:54:00.000 (Mable, lab office, dialogue "Press X to open the main menu...").
  - Frames rejected: 1:04:00 (Taunie, outdoor dusk street, not indoor).
  - Other scouted interiors: Hotel Z lobby ~0:26:30-0:27:30 and 0:29:00; Taunie's hotel room ~0:32:30-0:34:30 (warm daylight through windows, no fluorescent); the Pokemon Center / research room ~0:42:30-0:43:30 (nurse, not a candidate subject); Mable's office ~0:52:00-0:55:00.

## Shots
- `indoor_lab_mable_01` (tr0023_00 Mable, clip speak02_loop frame 0, cutscene close-up, UI present). Boxes: hairstyle x2, face_skin x2, obj (glasses) x2. The face and glasses are shown in close-up with the dialogue box covering the body, so `subjectCut: bottom` and `faceBox` are used.
  - Start dE 8.5 (indoorCool), end dE 4.74 (restart 1), per-box dE: hairstyle_1 3.6, hairstyle_2 3.5, face_skin_1 3.9, face_skin_2 0.9, obj_1 3.2, obj_2 15.8.
- Rejected: `indoor_lab_taunie_01` (outdoor street at dusk, not this scene). Files deleted.

## Fit
- Final mean dE 4.74 (single shot, so this is an overfit-prone fit). Knobs: ambient 1.633, key 0.667, keyTemp 4300, rim 0, exposure 1.867, agx, contrast 1.5, saturation 0.817, shadow 0.55.
- Sanity: rim 0, key < 1.5, ambient carries the light, keyTemp in the lamp range, exposure in 0.8-2.2. All pass.

## Doubts
- The lighting in the Mable frame is warm daylight through a large window, not the cool-white fluorescent the scene brief describes. The fit moved keyTemp to 4300 (warm) to match, so this preset may not represent a fluorescent lab. Treat it as the Mable office look, not the lab.
- Only one shot. No second character or Pokemon was found with fluorescent interior light. Taunie, the President and the Secretary were not found in an indoor frame that met the brief in the scouted windows (0:20-1:10). A longer scout is needed.
- Subject identity: Mable confirmed by name tag and hair/glasses against out/za/tr0023_00/thumb.webp. Clip speak02_loop is a dialogue clip; the face skin is the most reliable region.
- obj_2 (glasses rim) dE 15.8 remains high; it may be a reflection or a glare on the lenses.
- UI: the dialogue box covers the lower body. Only the top portion of the body is used (subjectCut bottom).
