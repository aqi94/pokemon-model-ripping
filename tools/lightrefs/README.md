# Lighting references from gameplay video

Goal: viewer lighting presets that match what the game shows in play, one preset per scene type, fitted to
high-quality frames of YouTube playthroughs. Everything for one scene lives in `refs/<game>/<scene>/`; PNGs are
git-ignored (the JSON sidecars make them re-fetchable with `fetch_frames regrab`), JSON is tracked.

```
refs/za/<scene>/
  scout/search.json, scout/<videoId>.json, scout/<videoId>_<from>-<to>_sNN.png, scout/<videoId>_around_<t>_s00.png
  <slug>.png            1920-wide frame (ignored)           <slug>.json           sidecar (tracked)
  <slug>_regions.json   camera + boxes (tracked)            <slug>_probe.png/.json  material-id pass (ignored / tracked)
  <slug>_debug.png      boxes on both images (ignored)      <slug>_render.png     fitted render (ignored)
  fit.json              fitted env + scores (tracked)       notes.md              what was used and why (tracked)
```

Scenes (ids, profiles, seeds, candidate subjects, search hints): `tools/lightrefs/scenes.py`. Fit knobs per profile:
`fit_profiles.json`. Seed environments: `seed_envs.json` or any preset id from `viewer/src/scene/environment.ts`.

## Workflow for one scene

Dev server running (`cd viewer && npm run dev`), models already converted (`out/za/<id>/`). All commands from the repo root
with `./.venv/Scripts/python.exe`.

1. `python -m tools.lightrefs.fetch_frames search --scene <id> [--query ...] [--channel @Handle]` -> candidate videos
   (prefer `switch2` = native 1080p/4K60 uploads; no-commentary full-game walkthroughs are long but complete).
   `info --video <id> --scene <id>` prints chapters when the upload has them.
2. `scout --scene <id> --video <id> --from mm:ss --to mm:ss [--every 20]` -> labelled contact sheets in `scout/`
   (about 25 s per 10 minutes of video). Read them, note timecodes where the subject faces the camera, is sharp,
   has no HUD over the areas you need, and the lighting is clearly this scene.
3. `around --scene <id> --video <id> --at mm:ss [--span 2 --step 0.25]` -> a full-res mini sheet to pick the exact frame.
4. `grab --scene <id> --video <id> --at mm:ss.fff --slug <scene>_<subject>_NN --model <id> --name "<name>" --clip <guess>
   [--time day|dusk|night] [--weather sunny|overcast|rain|none] [--location "..."] [--light ...] [--indoor]
   --context cutscene|field|battle|vs [--confidence high|medium|low] [--ui] [--dof none|background|subject]`
   -> `<slug>.png` + `<slug>.json`. Aim for 2-4 frames per scene: two different characters and one Pokémon.
5. Write `<slug>_regions.json` with just: `model`, `clip`, `frame`, `screenshot`, `width`, `height`, a rough `camera`
   (any; it gets corrected), `subjectBox` = [x0,y0,x1,y1] of the subject's whole body on the screenshot (head to feet,
   generous by ~10 px), `yaw` = the camera's angle around the subject in degrees (0 = the subject faces the camera,
   +45 = the subject looks toward screen-LEFT (we see their left cheek), -45 = toward screen-right, 180 = from behind),
   optional `faceBox` = [x0,y0,x1,y1] of the FACE (hairline/brow to chin, cheek to cheek, no hair) on the screenshot for
   close-ups or when a dialogue box hides the body (the aligner then scales and positions on the face skin),
   and `"regions": []`. Then let the tool place the camera and the boxes:
   `node tools/fit_scene.mjs refs/za/<scene>/<slug>_regions.json --preset <seed> --align --auto-regions --debug-out refs/za/<scene>`
   `--align` moves the camera until the rendered model fills `subjectBox`; `--auto-regions` writes 2 pure
   single-material windows per material (box = renderBox). Read `<slug>_debug.png`: left = screenshot with the boxes
   (green), right = render (cyan). Every box must land on the SAME body part and material in both images. If the
   model faces the wrong way, flip the sign of `yaw` (or adjust by +-30) and rerun; if the pose differs (arm across
   the body, sitting), try another `clip`/`frame` from `out/za/<id>/manifest.json`. Delete or move any auto box that
   lands on a different part in the screenshot (edit `box` only; set `"manual": true` on boxes you place yourself so
   a later `--auto-regions` keeps them). Keep 5-10 boxes, skin ones weighted 1.5 (automatic).
6. Fit jointly: `node tools/fit_scene.mjs --scene refs/za/<scene> --seeds <a,b> --profile <profile> --fit --passes 3
   --write refs/za/<scene>/fit.json --debug-out refs/za/<scene>`. Accept mean dE < 8 (8-12 with a note); > 12 means
   boxes/camera/clip are wrong, not the lights. Lock implausible knobs (`--lock elevation=55`) and refit once.
7. `notes.md`: videos (id, channel, platform), timecodes, why each frame, clip/frame, final dE vs the Studio
   baseline, doubts. Never hand-edit `fit.json`.

Then (main agent) `python -m tools.lightrefs.build_presets --game za` -> `viewer/src/scene/zaPresets.ts`,
`refs/za/index.json` (Reference panel) and `out/za_lighting_sheet.png`.

## Clip / frame choice

Trainers: field idle `defaultwait01_loop` frame 0, dialogue `speak02_loop` or `glad01` (NOT `speak01_loop`: it is an upper-body layer clip whose root scale is 0 in the viewer, the model vanishes), VS screen `advent01_loop` frame 40,
battle `btl01_idle01` / `battlewait01_loop`, sitting `sit01_loop`. Pokémon: `defaultwait01_loop`, battle
`battleidle02`, walking `walk01_loop`. The clip list is in `out/za/<id>/manifest.json`.

## Regions JSON

```json
{
  "schema": 1, "slug": "indoor_hotel_manager_01", "scene": "indoor_hotel",
  "model": "za/tr0003_00", "clip": "speak01_loop", "frame": 12,
  "screenshot": "indoor_hotel_manager_01.png", "width": 1920, "height": 1080,
  "camera": { "fov": 35, "position": [0.1, 1.45, 2.2], "target": [0.0, 1.35, 0.0] },
  "subjectBox": [860, 120, 1180, 1050], "yaw": 20,
  "weight": 1.0, "skip": false,
  "regions": [
    { "name": "skin_forehead", "box": [900, 300, 940, 330], "renderBox": [905, 310, 945, 340], "weight": 1.5 },
    { "name": "jacket_lit", "box": [860, 520, 920, 580] }
  ]
}
```
`box` is in screenshot pixels, `renderBox` (optional, defaults to `box`) in render pixels; the render canvas has the
screenshot's size. `excludeBoxes` = [[x0,y0,x1,y1], ...] screenshot areas (dialogue box, HUD, other characters) that `--auto-regions` must not use.
`subjectBox` (screenshot px) + `yaw` drive `--align` (add `"subjectCut": "bottom"` when a dialogue box or object hides the
lower body: the box then ends at the occluder and the aligner scales on width and aligns the top); `--auto-regions` fills `regions` (entries get
`"auto": true`; entries with `"manual": true` survive a rerun). Camera: vertical `fov` in degrees, `position`/`target` in world metres (models stand on y = 0, a
trainer's head is around y = 1.4-1.6; the camera preset "front" looks along -Z from +Z).

## Sidecar JSON (`<slug>.json`)

`schema 1, slug, scene, game, source (youtube|manual), video {id,url,title,channel,platform,formatId,width,height,fps,vcodec} | null,
timestamp (s), timecode, grabbedAt, image {file,width,height,scaledFrom}, tags {timeOfDay,weather,location,indoor,lightType},
subject {kind,model,name,confidence}, clipGuess, context, quality {ui,motionBlur,dof,fade}, notes`.

## fit.json (written by fit_scene.mjs)

`schema, scene, profile, locks, seeds, bestSeed, passes, seedLog [{seed,start,end}], env (all Environment keys, ground hex),
score {mean, perShot {slug: {dE, weight, regions {name: dE}}}}, shots [slug], fittedAt, tool`.

## Quality rules

Reject frames with dialogue boxes over the regions you need (boxing above them is fine), menus/HUD, screen fades
(first/last second of a cut), motion blur, subject depth-of-field, bloom halos on white cloth. Prefer `switch2`
uploads; grab from the 2160p stream (done automatically: it is downscaled to 1920 wide, which averages out
compression noise). The fit compares region means, so each box must sit on one flat material in both images.
