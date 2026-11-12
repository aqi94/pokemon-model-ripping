// Square transparent renders of BDSP battle trainers in their `advent02_b` (VS intro) pose, falling back to advent_b then wait_b through the running viewer dev server.
//
//   node tools/render_bdsp_vs.mjs --out <dir> [--size 1024] [--clip advent02_b] [--frames 0.5,1] [--camera front] [--preset bdspVs]
//                                 [--ball RItem1|LItem1] [--look vs] [--tag _x] <game/id> ...
// Writes <id>_f<frame>.png per requested frame fraction; crop/fit to the final canvas afterwards (tools/fit_square.py).
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf('--' + n); return i < 0 ? d : args[i + 1]; };
const ids = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
const out = opt('out');
if (!out || !ids.length) { console.error('usage: node tools/render_bdsp_vs.mjs --out <dir> [--size N] [--clip c] [--frames f,f] <game/id> ...'); process.exit(1); }
fs.mkdirSync(out, { recursive: true });
const S = Number(opt('size', '1024'));
const frames = opt('frames', '1').split(',').map(Number);
const clip = opt('clip', 'advent02_b');
const base = opt('url', 'http://localhost:5173');
// --ball <bone>: parent the Poké Ball (bdsp/ob0204_00) to that attach bone (RItem1 / LItem1 ...); --tag names the output suffix.
const ball = opt('ball', '');
const tag = opt('tag', '');
// --variant N: select KHR_materials_variants entry N (SV trainers: 1 = ver_00 Scarlet, 2 = ver_01 Violet) before rendering.
const variant = Number(opt('variant', '-1'));
// --hide a,b / --show a,b: toggle mesh nodes (manifest part names) before rendering; --cy Y: centre of the --fixed window in metres (default 0.95), for zoomed crops.
const hide = opt('hide', '').split(',').filter(Boolean);
const show = opt('show', '').split(',').filter(Boolean);
const cy = Number(opt('cy', '0.95'));
const ballScale = Number(opt('ballscale', '1'));
// Camera: the VS images sit between the viewer's wide perspective and a flat orthographic view (feet seen slightly from above):
// the viewer camera is perspective-only, so a long lens is used: --fov 14 --pitch 8 by default (--fov 3 ~ orthographic, --fov 0 = viewer preset).
const fov = Number(opt('fov', '14'));
const pitch = Number(opt('pitch', '8'));
const yaw = Number(opt('yaw', '0'));
// --fixed H: ignore the model's own bounds; frame a fixed H-metre-tall window centred 0.95 m above the origin, so several models rendered
// this way share one scale and ground line (used to compose double-battle pairs).
const fixedH = Number(opt('fixed', '0'));
// --gaze x,y: aim both eyeball bones at the camera (0,0) or at a point offset from it (x right, y up, as a fraction of the camera distance).
const uvsign = opt('uvsign', '').split(',').filter(Boolean).map(Number); // eye-gaze offset signs x,y (default: the viewer's)
const gaze = opt('gaze', '').split(',').filter(Boolean).map(Number);
// --look <name>: use that saved look's `lighting` (out/<game>/<id>/configs.json, viewer Appearance > Saved looks) instead of the preset
// when the model has one (bdsp `vs` = lighting fitted to each Bulbapedia reference). Camera/pose stay this script's.
const lookName = opt('look', '');
// --env '{"exposure":2}': overrides on top of the preset / look lighting
const envOverride = opt('env', '') ? JSON.parse(opt('env', '')) : {};

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: S + 400, height: S + 40 }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.error('page error:', e.message));

for (const id of ids) {
  await page.goto(`${base}/model-viewer?id=${encodeURIComponent(id)}`);
  try {
    await page.waitForFunction(() => window.__viewer?.anim && window.__viewer.manifest, null, { timeout: 120000 });
  } catch { console.error(id, 'failed to load'); continue; }
  await page.evaluate(async ({ S, preset, envOverride, lookName, id }) => {
    const st = document.createElement('style');
    st.textContent = `.sidebar,.handle,.back,.title-chip{display:none!important}.viewer{display:block!important}
      .stage{position:fixed!important;left:0;top:0;width:${S}px!important;height:${S}px!important}`;
    document.head.appendChild(st);
    const vs = window.__viewer;
    const m = await import('/src/scene/environment.ts');
    vs.renderer.setPixelRatio(1);
    let lit = null;
    if (lookName) {
      const [g, mid] = id.split('/');
      try { const f = await (await fetch(`/__configs/${g}/${mid}`)).json(); lit = (f.looks ?? []).find((l) => l.name === lookName)?.lighting ?? null; } catch { /* no looks */ }
    }
    vs.setEnvironment({ ...(lit ?? m.LIGHTING_PRESETS.find((p) => p.id === preset).env), ...envOverride });
    vs.controls.enableDamping = false;
    await new Promise((r) => setTimeout(r, 300));
  }, { S, preset: opt('preset', 'bdspVs'), envOverride, lookName, id });
  if (variant >= 0) await page.evaluate(async (v) => { const vs = window.__viewer; if (vs.variants?.available) await vs.variants.select(v); await new Promise((r) => setTimeout(r, 400)); }, variant);
  if (ball) await page.evaluate(async (bone) => { const vs = window.__viewer; vs.setHeldOnlyInClips(false); await vs.setHeldItem('/models/bdsp/ob0204_00/model.glb', { bone, side: '', clips: [] }); }, ball);
  if (ball && ballScale !== 1) await page.evaluate((k) => window.__viewer.held.obj.scale.setScalar(k), ballScale);
  if (uvsign.length === 2) await page.evaluate(([x, y]) => { window.__viewer.uvSign = { x, y }; }, uvsign);
  if (hide.length || show.length) await page.evaluate(({ hide, show }) => { const vs = window.__viewer; for (const n of hide) vs.setPartVisible(n, false); for (const n of show) vs.setPartVisible(n, true); }, { hide, show });
  const used = await page.evaluate((clip) => { const a = window.__viewer.anim; const c = [clip, 'advent_b', 'wait_b'].find((n) => a.has(n)) ?? null; if (c) a.play(c); return c; }, clip);
  const threeUrl = await page.evaluate(() => performance.getEntriesByType('resource').map((r) => r.name).find((n) => /deps\/three\.js/.test(n)));
  for (const f of frames) {
    const url = await page.evaluate(async ({ f, camera, threeUrl, cam, gz }) => {
      const vs = window.__viewer, a = vs.anim;
      a.setPaused(true);
      const fc = Math.max(1, (a.state().frameCount || 1) - 1);
      a.scrubToFrame(Math.round(f * fc));
      vs.setCameraPreset(camera);
      // clip planes hug the model below: a long lens sits far away and a wide near/far range z-fights the thin face meshes
      if (cam.fov > 0) {
        const fi0 = vs.fitInfo(); const old = vs.camera.fov; const rad = Math.PI / 180;
        const fi = cam.fixedH > 0 ? { center: vs.controls.target.clone().set(0, cam.cy, 0), dist: (cam.fixedH / 2) / Math.tan(old * rad / 2), radius: fi0.radius } : fi0;
        vs.camera.fov = cam.fov;
        const d = fi.dist * Math.tan(old * rad / 2) / Math.tan(cam.fov * rad / 2);
        const dir = { x: Math.sin(cam.yaw * rad) * Math.cos(cam.pitch * rad), y: Math.sin(cam.pitch * rad), z: Math.cos(cam.yaw * rad) * Math.cos(cam.pitch * rad) };
        vs.camera.position.set(fi.center.x + dir.x * d, fi.center.y + dir.y * d, fi.center.z + dir.z * d);
        vs.controls.maxDistance = fi.radius * 1000; vs.controls.target.copy(fi.center); vs.camera.near = Math.max(d - fi.radius * 1.6, fi.radius / 100); vs.camera.far = d + fi.radius * 1.6; vs.camera.updateProjectionMatrix(); vs.camera.lookAt(fi.center); vs.controls.update();
      }
      if (gz.length === 2) {
        const T = await import(threeUrl);
        vs.root.updateMatrixWorld(true);
        const skinned = []; vs.gltf.scene.traverse((o) => { if (o.isSkinnedMesh) skinned.push(o); });
        for (const n of ['LEyeball', 'REyeball']) {
          const b = vs.findBone(n); if (!b) continue;
          // the eyeball's own forward axis = the character's forward (+Z) seen from its bind orientation
          let fLocal = null;
          for (const m of skinned) { const i = m.skeleton.bones.indexOf(b); if (i >= 0) { const q = new T.Quaternion().setFromRotationMatrix(m.skeleton.boneInverses[i].clone().invert()); fLocal = new T.Vector3(0, 0, 1).applyQuaternion(q.invert()); break; } }
          if (!fLocal) continue;
          const eye = b.getWorldPosition(new T.Vector3());
          const toCam = vs.camera.position.clone().sub(eye).normalize();
          const right = new T.Vector3().crossVectors(new T.Vector3(0, 1, 0), toCam).normalize().negate();
          const up = new T.Vector3().crossVectors(toCam, right).normalize();
          const target = vs.camera.position.clone().add(right.multiplyScalar(gz[0] * vs.camera.position.distanceTo(eye))).add(up.multiplyScalar(gz[1] * vs.camera.position.distanceTo(eye)));
          const want = target.sub(eye).normalize();
          const wq = b.getWorldQuaternion(new T.Quaternion());
          const cur = fLocal.clone().applyQuaternion(wq);
          const nq = new T.Quaternion().setFromUnitVectors(cur, want).multiply(wq);
          b.quaternion.copy(b.parent.getWorldQuaternion(new T.Quaternion()).invert().multiply(nq));
          b.updateMatrixWorld(true);
        }
      }
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      return vs.pngDataUrl();
    }, { f, camera: opt('camera', 'front'), threeUrl, cam: { fov, pitch, yaw, fixedH, cy }, gz: gaze });
    fs.writeFileSync(path.join(out, `${id.replace(/\//g, '_')}_f${f}${tag}.png`), Buffer.from(url.split(',')[1], 'base64'));
  }
  console.log(id, used);
}
await browser.close();
