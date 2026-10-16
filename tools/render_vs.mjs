// VS-screen shots of the major Z-A NPCs holding their ball (dev server running: cd viewer && npm run dev).
//
//   node tools/render_vs.mjs --out <dir> [--preset za_indoor_hotel] [--size 1280x720] [--hand right|left|both] [name ...]
// Writes <dir>/VS<Name>[_left|_right].png, transparent background (ViewerScene.pngDataUrl).
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

// name -> [model id, held ball model, {hand, yaw (deg about Y), margin}]. Balls are game logic (user-supplied), see memory za-npc-ball-table.
export const NPCS = {
  Gwynn: ['tr0018_00', 'ob0208_00', { pitch: 0.05 }],     // Nest
  Grisham: ['tr0009_00', 'ob0210_00'],   // Timer
  Griselle: ['tr0010_00', 'ob0215_00'],  // Quick
  Canari: ['tr0015_00', 'ob0215_00', { hide: '^obj_stone', yaw: -35, pitch: -0.1, fov: 40 }],    // Quick
  Corbeau: ['tr0021_00', 'ob0213_00', { pitch: -0.1 }],   // Dusk
  Philippe: ['tr0022_00', 'ob0220_00', { yaw: -10, roll: -20, pitch: -0.45, fov: 45, zoomOut: 1.4 }],  // Heavy
  Tarragon: ['tr0016_00', 'ob0209_00'],  // Repeat
  Jacinthe: ['tr0019_00', 'ob0211_00'],  // Luxury
  Lebanne: ['tr0020_00', 'ob0203_00', { hand: 'left', noBall: true, yaw: -55, roll: -20, pitch: 0.25, fov: 50, zoomOut: 1.1 }],   // Great ball per user list, but not held in her VS pose
  Ivor: ['tr0017_00', 'ob0202_00', { yaw: -25, pitch: -0.1, fov: 45, zoomOut: 1.25 }],      // Ultra
  Taunie: ['tr0001_00', 'ob0204_00'],    // rival_f; ball unknown, Poké Ball placeholder
  Urbain: ['tr0002_00', 'ob0204_00'],    // rival_m; ball unknown, Poké Ball placeholder
};

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf('--' + n); return i < 0 ? d : args[i + 1]; };
const names = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
const out = opt('out');
if (!out) { console.error('usage: node tools/render_vs.mjs --out <dir> [--preset id] [--size WxH] [--hand right|left|both] [name ...]'); process.exit(1); }
fs.mkdirSync(out, { recursive: true });
const [W, H] = opt('size', '1280x720').split('x').map(Number);
const hands = opt('hand', 'both') === 'both' ? ['right', 'left'] : [opt('hand')];
const base = opt('url', 'http://localhost:5173');
const todo = names.length ? names : Object.keys(NPCS);

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: W + 400, height: H + 40 }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.error('page error:', e.message));

for (const name of todo) {
  const [id, ball, cfg = {}] = NPCS[name] ?? [];
  if (!id) { console.error('unknown NPC', name); continue; }
  await page.goto(`${base}/model-viewer?id=${encodeURIComponent('za/' + id)}`);
  await page.waitForFunction(() => window.__viewer?.anim && window.__viewer.manifest, null, { timeout: 120000 });
  await page.evaluate(async ({ W, H, preset }) => {
    const st = document.createElement('style');
    st.textContent = `.sidebar,.handle,.back,.title-chip{display:none!important}.viewer{display:block!important}
      .stage{position:fixed!important;left:0;top:0;width:${W}px!important;height:${H}px!important}`;
    document.head.appendChild(st);
    const vs = window.__viewer;
    const m = await import('/src/scene/environment.ts');
    vs.renderer.setPixelRatio(1);
    vs.setEnvironment(m.LIGHTING_PRESETS.find((p) => p.id === preset).env);
    vs.controls.enableDamping = false;
    await new Promise((r) => setTimeout(r, 300));
  }, { W, H, preset: opt('preset', 'za_indoor_hotel') });
  for (const hand of (cfg.hand && opt('hand', 'both') === 'right' ? [cfg.hand] : hands)) {
    await page.evaluate(async ({ ball, hand, yaw, pitch, fov, zoomOut, roll, hide, noBall }) => {
      const vs = window.__viewer;
      if (fov) { vs.camera.fov = fov; vs.camera.updateProjectionMatrix(); }
      const prop = vs.manifest.props.find((p) => p.side === hand);
      await vs.setHeldItem(`/models/za/${ball}/model.glb`, prop);
      vs.setCameraPreset('vs');           // plays advent01_loop first
      vs.anim.setPaused(true);
      vs.anim.scrubToFrame(0);
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      // Re-frame so head and ball both land in the shot (the stock 'vs' framing only follows the body bounds).
      vs.root.updateMatrixWorld(true);
      const head = vs.findBone('head');
      const headP = head.getWorldPosition(vs.camera.position.clone().set(0, 0, 0));
      const ballP = vs.held.obj.getWorldPosition(headP.clone().set(0, 0, 0));
      const h = vs.fitInfo().h;
      const pts = [headP.clone().add(headP.clone().set(0, 0.115 * h, 0)), headP.clone().add(headP.clone().set(0, -0.14 * h, 0)),
        ballP.clone().add(ballP.clone().set(0.05, 0.05, 0)), ballP.clone().add(ballP.clone().set(-0.05, -0.05, 0))];
      const target = pts.reduce((a, p) => a.add(p), pts[0].clone().set(0, 0, 0)).multiplyScalar(1 / pts.length);
      const dir = vs.camera.position.clone().sub(vs.controls.target).normalize();
      const c = Math.cos(yaw * Math.PI / 180), sn = Math.sin(yaw * Math.PI / 180);
      dir.set(dir.x * c + dir.z * sn, 0, -dir.x * sn + dir.z * c).normalize(); dir.y = Number(pitch); dir.normalize();
      const fits = (d) => {
        vs.camera.position.copy(target).addScaledVector(dir, d);
        vs.camera.lookAt(target); vs.camera.updateMatrixWorld(true);
        return pts.every((p) => { const n = p.clone().project(vs.camera); return Math.abs(n.x) <= 0.8 / zoomOut && Math.abs(n.y) <= 0.85 / zoomOut; });
      };
      let lo = 0.1, hi = 4;
      for (let i = 0; i < 30; i++) { const mid = (lo + hi) / 2; if (fits(mid)) hi = mid; else lo = mid; }
      fits(hi);
      vs.controls.target.copy(target); vs.controls.update();
      if (roll) { // dutch tilt: positive = clockwise as seen through the lens
        const view = target.clone().sub(vs.camera.position).normalize();
        vs.camera.up.set(0, 1, 0).applyAxisAngle(view, -roll * Math.PI / 180); vs.camera.lookAt(target); vs.camera.updateMatrixWorld(true);
      }
      // Hide props that float out of frame (Canari: obj_stone). A clip's visibility key beats the user override, so drop the keys first.
      const re = new RegExp(hide + '|phone', 'i');
      for (const p of vs.manifest.parts) if (re.test(p.node)) { for (const c of vs.manifest.clips) if (c.visibility) delete c.visibility[p.node]; vs.setPartVisible(p.node, false); }
      if (noBall) { await vs.setHeldItem(null, undefined); await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))); }
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    }, { ball, hand, yaw: Number(opt('yaw', cfg.yaw ?? 0)), pitch: Number(opt('pitch', cfg.pitch ?? -0.15)), fov: Number(opt('fov', cfg.fov ?? 0)), zoomOut: Number(opt('zoomout', cfg.zoomOut ?? 1)), roll: Number(opt('roll', cfg.roll ?? 0)), hide: cfg.hide ?? 'phone', noBall: !!cfg.noBall });
    const url = await page.evaluate(() => window.__viewer.pngDataUrl());
    const file = path.join(out, `VS${name}${hands.length > 1 ? '_' + hand : ''}.png`);
    fs.writeFileSync(file, Buffer.from(url.split(',')[1], 'base64'));
    console.log(file);
  }
}
await browser.close();
