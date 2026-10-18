// Full-body idle renders on one shared scale and ground line, near-orthographic (long lens), transparent background.
// Dev server running: cd viewer && npm run dev
//   node tools/render_idle_ortho.mjs --out <dir> [--preset za_indoor_hotel] [--size 640x1024] [--fov 3] [--margin 0.03] [--tallest 1.929 (fixed frame height basis, to re-render a subset at the set's scale)] [--clip wait_b] [--env '{"detail":0.2,"shadow":0.4}' (overrides on top of the preset)] [--exposure 1.4 (x the preset's)] <game/id> ...
// Frame height = the tallest listed model x (1 + 2*margin); feet at `margin` above the bottom edge, so short models stay short.
// --group stem|all|<file.json>: which models share one frame height. `all` (default) = every listed model; `stem` = models whose
// ids differ only in the trailing _NN variant (outfits of one character: tr1000_00/_01) share it, anything else fits itself;
// a JSON file maps id -> group key (use it for models on one skeleton, e.g. the generic NPCs).
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf('--' + n); return i < 0 ? d : args[i + 1]; };
const ids = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
const out = opt('out');
if (!out || !ids.length) { console.error('usage: node tools/render_idle_ortho.mjs --out <dir> [--preset id] [--size WxH] [--fov 3] [--margin 0.03] <game/id> ...'); process.exit(1); }
fs.mkdirSync(out, { recursive: true });
const [W, H] = opt('size', '640x1024').split('x').map(Number);
const fov = Number(opt('fov', '3'));
const margin = Number(opt('margin', '0.03'));
const base = opt('url', 'http://localhost:5173');
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: W + 400, height: H + 40 }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.error('page error:', e.message));

async function load(id) {
  await page.goto(`${base}/model-viewer?id=${encodeURIComponent(id)}`);
  await page.waitForFunction(() => window.__viewer?.anim && window.__viewer.manifest, null, { timeout: 120000 });
  await page.evaluate(async ({ W, H, preset, exposure, clip, envOver }) => {
    const st = document.createElement('style');
    st.textContent = `.sidebar,.handle,.back,.title-chip{display:none!important}.viewer{display:block!important}
      .stage{position:fixed!important;left:0;top:0;width:${W}px!important;height:${H}px!important}`;
    document.head.appendChild(st);
    const vs = window.__viewer;
    const m = await import('/src/scene/environment.ts');
    vs.renderer.setPixelRatio(1);
    const env = m.LIGHTING_PRESETS.find((p) => p.id === preset).env;
    vs.setEnvironment({ ...env, exposure: env.exposure * exposure, ...envOver });
    vs.controls.enableDamping = false;
    if (clip && vs.anim.has(clip)) vs.anim.play(clip);
    vs.anim.setPaused(true); vs.anim.scrubToFrame(0);
    vs.setCameraPreset('front');
    await new Promise((r) => setTimeout(r, 300));
  }, { W, H, preset: opt('preset', 'za_indoor_hotel'), exposure: Number(opt('exposure', '1')), clip: opt('clip'), envOver: JSON.parse(opt('env', '{}')) });
  return page.evaluate(() => { const f = window.__viewer.fitInfo(); return { top: f.center.y + f.size.y / 2, bottom: f.center.y - f.size.y / 2, w: f.size.x }; });
}

const groupOpt = opt('group', 'all');
const groupMap = groupOpt.endsWith('.json') ? JSON.parse(fs.readFileSync(groupOpt, 'utf8')) : null;
const groupOf = (id) => groupMap ? (groupMap[id] ?? groupMap[id.split('/').pop()] ?? id)
  : groupOpt === 'stem' ? id.replace(/_\d\d$/, '') : 'all';
const tops = {};
for (const id of ids) { try { tops[id] = await load(id); } catch { console.error(id, 'failed'); } }
const tallestOf = {};
const fixedTallest = opt('tallest') ? Number(opt('tallest')) : null;
for (const id of Object.keys(tops)) { const g = groupOf(id); tallestOf[g] = fixedTallest ?? Math.max(tallestOf[g] ?? 0, tops[id].top); }
for (const [g, t] of Object.entries(tallestOf)) console.log('group', g, 'tallest', t.toFixed(3), 'm; frame', (t * (1 + 2 * margin)).toFixed(3), 'm');
for (const id of Object.keys(tops)) {
  const tallest = tallestOf[groupOf(id)], frameH = tallest * (1 + 2 * margin);
  await load(id);
  const url = await page.evaluate(async ({ frameH, margin, tallest, fov, W, H, flash }) => {
    const vs = window.__viewer;
    const rad = Math.PI / 180;
    const cy = frameH / 2 - margin * tallest; // y=0 (ground) sits `margin` above the bottom edge
    const d = (frameH / 2) / Math.tan(fov * rad / 2);
    const c = vs.fitInfo().center;
    vs.camera.fov = fov; vs.camera.aspect = W / H;
    vs.camera.position.set(c.x, cy, d);
    vs.controls.target.set(c.x, cy, 0);
    vs.controls.minDistance = 0; vs.controls.maxDistance = d * 10;
    vs.camera.near = d - 5; vs.camera.far = d + 5; vs.camera.updateProjectionMatrix();
    vs.camera.lookAt(c.x, cy, 0); vs.controls.update();
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    // Flashlight glow (BDSP Policeman): additive sprites on the `Light` bone, sizes/colours fitted to samples/bdsp_police_officer.png
    // (white-yellow core ~3 cm radius, yellow halo ~10 cm, faint spill ~16 cm; 1 px of that shot ~ 4.8 mm).
    if (flash) {
      const T = await import(performance.getEntriesByType('resource').map((r) => r.name).find((n) => /deps\/three\.js/.test(n)));
      const p = vs.findBone('Light')?.getWorldPosition(new T.Vector3());
      if (p) {
        const c = document.createElement('canvas'); c.width = c.height = 128;
        const g = c.getContext('2d'), gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
        gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.2, 'rgba(255,255,255,0.8)'); gr.addColorStop(0.5, 'rgba(255,255,255,0.2)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
        g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
        const tex = new T.CanvasTexture(c), grp = new T.Group(); grp.name = '__fx'; (vs.root.parent ?? vs.root).add(grp);
        for (const [size, color, op] of [[0.32, 0xffd23a, 0.22], [0.2, 0xffe060, 0.7], [0.09, 0xfff4c0, 1]]) {
          const sp = new T.Sprite(new T.SpriteMaterial({ map: tex, color, transparent: true, opacity: op, blending: T.AdditiveBlending, depthWrite: false, depthTest: false, toneMapped: false }));
          sp.position.copy(p); sp.scale.setScalar(size); sp.renderOrder = 999; grp.add(sp);
        }
      }
    }
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    return vs.pngDataUrl();
  }, { frameH, margin, tallest, fov, W, H, flash: id.endsWith('/tr0039_00') && opt('flashlight', '1') !== '0' });
  // A viewer/preview holding the PNG open makes Windows fail the write (EBUSY/UNKNOWN): retry, then keep going and report it.
  const file = path.join(out, id.split('/').pop() + '.png'), buf = Buffer.from(url.split(',')[1], 'base64');
  let saved = false;
  for (let t = 0; t < 5 && !saved; t++) {
    try { fs.writeFileSync(file, buf); saved = true; } catch (e) { await new Promise((r) => setTimeout(r, 1000)); }
  }
  if (!saved) { console.error(id, 'WRITE FAILED (file locked?)'); process.exitCode = 1; continue; }
  console.log(id);
}
await browser.close();
