// Full-body idle renders on one shared scale and ground line, near-orthographic (long lens), transparent background.
// Dev server running: cd viewer && npm run dev
//   node tools/render_idle_ortho.mjs --out <dir> [--preset za_indoor_hotel] [--size 640x1024] [--fov 3] [--margin 0.03] <game/id> ...
// Frame height = the tallest listed model x (1 + 2*margin); feet at `margin` above the bottom edge, so short models stay short.
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
    vs.anim.setPaused(true); vs.anim.scrubToFrame(0);
    vs.setCameraPreset('front');
    await new Promise((r) => setTimeout(r, 300));
  }, { W, H, preset: opt('preset', 'za_indoor_hotel') });
  return page.evaluate(() => { const f = window.__viewer.fitInfo(); return { top: f.center.y + f.size.y / 2, bottom: f.center.y - f.size.y / 2, w: f.size.x }; });
}

const tops = {};
for (const id of ids) { try { tops[id] = await load(id); } catch { console.error(id, 'failed'); } }
const tallest = Math.max(...Object.values(tops).map((t) => t.top));
const frameH = tallest * (1 + 2 * margin);
console.log('tallest', tallest.toFixed(3), 'm; frame', frameH.toFixed(3), 'm');
for (const id of Object.keys(tops)) {
  await load(id);
  const url = await page.evaluate(async ({ frameH, margin, tallest, fov, W, H }) => {
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
    return vs.pngDataUrl();
  }, { frameH, margin, tallest, fov, W, H });
  fs.writeFileSync(path.join(out, id.split('/').pop() + '.png'), Buffer.from(url.split(',')[1], 'base64'));
  console.log(id);
}
await browser.close();
