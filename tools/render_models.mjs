// Batch-render models through the running viewer dev server (cd viewer && npm run dev).
//
//   node tools/render_models.mjs --out <dir> [--preset studio] [--size 600x800] <game/id> ...
// PNGs have a transparent background (ViewerScene.pngDataUrl, same as the viewer's PNG export).
//
// Per model writes <id>_vs.png (VS camera, advent01_loop when present) and <id>_full.png (front camera, default clip).
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf('--' + n); return i < 0 ? d : args[i + 1]; };
const ids = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
const out = opt('out');
if (!out || !ids.length) { console.error('usage: node tools/render_models.mjs --out <dir> [--preset id] [--size WxH] <game/id> ...'); process.exit(1); }
fs.mkdirSync(out, { recursive: true });
const [W, H] = opt('size', '600x800').split('x').map(Number);
const base = opt('url', 'http://localhost:5173');

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: W + 400, height: H + 40 }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.error('page error:', e.message));

async function shot(file) {
  const url = await page.evaluate(async () => {
    const vs = window.__viewer;
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    return vs.pngDataUrl(); // transparent background, like the viewer's PNG export
  });
  fs.writeFileSync(file, Buffer.from(url.split(',')[1], 'base64'));
}

for (const id of ids) {
  const t0 = Date.now();
  await page.goto(`${base}/model-viewer?id=${encodeURIComponent(id)}`);
  try {
    await page.waitForFunction(() => window.__viewer?.anim && window.__viewer.manifest, null, { timeout: 120000 });
  } catch { console.error(id, 'failed to load'); continue; }
  const info = await page.evaluate(async ({ W, H, preset }) => {
    const st = document.createElement('style');
    st.textContent = `.sidebar,.handle,.back,.title-chip{display:none!important}.viewer{display:block!important}
      .stage{position:fixed!important;left:0;top:0;width:${W}px!important;height:${H}px!important}`;
    document.head.appendChild(st);
    const vs = window.__viewer;
    const m = await import('/src/scene/environment.ts');
    const pr = m.LIGHTING_PRESETS.find((p) => p.id === preset);
    if (!pr) throw new Error('unknown preset ' + preset);
    vs.renderer.setPixelRatio(1);
    vs.setEnvironment(pr.env);
    vs.controls.enableDamping = false;
    await new Promise((r) => setTimeout(r, 300));
    return { name: vs.manifest.displayName, advent: vs.anim.has('advent01_loop') };
  }, { W, H, preset: opt('preset', 'studio') });
  const stem = path.join(out, id.replace(/\//g, '_'));
  // Full body: default clip, front camera.
  await page.evaluate(() => { const vs = window.__viewer; vs.anim.setPaused(true); vs.anim.scrubToFrame(0); vs.setCameraPreset('front'); });
  await shot(stem + '_full.png');
  // VS close-up: advent01_loop frame 0, the pose setCameraPreset frames on.
  await page.evaluate(() => {
    const vs = window.__viewer;
    vs.setCameraPreset('vs');
    vs.anim.setPaused(true);
    vs.anim.scrubToFrame(0);
  });
  await shot(stem + '_vs.png');
  console.log(`${id.padEnd(16)} ${info.name.padEnd(24)} ${info.advent ? '' : '(no advent01_loop) '}${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
await browser.close();
