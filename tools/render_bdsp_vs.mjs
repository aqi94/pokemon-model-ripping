// Square transparent renders of BDSP battle trainers in their `advent02_b` (VS intro) pose, falling back to advent_b then wait_b through the running viewer dev server.
//
//   node tools/render_bdsp_vs.mjs --out <dir> [--size 1024] [--clip advent02_b] [--frames 0.5,1] [--camera front] [--preset studio] <game/id> ...
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

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: S + 400, height: S + 40 }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.error('page error:', e.message));

for (const id of ids) {
  await page.goto(`${base}/model-viewer?id=${encodeURIComponent(id)}`);
  try {
    await page.waitForFunction(() => window.__viewer?.anim && window.__viewer.manifest, null, { timeout: 120000 });
  } catch { console.error(id, 'failed to load'); continue; }
  await page.evaluate(async ({ S, preset }) => {
    const st = document.createElement('style');
    st.textContent = `.sidebar,.handle,.back,.title-chip{display:none!important}.viewer{display:block!important}
      .stage{position:fixed!important;left:0;top:0;width:${S}px!important;height:${S}px!important}`;
    document.head.appendChild(st);
    const vs = window.__viewer;
    const m = await import('/src/scene/environment.ts');
    vs.renderer.setPixelRatio(1);
    vs.setEnvironment(m.LIGHTING_PRESETS.find((p) => p.id === preset).env);
    vs.controls.enableDamping = false;
    await new Promise((r) => setTimeout(r, 300));
  }, { S, preset: opt('preset', 'studio') });
  const used = await page.evaluate((clip) => { const a = window.__viewer.anim; const c = [clip, 'advent_b', 'wait_b'].find((n) => a.has(n)) ?? null; if (c) a.play(c); return c; }, clip);
  for (const f of frames) {
    const url = await page.evaluate(async ({ f, camera }) => {
      const vs = window.__viewer, a = vs.anim;
      a.setPaused(true);
      const fc = Math.max(1, (a.state().frameCount || 1) - 1);
      a.scrubToFrame(Math.round(f * fc));
      vs.setCameraPreset(camera);
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      return vs.pngDataUrl();
    }, { f, camera: opt('camera', 'front') });
    fs.writeFileSync(path.join(out, `${id.replace(/\//g, '_')}_f${f}.png`), Buffer.from(url.split(',')[1], 'base64'));
  }
  console.log(id, used);
}
await browser.close();
