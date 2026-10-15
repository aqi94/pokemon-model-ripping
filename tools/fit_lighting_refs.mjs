// Fit one viewer lighting environment to many reference images at once (coordinate descent on a colour loss).
//
//   node tools/fit_lighting_refs.mjs --pairs pairs.json --out env.json [--size 384] [--rounds 4] [--start studio]
// pairs.json: [{ "id": "bdsp/tr0039_00", "ref": "<png>", "clip": "advent02_b", "frame": 0, "holdout": false }, ...]
// Every pair gets its own tab (dev server must run). Loss = mean |dRGB| over pixels where both images are opaque, after
// cropping each to its alpha box and fitting it to a 64x64 canvas (so framing/scale differences drop out). Held-out pairs
// are scored but never drive the search.
import { chromium } from 'playwright';
import fs from 'node:fs';

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf('--' + n); return i < 0 ? d : args[i + 1]; };
const pairs = JSON.parse(fs.readFileSync(opt('pairs'), 'utf8'));
const S = Number(opt('size', '384'));
const rounds = Number(opt('rounds', '4'));
const base = opt('url', 'http://localhost:5173');
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader'] });
const ctx = await browser.newContext({ viewport: { width: S + 400, height: S + 40 }, deviceScaleFactor: 1 });

const NORM = `
window.__norm = async (url) => {
  const img = new Image(); img.src = url; await img.decode();
  const c = document.createElement('canvas'); c.width = img.width; c.height = img.height;
  const g = c.getContext('2d', { willReadFrequently: true }); g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, c.width, c.height).data;
  let x0 = 1e9, y0 = 1e9, x1 = -1, y1 = -1;
  for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) if (d[(y * c.width + x) * 4 + 3] > 40) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  const N = 64, w = x1 - x0 + 1, h = y1 - y0 + 1, s = (N - 2) / Math.max(w, h);
  const o = document.createElement('canvas'); o.width = N; o.height = N;
  const og = o.getContext('2d', { willReadFrequently: true });
  og.imageSmoothingQuality = 'high';
  og.drawImage(c, x0, y0, w, h, (N - w * s) / 2, N - 1 - h * s, w * s, h * s);
  return og.getImageData(0, 0, N, N).data;
};`;

const pages = [];
for (const p of pairs) {
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.error('page error:', e.message));
  await page.goto(`${base}/model-viewer?id=${encodeURIComponent(p.id)}`);
  await page.waitForFunction(() => window.__viewer?.anim && window.__viewer.manifest, null, { timeout: 120000 });
  await page.addScriptTag({ content: NORM });
  await page.evaluate(async ({ S, clip, frame }) => {
    const st = document.createElement('style');
    st.textContent = `.sidebar,.handle,.back,.title-chip{display:none!important}.viewer{display:block!important}
      .stage{position:fixed!important;left:0;top:0;width:${S}px!important;height:${S}px!important}`;
    document.head.appendChild(st);
    const vs = window.__viewer;
    vs.renderer.setPixelRatio(1);
    vs.controls.enableDamping = false;
    const a = vs.anim; const c = [clip, 'advent_b', 'wait_b'].find((n) => a.has(n)); a.play(c); a.setPaused(true);
    const fc = Math.max(1, (a.state().frameCount || 1) - 1); a.scrubToFrame(Math.round(frame * fc));
    vs.setCameraPreset('front');
    await new Promise((r) => setTimeout(r, 300));
  }, { S, clip: p.clip ?? 'advent02_b', frame: p.frame ?? 0 });
  const b64 = fs.readFileSync(p.ref).toString('base64');
  p.refData = await page.evaluate((u) => window.__norm(u).then((a) => Array.from(a)), 'data:image/png;base64,' + b64);
  pages.push(page);
  console.log('loaded', p.id);
}

const presets = await pages[0].evaluate(async () => { const m = await import('/src/scene/environment.ts'); return m.LIGHTING_PRESETS.map((p) => ({ id: p.id, env: p.env })); });
let env = structuredClone(presets.find((p) => p.id === opt('start', 'studio')).env);

async function score(e, held) {
  const out = [];
  for (let i = 0; i < pages.length; i++) {
    if (!!pairs[i].holdout !== held) continue;
    const l = await pages[i].evaluate(async ({ e, ref }) => {
      const vs = window.__viewer; vs.setEnvironment(e);
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      const a = await window.__norm(vs.pngDataUrl());
      let sum = 0, n = 0;
      for (let k = 0; k < a.length; k += 4) if (a[k + 3] > 128 && ref[k + 3] > 128) { sum += Math.abs(a[k] - ref[k]) + Math.abs(a[k + 1] - ref[k + 1]) + Math.abs(a[k + 2] - ref[k + 2]); n++; }
      return n > 40 ? sum / (3 * n) : 80;
    }, { e, ref: pairs[i].refData });
    out.push(l);
  }
  return out.reduce((s, v) => s + v, 0) / Math.max(1, out.length);
}

const SPEC = { ambient: [0.2, 7, 0.5], key: [0, 4, 0.4], azimuth: [-90, 90, 15], elevation: [0, 80, 12], keyTemp: [3500, 9500, 600],
  skyTemp: [3500, 12000, 800], warmth: [-0.6, 0.6, 0.1], shadow: [0, 1, 0.15], rim: [0, 2, 0.3], detail: [0, 1, 0.2], exposure: [0.5, 3.5, 0.25],
  contrast: [0.7, 1.7, 0.1], saturation: [0.6, 1.8, 0.1], gi: [0, 2, 0.3], back: [0, 3, 0.4] };
console.log('start train', (await score(env, false)).toFixed(2), 'held-out', (await score(env, true)).toFixed(2));
let best = { v: 1e9 };
for (const tm of ['linear', 'aces', 'agx', 'neutral']) {
  for (const ex of [0.8, 1.2, 1.8]) {
    const e = { ...env, toneMapping: tm, exposure: ex };
    const v = await score(e, false);
    if (v < best.v) best = { v, e };
  }
}
env = best.e; let cur = best.v;
console.log('tone map', env.toneMapping, 'exposure', env.exposure, cur.toFixed(2));
for (let r = 0; r < rounds; r++) {
  const f = 0.5 ** r;
  for (const [k, [lo, hi, st]] of Object.entries(SPEC)) {
    for (const dir of [1, -1]) {
      const e = { ...env, [k]: Math.min(hi, Math.max(lo, env[k] + dir * st * f)) };
      if (e[k] === env[k]) continue;
      const v = await score(e, false);
      if (v < cur - 1e-3) { env = e; cur = v; break; }
    }
  }
  console.log('round', r, 'train', cur.toFixed(2), 'held-out', (await score(env, true)).toFixed(2));
}
fs.writeFileSync(opt('out'), JSON.stringify({ env, train: cur, heldOut: await score(env, true) }, null, 1));
console.log(JSON.stringify(env));
await browser.close();
