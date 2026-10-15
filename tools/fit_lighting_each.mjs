// Fit a lighting environment to EACH reference image separately (the Bulbapedia VS images were shot in different game ambients: night,
// indoors, daylight ...). Sequential, one tab at a time, coordinate descent from the `bdspVs` preset with a small pull toward it.
//
//   node tools/fit_lighting_each.mjs --pairs pairs.json --out lighting.json [--rounds 3] [--only id,id]
// pairs.json: [{ "label": "Cowgirl", "id": "bdsp/tr0018_00", "ref": "<png>" }, ...]
// Output: { "<id without game>": { "label", "base": loss before, "loss": loss after, "env": { only the params that differ from bdspVs } } }
// Camera, pose (advent02_b frame 0) and loss are those of tools/render_bdsp_vs.mjs / fit_lighting_refs.mjs (alpha-box-normalised 64x64 crops).
import { chromium } from 'playwright';
import fs from 'node:fs';

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf('--' + n); return i < 0 ? d : args[i + 1]; };
const pairs = JSON.parse(fs.readFileSync(opt('pairs'), 'utf8'));
const only = opt('only', '') ? opt('only', '').split(',') : null;
const rounds = Number(opt('rounds', '3'));
const S = Number(opt('size', '384'));
const base = opt('url', 'http://localhost:5173');
const out = fs.existsSync(opt('out')) ? JSON.parse(fs.readFileSync(opt('out'), 'utf8')) : {};
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

// [min, max, step]: deliberately narrow; the loss also pulls toward the default (REG)
const SPEC = { exposure: [1.2, 2.8, 0.3], ambient: [0.8, 3.2, 0.5], key: [0.3, 2.4, 0.4], azimuth: [-70, 70, 20], elevation: [0, 75, 18],
  keyTemp: [4500, 8500, 700], skyTemp: [4500, 9000, 800], warmth: [-0.25, 0.25, 0.08], shadow: [0.1, 0.8, 0.15], contrast: [0.8, 1.3, 0.1], saturation: [0.85, 1.25, 0.1] };
const REG = 1.5; // loss units per (normalised deviation)^2 summed over params

const rad = (a) => a;
const list = pairs.filter((p) => !only || only.includes(p.id.split('/')[1]));
for (const p of list) {
  const key = p.id.split('/')[1];
  if (out[key] && !opt('force', '')) continue;
  const page = await ctx.newPage();
  page.on('pageerror', (e) => console.error('page error:', e.message));
  try {
    await page.goto(`${base}/model-viewer?id=${encodeURIComponent(p.id)}`);
    await page.waitForFunction(() => window.__viewer?.anim && window.__viewer.manifest, null, { timeout: 120000 });
    await page.addScriptTag({ content: NORM });
    await page.evaluate(async ({ S }) => {
      const st = document.createElement('style');
      st.textContent = `.sidebar,.handle,.back,.title-chip{display:none!important}.viewer{display:block!important}
        .stage{position:fixed!important;left:0;top:0;width:${S}px!important;height:${S}px!important}`;
      document.head.appendChild(st);
      const vs = window.__viewer;
      vs.renderer.setPixelRatio(1); vs.controls.enableDamping = false;
      const a = vs.anim; const c = ['advent02_b', 'advent_b', 'wait_b'].find((n) => a.has(n)); a.play(c); a.setPaused(true); a.scrubToFrame(0);
      vs.setCameraPreset('front');
      const fi = vs.fitInfo(); const old = vs.camera.fov; const r = Math.PI / 180; const fov = 14, pitch = 8;
      vs.camera.fov = fov;
      const d = fi.dist * Math.tan(old * r / 2) / Math.tan(fov * r / 2);
      vs.camera.position.set(fi.center.x, fi.center.y + Math.sin(pitch * r) * d, fi.center.z + Math.cos(pitch * r) * d);
      vs.controls.maxDistance = fi.radius * 1000; vs.controls.target.copy(fi.center); vs.camera.near = fi.radius / 100; vs.camera.far = fi.radius * 200;
      vs.camera.updateProjectionMatrix(); vs.camera.lookAt(fi.center); vs.controls.update();
      await new Promise((r2) => setTimeout(r2, 300));
    }, { S });
    const refData = await page.evaluate((u) => window.__norm(u).then((a) => Array.from(a)), 'data:image/png;base64,' + fs.readFileSync(p.ref).toString('base64'));
    const start = await page.evaluate(async () => (await import('/src/scene/environment.ts')).LIGHTING_PRESETS.find((q) => q.id === 'bdspVs').env);
    const score = async (e) => {
      const l = await page.evaluate(async ({ e, ref }) => {
        const vs = window.__viewer; vs.setEnvironment(e);
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
        const a = await window.__norm(vs.pngDataUrl());
        let sum = 0, n = 0;
        for (let k = 0; k < a.length; k += 4) if (a[k + 3] > 128 && ref[k + 3] > 128) { sum += Math.abs(a[k] - ref[k]) + Math.abs(a[k + 1] - ref[k + 1]) + Math.abs(a[k + 2] - ref[k + 2]); n++; }
        return n > 40 ? sum / (3 * n) : 80;
      }, { e, ref: refData });
      let reg = 0;
      for (const [k, [lo, hi]] of Object.entries(SPEC)) reg += (((e[k] - start[k]) / (hi - lo)) ** 2);
      return { l, total: l + REG * reg };
    };
    let env = structuredClone(start);
    const b0 = await score(env); let cur = b0.total;
    for (let r = 0; r < rounds; r++) {
      const f = 0.6 ** r;
      for (const [k, [lo, hi, st]] of Object.entries(SPEC)) {
        for (const dir of [1, -1]) {
          const e = { ...env, [k]: Math.min(hi, Math.max(lo, env[k] + dir * st * f)) };
          if (e[k] === env[k]) continue;
          const v = await score(e);
          if (v.total < cur - 1e-3) { env = e; cur = v.total; break; }
        }
      }
    }
    const fin = await score(env);
    const diff = {};
    for (const k of Object.keys(env)) if (env[k] !== start[k]) diff[k] = typeof env[k] === 'number' ? Math.round(env[k] * 1000) / 1000 : env[k];
    out[key] = { label: p.label, base: Math.round(b0.l * 100) / 100, loss: Math.round(fin.l * 100) / 100, env: diff };
    fs.writeFileSync(opt('out'), JSON.stringify(out, null, 1));
    console.log(key.padEnd(10), (p.label ?? '').padEnd(26), 'loss', b0.l.toFixed(1), '->', fin.l.toFixed(1), JSON.stringify(diff));
  } catch (e) { console.error(key, 'failed', String(e).slice(0, 120)); }
  await page.close();
}
await browser.close();
