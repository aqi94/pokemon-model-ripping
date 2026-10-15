// Compare (and optionally fit) a viewer lighting environment against an in-game screenshot.
//
//   node tools/match_lighting.mjs samples/detective_handclap01_regions.json [--preset indoorWindow] [--env env.json]
//        [--out render.png] [--fit] [--url http://localhost:5173]
//
// Needs the viewer dev server running (cd viewer && npm run dev). The regions file names the model, clip, frame,
// camera and the screenshot, plus boxes [x0,y0,x1,y1] in screenshot pixels; the viewer canvas is sized to the
// screenshot and the camera placed so the same boxes cover the same body parts in both images (optional
// "renderBox" per region when they don't). Background pixels (magenta) are ignored in the render.
// Prints mean colours and CIE76 dE per region; --fit runs coordinate descent and prints the best env JSON.
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf('--' + n); return i < 0 ? d : args[i + 1]; };
const flag = (n) => args.includes('--' + n);
const regionsPath = args.find((a) => a.endsWith('.json') && args[args.indexOf(a) - 1] !== '--env');
if (!regionsPath) { console.error('usage: node tools/match_lighting.mjs <regions.json> [--preset id] [--env env.json] [--out png] [--fit]'); process.exit(1); }
const spec = JSON.parse(fs.readFileSync(regionsPath, 'utf8'));
if (opt('frame')) spec.frame = Number(opt('frame'));
const base = opt('url', 'http://localhost:5173');
const shotPath = path.resolve(path.dirname(regionsPath), spec.screenshot);
const shotUrl = 'data:image/png;base64,' + fs.readFileSync(shotPath).toString('base64');

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: spec.width + 400, height: spec.height + 40 }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.error('page error:', e.message));
await page.goto(`${base}/model-viewer?id=${encodeURIComponent(spec.model)}`);
await page.waitForFunction(() => window.__viewer?.anim && window.__viewer.manifest, null, { timeout: 120000 });
const envMod = await page.evaluate(async () => {
  const m = await import('/src/scene/environment.ts');
  return { presets: m.LIGHTING_PRESETS, def: m.DEFAULT_ENV };
});

let env = { ...envMod.def };
if (opt('preset')) env = { ...envMod.presets.find((p) => p.id === opt('preset')).env };
if (opt('env')) env = { ...env, ...JSON.parse(fs.readFileSync(opt('env'), 'utf8')) };

// Fixed-size canvas, posed and framed like the screenshot.
await page.evaluate(async ({ spec, shotUrl }) => {
  const st = document.createElement('style');
  st.textContent = `.sidebar,.handle,.back,.title-chip{display:none!important}.viewer{display:block!important}
    .stage{position:fixed!important;left:0;top:0;width:${spec.width}px!important;height:${spec.height}px!important}`;
  document.head.appendChild(st);
  const vs = window.__viewer;
  vs.renderer.setPixelRatio(1);
  vs.setBackground('#ff00ff');
  vs.anim.play(spec.clip);
  vs.anim.setPaused(true);
  vs.anim.scrubToFrame(spec.frame);
  await new Promise((r) => setTimeout(r, 300));
  const c = spec.camera;
  vs.controls.enableDamping = false;
  vs.camera.fov = c.fov;
  vs.camera.position.set(...c.position);
  vs.controls.target.set(...c.target);
  vs.camera.updateProjectionMatrix();
  vs.controls.update();
  const img = new Image();
  img.src = shotUrl;
  await img.decode();
  const cv = document.createElement('canvas');
  cv.width = img.width; cv.height = img.height;
  const ctx = cv.getContext('2d');
  ctx.drawImage(img, 0, 0);
  window.__shot = ctx.getImageData(0, 0, img.width, img.height);
}, { spec, shotUrl });
await page.waitForTimeout(500);

// The ground colour is fitted as three channels (groundR/G/B, sRGB 0..1) and folded back into the hex.
const hex2 = (v) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, '0');
const withGround = (e) => (e.groundR === undefined ? e : { ...e, ground: '#' + hex2(e.groundR) + hex2(e.groundG) + hex2(e.groundB) });
const splitGround = (e) => { const h = e.ground.replace('#', ''); return { ...e, groundR: parseInt(h.slice(0, 2), 16) / 255, groundG: parseInt(h.slice(2, 4), 16) / 255, groundB: parseInt(h.slice(4, 6), 16) / 255 }; };

/** Render with env, return per-region mean sRGB of render and screenshot. */
async function measure(e0) {
  const e = withGround(e0);
  return page.evaluate(async ({ e, spec }) => {
    const vs = window.__viewer;
    vs.setEnvironment(e);
    vs.anim.scrubToFrame(spec.frame);
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    vs.renderer.render(vs.scene, vs.camera);
    const src = vs.renderer.domElement;
    const cv = document.createElement('canvas');
    cv.width = src.width; cv.height = src.height;
    const ctx = cv.getContext('2d');
    ctx.drawImage(src, 0, 0);
    const ren = ctx.getImageData(0, 0, cv.width, cv.height);
    window.__lastRender = cv;
    const mean = (img, [x0, y0, x1, y1], skipKey) => {
      let r = 0, g = 0, b = 0, n = 0;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
        const i = (y * img.width + x) * 4;
        const R = img.data[i], G = img.data[i + 1], B = img.data[i + 2];
        if (skipKey && R > 200 && B > 200 && G < 80) continue;
        r += R; g += G; b += B; n++;
      }
      return n ? [r / n, g / n, b / n, n] : null;
    };
    return spec.regions.map((rg) => ({ name: rg.name, shot: mean(window.__shot, rg.box, false), render: mean(ren, rg.renderBox ?? rg.box, true) }));
  }, { e, spec });
}

const lab = ([r, g, b]) => {
  const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const [R, G, B] = [lin(r), lin(g), lin(b)];
  const X = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047, Y = 0.2126 * R + 0.7152 * G + 0.0722 * B, Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883;
  const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
  return [116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z))];
};
const dE = (a, b) => { const A = lab(a), B = lab(b); return Math.hypot(A[0] - B[0], A[1] - B[1], A[2] - B[2]); };
function score(res) {
  let tot = 0, w = 0;
  for (const r of res) {
    if (!r.shot || !r.render || r.render[3] < 20) { tot += 50; w++; continue; }
    const wt = spec.regions.find((x) => x.name === r.name).weight ?? 1;
    tot += wt * dE(r.shot, r.render); w += wt;
  }
  return tot / w;
}
const report = (res) => {
  for (const r of res) {
    const f = (c) => c ? c.slice(0, 3).map((v) => Math.round(v).toString().padStart(3)).join(',') : '   -   ';
    console.log(`  ${r.name.padEnd(14)} shot ${f(r.shot)}  render ${f(r.render)}  dE ${r.shot && r.render ? dE(r.shot, r.render).toFixed(1) : '-'}`);
  }
};

let res = await measure(env);
let best = score(res);
console.log(`start: mean dE ${best.toFixed(2)}`);
report(res);

if (flag('fit')) {
  env = splitGround(env);
  const knobs = (spec.fit ?? [
    ['ambient', 0, 4, 0.2], ['key', 0, 4, 0.2], ['keyTemp', 1500, 12000, 500], ['azimuth', -180, 180, 20], ['elevation', 0, 90, 10],
    ['back', 0, 4, 0.2], ['backAzimuth', -180, 180, 20], ['backElevation', 0, 90, 10], ['backTemp', 1500, 12000, 500],
    ['skyTemp', 1500, 12000, 500], ['exposure', 0.5, 2, 0.1], ['contrast', 0.5, 1.5, 0.05], ['saturation', 0, 2, 0.1],
    ['shadow', 0, 1, 0.1], ['gi', 0, 2, 0.2], ['groundR', 0, 1, 0.1], ['groundG', 0, 1, 0.1], ['groundB', 0, 1, 0.1],
  ]);
  const tms = ['linear', 'aces', 'agx', 'neutral'];
  for (let pass = 0; pass < Number(opt('passes', 4)); pass++) {
    for (const tm of tms) {
      if (tm === env.toneMapping) continue;
      const s = score(await measure({ ...env, toneMapping: tm }));
      if (s < best - 1e-3) { best = s; env.toneMapping = tm; }
    }
    for (const [k, lo, hi, st0] of knobs) {
      const st = st0 / (pass + 1);
      for (const dir of [1, -1]) {
        for (;;) {
          const v = Math.min(hi, Math.max(lo, +(env[k] + dir * st).toFixed(4)));
          if (v === env[k]) break;
          const s = score(await measure({ ...env, [k]: v }));
          if (s < best - 1e-3) { best = s; env[k] = v; } else break;
        }
      }
    }
    console.log(`pass ${pass + 1}: mean dE ${best.toFixed(2)}`);
  }
  res = await measure(env);
  report(res);
  env = withGround(env);
  for (const k of ['groundR', 'groundG', 'groundB']) delete env[k];
  for (const k of Object.keys(env)) if (typeof env[k] === 'number') env[k] = +env[k].toFixed(k.endsWith('Temp') ? 0 : 2);
  console.log(JSON.stringify(env));
}

if (opt('out')) {
  await measure(env);
  // Saved like every PNG export: transparent background (the magenta key is only for sampling).
  const url = await page.evaluate(() => window.__viewer.pngDataUrl());
  fs.writeFileSync(opt('out'), Buffer.from(url.split(',')[1], 'base64'));
  console.log('wrote', opt('out'));
}
await browser.close();
