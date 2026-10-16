// Fit one viewer lighting environment to several in-game screenshots of a scene (dev server running).
//
//   node tools/fit_scene.mjs (--scene refs/za/<scene> | a_regions.json [b_regions.json ...])
//        [--preset id | --env f.json | --seeds studio,night,indoorWarm,f.json] [--profile outdoor|indoor|night]
//        [--lock rim=0,gi=1] [--fit] [--passes 3] [--restarts 1] [--skip slug[,slug]] [--only slug[,slug]]
//        [--write refs/za/<scene>/fit.json] [--debug-out dir] [--probe] [--align] [--auto-regions] [--url http://localhost:5173]
//
// --align: moves the camera (distance + pan, keeping its direction) until the material-id pass's modelBox matches
// "subjectBox" ([x0,y0,x1,y1] on the screenshot) and saves the new camera into the regions file.
// --auto-regions: replaces the regions with pure single-material windows found by the probe (box = renderBox; any
// region marked "manual": true is kept). Together with --align this turns a frame + a subject box + a clip into a
// fittable regions file; the debug image then shows whether the screenshot side of each box lands on the same part.
//
// Each regions JSON (schema: tools/lightrefs/README.md) names a model, clip, frame, camera pose and boxes on flat
// areas of the screenshot (+ optional renderBox on the render). Score = weighted mean CIE76 dE over all regions
// of all shots. --probe renders a material-id pass per shot (unique flat colour per material) and writes
// <slug>_probe.png/.json (per-material bbox/centroid + the model's box) to help place renderBoxes and the camera.
// --debug-out writes <slug>_debug.png: screenshot with boxes | render with renderBoxes | per-region numbers.
// --fit runs greedy coordinate descent from every seed (plus a light-direction perturbation restart), keeps the
// best, and --write stores fit.json + <slug>_render.png (transparent, like every PNG export).
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf('--' + n); return i < 0 ? d : args[i + 1]; };
const flag = (n) => args.includes('--' + n);
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '..');
const PROFILES = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools/lightrefs/fit_profiles.json'), 'utf8'));
const SEED_ENVS = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools/lightrefs/seed_envs.json'), 'utf8'));

// ---- shots ---------------------------------------------------------------------------------------------------
let files = [];
const sceneDir = opt('scene');
if (sceneDir) files = fs.readdirSync(sceneDir).filter((f) => f.endsWith('_regions.json')).map((f) => path.join(sceneDir, f));
else files = args.filter((a, i) => a.endsWith('_regions.json') || (a.endsWith('.json') && !(i > 0 && args[i - 1].startsWith('--')) && !a.includes('seed')));
const skip = new Set((opt('skip', '') || '').split(',').filter(Boolean));
const only = new Set((opt('only', '') || '').split(',').filter(Boolean));
const shots = [];
for (const f of files) {
  const spec = JSON.parse(fs.readFileSync(f, 'utf8'));
  spec.slug ??= path.basename(f).replace(/_regions\.json$/, '');
  if (spec.skip || skip.has(spec.slug) || (only.size && !only.has(spec.slug))) continue;
  spec._file = f;
  spec._shotPath = path.resolve(path.dirname(f), spec.screenshot);
  if (!fs.existsSync(spec._shotPath)) { console.error(`missing screenshot ${spec._shotPath} (run fetch_frames regrab)`); process.exit(1); }
  shots.push(spec);
}
if (!shots.length) { console.error('no shots (give --scene <dir> or *_regions.json files)'); process.exit(1); }
console.log(`${shots.length} shot(s): ${shots.map((s) => `${s.slug} [${s.model} ${s.clip}@${s.frame}]`).join(', ')}`);

// ---- browser: one page per model -----------------------------------------------------------------------------
const base = opt('url', 'http://localhost:5173');
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader'] });
const pages = new Map();
const maxW = Math.max(...shots.map((s) => s.width)), maxH = Math.max(...shots.map((s) => s.height));
for (const s of shots) {
  if (pages.has(s.model)) continue;
  const page = await browser.newPage({ viewport: { width: maxW + 420, height: maxH + 60 }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('page error:', e.message));
  await page.goto(`${base}/model-viewer?id=${encodeURIComponent(s.model)}`);
  await page.waitForFunction(() => window.__viewer?.anim && window.__viewer.manifest, null, { timeout: 120000 });
  await page.evaluate(() => {
    const st = document.createElement('style');
    st.id = 'fit-style';
    st.textContent = `.sidebar,.handle,.back,.title-chip,.leftrail,.toolbox,.optionsbar,.timeline{display:none!important}.viewer{display:block!important}
      .stage{position:fixed!important;left:0;top:0;width:800px!important;height:600px!important}`;
    document.head.appendChild(st);
    const vs = window.__viewer;
    vs.renderer.setPixelRatio(1);
    vs.controls.enableDamping = false;
    vs.setBackground('#ff00ff');
    window.__shots = {};
  });
  pages.set(s.model, page);
}
const envMod = await [...pages.values()][0].evaluate(async () => { const m = await import('/src/scene/environment.ts'); return { presets: m.LIGHTING_PRESETS, def: m.DEFAULT_ENV }; });
// screenshots into the pages
for (const s of shots) {
  const url = 'data:image/png;base64,' + fs.readFileSync(s._shotPath).toString('base64');
  await pages.get(s.model).evaluate(async ({ slug, url }) => {
    const img = new Image(); img.src = url; await img.decode();
    const cv = document.createElement('canvas'); cv.width = img.width; cv.height = img.height;
    const ctx = cv.getContext('2d'); ctx.drawImage(img, 0, 0);
    window.__shots[slug] = { data: ctx.getImageData(0, 0, img.width, img.height), canvas: cv };
  }, { slug: s.slug, url });
}

// ---- env helpers ----------------------------------------------------------------------------------------------
const hex2 = (v) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, '0');
const withGround = (e) => (e.groundR === undefined ? e : { ...e, ground: '#' + hex2(e.groundR) + hex2(e.groundG) + hex2(e.groundB) });
const splitGround = (e) => { const h = e.ground.replace('#', ''); return { ...e, groundR: parseInt(h.slice(0, 2), 16) / 255, groundG: parseInt(h.slice(2, 4), 16) / 255, groundB: parseInt(h.slice(4, 6), 16) / 255 }; };
const cleanEnv = (e0) => {
  const e = withGround(e0);
  for (const k of ['groundR', 'groundG', 'groundB']) delete e[k];
  // explicit colours would override the Kelvin knobs the fitter moves
  for (const k of ['keyColor', 'skyColor', 'backColor']) delete e[k];
  for (const k of Object.keys(e)) if (typeof e[k] === 'number') e[k] = +e[k].toFixed(k.endsWith('Temp') ? 0 : 3);
  return e;
};
function seedEnv(name) {
  if (name.endsWith('.json')) return { ...envMod.def, ...JSON.parse(fs.readFileSync(name, 'utf8')) };
  const pr = envMod.presets.find((p) => p.id === name);
  if (pr) return { ...pr.env };
  if (SEED_ENVS[name]) return { ...envMod.def, ...SEED_ENVS[name] };
  throw new Error(`unknown seed ${name}`);
}
let seeds = [];
if (opt('seeds')) seeds = opt('seeds').split(',').map((n) => [n, seedEnv(n.trim())]);
else if (opt('preset')) seeds = [[opt('preset'), seedEnv(opt('preset'))]];
else if (opt('env')) seeds = [[opt('env'), seedEnv(opt('env'))]];
else seeds = [['default', { ...envMod.def }]];
const profile = PROFILES[opt('profile', 'outdoor')] ?? PROFILES.outdoor;
const locks = { ...(profile.lock ?? {}) };
for (const kv of (opt('lock', '') || '').split(',').filter(Boolean)) { const [k, v] = kv.split('='); locks[k] = Number(v); }
const applyLocks = (e) => { for (const [k, v] of Object.entries(locks)) e[k] = v; return e; };

// ---- measurement ----------------------------------------------------------------------------------------------
/** The dev server hot-reloads the page when viewer sources change; re-create our page state when that happened. */
async function ensurePage(page, model) {
  const ok = await page.evaluate(() => !!(window.__shots && window.__viewer?.anim && window.__viewer.manifest && document.getElementById('fit-style')));
  if (ok) return;
  console.log(`  (page for ${model} was reloaded; re-initialising)`);
  await page.waitForFunction(() => window.__viewer?.anim && window.__viewer.manifest, null, { timeout: 120000 });
  await page.evaluate(() => {
    let st = document.getElementById('fit-style');
    if (!st) { st = document.createElement('style'); st.id = 'fit-style'; document.head.appendChild(st); }
    st.textContent = `.sidebar,.handle,.back,.title-chip,.leftrail,.toolbox,.optionsbar,.timeline{display:none!important}.viewer{display:block!important}
      .stage{position:fixed!important;left:0;top:0;width:800px!important;height:600px!important}`;
    const vs = window.__viewer;
    vs.renderer.setPixelRatio(1);
    vs.controls.enableDamping = false;
    vs.setBackground('#ff00ff');
    window.__shots = {};
  });
  for (const s of shots) {
    if (s.model !== model) continue;
    const url = 'data:image/png;base64,' + fs.readFileSync(s._shotPath).toString('base64');
    await page.evaluate(async ({ slug, url }) => {
      const img = new Image(); img.src = url; await img.decode();
      const cv = document.createElement('canvas'); cv.width = img.width; cv.height = img.height;
      const ctx = cv.getContext('2d'); ctx.drawImage(img, 0, 0);
      window.__shots[slug] = { data: ctx.getImageData(0, 0, img.width, img.height), canvas: cv };
    }, { slug: s.slug, url });
  }
}

async function setupShot(page, spec) {
  await ensurePage(page, spec.model);
  await page.evaluate(async ({ spec }) => {
    const st = document.getElementById('fit-style');
    st.textContent = st.textContent.replace(/width:\d+px!important;height:\d+px!important/, `width:${spec.width}px!important;height:${spec.height}px!important`);
    const vs = window.__viewer;
    if (vs.anim.current !== spec.clip) vs.anim.play(spec.clip);
    vs.anim.setPaused(true);
    vs.anim.scrubToFrame(spec.frame);
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    const c = spec.camera;
    vs.setCameraPose(c.fov, c.position, c.target);
  }, { spec: { width: spec.width, height: spec.height, clip: spec.clip, frame: spec.frame, camera: spec.camera } });
}

async function measureShot(page, spec, env) {
  await setupShot(page, spec);
  return page.evaluate(async ({ e, spec }) => {
    const vs = window.__viewer;
    vs.setEnvironment(e);
    vs.anim.scrubToFrame(spec.frame);
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    vs.renderer.render(vs.scene, vs.camera);
    const src = vs.renderer.domElement;
    const cv = document.createElement('canvas'); cv.width = src.width; cv.height = src.height;
    const ctx = cv.getContext('2d'); ctx.drawImage(src, 0, 0);
    const ren = ctx.getImageData(0, 0, cv.width, cv.height);
    window.__lastRender = cv;
    const mean = (img, [x0, y0, x1, y1], skipKey) => {
      let r = 0, g = 0, b = 0, n = 0;
      for (let y = Math.max(0, y0 | 0); y < Math.min(img.height, y1 | 0); y++) for (let x = Math.max(0, x0 | 0); x < Math.min(img.width, x1 | 0); x++) {
        const i = (y * img.width + x) * 4;
        const R = img.data[i], G = img.data[i + 1], B = img.data[i + 2];
        if (skipKey && R > 200 && B > 200 && G < 80) continue;
        r += R; g += G; b += B; n++;
      }
      return n ? [r / n, g / n, b / n, n] : null;
    };
    const shot = window.__shots[spec.slug].data;
    return spec.regions.map((rg) => ({ name: rg.name, weight: rg.weight ?? 1, shot: mean(shot, rg.box, false), render: mean(ren, rg.renderBox ?? rg.box, true) }));
  }, { e: withGround(env), spec: { slug: spec.slug, frame: spec.frame, regions: spec.regions } });
}

const lab = ([r, g, b]) => {
  const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const [R, G, B] = [lin(r), lin(g), lin(b)];
  const X = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047, Y = 0.2126 * R + 0.7152 * G + 0.0722 * B, Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883;
  const f = (t) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
  return [116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z))];
};
const dE = (a, b) => { const A = lab(a), B = lab(b); return Math.hypot(A[0] - B[0], A[1] - B[1], A[2] - B[2]); };

async function measure(env) {
  const out = [];
  for (const s of shots) out.push({ spec: s, regions: await measureShot(pages.get(s.model), s, env) });
  return out;
}
function shotScore(res) {
  let tot = 0, w = 0;
  for (const r of res.regions) {
    const wt = r.weight;
    if (!r.shot || !r.render || r.render[3] < 20) { tot += 50 * wt; w += wt; continue; }
    tot += wt * dE(r.shot, r.render); w += wt;
  }
  return w ? tot / w : 50;
}
function score(all) {
  let tot = 0, w = 0;
  for (const r of all) { const wt = r.spec.weight ?? 1; tot += wt * shotScore(r); w += wt; }
  return tot / w;
}
function report(all) {
  for (const r of all) {
    console.log(`  ${r.spec.slug}: mean dE ${shotScore(r).toFixed(2)}`);
    for (const x of r.regions) {
      const f = (c) => c ? c.slice(0, 3).map((v) => Math.round(v).toString().padStart(3)).join(',') : '   -   ';
      console.log(`    ${x.name.padEnd(16)} shot ${f(x.shot)}  render ${f(x.render)}  dE ${x.shot && x.render ? dE(x.shot, x.render).toFixed(1) : '-'}${x.render && x.render[3] < 20 ? '  (render box empty!)' : ''}`);
    }
  }
}

// ---- probe: material-id pass ----------------------------------------------------------------------------------
async function probeShot(page, spec, outDir) {
  await setupShot(page, spec);
  const res = await page.evaluate(async ({ frame }) => {
    const vs = window.__viewer;
    const THREE = vs.constructor.THREE; // the viewer's three instance (a second copy would not render)
    const saved = [];
    const colors = new Map();
    let n = 0;
    const idColor = (name) => {
      if (!colors.has(name)) { n++; const h = (n * 0.61803398875) % 1; const c = new THREE.Color().setHSL(h, 0.9, 0.5); colors.set(name, c); }
      return colors.get(name);
    };
    vs.root.traverse((o) => {
      if (!o.isMesh || !o.visible) return;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      const repl = mats.map((m) => new THREE.MeshBasicMaterial({ color: idColor(m.name || 'mat'), skinning: true, morphTargets: !!o.morphTargetInfluences, side: m.side, toneMapped: false }));
      saved.push([o, o.material]);
      o.material = Array.isArray(o.material) ? repl : repl[0];
    });
    const tm = vs.renderer.toneMapping; vs.renderer.toneMapping = THREE.NoToneMapping;
    vs.anim.scrubToFrame(frame);
    vs.renderer.render(vs.scene, vs.camera);
    const src = vs.renderer.domElement;
    const cv = document.createElement('canvas'); cv.width = src.width; cv.height = src.height;
    const ctx = cv.getContext('2d'); ctx.drawImage(src, 0, 0);
    const img = ctx.getImageData(0, 0, cv.width, cv.height);
    for (const [o, m] of saved) o.material = m;
    vs.renderer.toneMapping = tm;
    // classify pixels by nearest id colour (sRGB bytes)
    const table = [...colors.entries()].map(([name, c]) => ({ name, rgb: [c.r, c.g, c.b].map((v) => Math.round((v <= 0.0031308 ? v * 12.92 : 1.055 * v ** (1 / 2.4) - 0.055) * 255)), n: 0, sx: 0, sy: 0, x0: 1e9, y0: 1e9, x1: -1, y1: -1 }));
    const model = { n: 0, x0: 1e9, y0: 1e9, x1: -1, y1: -1 };
    for (let y = 0; y < img.height; y++) for (let x = 0; x < img.width; x++) {
      const i = (y * img.width + x) * 4; const R = img.data[i], G = img.data[i + 1], B = img.data[i + 2];
      if (R > 200 && B > 200 && G < 80) continue;
      let best = null, bd = 1e9;
      for (const t of table) { const d = (t.rgb[0] - R) ** 2 + (t.rgb[1] - G) ** 2 + (t.rgb[2] - B) ** 2; if (d < bd) { bd = d; best = t; } }
      if (!best || bd > 60 * 60) continue;
      best.n++; best.sx += x; best.sy += y; best.x0 = Math.min(best.x0, x); best.y0 = Math.min(best.y0, y); best.x1 = Math.max(best.x1, x); best.y1 = Math.max(best.y1, y);
      model.n++; model.x0 = Math.min(model.x0, x); model.y0 = Math.min(model.y0, y); model.x1 = Math.max(model.x1, x); model.y1 = Math.max(model.y1, y);
    }
    const materials = {};
    for (const t of table) if (t.n) materials[t.name] = { color: '#' + t.rgb.map((v) => v.toString(16).padStart(2, '0')).join(''), pixels: t.n, bbox: [t.x0, t.y0, t.x1 + 1, t.y1 + 1], centroid: [Math.round(t.sx / t.n), Math.round(t.sy / t.n)] };
    // per-pixel material index map, kept in the page for the automatic region search (autoRegions)
    const idx = new Int16Array(img.width * img.height).fill(-1);
    if (model.n) for (let y = 0; y < img.height; y++) for (let x = 0; x < img.width; x++) {
      const i = (y * img.width + x) * 4; const R = img.data[i], G = img.data[i + 1], B = img.data[i + 2];
      if (R > 200 && B > 200 && G < 80) continue;
      let best = -1, bd = 1e9;
      for (let k = 0; k < table.length; k++) { const t = table[k]; const d = (t.rgb[0] - R) ** 2 + (t.rgb[1] - G) ** 2 + (t.rgb[2] - B) ** 2; if (d < bd) { bd = d; best = k; } }
      if (best >= 0 && bd <= 60 * 60) idx[y * img.width + x] = best;
    }
    // percentile-trimmed boxes (2nd..98th) per material: nearest-colour classification leaves stray pixels on
    // anti-aliased edges, which made an "eye" box span half the body
    const pb = (k) => {
      const xs = new Uint32Array(img.width), ys = new Uint32Array(img.height); let n = 0;
      for (let y = 0; y < img.height; y++) for (let x = 0; x < img.width; x++) if (idx[y * img.width + x] === k) { xs[x]++; ys[y]++; n++; }
      if (!n) return null;
      const cut = (h, len) => { let lo = 0, hi = len - 1, acc = 0; const t = n * 0.02; while (lo < len && acc + h[lo] < t) acc += h[lo++]; acc = 0; while (hi >= 0 && acc + h[hi] < t) acc += h[hi--]; return [lo, hi + 1]; };
      const [x0, x1] = cut(xs, img.width), [y0, y1] = cut(ys, img.height);
      return [x0, y0, x1, y1];
    };
    for (let k = 0; k < table.length; k++) { const t = table[k]; if (t.n && materials[t.name]) materials[t.name].pbbox = pb(k); }
    window.__probe = { idx, names: table.map((t) => t.name), bboxes: table.map((t) => [t.x0, t.y0, t.x1 + 1, t.y1 + 1]), width: img.width, height: img.height, modelH: model.n ? model.y1 - model.y0 : 0 };
    return { png: cv.toDataURL('image/png'), json: { modelBox: model.n ? [model.x0, model.y0, model.x1 + 1, model.y1 + 1] : null, pixels: model.n, materials, width: img.width, height: img.height } };
  }, { frame: spec.frame });
  fs.writeFileSync(path.join(outDir, `${spec.slug}_probe.png`), Buffer.from(res.png.split(',')[1], 'base64'));
  fs.writeFileSync(path.join(outDir, `${spec.slug}_probe.json`), JSON.stringify(res.json, null, 1));
  const mb = res.json.modelBox;
  console.log(`  ${spec.slug}: probe modelBox ${mb ? mb.join(',') : 'none'} (${res.json.pixels} px); materials: ${Object.entries(res.json.materials).map(([k, v]) => `${k}@${v.centroid.join(',')}`).join(' ')}`);
  return res.json;
}

// ---- camera alignment + automatic regions -----------------------------------------------------------------------
const saveSpec = (spec) => {
  const out = {};
  for (const [k, v] of Object.entries(spec)) if (!k.startsWith('_')) out[k] = v;
  fs.writeFileSync(spec._file, JSON.stringify(out, null, 1));
};

/** Move the camera (distance + pan, same direction) until the probe's modelBox matches spec.subjectBox on the screenshot. */
async function alignShot(page, spec, outDir) {
  if (!spec.subjectBox && !spec.faceBox) { console.log(`  ${spec.slug}: no subjectBox / faceBox, cannot align`); return null; }
  if (spec.yaw !== undefined) {
    // "yaw" = camera azimuth around the target in degrees: 0 = in front of the model looking at its face;
    // positive = the camera orbits to the model's left, so the model looks toward screen-LEFT and we see its left
    // cheek; negative = the model looks toward screen-right. Keeps the camera's height and distance.
    await setupShot(page, spec);
    spec.camera = await page.evaluate(({ yaw }) => {
      const vs = window.__viewer; const cam = vs.camera, tgt = vs.controls.target.clone();
      const d = cam.position.clone().sub(tgt); const horiz = Math.hypot(d.x, d.z); const a = yaw * Math.PI / 180;
      const np = [tgt.x + Math.sin(a) * horiz, cam.position.y, tgt.z + Math.cos(a) * horiz];
      vs.setCameraPose(cam.fov, np, tgt.toArray());
      return { fov: cam.fov, position: np.map((v) => +v.toFixed(4)), target: tgt.toArray().map((v) => +v.toFixed(4)) };
    }, { yaw: spec.yaw });
  }
  let probe = null;
  for (let it = 0; it < 8; it++) {
    probe = await probeShot(page, spec, outDir);
    const mb = probe.modelBox;
    if (!mb) { console.log(`  ${spec.slug}: model not visible; move the camera back (position.z) or check the clip`); return probe; }
    let r, dcx, dcy, what;
    if (spec.faceBox) {
      // Face cue (close-ups, dialogue boxes over the body): the render's face = union of the face-skin / eye / brow
      // material boxes (hair is excluded on purpose: long hair reaches the legs and would swallow the box).
      const hb = spec.faceBox;
      // the face-skin material alone (trimmed box); eyes/brows only as a fallback for models without one
      let rh = null;
      const face = Object.entries(probe.materials).filter(([name]) => /face/i.test(name) && /skin/i.test(name)).sort((a, b) => b[1].pixels - a[1].pixels)[0]
        ?? Object.entries(probe.materials).filter(([name]) => /face/i.test(name) && !/hair/i.test(name)).sort((a, b) => b[1].pixels - a[1].pixels)[0];
      if (face) rh = face[1].pbbox ?? face[1].bbox;
      if (!rh) { console.log(`  ${spec.slug}: no face material in the render, cannot align on the face`); return probe; }
      const headCutTop = hb[1] <= 6;
      r = headCutTop ? (hb[2] - hb[0]) / (rh[2] - rh[0]) : (hb[3] - hb[1]) / (rh[3] - rh[1]);
      dcx = (hb[0] + hb[2]) / 2 - (rh[0] + rh[2]) / 2;
      dcy = headCutTop ? hb[3] - rh[3] : (hb[1] + hb[3]) / 2 - (rh[1] + rh[3]) / 2;
      what = `face ${headCutTop ? 'w ' + (rh[2] - rh[0]) + ' vs ' + (hb[2] - hb[0]) + ' (cut at the top)' : 'h ' + (rh[3] - rh[1]) + ' vs ' + (hb[3] - hb[1])}`;
    } else {
      const sb = spec.subjectBox;
      // A subject cut by the frame edge (waist-up shots) has no full height: scale on width and align the visible edge.
      const cutBottom = spec.subjectCut === 'bottom' || sb[3] >= spec.height - 6, cutTop = spec.subjectCut === 'top' || sb[1] <= 6;
      const mh = mb[3] - mb[1], sh = sb[3] - sb[1], mw = mb[2] - mb[0], sw = sb[2] - sb[0];
      r = (cutBottom || cutTop) ? sw / mw : sh / mh;
      dcx = (sb[0] + sb[2]) / 2 - (mb[0] + mb[2]) / 2;
      // bottom cut (dialogue box, frame edge): align the heads; only the top cut: align the feet; otherwise centres
      dcy = cutBottom ? sb[1] - mb[1] : cutTop ? sb[3] - mb[3] : (sb[1] + sb[3]) / 2 - (mb[1] + mb[3]) / 2;
      what = `modelBox ${cutBottom || cutTop ? 'w ' + mw + ' vs subject ' + sw + ' (cut)' : 'h ' + mh + ' vs subject ' + sh}`;
    }
    r = Math.min(2.5, Math.max(0.4, r)); // never jump more than 2.5x per step: a clipped box must not run away
    console.log(`  align ${it + 1}: ${what} (x${r.toFixed(3)}), offset ${dcx.toFixed(0)},${dcy.toFixed(0)} px`);
    if (Math.abs(r - 1) < 0.02 && Math.abs(dcx) < spec.height * 0.004 && Math.abs(dcy) < spec.height * 0.004) break;
    const cam = await page.evaluate(({ r, dcx, dcy, H }) => {
      const vs = window.__viewer; const THREE = vs.constructor.THREE;
      const cam = vs.camera, tgt = vs.controls.target.clone();
      const dir = cam.position.clone().sub(tgt); const d = dir.length(); dir.normalize();
      const newD = d / r;
      const worldPerPx = 2 * newD * Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2) / H;
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(cam.quaternion), up = new THREE.Vector3(0, 1, 0).applyQuaternion(cam.quaternion);
      // the model must move by (+dcx, +dcy) on screen: move the camera the opposite way (screen y points down)
      const shift = right.clone().multiplyScalar(-dcx * worldPerPx).add(up.clone().multiplyScalar(dcy * worldPerPx));
      const nt = tgt.add(shift), np = nt.clone().add(dir.multiplyScalar(newD));
      vs.setCameraPose(cam.fov, np.toArray(), nt.toArray());
      return { fov: cam.fov, position: np.toArray().map((v) => +v.toFixed(4)), target: nt.toArray().map((v) => +v.toFixed(4)) };
    }, { r, dcx, dcy, H: spec.height });
    spec.camera = cam;
  }
  saveSpec(spec);
  console.log(`  ${spec.slug}: camera -> fov ${spec.camera.fov} position ${spec.camera.position.join(',')} target ${spec.camera.target.join(',')} (saved)`);
  return probe;
}

/** Replace the regions with windows that are (a) one material in the render (material-id pass), (b) flat in colour
 *  on both the render (seed env) and the screenshot, and (c) not a wildly different colour (a tree behind the arm,
 *  a sign, UI). box = renderBox because the camera is aligned. Regions marked "manual": true are kept. */
async function autoRegions(page, spec, probe, env) {
  if (!probe?.modelBox) { console.log(`  ${spec.slug}: no model in the render, no auto regions`); return; }
  await measureShot(page, spec, env); // renders with the seed env, fills window.__lastRender
  const LIM = Number(opt('flat', 18)), SANE = Number(opt('sane', 35)), PER = Number(opt('per-material', 2));
  const excl = spec.excludeBoxes ?? [];
  const found = await page.evaluate(({ slug, LIM, PER, excl }) => {
    const P = window.__probe;
    const excluded = (x, y, w) => excl.some((b) => x < b[2] && x + w > b[0] && y < b[3] && y + w > b[1]);
    const shot = window.__shots[slug].data, ren = window.__lastRender.getContext('2d').getImageData(0, 0, P.width, P.height);
    const ws = Math.max(12, Math.min(40, Math.round(P.modelH / 24))), stride = Math.max(3, ws >> 2);
    const stat = (img, x0, y0) => {
      let n = 0, s = [0, 0, 0], ss = [0, 0, 0];
      for (let y = y0; y < y0 + ws; y++) for (let x = x0; x < x0 + ws; x++) {
        const i = (y * img.width + x) * 4;
        for (let c = 0; c < 3; c++) { const v = img.data[i + c]; s[c] += v; ss[c] += v * v; }
        n++;
      }
      let sd = 0; const m = [0, 0, 0];
      for (let c = 0; c < 3; c++) { m[c] = s[c] / n; sd += Math.sqrt(Math.max(0, ss[c] / n - m[c] * m[c])); }
      return { std: sd / 3, mean: m };
    };
    const out = {};
    for (let k = 0; k < P.names.length; k++) {
      if (/eye|lash|brow|teeth|tongue|mouth|lens|glass/i.test(P.names[k])) continue;
      const [bx0, by0, bx1, by1] = P.bboxes[k];
      if (bx1 - bx0 < ws || by1 - by0 < ws) continue;
      const wins = [];
      for (let y = by0; y + ws <= by1; y += stride) for (let x = bx0; x + ws <= bx1; x += stride) {
        let n = 0;
        for (let yy = y; yy < y + ws; yy++) for (let xx = x; xx < x + ws; xx++) if (P.idx[yy * P.width + xx] === k) n++;
        if (n / (ws * ws) < 0.95) continue;
        if (excluded(x, y, ws)) continue; // dialogue box, HUD, another character on the screenshot side
        const r = stat(ren, x, y);
        if (r.std > LIM) continue;
        const sh = stat(shot, x, y);
        if (sh.std > LIM) continue;
        wins.push({ box: [x, y, x + ws, y + ws], shot: sh.std, render: r.std, shotMean: sh.mean, renderMean: r.mean });
      }
      wins.sort((a, b) => a.shot + a.render - b.shot - b.render);
      out[P.names[k]] = wins;
    }
    return { out, ws };
  }, { slug: spec.slug, LIM, PER, excl });
  const manual = (spec.regions ?? []).filter((r) => r.manual);
  const regions = [];
  let dropped = 0;
  for (const [mat, wins] of Object.entries(found.out)) {
    const pick = [];
    for (const w of wins) {
      const d0 = dE(w.shotMean, w.renderMean);
      if (flag('verbose')) console.log(`    cand ${mat} ${w.box.join(',')} std shot ${w.shot.toFixed(1)} render ${w.render.toFixed(1)} dE0 ${d0.toFixed(1)}`);
      if (d0 > SANE) { dropped++; continue; }
      if (pick.every((q) => Math.hypot(w.box[0] - q.box[0], w.box[1] - q.box[1]) > found.ws * 1.5)) pick.push(w);
      if (pick.length >= PER) break;
    }
    pick.forEach((w, i) => regions.push({ name: `${mat}_${i + 1}`, box: w.box, renderBox: w.box, weight: /skin/i.test(mat) ? 1.5 : 1, auto: true, flat: [+w.shot.toFixed(1), +w.render.toFixed(1)], dE0: +dE(w.shotMean, w.renderMean).toFixed(1) }));
  }
  spec.regions = [...manual, ...regions];
  saveSpec(spec);
  console.log(`  ${spec.slug}: ${regions.length} auto regions (${[...new Set(regions.map((r) => r.name.replace(/_\d+$/, '')))].join(', ') || 'none'}; window ${found.ws}px, flat <= ${LIM}${dropped ? `, ${dropped} rejected for colour` : ''}) + ${manual.length} manual (saved)`);
}

// ---- debug image ----------------------------------------------------------------------------------------------
async function debugShot(page, spec, res, outDir) {
  const rows = res.regions.map((x) => `${x.name}  shot ${x.shot ? x.shot.slice(0, 3).map(Math.round).join(',') : '-'}  render ${x.render ? x.render.slice(0, 3).map(Math.round).join(',') : '-'}  dE ${x.shot && x.render ? dE(x.shot, x.render).toFixed(1) : '-'}`);
  const url = await page.evaluate(({ slug, regions, rows, title }) => {
    const shot = window.__shots[slug].canvas, ren = window.__lastRender;
    const W = shot.width, H = shot.height, foot = 18 * (rows.length + 1) + 8;
    const cv = document.createElement('canvas'); cv.width = W * 2 + 10; cv.height = H + foot;
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#404040'; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.drawImage(shot, 0, 0);
    // render: replace the magenta key with grey
    const tmp = document.createElement('canvas'); tmp.width = ren.width; tmp.height = ren.height;
    const tctx = tmp.getContext('2d'); tctx.drawImage(ren, 0, 0);
    const img = tctx.getImageData(0, 0, tmp.width, tmp.height);
    for (let i = 0; i < img.data.length; i += 4) if (img.data[i] > 200 && img.data[i + 2] > 200 && img.data[i + 1] < 80) { img.data[i] = img.data[i + 1] = img.data[i + 2] = 64; }
    tctx.putImageData(img, 0, 0);
    ctx.drawImage(tmp, W + 10, 0);
    ctx.lineWidth = 2; ctx.font = '13px sans-serif';
    for (const rg of regions) {
      const b = rg.box, rb = rg.renderBox ?? rg.box;
      ctx.strokeStyle = '#00ff00'; ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      ctx.fillStyle = '#00ff00'; ctx.fillText(rg.name, b[0] + 2, b[1] - 3);
      ctx.strokeStyle = '#00ffff'; ctx.strokeRect(W + 10 + rb[0], rb[1], rb[2] - rb[0], rb[3] - rb[1]);
      ctx.fillStyle = '#00ffff'; ctx.fillText(rg.name, W + 10 + rb[0] + 2, rb[1] - 3);
    }
    ctx.fillStyle = 'white'; ctx.font = '14px monospace';
    ctx.fillText(title, 4, H + 16);
    rows.forEach((r, i) => ctx.fillText(r, 4, H + 16 + 18 * (i + 1)));
    return cv.toDataURL('image/png');
  }, { slug: spec.slug, regions: spec.regions, rows, title: `${spec.slug}  ${spec.model} ${spec.clip}@${spec.frame}  mean dE ${shotScore(res).toFixed(2)}` });
  fs.writeFileSync(path.join(outDir, `${spec.slug}_debug.png`), Buffer.from(url.split(',')[1], 'base64'));
}

// ---- fit --------------------------------------------------------------------------------------------------------
async function descend(env0, label) {
  let env = applyLocks(splitGround({ ...env0 }));
  // seeds may sit outside the profile's physical bounds; start inside them
  for (const [k, lo, hi] of profile.knobs) if (!(k in locks) && typeof env[k] === 'number') env[k] = Math.min(hi, Math.max(lo, env[k]));
  let best = score(await measure(env));
  const start = best;
  const knobs = profile.knobs.filter(([k]) => !(k in locks));
  const tms = ['linear', 'aces', 'agx', 'neutral'];
  const passes = Number(opt('passes', 3));
  for (let pass = 0; pass < passes; pass++) {
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
    console.log(`  [${label}] pass ${pass + 1}: mean dE ${best.toFixed(2)}`);
  }
  return { env, best, start };
}

const outDir = opt('debug-out');
if (outDir) fs.mkdirSync(outDir, { recursive: true });

if (flag('align') || flag('auto-regions')) {
  for (const s of shots) {
    const dir = outDir ?? path.dirname(s._file);
    const probe = flag('align') ? await alignShot(pages.get(s.model), s, dir) : await probeShot(pages.get(s.model), s, dir);
    if (flag('auto-regions')) await autoRegions(pages.get(s.model), s, probe, applyLocks({ ...seeds[0][1] }));
  }
} else if (flag('probe')) {
  console.log('probe (material-id pass):');
  for (const s of shots) await probeShot(pages.get(s.model), s, outDir ?? path.dirname(s._file));
}

let env = applyLocks({ ...seeds[0][1] });
let res = await measure(env);
let best = score(res);
console.log(`start [${seeds[0][0]}]: mean dE ${best.toFixed(2)}`);
report(res);
let bestSeed = seeds[0][0];
const seedLog = [];

if (flag('fit')) {
  let winner = null;
  for (const [name, e] of seeds) {
    const r = await descend(e, name);
    seedLog.push({ seed: name, start: +r.start.toFixed(2), end: +r.best.toFixed(2) });
    if (!winner || r.best < winner.best) { winner = r; bestSeed = name; }
  }
  for (let i = 0; i < Number(opt('restarts', 1)); i++) {
    const pert = { ...withGround(winner.env) };
    for (const k of ['azimuth', 'backAzimuth']) if (!(k in locks)) pert[k] = ((pert[k] + 40 * (i % 2 ? -1 : 1) + 540) % 360) - 180;
    if (!('elevation' in locks)) pert.elevation = Math.min(90, Math.max(0, pert.elevation + (i % 2 ? -25 : 25)));
    if (!('exposure' in locks)) pert.exposure = Math.min(2.5, Math.max(0.5, pert.exposure + (i % 2 ? -0.3 : 0.3)));
    const r = await descend(pert, `restart ${i + 1}`);
    seedLog.push({ seed: `restart ${i + 1}`, start: +r.start.toFixed(2), end: +r.best.toFixed(2) });
    if (r.best < winner.best - 1e-3) { winner = r; bestSeed = `restart ${i + 1}`; }
  }
  env = winner.env; best = winner.best;
  console.table(seedLog);
  res = await measure(env);
  report(res);
  console.log(JSON.stringify(cleanEnv(env)));
}

if (outDir) for (const r of res) await debugShot(pages.get(r.spec.model), r.spec, r, outDir);

if (opt('write')) {
  const out = {
    schema: 1, scene: shots[0].scene ?? path.basename(path.dirname(shots[0]._file)), profile: opt('profile', 'outdoor'), locks,
    seeds: seeds.map(([n]) => n), bestSeed, passes: Number(opt('passes', 3)), seedLog, env: cleanEnv(env),
    score: { mean: +best.toFixed(2), perShot: Object.fromEntries(res.map((r) => [r.spec.slug, { dE: +shotScore(r).toFixed(2), weight: r.spec.weight ?? 1, regions: Object.fromEntries(r.regions.map((x) => [x.name, x.shot && x.render ? +dE(x.shot, x.render).toFixed(1) : null])) }])) },
    shots: res.map((r) => r.spec.slug), fittedAt: new Date().toISOString(), tool: 'fit_scene.mjs',
  };
  fs.mkdirSync(path.dirname(opt('write')), { recursive: true });
  fs.writeFileSync(opt('write'), JSON.stringify(out, null, 1));
  console.log('wrote', opt('write'));
  for (const r of res) {
    const page = pages.get(r.spec.model);
    await measureShot(page, r.spec, env);
    const url = await page.evaluate(() => window.__viewer.pngDataUrl());
    fs.writeFileSync(path.join(path.dirname(opt('write')), `${r.spec.slug}_render.png`), Buffer.from(url.split(',')[1], 'base64'));
  }
}
await browser.close();
