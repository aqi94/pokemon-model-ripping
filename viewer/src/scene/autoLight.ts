/**
 * Auto lighting: fit the Environment so the render matches an aligned reference screenshot.
 *
 * 1. Material-id pass (flat colour per material, cutout alpha kept, held item + blended materials = occluders),
 *    eroded 3 px so silhouette edges and seams between materials are not compared.
 * 2. Per material: the L* distribution as 5 equal-count tone bins (mean Lab each; pose-insensitive) + Lab means on a
 *    6x6 grid over the model's box (which side is lit). CIE76 per term, soft-capped above 20 (occluders, UI, pose).
 * 3. Optional registration: slide the screenshot up to +-16 px (probe scale) to absorb the camera solve's residual.
 * 4. Score every preset, then grid the key direction and pattern-search the identifiable knobs (exposure and tone
 *    mapping stay the preset's: exposure trades 1:1 with key + ambient), with a prior pulling toward the preset.
 *
 * Prototype measurements (synthetic Corbeau): preset 7.5 -> 0.9 loss; real Z-A shots median dE 12-20 -> 8-18.
 */
import * as THREE from 'three';
import type { Environment } from './environment';
import type { LightingPreset } from './environment';

/** The screenshot as it appears over the canvas, drawn at probe size; alpha 0 = no reference there / excluded. */
export type TargetDrawer = (ctx: CanvasRenderingContext2D, W: number, H: number) => void;

export interface ProbeHost {
  probe<T>(W: number, H: number, fn: (render: () => ImageData) => T): T;
  setEnvironment(env: Environment): void;
  gltf?: { scene: THREE.Object3D };
  heldObject(): THREE.Object3D | null;
  canvasRect(): DOMRect;
}

export interface AutoLightOptions {
  presets: LightingPreset[];
  /** where the fit starts when `scanPresets` is false */
  start: Environment;
  scanPresets: boolean;
  register: boolean;
  onProgress?: (msg: string) => void;
  cancelled?: () => boolean;
}

export interface AutoLightResult {
  env: Environment;
  presetId: string | null;
  /** loss of the user's current lighting, of the best preset, and after fitting (lower = closer) */
  before: number;
  presetLoss: number;
  after: number;
  /** median per-material colour difference (CIE76) before / after */
  dEBefore: number;
  dEAfter: number;
  shift: [number, number];
  usablePx: number;
  evals: number;
  ms: number;
}

// ---- colour -----------------------------------------------------------------------------------------------------
const LIN = new Float32Array(256);
for (let i = 0; i < 256; i++) { const c = i / 255; LIN[i] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; }
const fLab = (t: number) => (t > 216 / 24389 ? Math.cbrt(t) : (24389 / 27 * t + 16) / 116);
function lab(r: number, g: number, b: number, out: Float32Array) {
  const R = LIN[r], G = LIN[g], B = LIN[b];
  const X = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047, Y = 0.2126 * R + 0.7152 * G + 0.0722 * B, Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883;
  const fy = fLab(Y);
  out[0] = 116 * fy - 16; out[1] = 500 * (fLab(X) - fy); out[2] = 200 * (fy - fLab(Z));
}

// ---- id pass ----------------------------------------------------------------------------------------------------
const OCCLUDER = 0; // id colour slot for things never compared (held item, blended / additive materials)
const idBytes = (k: number): [number, number, number] => [((k * 53) % 200) + 30, ((k * 97) % 200) + 30, (k % 200) + 20];

interface Ids { W: number; H: number; ids: Int16Array; valid: Uint8Array; names: string[]; bbox: [number, number, number, number] }

function idPass(host: ProbeHost, W: number, H: number, erode = 3): Ids {
  const root = host.gltf?.scene;
  if (!root) throw new Error('No model loaded');
  const held = host.heldObject();
  const names: string[] = [];
  const idOf = new Map<string, number>();
  const saved: [THREE.Mesh, THREE.Material | THREE.Material[]][] = [];
  const made: THREE.Material[] = [];
  const isUnder = (o: THREE.Object3D, anc: THREE.Object3D | null) => { for (let p: THREE.Object3D | null = o; p; p = p.parent) if (p === anc) return true; return false; };
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const occl = !!held && isUnder(mesh, held);
    const mats = ([] as THREE.Material[]).concat(mesh.material);
    const repl = mats.map((m) => {
      const sm = m as THREE.MeshStandardMaterial;
      let k = OCCLUDER;
      if (!occl && !m.transparent && m.blending === THREE.NormalBlending) {
        const key = m.name || m.uuid;
        if (!idOf.has(key)) { idOf.set(key, names.length + 1); names.push(key); }
        k = idOf.get(key)!;
      }
      const [r, g, b] = idBytes(k);
      const c = new THREE.Color().setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
      const bm = new THREE.MeshBasicMaterial({ color: c, side: m.side, toneMapped: false, alphaTest: m.alphaTest, map: m.alphaTest > 0 ? sm.map ?? null : null });
      // cutouts keep the texture's alpha but not its colour (the id colour must stay exact)
      bm.onBeforeCompile = (sh) => { sh.fragmentShader = sh.fragmentShader.replace('#include <map_fragment>', '#ifdef USE_MAP\n diffuseColor.a *= texture2D(map, vMapUv).a;\n#endif'); };
      made.push(bm);
      return bm;
    });
    saved.push([mesh, mesh.material]);
    mesh.material = Array.isArray(mesh.material) ? repl : repl[0];
  });
  let img: ImageData;
  try { img = host.probe(W, H, (render) => render()); } finally {
    for (const [m, mat] of saved) m.material = mat;
    for (const m of made) m.dispose();
  }
  const d = img.data, n = W * H;
  const table = [idBytes(OCCLUDER), ...names.map((_, i) => idBytes(i + 1))];
  const ids = new Int16Array(n).fill(-1);
  for (let p = 0; p < n; p++) {
    if (d[p * 4 + 3] < 250) continue; // background / AA edge
    const R = d[p * 4], G = d[p * 4 + 1], B = d[p * 4 + 2];
    let best = -1, bd = 1e9;
    for (let k = 0; k < table.length; k++) { const t = table[k]; const e = (t[0] - R) ** 2 + (t[1] - G) ** 2 + (t[2] - B) ** 2; if (e < bd) { bd = e; best = k; } }
    if (bd < 144) ids[p] = best - 1; // -1 = occluder
    else ids[p] = -2; // unknown colour: treat like an edge
  }
  const valid = new Uint8Array(n);
  const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]];
  let x0 = W, y0 = H, x1 = 0, y1 = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const p = y * W + x, id = ids[p];
    if (id === -1 && d[p * 4 + 3] < 250) continue;
    if (d[p * 4 + 3] >= 250) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    if (id < 0) continue;
    let ok = true;
    for (const [dx, dy] of dirs) for (let r = 1; r <= erode && ok; r++) {
      const xx = x + dx * r, yy = y + dy * r;
      if (xx < 0 || yy < 0 || xx >= W || yy >= H || ids[yy * W + xx] !== id) ok = false;
    }
    if (ok) valid[p] = 1;
  }
  return { W, H, ids, valid, names, bbox: [x0, y0, x1 + 1, y1 + 1] };
}

// ---- statistics -------------------------------------------------------------------------------------------------
const NB = 5, MINC = 12, MINM = 40, NL = 128, GRID = 6;
interface Stats { mats: ({ n: number; bins: number[][] } | null)[]; cells: Map<number, number[]> }

/** `mask` = pixels to use (model id pass AND reference present). */
function stats(img: ImageData, I: Ids, mask: Uint8Array): Stats {
  const { W, H, ids } = I, d = img.data, nM = I.names.length;
  const hn = new Float32Array(nM * NL), hl = new Float32Array(nM * NL), ha = new Float32Array(nM * NL), hb = new Float32Array(nM * NL);
  const [bx0, by0, bx1, by1] = I.bbox;
  const cw = Math.max(1, bx1 - bx0) / GRID, ch = Math.max(1, by1 - by0) / GRID;
  const cs = new Float32Array(nM * GRID * GRID * 4);
  const t = new Float32Array(3);
  for (let p = 0; p < W * H; p++) {
    if (!mask[p]) continue;
    lab(d[p * 4], d[p * 4 + 1], d[p * 4 + 2], t);
    const m = ids[p];
    const bin = m * NL + Math.min(NL - 1, Math.max(0, (t[0] / 100 * NL) | 0));
    hn[bin]++; hl[bin] += t[0]; ha[bin] += t[1]; hb[bin] += t[2];
    const x = p % W, y = (p / W) | 0;
    const ci = ((m * GRID + Math.min(GRID - 1, Math.max(0, ((x - bx0) / cw) | 0))) * GRID + Math.min(GRID - 1, Math.max(0, ((y - by0) / ch) | 0))) * 4;
    cs[ci] += t[0]; cs[ci + 1] += t[1]; cs[ci + 2] += t[2]; cs[ci + 3]++;
  }
  const mats: Stats['mats'] = [];
  for (let m = 0; m < nM; m++) {
    let n = 0; for (let k = 0; k < NL; k++) n += hn[m * NL + k];
    if (n < MINM) { mats.push(null); continue; }
    const bins = Array.from({ length: NB }, () => [0, 0, 0, 0]);
    let acc = 0;
    for (let k = 0; k < NL; k++) {
      const c = hn[m * NL + k]; if (!c) continue;
      // spread this histogram bin over the equal-count tone bins it straddles
      let left = c, pos = acc;
      while (left > 1e-6) {
        const rb = Math.min(NB - 1, Math.floor(pos / n * NB));
        const take = rb === NB - 1 ? left : Math.min(left, Math.max((rb + 1) * n / NB - pos, 1e-6));
        const fr = take / c, B = bins[rb];
        B[0] += hl[m * NL + k] * fr; B[1] += ha[m * NL + k] * fr; B[2] += hb[m * NL + k] * fr; B[3] += take;
        left -= take; pos += take;
      }
      acc += c;
    }
    mats.push({ n, bins: bins.map((B) => [B[0] / B[3], B[1] / B[3], B[2] / B[3]]) });
  }
  const cells = new Map<number, number[]>();
  for (let i = 0; i < cs.length; i += 4) if (cs[i + 3] >= MINC) cells.set(i, [cs[i] / cs[i + 3], cs[i + 1] / cs[i + 3], cs[i + 2] / cs[i + 3], cs[i + 3]]);
  return { mats, cells };
}

const soft = (e: number, c = 20) => (e < c ? e : c + Math.sqrt(e - c) * 2);
function loss(a: Stats, b: Stats, wCell = 0.5): { loss: number; dE: number } {
  let tot = 0, w = 0, plain = 0, pw = 0;
  for (let m = 0; m < a.mats.length; m++) {
    const A = a.mats[m], B = b.mats[m]; if (!A || !B) continue;
    const wt = Math.sqrt(Math.min(A.n, B.n));
    let e = 0;
    for (let k = 0; k < NB; k++) e += soft(Math.hypot(A.bins[k][0] - B.bins[k][0], A.bins[k][1] - B.bins[k][1], A.bins[k][2] - B.bins[k][2]));
    tot += (1 - wCell) * wt * e / NB; w += (1 - wCell) * wt;
    const mid = NB >> 1;
    plain += wt * Math.hypot(A.bins[mid][0] - B.bins[mid][0], A.bins[mid][1] - B.bins[mid][1], A.bins[mid][2] - B.bins[mid][2]); pw += wt;
  }
  let ct = 0, cwt = 0;
  for (const [k, A] of a.cells) {
    const B = b.cells.get(k); if (!B) continue;
    const wt = Math.sqrt(Math.min(A[3], B[3]));
    ct += wt * soft(Math.hypot(A[0] - B[0], A[1] - B[1], A[2] - B[2])); cwt += wt;
  }
  if (cwt && w) { tot += wCell * (ct / cwt) * w / (1 - wCell); w += wCell * w / (1 - wCell); }
  return { loss: w ? tot / w : 99, dE: pw ? plain / pw : 99 };
}

// ---- knobs ------------------------------------------------------------------------------------------------------
type Knob = 'ambient' | 'key' | 'azimuth' | 'elevation' | 'keyTemp' | 'skyTemp' | 'shadow' | 'saturation' | 'contrast' | 'back' | 'gi' | 'groundR' | 'groundG' | 'groundB';
/** [min, max, initial step]. Exposure and tone mapping are not here on purpose. */
const KNOBS: Record<Knob, [number, number, number]> = {
  ambient: [0, 5, 0.4], key: [0, 5, 0.4], azimuth: [-180, 180, 40], elevation: [0, 85, 20],
  keyTemp: [2000, 12000, 1500], skyTemp: [2000, 12000, 1500],
  shadow: [0, 1, 0.2], saturation: [0.5, 1.6, 0.2], contrast: [0.6, 1.4, 0.15], back: [0, 4, 0.5], gi: [0, 2, 0.4],
  groundR: [0.15, 1, 0.2], groundG: [0.15, 1, 0.2], groundB: [0.15, 1, 0.2],
};
const KNOB_LIST = Object.keys(KNOBS) as Knob[];
/** Prior: loss units for moving one initial step away from the starting preset (keeps a poor match from drifting
 *  into implausible looks, e.g. saturation 0.57 + a pink ground on the Taunie street shot). */
// FITTED: auto-lighting fitting prior on the Taunie street shot (samples/), 2026-10-19 (engine-rules debt)
const PRIOR = 0.12;

type Flat = Environment & { groundR: number; groundG: number; groundB: number };
const hx = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, '0');
function flatten(e: Environment, fitTemps: boolean): Flat {
  const h = (e.ground || '#808080').replace('#', '');
  const f: Flat = { ...e, groundR: parseInt(h.slice(0, 2), 16) / 255, groundG: parseInt(h.slice(2, 4), 16) / 255, groundB: parseInt(h.slice(4, 6), 16) / 255 };
  // explicit colours override the Kelvin knobs: drop them when the temperatures are fitted
  if (fitTemps) { delete f.keyColor; delete f.skyColor; }
  return f;
}
function unflatten(f: Flat): Environment {
  const { groundR, groundG, groundB, ...e } = f;
  return { ...e, ground: '#' + hx(groundR) + hx(groundG) + hx(groundB) };
}

// ---- driver -----------------------------------------------------------------------------------------------------
const tick = () => new Promise<void>((r) => setTimeout(r, 0));

export async function autoLight(host: ProbeHost, drawTarget: TargetDrawer, o: AutoLightOptions): Promise<AutoLightResult> {
  const T0 = performance.now();
  const say = (s: string) => o.onProgress?.(s);
  const stop = () => { if (o.cancelled?.()) throw new Error('cancelled'); };
  // probe size: canvas aspect (the camera projection must not change), longest side 440 px
  const cr = host.canvasRect();
  const k = 440 / Math.max(cr.width, cr.height);
  const W = Math.max(64, Math.round(cr.width * k)), H = Math.max(64, Math.round(cr.height * k));

  say('Material pass…');
  const I = idPass(host, W, H);
  await tick(); stop();

  // reference at probe size; alpha marks where it exists (outside the image / excluded areas = 0)
  const tc = document.createElement('canvas'); tc.width = W; tc.height = H;
  const tctx = tc.getContext('2d', { willReadFrequently: true })!;
  drawTarget(tctx, W, H);
  const raw = tctx.getImageData(0, 0, W, H);

  const shifted = (dx: number, dy: number) => {
    const out = new ImageData(W, H);
    for (let y = 0; y < H; y++) {
      const sy = y + dy;
      for (let x = 0; x < W; x++) {
        const sx = x + dx, i = (y * W + x) * 4;
        if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue; // alpha stays 0
        const j = (sy * W + sx) * 4;
        out.data[i] = raw.data[j]; out.data[i + 1] = raw.data[j + 1]; out.data[i + 2] = raw.data[j + 2]; out.data[i + 3] = raw.data[j + 3];
      }
    }
    return out;
  };
  const maskFor = (img: ImageData) => {
    const m = new Uint8Array(W * H);
    for (let p = 0; p < W * H; p++) if (I.valid[p] && img.data[p * 4 + 3] >= 250) m[p] = 1;
    return m;
  };

  let target = raw, mask = maskFor(raw), T = stats(target, I, mask);
  let usable = mask.reduce((s, v) => s + v, 0);
  if (usable < 200) throw new Error('Too few comparable pixels: is the model aligned over the reference (and not excluded)?');

  let evals = 0;
  const renderEnv = (env: Environment, render: () => ImageData) => { host.setEnvironment(env); evals++; return render(); };
  /** evaluate a batch of envs inside one probe (one canvas resize) */
  const evalMany = (envs: Environment[]) => host.probe(W, H, (render) => envs.map((e) => loss(T, stats(renderEnv(e, render), I, mask))));

  // 1. current lighting + presets
  say('Comparing presets…');
  const [cur] = evalMany([o.start]);
  let startEnv = o.start, presetId: string | null = null, presetLoss = cur.loss;
  if (o.scanPresets) {
    const res: { id: string; l: number }[] = [];
    for (let i = 0; i < o.presets.length; i += 4) {
      const chunk = o.presets.slice(i, i + 4);
      evalMany(chunk.map((p) => p.env)).forEach((r, j) => res.push({ id: chunk[j].id, l: r.loss }));
      say(`Comparing presets… ${Math.min(i + 4, o.presets.length)}/${o.presets.length}`);
      await tick(); stop();
    }
    res.sort((a, b) => a.l - b.l);
    if (res[0] && res[0].l < cur.loss) { presetId = res[0].id; presetLoss = res[0].l; startEnv = o.presets.find((p) => p.id === presetId)!.env; }
  }

  // 2. registration: slide the screenshot (the camera solve leaves a few px). Scored on edges, not colours: the
  //    screenshot's luminance gradient sampled along the id pass's boundaries (silhouette + material borders). Colour
  //    scores are biased while the lighting is still wrong (they moved a pixel-exact reference by 5 px).
  let shift: [number, number] = [0, 0];
  if (o.register) {
    say('Registering…');
    const lum = new Float32Array(W * H);
    for (let p = 0; p < W * H; p++) lum[p] = raw.data[p * 4 + 3] < 250 ? -1 : 0.2126 * raw.data[p * 4] + 0.7152 * raw.data[p * 4 + 1] + 0.0722 * raw.data[p * 4 + 2];
    const grad = new Float32Array(W * H);
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const p = y * W + x;
      if (lum[p] < 0 || lum[p - 1] < 0 || lum[p + 1] < 0 || lum[p - W] < 0 || lum[p + W] < 0) continue;
      grad[p] = Math.hypot(lum[p + 1] - lum[p - 1], lum[p + W] - lum[p - W]);
    }
    const edges: number[] = [];
    for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
      const p = y * W + x, id = I.ids[p];
      if (id !== I.ids[p + 1] || id !== I.ids[p + W]) edges.push(p);
    }
    const score = (dx: number, dy: number) => {
      let sum = 0, n = 0;
      for (const p of edges) {
        const x = (p % W) + dx, y = ((p / W) | 0) + dy;
        if (x < 1 || y < 1 || x >= W - 1 || y >= H - 1) continue;
        sum += grad[y * W + x]; n++;
      }
      return n ? sum / n : 0;
    };
    const s0 = score(0, 0);
    let best = { dx: 0, dy: 0, s: s0 };
    const R = 16;
    for (let dy = -R; dy <= R; dy += 2) for (let dx = -R; dx <= R; dx += 2) { const v = score(dx, dy); if (v > best.s) best = { dx, dy, s: v }; }
    const c = { ...best };
    for (let dy = c.dy - 1; dy <= c.dy + 1; dy++) for (let dx = c.dx - 1; dx <= c.dx + 1; dx++) { const v = score(dx, dy); if (v > best.s) best = { dx, dy, s: v }; }
    // only move when the edges clearly line up better (noise / pose differences otherwise win by a hair)
    if ((best.dx || best.dy) && best.s > s0 * 1.05) {
      shift = [best.dx, best.dy];
      target = shifted(best.dx, best.dy); mask = maskFor(target); T = stats(target, I, mask);
      usable = mask.reduce((s, v) => s + v, 0);
    }
    await tick(); stop();
  }
  const [base] = evalMany([o.start]);
  const [pre] = evalMany([startEnv]);

  // 3. continuous fit from the starting preset
  const fitTemps = true;
  const f0 = flatten(startEnv, fitTemps);
  const env = { ...f0 };
  const prior = (e: Flat) => {
    let s = 0;
    for (const kk of KNOB_LIST) {
      if (kk === 'azimuth' || kk === 'elevation') continue; // direction comes from the data (multi-modal; the grid finds it)
      s += (((e[kk] as number) - (f0[kk] as number)) / KNOBS[kk][2]) ** 2;
    }
    return PRIOR * s;
  };
  const first = evalMany([unflatten(env)])[0];
  let best = { ...first, total: first.loss + prior(env) };
  say('Fitting light direction…');
  const dirs: [number, number][] = [];
  for (const az of [-135, -90, -45, 0, 45, 90, 135, 180]) for (const el of [10, 40, 70]) dirs.push([az, el]);
  for (let i = 0; i < dirs.length; i += 6) {
    host.probe(W, H, (render) => {
      for (const [az, el] of dirs.slice(i, i + 6)) {
        const e = { ...env, azimuth: az, elevation: el };
        const r = loss(T, stats(renderEnv(unflatten(e), render), I, mask));
        const tot = r.loss + prior(e);
        if (tot < best.total) { best = { ...r, total: tot }; env.azimuth = az; env.elevation = el; }
      }
    });
    await tick(); stop();
  }
  const step = Object.fromEntries(KNOB_LIST.map((kk) => [kk, KNOBS[kk][2]])) as Record<Knob, number>;
  const MAX_EVALS = 260;
  const startEvals = evals;
  for (let round = 0; round < 6 && evals - startEvals < MAX_EVALS; round++) {
    say(`Fitting… pass ${round + 1}/6`);
    for (const kk of KNOB_LIST) {
      const [lo, hi] = KNOBS[kk];
      host.probe(W, H, (render) => {
        for (const dir of [1, -1]) {
          let moved = false;
          for (;;) {
            if (evals - startEvals >= MAX_EVALS) break;
            let v = (env[kk] as number) + dir * step[kk];
            v = kk === 'azimuth' ? ((v + 540) % 360) - 180 : Math.min(hi, Math.max(lo, v));
            if (v === env[kk]) break;
            const e = { ...env, [kk]: v };
            const r = loss(T, stats(renderEnv(unflatten(e), render), I, mask));
            const tot = r.loss + prior(e);
            if (tot < best.total - 0.005) { best = { ...r, total: tot }; (env[kk] as number) = v; moved = true; } else break;
          }
          if (moved) break;
        }
      });
      await tick(); stop();
    }
    for (const kk of KNOB_LIST) step[kk] /= 2;
  }
  const out = unflatten(env);
  const [fin] = evalMany([out]);
  host.setEnvironment(o.start); // the caller applies the result (and keeps the undo)
  return {
    env: out, presetId, before: base.loss, presetLoss: pre.loss, after: fin.loss,
    dEBefore: base.dE, dEAfter: fin.dE, shift, usablePx: usable, evals, ms: performance.now() - T0,
  };
}
