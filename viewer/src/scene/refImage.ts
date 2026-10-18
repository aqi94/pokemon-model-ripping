/** Free-form reference image pinned to the viewport (independent of the 3D camera). */

const BASE = import.meta.env.BASE_URL;

export interface RefImage {
  src: string;
  name: string;
  /** path under samples/ when it came from the local folder; null for uploads / drops (object URLs, session only) */
  samplePath: string | null;
  w: number;
  h: number;
}

/** x/y = image centre as a fraction of the stage; scale multiplies the contain-fit size; rotation in degrees. */
export interface RefTransform { x: number; y: number; scale: number; rotation: number; flipX: boolean }
export type RefLayer = 'background' | 'overlay';
export interface RefSettings { visible: boolean; layer: RefLayer; opacity: number; transform: RefTransform }

export interface SampleFile { path: string; size: number; mtime: number }

export const DEFAULT_TRANSFORM: RefTransform = { x: 0.5, y: 0.5, scale: 1, rotation: 0, flipX: false };
export const DEFAULT_SETTINGS: RefSettings = { visible: true, layer: 'overlay', opacity: 0.5, transform: DEFAULT_TRANSFORM };

export const sampleUrl = (p: string) => `${BASE}samples/${p.split('/').map(encodeURIComponent).join('/')}`;

/** null = no local samples folder (static web build): only upload / drag-drop are offered. */
export async function listSamples(): Promise<SampleFile[] | null> {
  try {
    const r = await fetch(`${BASE}__samples.json`, { cache: 'no-store' });
    if (!r.ok || !(r.headers.get('content-type') ?? '').includes('json')) return null;
    return await r.json();
  } catch { return null; }
}

/** Size of the image at scale 1: contain-fit inside the stage. */
export function fitSize(img: { w: number; h: number }, stageW: number, stageH: number) {
  const k = Math.min(stageW / img.w, stageH / img.h);
  return { w: img.w * k, h: img.h * k };
}

export function loadImageSize(src: string): Promise<{ w: number; h: number }> {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res({ w: im.naturalWidth, h: im.naturalHeight });
    im.onerror = () => rej(new Error('Could not load image'));
    im.src = src;
  });
}

// --- persistence (local samples only: uploads are object URLs that die with the page) ---
const LAST_KEY = 'refImage.last';
const TRANSFORMS_KEY = 'refImage.transforms';

export function storeLast(samplePath: string | null, s: RefSettings) {
  try {
    if (samplePath) localStorage.setItem(LAST_KEY, JSON.stringify({ samplePath, settings: s }));
    else localStorage.removeItem(LAST_KEY);
    if (samplePath) {
      const all = JSON.parse(localStorage.getItem(TRANSFORMS_KEY) ?? '{}');
      all[samplePath] = s.transform;
      localStorage.setItem(TRANSFORMS_KEY, JSON.stringify(all));
    }
  } catch { /* storage unavailable */ }
}

export function loadLast(): { samplePath: string; settings: RefSettings } | null {
  try {
    const v = JSON.parse(localStorage.getItem(LAST_KEY) ?? 'null');
    if (!v?.samplePath) return null;
    return { samplePath: v.samplePath, settings: { ...DEFAULT_SETTINGS, ...v.settings, transform: { ...DEFAULT_TRANSFORM, ...v.settings?.transform } } };
  } catch { return null; }
}

export function transformFor(samplePath: string): RefTransform | null {
  try {
    const t = JSON.parse(localStorage.getItem(TRANSFORMS_KEY) ?? '{}')[samplePath];
    return t ? { ...DEFAULT_TRANSFORM, ...t } : null;
  } catch { return null; }
}

// --- alignment pins: image point <-> model surface point, solved into a camera (camSolve.ts) ---

/** A point on the model surface: mesh (traversal index + name check), triangle vertex indices, barycentric weights. */
export interface SurfaceAnchor { mesh: number; name: string; tri: [number, number, number]; bary: [number, number, number] }
/** u,v = position on the reference image (0..1, top-left origin, in the image's own pixels: survives move/scale/flip). */
export interface RefPin { id: number; u: number; v: number; anchor: SurfaceAnchor | null }

/** Image-relative point -> stage px (same maths as the CSS in RefImageLayer: translate(-50%,-50%) rotate scaleX). */
export function imageToStage(u: number, v: number, img: { w: number; h: number }, t: RefTransform, stageW: number, stageH: number) {
  const f = fitSize(img, stageW, stageH);
  const bw = f.w * t.scale, bh = f.h * t.scale;
  let lx = (u - 0.5) * bw;
  const ly = (v - 0.5) * bh;
  if (t.flipX) lx = -lx;
  const a = t.rotation * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  return { x: t.x * stageW + lx * c - ly * s, y: t.y * stageH + lx * s + ly * c };
}

export function stageToImage(x: number, y: number, img: { w: number; h: number }, t: RefTransform, stageW: number, stageH: number) {
  const f = fitSize(img, stageW, stageH);
  const bw = f.w * t.scale, bh = f.h * t.scale;
  const dx = x - t.x * stageW, dy = y - t.y * stageH;
  const a = -t.rotation * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  let lx = dx * c - dy * s;
  const ly = dx * s + dy * c;
  if (t.flipX) lx = -lx;
  return { u: lx / bw + 0.5, v: ly / bh + 0.5 };
}

const PINS_KEY = 'refImage.pins';
/** Pins are stored per sample image AND model (anchors only make sense on the model they were picked on). */
export function storePins(samplePath: string, model: string, pins: RefPin[]) {
  try {
    const all = JSON.parse(localStorage.getItem(PINS_KEY) ?? '{}');
    const k = `${model}|${samplePath}`;
    if (pins.length) all[k] = pins; else delete all[k];
    localStorage.setItem(PINS_KEY, JSON.stringify(all));
  } catch { /* storage unavailable */ }
}

export function loadPins(samplePath: string, model: string): RefPin[] {
  try {
    const v = JSON.parse(localStorage.getItem(PINS_KEY) ?? '{}')[`${model}|${samplePath}`];
    return Array.isArray(v) ? v : [];
  } catch { return []; }
}

/** Area of the reference to ignore when fitting the lighting (dialogue box, other characters...), image-relative. */
export interface RefBox { u0: number; v0: number; u1: number; v1: number }
const EXCL_KEY = 'refImage.excludes';
export function storeExcludes(samplePath: string, boxes: RefBox[]) {
  try {
    const all = JSON.parse(localStorage.getItem(EXCL_KEY) ?? '{}');
    if (boxes.length) all[samplePath] = boxes; else delete all[samplePath];
    localStorage.setItem(EXCL_KEY, JSON.stringify(all));
  } catch { /* storage unavailable */ }
}
export function loadExcludes(samplePath: string): RefBox[] {
  try { const v = JSON.parse(localStorage.getItem(EXCL_KEY) ?? '{}')[samplePath]; return Array.isArray(v) ? v : []; } catch { return []; }
}
