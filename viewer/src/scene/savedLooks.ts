/**
 * Saved looks: named per-model configurations (appearance, held item, animation, camera; optional lighting) that can be
 * re-applied in one click, one of them the default that auto-loads when the model opens (`?look=<name>` picks one).
 *
 * Storage: local deployment = `out/<game>/<id>/configs.json` through the dev/preview server's `/__configs/<game>/<id>`
 * endpoint (vite.config.ts); a static build (no endpoint) = localStorage `looks:<game>/<id>`.
 */
import { useSyncExternalStore } from 'react';
import type { Environment } from './environment';
import type { HologramState } from './hologram';
import type { RefTransform } from './refImage';

type Vec3 = [number, number, number];

export interface Look {
  name: string;
  /** auto-applied when the model opens (at most one per model) */
  default?: boolean;
  savedAt?: string;
  appearance?: {
    /** variant group name -> variant index (skin tone, shiny...) */
    variantSel?: Record<string, number>;
    /** part group name -> option id (modular outfits) */
    groupSel?: Record<string, string>;
    /** first material of a palette entry -> its materials + 1-based zone per colour layer */
    palettes?: Record<string, { materials: string[]; indices: number[] }>;
    /** explicit part checkbox overrides (node -> visible), applied after the group options */
    partsVis?: Record<string, boolean>;
    hologram?: HologramState;
    /** global viewer option: applied too, so it carries over to the next model */
    albedo?: 'baked' | 'raw' | 'off';
  };
  held?: { item: string; bone: string; only: boolean };
  anim?: { clip: string | null; frame: number; paused: boolean; timeScale: number };
  camera?: {
    /** effective vertical FOV (deg) */
    fov: number; position: Vec3; target: Vec3;
    /** camera roll (deg), Camera panel; absent = 0 */
    roll?: number;
    /** solved lens lock; `image` locks are only re-attached with the same reference sample, else they fall back to `canvas` */
    lens?: { kind: 'image' | 'canvas'; fov: number; roll: number; samplePath: string | null; transform?: RefTransform };
  };
  /** optional, mostly written by tools; applied as the current (Custom) environment */
  lighting?: Environment;
}

export interface LooksFile { version: 1; looks: Look[] }
export type LooksBackend = 'server' | 'local';
export interface LooksEntry { looks: Look[]; backend: LooksBackend; loaded: boolean }

const BASE = import.meta.env.BASE_URL;
const EMPTY: LooksEntry = { looks: [], backend: 'local', loaded: false };
const store = new Map<string, LooksEntry>();
const listeners = new Set<() => void>();
const keyOf = (game: string, id: string) => `${game}/${id}`;
const lsKey = (k: string) => `looks:${k}`;
const endpoint = (game: string, id: string) => `${BASE}__configs/${encodeURIComponent(game)}/${encodeURIComponent(id)}`;

/** Drops malformed entries; keeps unknown fields (tools may add some). */
export function sanitizeLooks(raw: unknown): Look[] {
  const arr = Array.isArray(raw) ? raw : (raw && typeof raw === 'object' && Array.isArray((raw as LooksFile).looks)) ? (raw as LooksFile).looks : [];
  const seen = new Set<string>();
  const out: Look[] = [];
  for (const l of arr) {
    if (!l || typeof l !== 'object' || typeof l.name !== 'string' || !l.name.trim()) continue;
    const k = l.name.trim().toLowerCase();
    if (seen.has(k)) continue;
    seen.add(k); out.push({ ...l, name: l.name.trim() });
  }
  let def = false; // at most one default
  for (const l of out) { if (l.default && !def) def = true; else if (l.default) delete l.default; }
  return out;
}

function readLocal(k: string): Look[] {
  try { return sanitizeLooks(JSON.parse(localStorage.getItem(lsKey(k)) ?? '[]')); } catch { return []; }
}
function writeLocal(k: string, looks: Look[]) {
  try {
    if (looks.length) localStorage.setItem(lsKey(k), JSON.stringify({ version: 1, looks }));
    else localStorage.removeItem(lsKey(k));
  } catch { /* storage full / blocked */ }
}

function set(k: string, e: LooksEntry) { store.set(k, e); listeners.forEach((l) => l()); }

/** Loads the looks of one model (server endpoint first, localStorage when there is none). */
export async function loadLooks(game: string, id: string): Promise<LooksEntry> {
  const k = keyOf(game, id);
  let entry: LooksEntry;
  try {
    const r = await fetch(endpoint(game, id), { cache: 'no-store' });
    if (!r.ok || !(r.headers.get('content-type') ?? '').includes('json')) throw new Error('no endpoint');
    entry = { looks: sanitizeLooks(await r.json()), backend: 'server', loaded: true };
  } catch {
    entry = { looks: readLocal(k), backend: 'local', loaded: true };
  }
  set(k, entry);
  return entry;
}

async function persist(game: string, id: string, looks: Look[]) {
  const k = keyOf(game, id);
  const cur = store.get(k) ?? EMPTY;
  set(k, { ...cur, looks, loaded: true });
  if (cur.backend === 'server') {
    try {
      const r = await fetch(endpoint(game, id), { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ version: 1, looks }) });
      if (r.ok) return;
    } catch { /* fall through */ }
    set(k, { looks, backend: 'local', loaded: true }); // endpoint went away: keep the change in this browser
  }
  writeLocal(k, looks);
}

export const getLooks = (game: string, id: string): LooksEntry => store.get(keyOf(game, id)) ?? EMPTY;

export function useLooks(game: string, id: string): LooksEntry {
  return useSyncExternalStore((cb) => { listeners.add(cb); return () => { listeners.delete(cb); }; }, () => getLooks(game, id));
}

/** Saves `look`; an existing look with the same name (case-insensitive) is overwritten in place. */
export async function saveLook(game: string, id: string, look: Look) {
  const looks = getLooks(game, id).looks;
  const name = look.name.trim();
  const entry: Look = { ...structuredClone(look), name, savedAt: new Date().toISOString() };
  const i = looks.findIndex((l) => l.name.toLowerCase() === name.toLowerCase());
  let next = i >= 0 ? looks.map((l, j) => (j === i ? entry : l)) : [...looks, entry];
  if (entry.default) next = next.map((l) => (l === entry ? l : (({ default: _d, ...rest }) => rest)(l)));
  await persist(game, id, next);
}

export async function deleteLook(game: string, id: string, name: string) {
  await persist(game, id, getLooks(game, id).looks.filter((l) => l.name !== name));
}

/** Makes `name` the default look (null = none). */
export async function setDefaultLook(game: string, id: string, name: string | null) {
  await persist(game, id, getLooks(game, id).looks.map((l) => {
    const { default: _d, ...rest } = l;
    return l.name === name ? { ...rest, default: true } : rest;
  }));
}

/** The look to auto-apply on open: `?look=<name>` (case-insensitive) when it exists, else the default. */
export function startupLook(looks: Look[], urlName: string | null): Look | null {
  if (urlName) {
    const l = looks.find((x) => x.name.toLowerCase() === urlName.trim().toLowerCase());
    if (l) return l;
  }
  return looks.find((l) => l.default) ?? null;
}
