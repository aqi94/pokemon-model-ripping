import type { GameIndex, GameSummary, RefIndex } from '../types';

// Only the game (domain) being viewed is ever loaded; switching domains drops the previous cache entry.
const cache = new Map<string, Promise<GameIndex>>();
const BASE = import.meta.env.BASE_URL;
const GLB_BASE = (import.meta.env.VITE_GLB_BASE as string | undefined)?.replace(/\/$/, '');

/** URL of a file under out/ (`za/<id>/manifest.json`). GLBs come from VITE_GLB_BASE (GitHub release assets) when set. */
export function modelAsset(rel: string): string {
  if (GLB_BASE && rel.endsWith('.glb')) return `${GLB_BASE}/${rel.split('/').join('__')}`;
  return `${BASE}models/${rel}`;
}

let gamesP: Promise<Record<string, GameSummary>> | null = null;

export function loadGames(): Promise<Record<string, GameSummary>> {
  return (gamesP ??= fetch(`${BASE}models/games.json`).then((r) => { if (!r.ok) throw new Error('games.json missing: run `python -m pipeline index`'); return r.json(); }));
}

export function loadGameIndex(game: string): Promise<GameIndex> {
  for (const k of [...cache.keys()]) if (k !== game) cache.delete(k);
  let p = cache.get(game);
  if (!p) {
    p = fetch(`${BASE}models/${game}/index.json`).then((r) => { if (!r.ok) throw new Error(`No models for "${game}"`); return r.json(); });
    p.catch(() => cache.delete(game));
    cache.set(game, p);
  }
  return p;
}

/** Calibrated screenshot index for the Reference panel; null when the game has none (404). */
export function loadRefIndex(game: string): Promise<RefIndex | null> {
  return fetch(`${BASE}refs/${game}/index.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null);
}
