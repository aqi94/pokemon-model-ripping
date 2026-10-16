import { useSyncExternalStore } from 'react';
import { DEFAULT_ENV, LIGHTING_PRESETS, type Environment, type LightingPreset } from './environment';

const PRESETS_KEY = 'userLightPresets';
const LAST_ENV_KEY = 'lastLightEnv';

/** Merged over DEFAULT_ENV so saves made before an Environment field existed still load. */
const revive = (e: Partial<Environment>): Environment => ({ ...DEFAULT_ENV, ...e });

function read(): LightingPreset[] {
  try {
    const raw = JSON.parse(localStorage.getItem(PRESETS_KEY) ?? '[]');
    if (!Array.isArray(raw)) return [];
    return raw.filter((r) => r && typeof r.id === 'string' && typeof r.label === 'string' && r.env).map((r) => ({ id: r.id, label: r.label, env: revive(r.env) }));
  } catch { return []; }
}

let presets = read();
const listeners = new Set<() => void>();
const commit = (next: LightingPreset[]) => {
  presets = next;
  try { localStorage.setItem(PRESETS_KEY, JSON.stringify(next)); } catch { /* ignore */ }
  listeners.forEach((l) => l());
};

export const USER_PREFIX = 'user:';

export function useUserPresets(): LightingPreset[] {
  return useSyncExternalStore((cb) => { listeners.add(cb); return () => { listeners.delete(cb); }; }, () => presets);
}

/** Saves under `label`; an existing preset with the same name (case-insensitive) is overwritten. Returns its id. */
export function saveUserPreset(label: string, env: Environment): string {
  const name = label.trim();
  const existing = presets.find((p) => p.label.toLowerCase() === name.toLowerCase());
  const id = existing?.id ?? USER_PREFIX + Date.now().toString(36);
  const entry: LightingPreset = { id, label: name, env: structuredClone(env) };
  commit(existing ? presets.map((p) => (p.id === id ? entry : p)) : [...presets, entry]);
  return id;
}

export function deleteUserPreset(id: string) { commit(presets.filter((p) => p.id !== id)); }

export const findPreset = (id: string): LightingPreset | undefined => LIGHTING_PRESETS.find((p) => p.id === id) ?? presets.find((p) => p.id === id);

/** Id of the built-in or user preset equal to `env`, else null. Built-ins win ties. */
export function matchAnyPreset(env: Environment, match: (e: Environment) => string | null): string | null {
  const b = match(env);
  if (b) return b;
  const same = (a: Environment, c: Environment) => {
    const ks = new Set([...Object.keys(a), ...Object.keys(c)]) as Set<keyof Environment>;
    return [...ks].every((k) => JSON.stringify(a[k] ?? null) === JSON.stringify(c[k] ?? null));
  };
  return presets.find((p) => same(p.env, env))?.id ?? null;
}

export function loadLastEnv(): Environment {
  try { const raw = localStorage.getItem(LAST_ENV_KEY); if (raw) return revive(JSON.parse(raw)); } catch { /* ignore */ }
  return { ...DEFAULT_ENV };
}
export function storeLastEnv(env: Environment) { try { localStorage.setItem(LAST_ENV_KEY, JSON.stringify(env)); } catch { /* ignore */ } }
