// Lighting follows the usual game-engine model (UE/Unity HDRP/Frostbite):
// light colour as colour temperature, key + back (rim) light + sky/ground
// ambient (the L1 approximation of a light probe), exposure + filmic tone
// mapping, and named profiles that swap the whole set (post-process volumes).

import { SV_PRESETS } from './svPresets';
import { ZA_PRESETS } from './zaPresets';

export type ToneMap = 'linear' | 'aces' | 'agx' | 'neutral' | 'hable';

/** John Hable's piecewise filmic curve ("Filmic Worlds" user params); Game Freak's `PfxTonemapMode 2` stores exactly these. */
export interface HableParams {
  toeStrength: number;     // 0..1
  toeLength: number;       // 0..1
  shoulderStrength: number; // F-stops of extra white point, 0..1 in the UI space
  shoulderLength: number;  // 0..1
  shoulderAngle: number;   // 0..1
  gamma: number;
}

export interface Environment {
  ambient: number;     // hemisphere intensity
  key: number;         // key light intensity
  azimuth: number;     // degrees, 0 = straight in front of the model (+Z), positive toward +X
  elevation: number;   // degrees above the horizon
  keyTemp: number;     // key colour temperature, Kelvin
  back: number;        // back/rim light intensity (window, street lamp, neon)
  backAzimuth: number; // degrees, 180 = straight behind the model
  backElevation: number;
  backTemp: number;
  skyTemp: number;     // ambient sky colour temperature
  ground: string;      // ambient ground (bounce) colour
  warmth: number;      // -1 cool .. 1 warm, white-balance shift on top of everything
  shadow: number;      // 0..1 strength of the shadow colour
  rim: number;         // multiplier on the material rim term
  detail: number;      // normal map / AO strength
  exposure: number;
  toneMapping: ToneMap;
  contrast: number;
  saturation: number;
  gi: number;          // ambient tint toward the material's shadow colours (x ShadowingGIGain)
  // Optional explicit light colours (sRGB hex). When present they replace the Kelvin fields, which the
  // game-data presets cannot express (a pink GI tint has no blackbody temperature). Moving a Kelvin slider clears them.
  keyColor?: string;
  skyColor?: string;
  backColor?: string;
  hable?: HableParams; // only read when toneMapping === 'hable'
}

/** Tuned against the VS screenshots (neutral white light, linear output). */
export const DEFAULT_ENV: Environment = {
  ambient: 1.9, key: 1.2, azimuth: -26.6, elevation: 38.8, keyTemp: 6500,
  back: 0, backAzimuth: 160, backElevation: 25, backTemp: 6500, skyTemp: 6500, ground: '#f6ecd8',
  warmth: 0, shadow: 0.65, rim: 1, detail: 0.5, exposure: 1, toneMapping: 'linear', contrast: 1, saturation: 1, gi: 1,
};

/** Scarlet/Violet's master post-process curve (romfs/env/master/master.trlgt Global.Pfx*), the default when 'hable' is picked by hand. */
export const DEFAULT_HABLE: HableParams = { toeStrength: 0.085, toeLength: 0.5, shoulderStrength: 0.9999, shoulderLength: 0.5, shoulderAngle: 0, gamma: 0.967 };

/** `game` records which game's data a preset came from (provenance only; every preset is offered for every model). */
export interface LightingPreset { id: string; label: string; env: Environment; game?: string }

const p = (id: string, label: string, e: Partial<Environment>, game?: string): LightingPreset => ({ id, label, env: { ...DEFAULT_ENV, ...e }, game });

// Temperatures are the standard reference values: sun 5500-6500 K, overcast
// sky ~7000 K, incandescent / warm LED 2700-3000 K, fluorescent ~4300 K,
// sodium street lamp ~2000 K, moonlight rendered ~4100 K at low exposure.
// Calibrated entries say so in their comment; SV entries come from the game's own light rigs (svPresets.ts, generated).
export const LIGHTING_PRESETS: LightingPreset[] = [
  p('studio', 'Studio (VS)', {}),
  // Fitted with tools/fit_lighting_refs.mjs to 15 BDSP trainers vs their Bulbapedia VS images (mean |dRGB| 41.7 -> 21.6, 5 held-out
  // trainers 49.8 -> 23.7): mostly exposure (the game is much brighter than Studio) plus a flatter, lower key. Linear output.
  p('bdspVs', 'BDSP VS render', { ambient: 1.84, azimuth: -32, elevation: 16, keyTemp: 6125, back: 0.1, skyTemp: 5900, shadow: 0.39, detail: 0.4,
    exposure: 1.8, contrast: 0.9 }, 'bdsp'),
  // Z-A scene presets fitted to gameplay frames (refs/za/<scene>/, tools/fit_scene.mjs, generated zaPresets.ts).
  ...ZA_PRESETS,
  ...SV_PRESETS,
  // The hand-fitted 'night' (Grisham VS, dE 8.4) and 'indoorWindow' (detective, dE 4.9) presets became
  // za_battle_zone_night and za_indoor_hotel in zaPresets.ts (same screenshots, refitted by tools/fit_scene.mjs).
];
// The old uncalibrated guesses (day / overcast / sunset / indoorWarm / indoorCool) now live in
// tools/lightrefs/seed_envs.json as fit seeds only.

const same = (a: unknown, b: unknown) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

export function matchPreset(env: Environment): string | null {
  const keys = (e: Environment) => (Object.keys(e) as (keyof Environment)[]).filter((k) => e[k] !== undefined);
  for (const pr of LIGHTING_PRESETS) {
    const ks = new Set([...keys(pr.env), ...keys(env)]);
    if ([...ks].every((k) => same(pr.env[k], env[k]))) return pr.id;
  }
  return null;
}

/** Colour temperature -> linear RGB (0..1), Tanner Helland's blackbody fit, normalised so 6500 K is white. */
export function kelvinToRGB(k: number): [number, number, number] {
  const raw = (kk: number) => {
    const t = Math.min(40000, Math.max(1000, kk)) / 100;
    const r = t <= 66 ? 255 : 329.698727446 * Math.pow(t - 60, -0.1332047592);
    const g = t <= 66 ? 99.4708025861 * Math.log(t) - 161.1195681661 : 288.1221695283 * Math.pow(t - 60, -0.0755148492);
    const b = t >= 66 ? 255 : t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307;
    // sRGB-encoded fit -> linear
    return [r, g, b].map((c) => { const s = Math.min(255, Math.max(0, c)) / 255; return s <= 0.04045 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4); });
  };
  const c = raw(k), w = raw(6500);
  const out = c.map((v, i) => v / w[i]);
  const m = Math.max(...out);
  return [out[0] / m, out[1] / m, out[2] / m];
}
