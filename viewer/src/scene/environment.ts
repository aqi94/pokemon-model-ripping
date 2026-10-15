// Lighting follows the usual game-engine model (UE/Unity HDRP/Frostbite):
// light colour as colour temperature, key + back (rim) light + sky/ground
// ambient (the L1 approximation of a light probe), exposure + filmic tone
// mapping, and named profiles that swap the whole set (post-process volumes).

export type ToneMap = 'linear' | 'aces' | 'agx' | 'neutral';

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
}

/** Tuned against the VS screenshots (neutral white light, linear output). */
export const DEFAULT_ENV: Environment = {
  ambient: 1.9, key: 1.2, azimuth: -26.6, elevation: 38.8, keyTemp: 6500,
  back: 0, backAzimuth: 160, backElevation: 25, backTemp: 6500, skyTemp: 6500, ground: '#f6ecd8',
  warmth: 0, shadow: 0.65, rim: 1, detail: 0.5, exposure: 1, toneMapping: 'linear', contrast: 1, saturation: 1, gi: 1,
};

export interface LightingPreset { id: string; label: string; env: Environment }

const p = (id: string, label: string, e: Partial<Environment>): LightingPreset => ({ id, label, env: { ...DEFAULT_ENV, ...e } });

// Temperatures are the standard reference values: sun 5500-6500 K, overcast
// sky ~7000 K, incandescent / warm LED 2700-3000 K, fluorescent ~4300 K,
// sodium street lamp ~2000 K, moonlight rendered ~4100 K at low exposure.
// Only "Indoor, window" is calibrated against a screenshot
// (samples/detective_handclap01_*, tools/match_lighting.mjs).
export const LIGHTING_PRESETS: LightingPreset[] = [
  p('studio', 'Studio (VS)', {}),
  // Fitted with tools/fit_lighting_refs.mjs to 15 BDSP trainers vs their Bulbapedia VS images (mean |dRGB| 41.7 -> 21.6, 5 held-out
  // trainers 49.8 -> 23.7): mostly exposure (the game is much brighter than Studio) plus a flatter, lower key. Linear output.
  p('bdspVs', 'BDSP VS render', { ambient: 1.84, azimuth: -32, elevation: 16, keyTemp: 6125, back: 0.1, skyTemp: 5900, shadow: 0.39, detail: 0.4,
    exposure: 1.8, contrast: 0.9 }),
  p('day', 'Day, street', { ambient: 2.2, key: 1.4, azimuth: -35, elevation: 55, keyTemp: 5800, skyTemp: 8000, ground: '#b5a890',
    back: 0.4, backTemp: 8000, rim: 0.5, toneMapping: 'agx', exposure: 1.6, contrast: 1.3 }),
  p('overcast', 'Overcast', { ambient: 3.0, key: 0.3, elevation: 60, keyTemp: 7000, skyTemp: 7000, ground: '#a8a8a4',
    rim: 0.3, shadow: 0.4, toneMapping: 'agx', exposure: 1.7, contrast: 1.25 }),
  p('sunset', 'Sunset', { ambient: 1.8, key: 1.6, azimuth: -60, elevation: 8, keyTemp: 3000, skyTemp: 9000, ground: '#8a6a70',
    back: 0.8, backAzimuth: 150, backElevation: 10, backTemp: 2600, rim: 0.6, toneMapping: 'agx', exposure: 1.6, contrast: 1.3 }),
  // Fitted to samples/VSGrisham_advent_loop_night* (Grisham tr0009_00, mean dE 8.4 vs 15.9 for Studio and 20.5 for the
  // old guess): bright artificial night = strong cool ambient (sky 12000 K), cool back light from behind, modest high
  // key, light rim; ACES. The shirt shadow side stays too bright (pose differs from the screenshot).
  p('night', 'Night, battle zone', { ambient: 3.76, key: 0.6, azimuth: -37, elevation: 56, keyTemp: 6600, skyTemp: 12000, ground: '#e06941',
    back: 2.1, backAzimuth: 122, backElevation: 29, backTemp: 12000, shadow: 0.72, rim: 0.16, gi: 1.09,
    toneMapping: 'aces', exposure: 1.22, contrast: 1.22, saturation: 0.85 }),
  // Fitted to samples/detective_handclap01_* (mean dE 4.9 vs 13.4 for Studio): light on her front is the room's
  // warm bounce, not a key; the window behind her is a cool back light; no rim glow; AgX + contrast.
  p('indoorWindow', 'Indoor, window behind', { ambient: 2.75, key: 0.25, azimuth: 120, elevation: 0, keyTemp: 5900, skyTemp: 6000, ground: '#af8680',
    back: 0.7, backAzimuth: 167, backElevation: 25, backTemp: 9000, shadow: 0.35, rim: 0, gi: 0.1,
    toneMapping: 'agx', exposure: 1.7, contrast: 1.35 }),
  p('indoorWarm', 'Indoor, lamp-lit', { ambient: 2.4, key: 0.6, azimuth: -40, elevation: 60, keyTemp: 2900, skyTemp: 3800, ground: '#a08068',
    rim: 0.2, toneMapping: 'agx', exposure: 1.7, contrast: 1.3 }),
  p('indoorCool', 'Indoor, fluorescent', { ambient: 2.8, key: 0.6, azimuth: 0, elevation: 75, keyTemp: 4300, skyTemp: 5000, ground: '#b8bcb8',
    rim: 0.3, shadow: 0.4, toneMapping: 'agx', exposure: 1.6, contrast: 1.25 }),
];

export function matchPreset(env: Environment): string | null {
  for (const pr of LIGHTING_PRESETS) {
    if ((Object.keys(pr.env) as (keyof Environment)[]).every((k) => pr.env[k] === env[k])) return pr.id;
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
