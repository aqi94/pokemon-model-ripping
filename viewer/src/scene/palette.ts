/** Runtime recolour of colour-table materials (modular NPCs). Mirrors pipeline apply_color_table(masked_only=True). */

export interface PaletteData {
  albedo: number; // glTF texture indices
  mask: number; // RGB of the layer mask (opaque image)
  maskA: number; // its alpha channel as grey (opaque image)
  divide: number;
  indices: number[]; // 1-based zone per layer
  scales: number[];
  layers: number[];
  lit: number[][]; // sRGB 0..255 per zone
  shadow: number[][];
}

const S2L = new Float32Array(256);
for (let i = 0; i < 256; i++) { const c = i / 255; S2L[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
const L2S_N = 4096;
const L2S = new Uint8ClampedArray(L2S_N + 1);
for (let i = 0; i <= L2S_N; i++) { const c = i / L2S_N; L2S[i] = Math.round((c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055) * 255); }

/** Zone 0 on a layer means "same zone as layer 1" (a lip layer painted in the skin colour = no lipstick). */
const zoneLin = (table: number[][], zone1: number): [number, number, number] => {
  const z = Math.min(Math.max(zone1, 1), table.length) - 1;
  const c = table[z];
  return [S2L[c[0]], S2L[c[1]], S2L[c[2]]];
};

/** Both outputs are RGBA8 buffers the size of the albedo; alpha is copied from it. */
export function bakePalette(alb: Uint8ClampedArray, mask: Uint8ClampedArray, pal: PaletteData, indices: number[]) {
  const n = alb.length / 4;
  const lit = new Uint8ClampedArray(alb.length);
  const shadow = new Uint8ClampedArray(alb.length);
  const zoneOf = (c: number) => (indices[c] || indices[0]) ?? 1;
  const lc = [0, 1, 2, 3].map((c) => zoneLin(pal.lit, zoneOf(c)));
  const sc = [0, 1, 2, 3].map((c) => zoneLin(pal.shadow, zoneOf(c)));
  for (let p = 0, o = 0; p < n; p++, o += 4) {
    let tl0 = 1, tl1 = 1, tl2 = 1, ts0 = 1, ts1 = 1, ts2 = 1;
    for (let c = 0; c < 4; c++) {
      const w = (mask[o + c] / 255) * (pal.scales[c] ?? 1);
      if (w <= 0) continue;
      tl0 += (lc[c][0] - tl0) * w; tl1 += (lc[c][1] - tl1) * w; tl2 += (lc[c][2] - tl2) * w;
      ts0 += (sc[c][0] - ts0) * w; ts1 += (sc[c][1] - ts1) * w; ts2 += (sc[c][2] - ts2) * w;
    }
    const r = S2L[alb[o]], g = S2L[alb[o + 1]], b = S2L[alb[o + 2]];
    lit[o] = L2S[Math.min(L2S_N, Math.round(r * tl0 * L2S_N))];
    lit[o + 1] = L2S[Math.min(L2S_N, Math.round(g * tl1 * L2S_N))];
    lit[o + 2] = L2S[Math.min(L2S_N, Math.round(b * tl2 * L2S_N))];
    shadow[o] = L2S[Math.min(L2S_N, Math.round(r * ts0 * L2S_N))];
    shadow[o + 1] = L2S[Math.min(L2S_N, Math.round(g * ts1 * L2S_N))];
    shadow[o + 2] = L2S[Math.min(L2S_N, Math.round(b * ts2 * L2S_N))];
    lit[o + 3] = shadow[o + 3] = alb[o + 3];
  }
  return { lit, shadow };
}
