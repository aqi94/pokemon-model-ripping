import * as THREE from 'three';

/**
 * UNITE flame look (user feedback 2026-11-02: "orange/yellow hybrid with blurred edges"), shared by the stand-in
 * emitters (particles.ts) and the models' own flame materials (manifest `extras.fireMaterials`, material
 * `extras.fire` from `importer.fire_effects`: Mega Charizard X's blue tail, Mega Y's orange one).
 * Checked against `samples/unite_fx/unite_fx_Fire_Blast_Charizard_UNITE_screen.png` (yellow core ~sRGB 253,228,2,
 * orange rim ~221,127,0, soft glowing edge):
 *   - the hard cut-out mask is blurred once on load (`SOFT_NEAR` = alpha, `SOFT_WIDE` = colour; FITTED);
 *   - rim colour = the mesh's per-vertex colour (Unity Fx shaders multiply their masks by it), core = HOME's FireCore
 *     `_LayerColor`, blended along HOME's `_LerpTex` ramp (`lerpProfile`). INFERRED: HOME drives that ramp with its own
 *     `_LerpTex`, not with the mask; UNITE's fx shader is not in the rip (its `PerObjectBloom` renderer feature would
 *     explain the glow).
 *
 * Along-flame colouring (user, 2026-11-08: two dominant colours along the flame): when the importer ships `_FLAMET`
 * (0 root .. 1 tip, from the mesh's principal axis, root = end nearest the emitter joint) the colour is
 * mix(HOME `_BaseColor`, `_LayerColor`, ramp(1 - t)). HOME's orientation, from the data: the FireCore card is skinned
 * TailA01 (root, glTF V ~1) .. TailA04 (tip, V ~0); `_LerpTex` row 0 (Unity V = 1, ramp 0 = `_BaseColor`) maps to
 * glTF V 0, i.e. the TIP, and its last row (ramp 1 = `_LayerColor`) to the root. Assumes the shader samples `_LerpTex`
 * with the mesh UV (the shader is not in the rip). The blurred mask only drives alpha then.
 *
 * Motion (2026-11-10, ledger `fx:homeMotion`): HOME's looping idle clip scrolls its fire core's two noise layers
 * (`_Blend0Tex`, `_Blend1Tex`) at constant speeds (`unite_home_fx.json` `noise[].speed`, UV/s). Here the UNITE silhouette
 * stays fixed (it is the alpha) and the noise moves each pixel along the colour ramp: n = clamp((a + b) / p99) - mean,
 * colour = ramp(x + n). User decision 2026-11-10: keep the approved colours on average and let them ripple. Taking HOME's
 * layering literally (product of the layers over the `_BaseColor` card) left the core on 4% of the card: a red flame.
 * INFERRED: the combination (HOME's shader is not in the rip) and the V direction (`noiseDir`, root -> tip).
 */

/** Blur radii as a fraction of the mask's width: NEAR = the soft edge (alpha), WIDE = the colour ramp (a rim band
 *  inside the edge, the core colour deeper in). FITTED to the Fire Blast still (soft edge ~1/4 of the flame width). */
export const SOFT_NEAR = 0.12;
export const SOFT_WIDE = 0.15;

/** The mask channel as an opaque grey image blurred by `frac` x width, wrapping like a repeat sampler (drawn 3 x 3 and
 *  cropped). Done once on the CPU side: shader taps wide enough for this blur band visibly. */
export function blurredMask(src: THREE.Texture, channel: 'a' | 'r', frac: number): THREE.Texture | null {
  const img = src.image as CanvasImageSource & { width: number; height: number };
  if (!img?.width || !img?.height) return null;
  const w = img.width, h = img.height;
  const grey = document.createElement('canvas');
  grey.width = w; grey.height = h;
  const g = grey.getContext('2d', { willReadFrequently: true });
  if (!g) return null;
  g.drawImage(img, 0, 0);
  const d = g.getImageData(0, 0, w, h);
  for (let i = 0; i < d.data.length; i += 4) {
    const v = channel === 'a' ? d.data[i + 3] : d.data[i];
    d.data[i] = d.data[i + 1] = d.data[i + 2] = v; d.data[i + 3] = 255;
  }
  g.putImageData(d, 0, 0);
  const tiled = document.createElement('canvas');
  tiled.width = 3 * w; tiled.height = 3 * h;
  const tc = tiled.getContext('2d')!;
  for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) tc.drawImage(grey, x * w, y * h);
  const blur = document.createElement('canvas');
  blur.width = 3 * w; blur.height = 3 * h;
  const bc = blur.getContext('2d');
  if (!bc) return null;
  bc.filter = `blur(${Math.max(1, frac * w).toFixed(1)}px)`;
  bc.drawImage(tiled, 0, 0);
  const out = document.createElement('canvas');
  out.width = w; out.height = h;
  out.getContext('2d')!.drawImage(blur, w, h, w, h, 0, 0, w, h);
  const tex = new THREE.CanvasTexture(out);
  tex.flipY = src.flipY;
  tex.wrapS = src.wrapS; tex.wrapT = src.wrapT;
  tex.colorSpace = THREE.NoColorSpace;
  return tex;
}

/** GLSL: uniforms + `fireRamp(x)` (HOME's 8-sample `_LerpTex` profile, linear in between). */
export const FIRE_GLSL = `
uniform sampler2D uFireNear;
uniform sampler2D uFireWide;
uniform vec3 uFireCore;
uniform vec3 uFireBase;
uniform float uFireAlong;
varying float vFlameT;
uniform float uFireProfile[8];
uniform float uFireOn;
uniform sampler2D uFireNoise0;
uniform sampler2D uFireNoise1;
uniform vec4 uFireNoiseST0; // scale.xy, offset.xy (already x time, wrapped)
uniform vec4 uFireNoiseST1;
uniform float uFireNoiseNorm;
uniform float uFireNoiseMean;
uniform float uFireNoiseOn;
// HOME's two scrolling layers summed and renormalised, minus their mean: a zero-average nudge along the colour ramp
float fireNoise(vec2 uv) {
  if (uFireNoiseOn < 0.5) return 0.0;
  float a = texture2D(uFireNoise0, uv * uFireNoiseST0.xy + uFireNoiseST0.zw).r;
  float b = texture2D(uFireNoise1, uv * uFireNoiseST1.xy + uFireNoiseST1.zw).r;
  return clamp((a + b) / max(uFireNoiseNorm, 1e-3), 0.0, 1.0) - uFireNoiseMean;
}
float fireRamp(float x) {
  float f = clamp(x, 0.0, 1.0) * 7.0;
  int i = int(min(floor(f), 6.0));
  float a = 0.0, b = 0.0;
  for (int k = 0; k < 7; k++) if (k == i) { a = uFireProfile[k]; b = uFireProfile[k + 1]; }
  return mix(a, b, f - float(i));
}
float fireAlpha(vec2 uv) { return smoothstep(0.3, 0.8, texture2D(uFireNear, uv).r); }
vec3 fireColour(vec2 uv, vec3 rim) {
  // HOME: ramp 0 (base) at the tip; the core card (ramp colour) shows through its scrolling noise over the outline card
  float n = fireNoise(uv);
  if (uFireAlong > 0.5) return mix(uFireBase, uFireCore, fireRamp(1.0 - vFlameT + n));
  return mix(rim, uFireCore, fireRamp(texture2D(uFireWide, uv).r + n));
}
`;

export interface NoiseLayer { scale: number[]; speed: number[]; map?: { index: number } }
export interface FireSpec {
  layerColor: number[]; lerpProfile?: number[] | null; baseColor?: number[]; alongFlame?: boolean;
  noise?: NoiseLayer[]; noiseNorm?: number; noiseMean?: number; noiseDir?: number;
}

export type FireUniforms = {
  uFireNear: { value: THREE.Texture | null }; uFireWide: { value: THREE.Texture | null };
  uFireCore: { value: THREE.Color }; uFireBase: { value: THREE.Color }; uFireAlong: { value: number }; uFireProfile: { value: number[] }; uFireOn: { value: number };
  uFireNoise0: { value: THREE.Texture | null }; uFireNoise1: { value: THREE.Texture | null };
  uFireNoiseST0: { value: THREE.Vector4 }; uFireNoiseST1: { value: THREE.Vector4 };
  uFireNoiseNorm: { value: number }; uFireNoiseMean: { value: number }; uFireNoiseOn: { value: number };
};

export function fireUniforms(spec: FireSpec): FireUniforms {
  const lc = spec.layerColor;
  const bc = spec.baseColor ?? lc;
  const prof = spec.lerpProfile?.length === 8 ? spec.lerpProfile.slice() : [0, 0, 0, 0, 0.224, 0.561, 0.872, 1];
  return {
    uFireNear: { value: null }, uFireWide: { value: null },
    uFireCore: { value: new THREE.Color().setRGB(lc[0], lc[1], lc[2], THREE.SRGBColorSpace) },
    uFireBase: { value: new THREE.Color().setRGB(bc[0], bc[1], bc[2], THREE.SRGBColorSpace) },
    uFireAlong: { value: spec.alongFlame ? 1 : 0 },
    uFireProfile: { value: prof }, uFireOn: { value: 0 },
    uFireNoise0: { value: null }, uFireNoise1: { value: null },
    uFireNoiseST0: { value: new THREE.Vector4(1, 1, 0, 0) }, uFireNoiseST1: { value: new THREE.Vector4(1, 1, 0, 0) },
    uFireNoiseNorm: { value: spec.noiseNorm ?? 1 }, uFireNoiseMean: { value: spec.noiseMean ?? 0.5 }, uFireNoiseOn: { value: 0 },
  };
}

/** Switch the scrolling noise on once both HOME layers are loaded (repeat-wrapped, linear). */
export function setFireNoise(u: FireUniforms, spec: FireSpec, maps: (THREE.Texture | null)[]) {
  if (!spec.noise || spec.noise.length < 2 || !maps[0] || !maps[1]) return;
  for (const t of maps) { t!.wrapS = t!.wrapT = THREE.RepeatWrapping; t!.colorSpace = THREE.NoColorSpace; t!.needsUpdate = true; }
  u.uFireNoise0.value = maps[0]; u.uFireNoise1.value = maps[1];
  u.uFireNoiseST0.value.set(spec.noise[0].scale[0] ?? 1, spec.noise[0].scale[1] ?? 1, 0, 0);
  u.uFireNoiseST1.value.set(spec.noise[1].scale[0] ?? 1, spec.noise[1].scale[1] ?? 1, 0, 0);
  u.uFireNoiseOn.value = 1;
}

/** Advance the noise to `t` seconds of effect time: offset = speed x t (V by `noiseDir`, see the header), wrapped. */
export function updateFireNoise(u: FireUniforms, spec: FireSpec, t: number) {
  if (!spec.noise) return;
  const dir = spec.noiseDir ?? 1;
  const st = [u.uFireNoiseST0.value, u.uFireNoiseST1.value];
  spec.noise.slice(0, 2).forEach((n, i) => {
    const fu = (n.speed[0] ?? 0) * t, fv = -dir * (n.speed[1] ?? 0) * t;
    st[i].z = fu - Math.floor(fu); st[i].w = fv - Math.floor(fv);
  });
}

/** Blur `mask` into the uniforms and switch the look on. Returns the textures made (the caller disposes them). */
export function setFireMask(u: FireUniforms, mask: THREE.Texture, channel: 'a' | 'r'): THREE.Texture[] {
  const near = blurredMask(mask, channel, SOFT_NEAR), wide = blurredMask(mask, channel, SOFT_WIDE);
  if (!near || !wide) return [];
  u.uFireNear.value = near; u.uFireWide.value = wide; u.uFireOn.value = 1;
  return [near, wide];
}

/**
 * A model's own flame material (glTF effect material: black base, emissive = the mask texture, BLEND, COLOR_0 = the
 * per-vertex rim colour): its final colour/alpha are replaced by the fire look. Install once, before the game-shading
 * patch captures onBeforeCompile (the two chain). The mask = the material's own texture alpha (the importer turns grey
 * masks into alpha). Returns whether it installed.
 */
export function installFire(mat: THREE.MeshStandardMaterial, loadTexture?: (index: number) => Promise<THREE.Texture | null>): boolean {
  const spec = mat.userData?.fire as FireSpec | undefined;
  if (!spec?.layerColor) return false;
  if (mat.userData.__fire) return true;
  const mask = mat.map ?? mat.emissiveMap;
  if (!mask) return false;
  const u = fireUniforms(spec);
  mat.userData.__fire = u;
  mat.userData.__fireTex = setFireMask(u, mask, 'a'); // glTF textures are decoded by the time materials are set up
  if (loadTexture && spec.noise?.length === 2 && spec.noise.every((n) => typeof n.map?.index === 'number')) {
    void Promise.all(spec.noise.map((n) => loadTexture(n.map!.index))).then((maps) => {
      // the GLB texture is shared with nothing else, but clone so wrap/colour-space changes stay local
      const own = maps.map((t) => t?.clone() ?? null);
      (mat.userData.__fireTex as THREE.Texture[]).push(...own.filter((t): t is THREE.Texture => !!t));
      setFireNoise(u, spec, own);
    });
  }
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = function (this: THREE.Material, shader, renderer) {
    prev?.call(this, shader, renderer);
    Object.assign(shader.uniforms, u);
    // vFlameT is declared by FIRE_GLSL in the fragment shader, so the vertex shader always writes it
    shader.vertexShader = shader.vertexShader
      .replace('void main() {', (spec.alongFlame ? 'attribute float _flamet;\n' : '') + 'varying float vFlameT;\nvoid main() {')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n  vFlameT = ' + (spec.alongFlame ? '_flamet' : '0.0') + ';');
    shader.fragmentShader = shader.fragmentShader
      .replace('void main() {', FIRE_GLSL + '\nvoid main() {')
      .replace('#include <dithering_fragment>', `#include <dithering_fragment>
#if defined( USE_MAP ) || defined( USE_EMISSIVEMAP )
  if (uFireOn > 0.5) {
  #ifdef USE_MAP
    vec2 fireUv = vMapUv;
  #else
    vec2 fireUv = vEmissiveMapUv;
  #endif
  #ifdef USE_COLOR
    vec3 fireRim = vColor.rgb;
  #else
    vec3 fireRim = emissive;
  #endif
    gl_FragColor = linearToOutputTexel(vec4(fireColour(fireUv, fireRim), fireAlpha(fireUv) * opacity));
  }
#endif`);
  };
  mat.needsUpdate = true;
  return true;
}

/**
 * HOME-painted flames on UNITE parts (`extras.homeFire`, importer `home_parts`: Skeledirge's lobby flame strands carry
 * HOME `FlameSkin`'s own UVs and `BodyC` texture). HOME's looping idle clip scrolls that texture at a constant speed
 * (`scroll`, glTF UV/s, the eye helpers' convention); the clone keeps the shift local to this material.
 */
export function installHomeFire(mat: THREE.MeshStandardMaterial): boolean {
  const hf = mat.userData?.homeFire as { scroll?: number[] } | undefined;
  if (!hf?.scroll || mat.userData.__homeFire) return !!mat.userData?.__homeFire;
  const src = mat.emissiveMap ?? mat.map;
  if (!src) return false;
  const t = src.clone();
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.needsUpdate = true;
  if (mat.map === src || mat.map?.source === src.source) mat.map = t;
  if (mat.emissiveMap) mat.emissiveMap = t;
  mat.userData.__homeFire = t;
  mat.needsUpdate = true;
  return true;
}

export function updateHomeFire(mat: THREE.MeshStandardMaterial, t: number) {
  const tex = mat.userData.__homeFire as THREE.Texture | undefined;
  const sc = (mat.userData.homeFire as { scroll: number[] }).scroll;
  if (!tex) return;
  const u = sc[0] * t, v = sc[1] * t;
  tex.offset.set(u - Math.floor(u), v - Math.floor(v));
}
