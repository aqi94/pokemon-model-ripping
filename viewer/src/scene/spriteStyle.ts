import * as THREE from 'three';
import { GIFEncoder, quantize } from 'gifenc';

/**
 * 3DS battle-sprite look (the X/Y-ORAS rips on play.pokemonshowdown.com/sprites/ani): the scene is rendered at a low
 * resolution without anti-aliasing (textures stay filtered, edges are hard), then a screen-space pass draws a 1-px
 * outline: a fixed near-black line around the silhouette and the surface's own colour darkened where one part
 * overlaps another (Pikachu's arm over its body). Measured on the Showdown sprites (pikachu, charizard, bulbasaur,
 * lucario, garchomp, front + back): 1-bit alpha, silhouette ~(39,39,38), inner lines ~0.4x the surface, one global
 * 16-entry palette, 30 ms frames.
 */

// FITTED: silhouette colour, mean edge colour of the Showdown ani sprites (pikachu, charizard, lucario), 2026-10-20
export const SPRITE_SILHOUETTE = '#272726';
// FITTED: inner-line darkness, Showdown pikachu/charizard inner contour vs adjacent surface (sRGB bytes), 2026-10-20
export const SPRITE_INNER_DARKNESS = 0.4;
// FITTED: depth jump (x model radius) that counts as an overlap edge, eyeballed on ZA Pikachu/Charizard, 2026-10-20
export const SPRITE_DEPTH_REL = 0.03;
// FITTED: crease line where neighbouring view-space normals differ by more than ~32 deg (cos 0.85): mouth, nostrils,
// brow and claws of the Showdown Charizard, eyeballed on ZA Charizard/Pikachu, 2026-10-20. A crease also needs an
// edge-on nearer side (SPRITE_CONTOUR_COS).
export const SPRITE_CREASE_COS = 0.85;
// FITTED: crease lines only where the nearer side is edge-on (|n.v| < 0.35), the hull-like contour test that removes
// eye-socket / whisker-root / armpit lines (surfaces that merely meet), ZA Meowth/Gengar/Pikachu/Charizard, 2026-10-20
export const SPRITE_CONTOUR_COS = 0.35;

const VERT = `out vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const EDGE_FRAG = `
precision highp float;
uniform sampler2D tColor;
uniform sampler2D tDepth;
uniform sampler2D tNormal;
uniform float uCrease;
uniform float uContour;
uniform mat4 uInvProj;
uniform vec3 uSil;
uniform float uDark;
uniform float uThresh;
uniform float uNear;
uniform float uFar;
uniform bool uOrtho;
out vec4 outColor;
float lin(float d) {
  if (uOrtho) return uNear + d * (uFar - uNear);
  float z = d * 2.0 - 1.0;
  return 2.0 * uNear * uFar / (uFar + uNear - z * (uFar - uNear));
}
// Opaque, or an additive glow drawn over the cleared (alpha 0) background: Charizard's tail flame.
bool isSolid(vec4 c) { return c.a >= 0.5 || max(c.r, max(c.g, c.b)) > 0.3; }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 sz = textureSize(tColor, 0);
  vec4 c = texelFetch(tColor, p, 0);
  bool solid = isSolid(c);
  float d = lin(texelFetch(tDepth, p, 0).r);
  vec3 n = texelFetch(tNormal, p, 0).xyz * 2.0 - 1.0;
  // View ray of this pixel (through the inverse projection, so a lens shift / view offset is respected).
  vec2 ndc = (gl_FragCoord.xy / vec2(sz)) * 2.0 - 1.0;
  vec3 vdir = uOrtho ? vec3(0.0, 0.0, -1.0) : normalize((uInvProj * vec4(ndc, 1.0, 1.0)).xyz);
  ivec2 off[4] = ivec2[4](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1));
  bool nearSolid = false, nearer = false;
  for (int i = 0; i < 4; i++) {
    ivec2 q = clamp(p + off[i], ivec2(0), sz - 1);
    vec4 cq = texelFetch(tColor, q, 0);
    if (!isSolid(cq)) continue;
    nearSolid = true;
    if (!solid) continue;
    float dq = lin(texelFetch(tDepth, q, 0).r);
    if (d - dq > uThresh) { nearer = true; continue; }
    // Crease (fold, mouth, nostril): a 1-px line on the farther side of the fold only, so it never doubles, and only
    // where the nearer side turns edge-on (a contour). Surfaces that merely meet (eye set into the face, whisker root,
    // armpit) are both camera-facing there and get no line, like the hull outline.
    vec3 nq = texelFetch(tNormal, q, 0).xyz * 2.0 - 1.0;
    if (d >= dq && dot(n, nq) < uCrease && abs(dot(nq, vdir)) < uContour) nearer = true;
  }
  if (!solid) { outColor = nearSolid ? vec4(uSil, 1.0) : vec4(0.0); return; }
  // Blended pixels (lenses, effects) carry premultiplied-looking colour at partial alpha: un-premultiply.
  vec3 rgb = c.a < 0.999 && c.a >= 0.5 ? c.rgb / c.a : c.rgb;
  outColor = vec4(nearer ? rgb * uDark : rgb, 1.0);
}`;

const BLIT_FRAG = `
precision highp float;
uniform sampler2D tSrc;
uniform vec3 uBg;
uniform bool uHasBg;
in vec2 vUv;
out vec4 outColor;
void main() {
  vec4 c = texture(tSrc, vUv);
  outColor = uHasBg ? vec4(mix(uBg, c.rgb, c.a), 1.0) : c;
}`;

export interface SpriteFrame { width: number; height: number; data: Uint8Array }

export class SpritePass {
  private rtScene: THREE.WebGLRenderTarget;
  private rtOut: THREE.WebGLRenderTarget;
  /** Geometric view-space normals (no normal maps) for crease lines. */
  private rtNormal: THREE.WebGLRenderTarget;
  private normalMat = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
  private quadScene = new THREE.Scene();
  private quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  private quad: THREE.Mesh;
  private edge: THREE.ShaderMaterial;
  private blitMat: THREE.ShaderMaterial;
  silhouette = new THREE.Color(SPRITE_SILHOUETTE);

  constructor() {
    const depthTexture = new THREE.DepthTexture(1, 1);
    depthTexture.type = THREE.UnsignedIntType;
    this.rtScene = new THREE.WebGLRenderTarget(1, 1, { samples: 0, depthTexture, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    // three only tone-maps and sRGB-encodes when drawing to the screen, or to an XR target: flag ours as one so the
    // sprite has exactly the live view's colours (stored as sRGB bytes in a linear RGBA8 texture, no double encode).
    (this.rtScene as unknown as { isXRRenderTarget: boolean }).isXRRenderTarget = true;
    this.rtScene.texture.colorSpace = THREE.SRGBColorSpace;
    this.rtOut = new THREE.WebGLRenderTarget(1, 1, { samples: 0, depthBuffer: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    this.rtNormal = new THREE.WebGLRenderTarget(1, 1, { samples: 0, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    this.edge = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: VERT, fragmentShader: EDGE_FRAG, depthTest: false, depthWrite: false,
      uniforms: {
        tColor: { value: this.rtScene.texture }, tDepth: { value: depthTexture }, tNormal: { value: null as THREE.Texture | null }, uCrease: { value: SPRITE_CREASE_COS }, uContour: { value: SPRITE_CONTOUR_COS }, uInvProj: { value: new THREE.Matrix4() }, uSil: { value: new THREE.Vector3() },
        uDark: { value: SPRITE_INNER_DARKNESS }, uThresh: { value: 0.05 }, uNear: { value: 0.1 }, uFar: { value: 100 }, uOrtho: { value: false },
      },
    });
    this.blitMat = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3, vertexShader: VERT, fragmentShader: BLIT_FRAG, depthTest: false, depthWrite: false, transparent: false,
      uniforms: { tSrc: { value: this.rtOut.texture }, uBg: { value: new THREE.Vector3() }, uHasBg: { value: true } },
    });
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.edge);
    this.quad.frustumCulled = false;
    this.quadScene.add(this.quad);
  }

  get size(): [number, number] { return [this.rtOut.width, this.rtOut.height]; }

  /**
   * Renders `scene` at w x h into the low-res target and runs the outline pass into the output target. The caller hides
   * what must not be in a sprite (background, ground, helpers). `radius` = model bounding radius (sets the overlap
   * depth threshold).
   */
  render(r: THREE.WebGLRenderer, scene: THREE.Scene, cam: THREE.PerspectiveCamera | THREE.OrthographicCamera, w: number, h: number, radius: number) {
    w = Math.max(1, Math.round(w)); h = Math.max(1, Math.round(h));
    if (this.rtScene.width !== w || this.rtScene.height !== h) { this.rtScene.setSize(w, h); this.rtOut.setSize(w, h); this.rtNormal.setSize(w, h); }
    const prevRt = r.getRenderTarget();
    const clear = r.getClearColor(new THREE.Color()), clearA = r.getClearAlpha();
    const bg = scene.background;
    scene.background = null;
    r.setClearColor(0x000000, 0);
    r.setRenderTarget(this.rtScene);
    r.clear();
    r.render(scene, cam);
    this.renderNormals(r, scene, cam);
    scene.background = bg;
    const u = this.edge.uniforms;
    const sil = this.silhouette.getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace); // the pass works in sRGB bytes / 255
    u.uSil.value.set(sil.r, sil.g, sil.b);
    u.uThresh.value = SPRITE_DEPTH_REL * radius;
    u.tNormal.value = this.rtNormal.texture;
    u.uNear.value = cam.near; u.uFar.value = cam.far;
    u.uInvProj.value.copy(cam.projectionMatrixInverse);
    u.uOrtho.value = (cam as THREE.OrthographicCamera).isOrthographicCamera === true;
    this.quad.material = this.edge;
    r.setRenderTarget(this.rtOut);
    r.clear();
    r.render(this.quadScene, this.quadCam);
    r.setRenderTarget(prevRt);
    r.setClearColor(clear, clearA);
  }

  /**
   * View-space normals of the opaque, depth-writing meshes only: blended overlays (eyelid / layer copies of the same
   * triangles), gas and effects would z-fight or cover the surface and speckle the crease test.
   */
  private renderNormals(r: THREE.WebGLRenderer, scene: THREE.Scene, cam: THREE.Camera) {
    const swapped: [THREE.Mesh, THREE.Material | THREE.Material[]][] = [];
    const hidden: THREE.Object3D[] = [];
    scene.traverseVisible((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      const mats = ([] as THREE.Material[]).concat(m.material);
      const solid = (x: THREE.Material) => x.visible && !x.transparent && x.depthWrite && !x.userData?.gas;
      if (!mats.some(solid)) { hidden.push(m); return; }
      swapped.push([m, m.material]);
      if (Array.isArray(m.material)) m.material = mats.map((x) => (solid(x) ? this.normalMat : this.hideMat));
      else m.material = this.normalMat;
    });
    for (const o of hidden) o.visible = false;
    try {
      r.setRenderTarget(this.rtNormal);
      r.clear();
      r.render(scene, cam);
    } finally {
      for (const [m, mat] of swapped) m.material = mat;
      for (const o of hidden) o.visible = true;
    }
  }
  private hideMat = (() => { const m = new THREE.MeshBasicMaterial(); m.visible = false; return m; })();

  /** Draws the last output to the current target (the canvas) with nearest upscaling; `bg` null = keep transparency. */
  blit(r: THREE.WebGLRenderer, bg: THREE.Color | null) {
    const u = this.blitMat.uniforms;
    u.uHasBg.value = !!bg;
    if (bg) { const s = bg.getRGB({ r: 0, g: 0, b: 0 }, THREE.SRGBColorSpace); u.uBg.value.set(s.r, s.g, s.b); }
    this.quad.material = this.blitMat;
    r.render(this.quadScene, this.quadCam);
  }

  /** The last output as top-row-first RGBA bytes. */
  read(r: THREE.WebGLRenderer): SpriteFrame {
    const w = this.rtOut.width, h = this.rtOut.height;
    const raw = new Uint8Array(w * h * 4);
    r.readRenderTargetPixels(this.rtOut, 0, 0, w, h, raw);
    const data = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) data.set(raw.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
    return { width: w, height: h, data };
  }

  dispose() {
    this.rtScene.depthTexture?.dispose();
    this.rtScene.dispose(); this.rtOut.dispose(); this.rtNormal.dispose(); this.normalMat.dispose();
    this.edge.dispose(); this.blitMat.dispose();
    this.quad.geometry.dispose();
  }
}

/** Alpha bounding box (inclusive) of a frame, or null if empty. */
export function frameBounds(f: SpriteFrame): { x0: number; y0: number; x1: number; y1: number } | null {
  let x0 = f.width, y0 = f.height, x1 = -1, y1 = -1;
  for (let y = 0; y < f.height; y++) for (let x = 0; x < f.width; x++) {
    if (f.data[(y * f.width + x) * 4 + 3] < 128) continue;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  return x1 < 0 ? null : { x0, y0, x1, y1 };
}

export interface GifOptions { colors: number; delayMs: number }

/**
 * Animated GIF from same-size frames: cropped to the union alpha box (+1 px), one global palette (index 0 =
 * transparent, the rest quantised from the opaque pixels of every frame), loops forever, disposal 2 like the
 * Showdown sprites.
 */
export function encodeSpriteGif(frames: SpriteFrame[], o: GifOptions): Blob {
  if (!frames.length) throw new Error('no frames');
  const W = frames[0].width, H = frames[0].height;
  let bx0 = W, by0 = H, bx1 = -1, by1 = -1;
  for (const f of frames) {
    const b = frameBounds(f);
    if (!b) continue;
    bx0 = Math.min(bx0, b.x0); by0 = Math.min(by0, b.y0); bx1 = Math.max(bx1, b.x1); by1 = Math.max(by1, b.y1);
  }
  if (bx1 < 0) { bx0 = by0 = 0; bx1 = by1 = 0; }
  bx0 = Math.max(0, bx0 - 1); by0 = Math.max(0, by0 - 1); bx1 = Math.min(W - 1, bx1 + 1); by1 = Math.min(H - 1, by1 + 1);
  const cw = bx1 - bx0 + 1, ch = by1 - by0 + 1;

  // Opaque pixels of every frame (strided if huge) feed the quantiser.
  let opaque = 0;
  for (const f of frames) for (let i = 3; i < f.data.length; i += 4) if (f.data[i] >= 128) opaque++;
  const stride = Math.max(1, Math.ceil(opaque / 2_000_000));
  const sample = new Uint8Array(Math.max(1, Math.ceil(opaque / stride)) * 4);
  let n = 0, k = 0;
  for (const f of frames) for (let i = 0; i < f.data.length; i += 4) {
    if (f.data[i + 3] < 128) continue;
    if (k++ % stride) continue;
    sample[n * 4] = f.data[i]; sample[n * 4 + 1] = f.data[i + 1]; sample[n * 4 + 2] = f.data[i + 2]; sample[n * 4 + 3] = 255; n++;
  }
  const colors = Math.max(2, Math.min(256, Math.round(o.colors)));
  const pal = quantize(sample.subarray(0, Math.max(1, n) * 4), colors - 1, { format: 'rgb565' }) as number[][];
  const palette: number[][] = [[0, 0, 0], ...pal.map((c) => [c[0], c[1], c[2]])];
  while (palette.length < 2) palette.push([0, 0, 0]);

  const cache = new Map<number, number>();
  const nearest = (r: number, g: number, b: number) => {
    const key = (r << 16) | (g << 8) | b;
    let idx = cache.get(key);
    if (idx !== undefined) return idx;
    let best = Infinity; idx = 1;
    for (let p = 1; p < palette.length; p++) {
      const c = palette[p], dr = c[0] - r, dg = c[1] - g, db = c[2] - b;
      const dist = 2 * dr * dr + 4 * dg * dg + 3 * db * db;
      if (dist < best) { best = dist; idx = p; }
    }
    cache.set(key, idx);
    return idx;
  };

  const gif = GIFEncoder();
  for (const f of frames) {
    const index = new Uint8Array(cw * ch);
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
      const s = ((by0 + y) * W + bx0 + x) * 4;
      index[y * cw + x] = f.data[s + 3] < 128 ? 0 : nearest(f.data[s], f.data[s + 1], f.data[s + 2]);
    }
    gif.writeFrame(index, cw, ch, { palette, delay: o.delayMs, transparent: true, transparentIndex: 0, dispose: 2, repeat: 0 });
  }
  gif.finish();
  return new Blob([gif.bytes()], { type: 'image/gif' });
}
