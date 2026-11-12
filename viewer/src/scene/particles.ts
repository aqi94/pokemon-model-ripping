import * as THREE from 'three';
import { blurredMask, setFireNoise, updateFireNoise, SOFT_NEAR, SOFT_WIDE, type FireSpec, type FireUniforms, type NoiseLayer } from './fireEffect';

/**
 * UNITE idle effects the rip keeps only as empty emitter Nulls in most skins (Charizard-line tail flame, Gastly's gas).
 * Nothing here is invented (user decision 2026-11-01): the viewer draws only what another skin's copy of the SAME
 * prefab kept as real meshes (`importer.attach_fx_donors`, manifest `extras.particles[].donor`), exactly as that
 * file places them:
 *   - geometry and rest transform relative to the prefab node, from the donor FBX (plain Charmander's `CharmanderTail`
 *     for `pref_fx_0004_battlefire_lv3`, Gastly skin 10's `mesh huo1` disc for `pref_fx_Gastly_idle_lv3`);
 *   - the cut-out = the first texture of the mesh's material block (`maskChannel`: alpha, or red for AssetStudio's opaque
 *     grey masks). INFERRED: that the first texture is the mask (it is the silhouette in both cases);
 *   - colour = the mesh's mean vertex colour (Unity Fx shaders multiply by it), else the same species' HOME effect
 *     colour (`unite_home_fx.json`, the fallback the user named);
 *   - `behindBody` meshes (their bounds contain the body centre) draw before the body without depth writes, so the body
 *     is painted over them, as the footage shows for Gastly.
 * Fire look: see fireEffect.ts (blurred mask, rim = the donor's per-vertex colour, core = HOME `_LayerColor`).
 * Smoke discs (Gastly's gas) turn to face the camera about their own centre (user decision 2026-11-02; the rip has no
 * billboard flag), in `onBeforeRender` so every camera (live, PNG, GIF) sees them face-on.
 * Motion: fire records carry HOME's two scrolling noise layers (`noise`, see fireEffect.ts): the colour flickers between
 * HOME's outline and core colours inside the fixed donor silhouette. UNITE's own shader animation is not in the rip.
 * Not drawn: donor meshes with no texture of their own.
 * Emitters without a donor draw nothing.
 */
export interface DonorMesh {
  name: string;
  material: string;
  textures: string[];
  maskChannel?: 'a' | 'r';
  vertexColour?: number[] | null;
  vertexColours?: number[] | null; // RGB per vertex (linear, like the mean)
  flameT?: number[] | null; // 0 root .. 1 tip per vertex (importer.flame_coordinate)
  noiseDir?: number; // which way HOME's V scroll runs on this mesh (importer.noise_direction)
  behindBody?: boolean;
  matrix: number[]; // 4x4 row-major, relative to the prefab node
  positions: number[];
  uvs: number[] | null;
  indices: number[];
}

export interface ParticleRecord {
  prefab: string;
  bone: string;
  translation: [number, number, number];
  rotation: [number, number, number, number];
  scale: [number, number, number];
  subtreeScale?: number;
  kind?: 'fire' | 'smoke' | null;
  baseColor?: number[];
  layerColor?: number[];
  lerpProfile?: number[];
  noise?: (NoiseLayer & { key: string })[];
  noiseNorm?: number;
  noiseMean?: number;
  loop?: number;
  donor?: { model: string; meshes: DonorMesh[] };
}

export type TextureSource = (key: string) => Promise<THREE.Texture | null>;

const VERT = `
attribute vec3 vcol;
attribute float flameT;
varying vec2 vUv;
varying vec3 vCol;
varying float vT;
void main() { vUv = uv; vCol = vcol; vT = flameT; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

const FRAG = `
varying vec2 vUv;
varying vec3 vCol;
varying float vT;
uniform float uAlong;
uniform vec3 uBase;
uniform sampler2D uMask;
uniform sampler2D uNear;
uniform sampler2D uWide;
uniform float uAlphaChannel;
uniform float uUseVcol;
uniform vec3 uColor;
uniform vec3 uCore;
uniform float uFire;
uniform float uProfile[8];
uniform sampler2D uFireNoise0;
uniform sampler2D uFireNoise1;
uniform vec4 uFireNoiseST0;
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
float maskAt(vec2 uv) { vec4 t = texture2D(uMask, uv); return uAlphaChannel > 0.5 ? t.a : t.r; }
float ramp(float x) {
  float f = clamp(x, 0.0, 1.0) * 7.0;
  int i = int(min(floor(f), 6.0));
  float a = 0.0, b = 0.0;
  for (int k = 0; k < 7; k++) if (k == i) { a = uProfile[k]; b = uProfile[k + 1]; }
  return mix(a, b, f - float(i));
}
void main() {
  float m;
  vec3 rim = uUseVcol > 0.5 ? vCol : uColor;
  vec3 col = rim;
  if (uFire > 0.5) {
    // pre-blurred masks: near = alpha (soft edge, solid inside), wide = position on the colour ramp
    m = smoothstep(0.3, 0.8, texture2D(uNear, vUv).r);
    // along the flame (root 0 .. tip 1): HOME's ramp 0 (_BaseColor) is at the tip, see fireEffect.ts
    // HOME's scrolling noise ripples the colour along the ramp (fireEffect.ts header)
    float n = fireNoise(vUv);
    col = uAlong > 0.5 ? mix(uBase, uCore, ramp(1.0 - vT + n)) : mix(rim, uCore, ramp(texture2D(uWide, vUv).r + n));
  } else {
    m = maskAt(vUv);
  }
  if (m < 0.004) discard;
  gl_FragColor = vec4(col, m);
  #include <colorspace_fragment>
}`;

const _c = new THREE.Vector3();
const _n = new THREE.Vector3();
const _d = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _m = new THREE.Matrix4();
const _t = new THREE.Matrix4();
const _nm = new THREE.Matrix3();

export class ParticleEmitter {
  /** Placed at bone world x rest offset each frame; lives in the top-level scene (outside the glTF tree, so the
   *  shading, wireframe, outline and STL passes that walk the model never see it). */
  readonly group = new THREE.Group();
  /** What Render > Particles and probe renders toggle. */
  readonly mesh = new THREE.Group();
  bone?: THREE.Object3D;
  private offset = new THREE.Matrix4();
  private count = 0;
  private owned: THREE.Texture[] = []; // blurred masks made here
  /** Per fire mesh: its noise uniforms + spec (HOME speeds, this mesh's direction), advanced by `setTime`. */
  private fires: { u: FireUniforms; spec: FireSpec }[] = [];

  constructor(readonly rec: ParticleRecord, textures?: TextureSource) {
    this.group.name = `particles:${rec.prefab}`;
    this.group.matrixAutoUpdate = false;
    this.group.add(this.mesh);
    this.mesh.userData.particles = true;
    this.offset.compose(new THREE.Vector3().fromArray(rec.translation), new THREE.Quaternion().fromArray(rec.rotation),
      new THREE.Vector3().fromArray(rec.scale));
    for (const dm of rec.donor?.meshes ?? []) this.addMesh(dm, textures);
  }

  get drawable() { return this.count > 0; }

  private addMesh(dm: DonorMesh, textures?: TextureSource) {
    if (!dm.textures.length || !dm.uvs) return;
    const vc = dm.vertexColour;
    let color: THREE.Color;
    if (vc) color = new THREE.Color(vc[0], vc[1], vc[2]); // linear, as the importer's vertex-colour tint
    else if (this.rec.baseColor) color = new THREE.Color().setRGB(this.rec.baseColor[0], this.rec.baseColor[1], this.rec.baseColor[2], THREE.SRGBColorSpace);
    else return; // no colour in UNITE or HOME: not drawn
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(dm.positions, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(dm.uvs, 2));
    const nv = dm.positions.length / 3;
    const perVertex = !!dm.vertexColours && dm.vertexColours.length === nv * 3;
    g.setAttribute('vcol', new THREE.Float32BufferAttribute(perVertex ? dm.vertexColours! : new Array(nv * 3).fill(0), 3));
    const along = !!dm.flameT && dm.flameT.length === nv && this.rec.kind === 'fire';
    g.setAttribute('flameT', new THREE.Float32BufferAttribute(along ? dm.flameT! : new Array(nv).fill(0), 1));
    g.setIndex(dm.indices);
    const lc = this.rec.layerColor;
    const prof = this.rec.lerpProfile;
    const fire = this.rec.kind === 'fire' && !!lc && prof?.length === 8;
    const mat = new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, side: THREE.DoubleSide, depthWrite: false, transparent: true,
      uniforms: {
        uMask: { value: null }, uAlphaChannel: { value: dm.maskChannel === 'a' ? 1 : 0 }, uColor: { value: color },
        uNear: { value: null }, uWide: { value: null },
        uUseVcol: { value: perVertex ? 1 : 0 }, uFire: { value: fire ? 1 : 0 },
        uAlong: { value: fire && along ? 1 : 0 },
        uBase: { value: this.rec.baseColor ? new THREE.Color().setRGB(this.rec.baseColor[0], this.rec.baseColor[1], this.rec.baseColor[2], THREE.SRGBColorSpace) : new THREE.Color() },
        uCore: { value: lc ? new THREE.Color().setRGB(lc[0], lc[1], lc[2], THREE.SRGBColorSpace) : new THREE.Color() },
        uProfile: { value: fire ? prof!.slice() : new Array(8).fill(0) },
        uFireNoise0: { value: null }, uFireNoise1: { value: null },
        uFireNoiseST0: { value: new THREE.Vector4(1, 1, 0, 0) }, uFireNoiseST1: { value: new THREE.Vector4(1, 1, 0, 0) },
        uFireNoiseNorm: { value: this.rec.noiseNorm ?? 1 }, uFireNoiseMean: { value: this.rec.noiseMean ?? 0.5 }, uFireNoiseOn: { value: 0 },
      },
    });
    const noise = this.rec.noise;
    if (fire && noise?.length === 2 && textures) {
      const spec: FireSpec = { layerColor: lc!, noise, noiseNorm: this.rec.noiseNorm, noiseMean: this.rec.noiseMean, noiseDir: dm.noiseDir ?? 1 };
      const u = mat.uniforms as unknown as FireUniforms;
      this.fires.push({ u, spec });
      void Promise.all(noise.map((n) => textures(n.key))).then((maps) => {
        const own = maps.map((t) => t?.clone() ?? null);
        this.owned.push(...own.filter((t): t is THREE.Texture => !!t));
        setFireNoise(u, spec, own);
      });
    }
    const obj = new THREE.Mesh(g, mat);
    obj.frustumCulled = false;
    obj.visible = false; // until its texture arrives
    if (dm.behindBody) {
      // opaque pass, before the body, no depth writes: the body is painted over it
      mat.transparent = false;
      mat.blending = THREE.CustomBlending;
      mat.blendSrc = THREE.SrcAlphaFactor; mat.blendDst = THREE.OneMinusSrcAlphaFactor;
      obj.renderOrder = -10;
    }
    obj.matrixAutoUpdate = false;
    obj.matrix.fromArray(dm.matrix).transpose(); // row-major -> three's column-major
    if (this.rec.kind === 'smoke') this.billboard(obj, dm);
    this.mesh.add(obj);
    this.count++;
    void textures?.(dm.textures[0]).then((t) => {
      if (!t) return;
      if (this.rec.kind === 'smoke') {
        // Polar UVs (U around, V centre -> rim): V is radial and must not wrap. With the GLB's repeat sampler,
        // filtering at the rim (V = 0) blended the mask's black top row with its white bottom row and drew a thin
        // solid circle at the disc's edge, outside the lumpy cut-out (no such line in the footage).
        t = t.clone();
        t.wrapS = THREE.RepeatWrapping;
        t.wrapT = THREE.ClampToEdgeWrapping;
        this.owned.push(t);
      }
      t.needsUpdate = true;
      mat.uniforms.uMask.value = t;
      if (fire) {
        const ch = dm.maskChannel === 'a' ? 'a' : 'r';
        const near = blurredMask(t, ch, SOFT_NEAR), wide = blurredMask(t, ch, SOFT_WIDE);
        if (!near || !wide) return;
        mat.uniforms.uNear.value = near;
        mat.uniforms.uWide.value = wide;
        this.owned.push(near, wide);
      }
      obj.visible = true;
    });
  }

  /** Turn a flat disc to face the drawing camera about its own centre (Gastly's gas). three multiplies matrixWorld
   *  into modelViewMatrix right after `onBeforeRender`, so rewriting matrixWorld there takes effect for that draw. */
  private billboard(obj: THREE.Mesh, dm: DonorMesh) {
    const p = dm.positions;
    const nv = p.length / 3;
    if (nv < 3) return;
    const centre = new THREE.Vector3();
    for (let i = 0; i < nv; i++) centre.add(_c.set(p[3 * i], p[3 * i + 1], p[3 * i + 2]));
    centre.divideScalar(nv);
    // plane normal: the largest triangle's (a flat disc, so any triangle would do)
    const normal = new THREE.Vector3();
    let best = 0;
    const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
    for (let t = 0; t + 2 < dm.indices.length; t += 3) {
      a.fromArray(p, 3 * dm.indices[t]); b.fromArray(p, 3 * dm.indices[t + 1]); c.fromArray(p, 3 * dm.indices[t + 2]);
      const cr = b.sub(a).cross(c.sub(a));
      const l = cr.length();
      if (l > best) { best = l; normal.copy(cr).divideScalar(l); }
    }
    if (best === 0) return;
    obj.onBeforeRender = (_r, _s, camera) => {
      // from the parent each draw: a frame may draw the disc more than once (sprite/probe passes)
      if (obj.parent) obj.matrixWorld.multiplyMatrices(obj.parent.matrixWorld, obj.matrix);
      _c.copy(centre).applyMatrix4(obj.matrixWorld);
      _n.copy(normal).applyMatrix3(_nm.getNormalMatrix(obj.matrixWorld)).normalize();
      if ((camera as THREE.OrthographicCamera).isOrthographicCamera) camera.getWorldDirection(_d).negate();
      else _d.setFromMatrixPosition(camera.matrixWorld).sub(_c).normalize();
      if (_n.dot(_d) < 0) _n.negate(); // double-sided: take the nearer face, so the disc never spins half a turn
      _q.setFromUnitVectors(_n, _d);
      _m.makeTranslation(_c.x, _c.y, _c.z).multiply(_t.makeRotationFromQuaternion(_q))
        .multiply(_t.makeTranslation(-_c.x, -_c.y, -_c.z));
      obj.matrixWorld.premultiply(_m);
    };
  }

  /** Effect clock (seconds): scrolls HOME's noise layers. */
  setTime(t: number) {
    for (const f of this.fires) updateFireNoise(f.u, f.spec, t);
  }

  /** Follow the bone (the effect has no animation of its own in the data). */
  update() {
    if (!this.bone) return;
    this.bone.updateWorldMatrix(true, false);
    this.group.matrix.multiplyMatrices(this.bone.matrixWorld, this.offset);
    let vis = true;
    for (let o: THREE.Object3D | null = this.bone; o; o = o.parent) if (!o.visible) vis = false;
    this.group.visible = vis;
    this.group.updateMatrixWorld(true);
  }

  dispose() {
    this.group.removeFromParent();
    for (const t of this.owned) t.dispose();
    this.group.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh) { m.geometry.dispose(); (m.material as THREE.Material).dispose(); }
    });
  }
}

/** Build the emitters that have donor geometry (bones found by sanitised name); add each `group` to the scene. */
export function buildParticles(root: THREE.Object3D, records: ParticleRecord[] | undefined, textures?: TextureSource): ParticleEmitter[] {
  const out: ParticleEmitter[] = [];
  for (const rec of records ?? []) {
    if (!rec.donor?.meshes.length) continue;
    const want = THREE.PropertyBinding.sanitizeNodeName(rec.bone);
    let bone: THREE.Object3D | undefined;
    root.traverse((o) => { if (!bone && (o as THREE.Bone).isBone && o.name === want) bone = o; });
    // emitter-only joints (Charmeleon's animated `Bone_Tail06` Null skins nothing, so glTF writes it as a plain node)
    if (!bone) root.traverse((o) => { if (!bone && o.name === want) bone = o; });
    if (!bone) { console.warn('particles: bone not found', rec.bone); continue; }
    const e = new ParticleEmitter(rec, textures);
    if (!e.drawable) { e.dispose(); continue; }
    e.bone = bone;
    out.push(e);
  }
  return out;
}
