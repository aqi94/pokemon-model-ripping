import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { Manifest, PropInfo } from '../types';
import { AnimationController, type PlaybackState } from './animation';
import { VariantSwitcher } from './variants';
import { loadGlb } from './loader';
import { download, exportCanvasPng } from './exporter';
import { PartHighlight } from './partHighlight';
import { buildStl, DEFAULT_STL, type StlOptions } from './stl';
import { DEFAULT_EXPORT, MAX_EXPORT_PX, type ExportOptions, type GlbExportOptions } from './exportOptions';
import type { GlbExportInput, PrimView } from './glbExport';
import { DEFAULT_ENV, DEFAULT_HABLE, kelvinToRGB, type Environment, type ToneMap } from './environment';
import { installCustomTone, installHable } from './hable';
import { installUniteShading, UNITE_K, UNITE_ON, UNITE_SCENE, UNITE_TONE_GLSL, uniteShadingKey, uniteSwitch } from './uniteShading';
import { applyGameShading, removeGameShading, syncUnlit } from './gameShading';
import { isReflective, makeReflectionEnv } from './reflection';
import { bakePalette, type PaletteData } from './palette';
import { installGas, setGasTexture, updateGas } from './gas';
import { buildParticles, type ParticleEmitter } from './particles';
import { installHologram, setHologram, updateHologram, type HoloParams, type HologramState } from './hologram';
import { COAT_ENV_INTENSITY, installCoatReflection, installEyeHighlight, setEyeHighlightNormal, setEyeHighlightOn } from './eyeHighlight';
import { defaultMatcap, installMatcap, isMatcapReflect, loadUniteMatcaps, setMatcapTexture, uniteKey } from './matcap';
import { installSphereMap } from './sphereMap';
import { installFire, installHomeFire, updateFireNoise, updateHomeFire, type FireSpec, type FireUniforms } from './fireEffect';
import type { SurfaceAnchor } from './refImage';
import { DEFAULT_OUTLINE, gameOutlineOf, makeOutlineMesh, modelHasGameOutline, updateOutlineUniform, type OutlineState } from './outline';
import { SpritePass, encodeSpriteGif, type SpriteFrame } from './spriteStyle';

export type ShadingMode = 'game' | 'pbr';
export type CameraPreset = 'fit' | 'vs' | 'front' | 'side' | 'back' | 'top' | 'idle' | 'battle' | 'battleBack';
/** Vertical FOV (deg) of every framing preset except the near-orthographic Idle shot; presets reset to it (an Idle shot
 *  or a lens solve leaves a narrow FOV behind, which used to make the next preset's fit distance exceed maxDistance). */
// FITTED: DEFAULT_FOV 35 deg chosen by eye (viewer default since the first version), 2026-10-19 (engine-rules debt)
export const DEFAULT_FOV = 35;
// FITTED: battle camera = Pokemon Showdown sprites/ani (front) + ani-back, read off pikachu/charizard/garchomp/snorlax/eevee:
// front yaw 20 deg toward the viewer's left, pitch 12 deg; back yaw 20 deg toward the right, pitch 22 deg; mild perspective
// fov 20; subject fills ~90% of the frame (bounding-sphere fit * 0.9). 2026-10-19
const BATTLE_SHOTS = {
  battle: { yaw: 20, pitch: 12 },
  battleBack: { yaw: 180 + 20, pitch: 22 },
} as const;
const BATTLE_FOV = 20;
const BATTLE_FILL = 0.9;
/** Camera as the Camera panel edits it: orbit around `target` plus world-space position (metres, degrees).
 *  yaw 0 = camera in front of the model (+Z), +90 = on its +X side (mirrored with the model); pitch + = looking down. */
export interface CameraOrbit {
  yaw: number; pitch: number; distance: number; roll: number; fov: number;
  target: [number, number, number]; position: [number, number, number];
}
/** Pitch stays off the poles, where OrbitControls' up vector degenerates. */
export const MAX_PITCH = 89.9;
function wrapDeg(d: number): number { return ((((d + 180) % 360) + 360) % 360) - 180; }
export type AlbedoMode = 'baked' | 'raw' | 'off';

const TONE_MAPS: Record<ToneMap, THREE.ToneMapping> = {
  linear: THREE.LinearToneMapping, aces: THREE.ACESFilmicToneMapping, agx: THREE.AgXToneMapping, neutral: THREE.NeutralToneMapping,
  hable: THREE.CustomToneMapping, // Hable filmic with the game's parameters, see hable.ts
  unite: THREE.CustomToneMapping, // UNITE's post curve, see uniteShading.ts
};

interface AlbedoState { baked: THREE.Texture | null; color: THREE.Color; raw?: THREE.Texture | null }

export class ViewerScene {
  static THREE = THREE; // for page.evaluate tooling (tools/*.mjs) that needs the viewer's own three instance
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(DEFAULT_FOV, 1, 0.01, 1000);
  controls: OrbitControls;
  root = new THREE.Group(); // mirror flips this
  gltf?: GLTF;
  anim?: AnimationController;
  variants?: VariantSwitcher;
  skeletonHelper?: THREE.SkeletonHelper;
  manifest?: Manifest;
  onFrame?: (s: PlaybackState) => void;
  /** Effective per-part visibility and the nodes currently driven by the playing clip. */
  onPartsState?: (visible: Record<string, boolean>, driven: string[]) => void;
  private userOverride = new Map<string, boolean>();
  private visKey = '';
  private lockPartsToClip = false;
  private uvMats = new Map<string, THREE.MeshStandardMaterial[]>();
  private uvDriven = new Set<string>();
  /** Sign of the eye-gaze texture offset on x / y (the shader is not in the rip: set empirically; three.js moves the image opposite to the offset, so the pupil follows the game offset with x negated and y as-is). */
  uvSign = { x: -1, y: 1 };
  private nodeCache = new Map<string, THREE.Object3D>();

  private reflEnv?: THREE.Texture;
  private held?: { obj: THREE.Object3D; prop: PropInfo; url: string; parser?: GLTF['parser'] };
  private env: Environment = { ...DEFAULT_ENV };
  private texCache = new WeakMap<THREE.Material, Record<string, THREE.Texture | null | 'pending'>>();
  private heldGen = 0;
  private albedoMode: AlbedoMode = 'baked';
  private albedoGen = 0;
  private albedoState = new WeakMap<THREE.Material, AlbedoState>();
  private heldOnlyInClips = true;
  private shadingMode: ShadingMode = 'game';
  /** Gas-shell materials (Gastly smoke) whose cloud map pans with `gasTime` (seconds of unpaused playback). */
  private gasMats = new Set<THREE.MeshStandardMaterial>();
  gasTime = 0;
  /** UNITE idle effects drawn from another skin's copy of the same prefab (particles.ts); they follow their bone. */
  private particles: ParticleEmitter[] = [];
  private particlesOn = true;
  /** UNITE flame materials with HOME's scrolling noise (fireEffect.ts), and the effect clock that moves them and the
   *  particle flames (seconds of unpaused playback, like `gasTime`). Render tab > Animate effects freezes it. */
  private fireMats = new Set<THREE.MeshStandardMaterial>();
  fxTime = 0;
  private fxAnimate = true;
  /** Appearance > Hologram (null/off = normal look); materials carrying the patch, for the scroll clock. */
  private hologram: HologramState | null = null;
  private holoMats = new Set<THREE.MeshStandardMaterial>();
  private holoTime = 0;
  private pbrLights: THREE.Object3D[] = [];
  private gameLights: THREE.Object3D[] = [];
  private lightFrame = { center: new THREE.Vector3(), radius: 1 };

  private clock = new THREE.Clock();
  private raf = 0;
  private key = new THREE.DirectionalLight(0xffffff, 2.2);
  private back = new THREE.DirectionalLight(0xffffff, 0);
  private hableKey = '';
  private ground: THREE.Mesh;
  private wireframe = false;
  private disposed = false;
  private lastEmit = 0;
  private ro: ResizeObserver;
  /** Fly navigation for large static scenes (kind "map"): WASD move, Q/E down/up, Shift = 4x. Orbit/pan/zoom still work. */
  private flyEnabled = false;
  private flyKeys = new Set<string>();
  private onFlyKey = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement | null;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
    const k = e.key.toLowerCase();
    if (!'wasdqe'.includes(k) || k.length !== 1) { if (k === 'shift') { if (e.type === 'keydown') this.flyKeys.add(k); else this.flyKeys.delete(k); } return; }
    if (e.type === 'keydown' && !e.ctrlKey && !e.metaKey) this.flyKeys.add(k); else this.flyKeys.delete(k);
  };
  private onFlyBlur = () => this.flyKeys.clear();

  /** `low` = reduced GPU load (no MSAA, pixel ratio 1, 1024 shadow map), used after a lost WebGL context. */
  constructor(private container: HTMLElement, private low = false) {
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: !low, alpha: true });
    } catch (e) {
      // e.g. Firefox/ANGLE "EGL_NO_CONFIG": the multisampled config is unavailable. Retry without MSAA, in low mode.
      if (low) throw e;
      this.low = true;
      try { sessionStorage.setItem('lowGpu', '1'); } catch { /* ignore */ }
      this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
    } // alpha: transparent PNG export
    this.renderer.setPixelRatio(this.low ? 1 : Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.LinearToneMapping; // linear = no curve, but honours toneMappingExposure
    this.renderer.toneMappingExposure = 1;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.addEventListener('webglcontextlost', this.onContextLost, false);
    this.renderer.domElement.addEventListener('webglcontextrestored', this.onContextRestored, false);
    this.scene.background = this.bgColor;
    this.scene.add(this.root);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x555566, 1.2);
    this.scene.add(hemi);
    this.key.position.set(3, 6, 4);
    this.key.castShadow = false;
    this.key.shadow.mapSize.set(this.low ? 1024 : 2048, this.low ? 1024 : 2048);
    this.key.shadow.bias = -0.0005;
    this.scene.add(this.key, this.key.target);
    this.scene.add(this.back, this.back.target);
    const fill = new THREE.DirectionalLight(0xaabbff, 0.6);
    fill.position.set(-4, 2, -3);
    this.scene.add(fill);
    this.pbrLights = [hemi, fill];
    const gameHemi = new THREE.HemisphereLight(0xffffff, 0xf6ecd8, 1.9);
    this.scene.add(gameHemi);
    this.gameLights = [gameHemi];
    this.applyRig();

    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShadowMaterial({ opacity: 0.35 }));
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.receiveShadow = true;
    this.ground.visible = false;
    this.scene.add(this.ground);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    const bufSize = new THREE.Vector2();
    // Hull width follows the target being drawn (the canvas, or the GIF export's render target).
    this.scene.onBeforeRender = (r) => {
      const rt = r.getRenderTarget();
      if (rt) bufSize.set(rt.width, rt.height); else r.getDrawingBufferSize(bufSize);
      updateOutlineUniform(this.outline.width, bufSize.x, bufSize.y);
    };
    window.addEventListener('keydown', this.onFlyKey);
    window.addEventListener('keyup', this.onFlyKey);
    window.addEventListener('blur', this.onFlyBlur);
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.resize();
    this.loop();
  }

  private resize() {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  async load(manifest: Manifest, glbUrl: string, onProgress?: (f: number) => void) {
    this.jawRigs = []; this.jawOpen = 0;
    this.manifest = manifest;
    const gltf = await loadGlb(glbUrl, onProgress);
    if (this.disposed) return;
    this.gltf = gltf;
    this.outlineMapCache.clear();
    this.holoMats.clear(); this.holoTime = 0;
    this.fireMats.clear(); this.fxTime = 0;
    this.root.add(gltf.scene);
    const staticScene = manifest.kind === 'map' || manifest.kind === 'prop';
    this.flyEnabled = manifest.kind === 'map';
    this.flyKeys.clear();

    // LOD>0 and hidden-by-default nodes
    for (const p of manifest.parts) {
      const obj = this.findNode(p.node);
      if (obj && (p.lod > 0 || !p.visible)) obj.visible = false;
    }
    gltf.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      m.castShadow = true;
      // static scenes are large and unskinned: let three cull off-screen meshes (skinned bounds are not reliable)
      m.frustumCulled = staticScene && !(m as THREE.SkinnedMesh).isSkinnedMesh;
      for (const mat of ([] as THREE.Material[]).concat(m.material)) {
        const s = mat as THREE.MeshStandardMaterial;
        if ('envMapIntensity' in s) s.envMapIntensity = 0.2;
        applyGameBlend(mat);
      }
    });

    this.variants = new VariantSwitcher(gltf);
    this.variants.onMaterialsChanged = () => { this.applyWireframe(); this.applyShading(); void this.applyAlbedo(); };
    if (this.variants.available) {
      for (const g of manifest.variantGroups) {
        const def = g.options.find((o) => o.default)?.variant ?? g.options[0]?.variant ?? 0;
        await this.variants.select(def, g.options.map((o) => o.variant));
      }
    }

    this.applyShading();
    this.anim = new AnimationController(gltf.scene, gltf.animations, manifest.clips);
    const start = manifest.defaultClip ?? manifest.clips.find((c) => c.default)?.name ?? manifest.clips[0]?.name;
    this.defaultClipName = start;
    if (start && this.anim.has(start)) this.anim.play(start);

    this.applyWireframe();
    this.uvMats.clear(); this.uvDriven.clear();
    this.gasTime = 0;
    this.buildParticles(manifest);
    this.updateVisibility();
    this.applyUvOffsets();
    this.recenter();
    if (manifest.kind === 'pokemon' && !this.pendingPreset) this.pendingPreset = 'battle';
    if (this.pendingPreset) { const p = this.pendingPreset; this.pendingPreset = undefined; this.setCameraPreset(p); }
  }

  private buildParticles(manifest: Manifest) {
    for (const e of this.particles) e.dispose();
    this.particles = [];
    const recs = manifest.extras?.particles;
    if (!this.gltf || !recs?.length) return;
    // effect textures stored in the GLB without a material (root extras.fxTextures: key -> texture index)
    const parser = this.gltf.parser;
    const fx = (parser.json?.extras?.fxTextures ?? {}) as Record<string, number>;
    const textures = async (key: string) => {
      const i = fx[key];
      if (i === undefined) return null;
      try { return (await parser.getDependency('texture', i)) as THREE.Texture; } catch { return null; }
    };
    this.particles = buildParticles(this.gltf.scene, recs, textures);
    for (const e of this.particles) { e.mesh.visible = this.particlesOn; this.scene.add(e.group); e.update(); }
  }

  /** Render tab > Animate effects: off freezes the flame noise at its current time. */
  setEffectsAnimated(on: boolean) { this.fxAnimate = on; }

  /** Render tab > Particles. */
  setParticles(on: boolean) {
    this.particlesOn = on;
    for (const e of this.particles) e.mesh.visible = on;
  }

  /** GLTFLoader de-duplicates names (mesh node "x" becomes "x_1" when the mesh is also called "x"), so match both. */
  findNode(name: string): THREE.Object3D | undefined {
    const want = new Set([THREE.PropertyBinding.sanitizeNodeName(name), THREE.PropertyBinding.sanitizeNodeName(name) + '_1']);
    let found: THREE.Object3D | undefined;
    this.gltf?.scene.traverse((o) => {
      if (!found && !(o as THREE.Bone).isBone && want.has(o.name) && ((o as THREE.Mesh).isMesh || o.type === 'Group')) found = o;
    });
    return found;
  }

  /** Model Parts hover (partHighlight.ts): the part's node, its meshes, and ghost materials while the part is hidden. */
  private hover?: { node: THREE.Object3D; meshes: Set<THREE.Mesh>; ghosts: Map<THREE.Mesh, THREE.Material | THREE.Material[]> };
  private highlight?: PartHighlight;

  /** Highlight a part from the Model Parts list (null clears). Applied per frame in the live loop only, so the parts state,
   *  saved looks and every export (which render outside the loop) never see it. */
  setHoverPart(node: string | null) {
    if (this.hover) {
      for (const g of this.hover.ghosts.values()) ([] as THREE.Material[]).concat(g).forEach((m) => m.dispose());
      this.hover = undefined;
    }
    const obj = node ? this.findNode(node) : undefined;
    if (!obj) return;
    const meshes = new Set<THREE.Mesh>();
    obj.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh && !m.userData.outline && !m.userData.holoPrepass) meshes.add(m); });
    this.hover = { node: obj, meshes, ghosts: new Map() };
  }

  private renderHovered() {
    const h = this.hover!;
    // a hidden part is drawn with its own materials at 5 % opacity
    let ghost = false;
    for (let p: THREE.Object3D | null = h.node; p && p !== this.scene; p = p.parent) if (!p.visible) ghost = true;
    const nodeWas = h.node.visible;
    const saved = new Map<THREE.Mesh, THREE.Material | THREE.Material[]>();
    const helpers: THREE.Object3D[] = [];
    if (ghost) {
      h.node.visible = true;
      for (const m of h.meshes) {
        let g = h.ghosts.get(m);
        if (!g) {
          const mk = (x: THREE.Material) => { const c = x.clone(); c.transparent = true; c.opacity = 0.05 * x.opacity; c.depthWrite = false; return c; };
          g = Array.isArray(m.material) ? m.material.map(mk) : mk(m.material);
          h.ghosts.set(m, g);
        }
        saved.set(m, m.material);
        m.material = g;
      }
      h.node.traverse((o) => { if (o.userData.outline && o.visible) { helpers.push(o); o.visible = false; } });
    }
    if (this.spriteLive()) this.renderSpriteLive();
    else this.renderer.render(this.scene, this.camera);
    for (const [m, mat] of saved) m.material = mat;
    helpers.forEach((o) => { o.visible = true; });
    h.node.visible = nodeWas;
    this.highlight ??= new PartHighlight();
    this.highlight.render(this.renderer, this.camera, this.root, h.meshes, ghost ? h.node : null, performance.now() / 1000);
  }

  setPartVisible(node: string, v: boolean) {
    this.userOverride.set(node, v);
    this.updateVisibility();
  }

  /** The part checkboxes / group options the user set (node -> visible); saved looks store these, not clip-driven state. */
  partOverrides(): Record<string, boolean> { return Object.fromEntries(this.userOverride); }
  clearPartOverrides() { this.userOverride.clear(); this.updateVisibility(); }

  /** When true the playing clip's visibility keys beat the user's checkboxes (game behaviour); when false the checkboxes win. */
  setLockPartsToClip(lock: boolean) {
    this.lockPartsToClip = lock;
    this.updateVisibility();
  }

  /**
   * Material texture-offset animation of the playing clip (BDSP eye gaze: the pupils sit in an atlas and the game slides it). Each driven
   * material gets its own clone of its map (the eyes share one texture), sampled linearly at the clip time; materials the clip does not
   * drive go back to offset 0.
   */
  private applyUvOffsets() {
    const man = this.manifest;
    if (!man || !this.gltf) return;
    const cur = this.anim?.current;
    const info = cur ? man.clips.find((c) => c.name === cur) : undefined;
    const tracks = info?.uvOffset;
    // Trinity eye atlases (.tracm UVScaleOffset): glTF-space offsets that snap between cells.
    const step = info?.uvOffsetMode === 'gltf_step';
    const time = this.anim ? this.anim.state().time : 0;
    const names = new Set<string>([...this.uvDriven, ...Object.keys(tracks ?? {})]);
    for (const name of names) {
      let mats = this.uvMats.get(name);
      if (!mats) {
        mats = [];
        this.gltf.scene.traverse((o) => {
          const m = o as THREE.Mesh;
          if (!m.isMesh) return;
          for (const mat of ([] as THREE.Material[]).concat(m.material)) if ((mat.name === name || mat.name.startsWith(name + '@')) && (mat as THREE.MeshStandardMaterial).isMeshStandardMaterial && !mats!.includes(mat as THREE.MeshStandardMaterial)) mats!.push(mat as THREE.MeshStandardMaterial);
        });
        this.uvMats.set(name, mats);
      }
      const tr = tracks?.[name];
      let x = 0, y = 0;
      if (tr && tr.times.length) {
        const t = tr.times, v = tr.values;
        let i = 0;
        while (i + 1 < t.length && t[i + 1] <= time) i++;
        if (!step && i + 1 < t.length && t[i + 1] > t[i]) {
          const k = Math.min(1, Math.max(0, (time - t[i]) / (t[i + 1] - t[i])));
          x = v[i][0] + (v[i + 1][0] - v[i][0]) * k; y = v[i][1] + (v[i + 1][1] - v[i][1]) * k;
        } else { x = v[i][0]; y = v[i][1]; }
        this.uvDriven.add(name);
      }
      for (const mat of mats) {
        const map = mat.map;
        if (!map) continue;
        let own = map;
        if (!map.userData.__uvOwned) { own = map.clone(); own.userData.__uvOwned = true; own.needsUpdate = true; mat.map = own; }
        const ox = step ? x : x * this.uvSign.x, oy = step ? y : y * this.uvSign.y;
        if (own.offset.x !== ox || own.offset.y !== oy) own.offset.set(ox, oy);
      }
    }
  }

  /** Rest visibility (manifest) -> user override -> explicit per-clip key at the current frame. LOD>0 stays hidden. */
  private updateVisibility() {
    const man = this.manifest;
    if (!man || !this.gltf) return;
    const cur = this.anim?.current;
    const info = cur ? man.clips.find((c) => c.name === cur) : undefined;
    const tracks = info?.visibility;
    const frame = tracks && this.anim ? this.anim.state().time * (info!.frameRate || 30) : 0;
    const vis: Record<string, boolean> = {};
    const driven: string[] = [];
    for (const p of man.parts) {
      let o = this.nodeCache.get(p.node);
      if (!o) { o = this.findNode(p.node); if (o) this.nodeCache.set(p.node, o); }
      if (!o) continue;
      let v: boolean;
      if (p.lod > 0) v = false;
      else {
        v = this.userOverride.get(p.node) ?? p.visible;
        const t = !this.lockPartsToClip && this.userOverride.has(p.node) ? undefined : tracks?.[p.node];
        if (t && t.length) {
          let last = t[0][1];
          for (const [f, val] of t) if (f <= frame + 1e-6) last = val; else break;
          // a clip key always wins; it only counts as "driving" (locks the checkbox) when it overrides what the user/rest state would show
          if ((last !== 0) !== v) driven.push(p.node);
          v = last !== 0;
        }
      }
      if (o.visible !== v) o.visible = v;
      if (p.lod === 0) vis[p.node] = v;
    }
    const key = JSON.stringify(vis) + '|' + driven.join(',');
    if (key !== this.visKey) { this.visKey = key; this.onPartsState?.(vis, driven); }
  }

  /** Distance at which a sphere of `radius` fills the narrower of the vertical / horizontal FOV (narrow canvases do not crop). */
  private fitDistance(radius: number): number {
    const v = THREE.MathUtils.degToRad(this.camera.fov);
    const h = 2 * Math.atan(Math.tan(v / 2) * (this.camera.aspect || 1));
    return (radius / Math.sin(Math.min(v, h) / 2)) * 0.95;
  }

  /** Clip planes + zoom limits that follow the camera distance `d` (a narrow FOV puts the camera far away). */
  // FITTED: zoom-out 10x / 20r, far +20r, near 1%, idle planes 0.05d..10d chosen by eye (no data), 2026-10-19 (engine-rules debt)
  private applyLimits(radius: number, d: number) {
    this.camera.near = Math.min(radius / 100, d * 0.01);
    this.camera.far = Math.max(radius * 100, d + radius * 20);
    this.controls.minDistance = Math.min(radius * 0.3, d * 0.5);
    this.controls.maxDistance = Math.max(radius * 20, d * 10);
    this.camera.updateProjectionMatrix();
  }

  /** Fit camera to visible meshes (LOD0), at `fov` (the default FOV unless an export asks for another). */
  recenter(fov = DEFAULT_FOV) {
    this.releaseLens();
    if (!this.exporting) this.userRoll = 0;
    if (!this.gltf) return;
    this.camera.fov = fov;
    this.camera.updateProjectionMatrix();
    this.root.updateMatrixWorld(true);
    const box = new THREE.Box3();
    this.gltf.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      for (let p: THREE.Object3D | null = m; p; p = p.parent) if (!p.visible) return;
      const g = m.geometry;
      let bb: THREE.Box3 | null = null;
      if ((m as THREE.SkinnedMesh).isSkinnedMesh) {
        const sm = m as THREE.SkinnedMesh;
        sm.computeBoundingBox();
        bb = sm.boundingBox;
      }
      if (!bb) { if (!g.boundingBox) g.computeBoundingBox(); bb = g.boundingBox; }
      if (bb) box.union(bb.clone().applyMatrix4(m.matrixWorld));
    });
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 1e-3);
    const map = this.manifest?.kind === 'map';
    // maps: a high three-quarter view of the whole area (like the game's overworld camera, pulled back); others: low 3/4
    // FITTED: map view direction / 0.8 distance chosen by eye on LGPE area02, 2026-10-20
    const dist = this.fitDistance(radius) * (map ? 0.8 : 1);
    this.applyLimits(radius, dist);
    const dir = (map ? new THREE.Vector3(0.35, 0.9, 1) : new THREE.Vector3(0.6, 0.25, 1)).normalize();
    this.camera.position.copy(center).addScaledVector(dir, dist);
    this.controls.target.copy(center);
    this.controls.update();

    this.ground.scale.setScalar(radius * 6);
    this.ground.position.set(center.x, box.min.y - radius * 0.002, center.z);
    this.lightFrame = { center: center.clone(), radius };
    this.positionKey();
    this.key.target.position.copy(center);
    this.back.target.position.copy(center);
    const sc = this.key.shadow.camera;
    sc.left = sc.bottom = -radius * 1.6;
    sc.right = sc.top = radius * 1.6;
    sc.near = radius * 0.1;
    sc.far = radius * 12;
    sc.updateProjectionMatrix();
  }

  // ---- jaw slider (UNITE mouths are bone-driven) ----
  private jawOpen = 0;
  private jawRigs: { bone: THREE.Object3D; delta: THREE.Quaternion; base: THREE.Quaternion; out: THREE.Quaternion }[] = [];

  /** Extra jaw opening on top of the playing clip: 0 = as animated, 1 = the widest pose any clip reaches (manifest extras.jaws). */
  setJawOpen(v: number) {
    this.jawOpen = v;
    if (!this.jawRigs.length) {
      for (const j of this.manifest?.extras?.jaws ?? []) {
        const bone = this.findBone(j.bone);
        if (bone) this.jawRigs.push({ bone, delta: new THREE.Quaternion(...j.open), base: new THREE.Quaternion(), out: new THREE.Quaternion() });
      }
    }
    this.applyJaw();
  }

  /** The mixer only rewrites a bone it has a track for, so remember the animated pose and re-derive from it every frame. */
  private applyJaw() {
    const id = new THREE.Quaternion();
    for (const r of this.jawRigs) {
      if (!r.bone.quaternion.equals(r.out)) r.base.copy(r.bone.quaternion); // the clip wrote it since we last did
      r.out.copy(r.base).multiply(id.identity().slerp(r.delta, this.jawOpen));
      r.bone.quaternion.copy(r.out);
    }
  }

  /** Find a skeleton bone by exact name (GLTFLoader may suffix duplicates with _1). */
  findBone(name: string): THREE.Object3D | undefined {
    const n = THREE.PropertyBinding.sanitizeNodeName(name);
    const want = new Set([n, n + '_1']);
    let found: THREE.Object3D | undefined;
    this.gltf?.scene.traverse((o) => { if (!found && want.has(o.name) && !this.isInHeld(o)) found = o; });
    return found;
  }

  private isInHeld(o: THREE.Object3D) {
    for (let p: THREE.Object3D | null = o; p; p = p.parent) if (p === this.held?.obj) return true;
    return false;
  }

  /** Attach a static GLB to a prop bone; url=null removes. Reuses the loaded model when only the hand changes. */
  async setHeldItem(url: string | null, prop: PropInfo | undefined) {
    const gen = ++this.heldGen;
    if (!url || !prop) { this.clearHeld(); return; }
    let obj = this.held?.url === url ? this.held.obj : undefined;
    let parser = this.held?.url === url ? this.held.parser : undefined;
    if (!obj) {
      const g = await loadGlb(url);
      if (gen !== this.heldGen || this.disposed) { disposeTree(g.scene); return; }
      this.clearHeld();
      obj = g.scene;
      parser = g.parser;
      obj.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) { m.frustumCulled = false; m.castShadow = true; } });
    }
    const bone = this.findBone(prop.bone);
    if (!bone) { console.warn('prop bone not found', prop.bone); this.clearHeld(); return; }
    obj.removeFromParent();
    obj.position.set(0, 0, 0); obj.quaternion.identity(); obj.scale.set(1, 1, 1);
    bone.add(obj);
    this.held = { obj, prop, url, parser };
    this.applyShading();
    this.applyHeldVisibility();
    void this.applyAlbedo();
  }

  private clearHeld() {
    if (this.held) { this.held.obj.removeFromParent(); disposeTree(this.held.obj); this.held = undefined; }
  }

  /** A held item is attached. */
  hasHeld() { return !!this.held; }

  /** GLB of the model as the view shows it (scene/glbExport.ts edits the original file): clips, hidden parts, colours, held item. */
  async glbExport(url: string, opts: GlbExportOptions, onPhase: (label: string) => void): Promise<{ blob: Blob; bytes: number }> {
    const gltf = this.gltf, man = this.manifest;
    if (!gltf || !man) throw new Error('No model loaded');
    const parser = gltf.parser;
    const assoc = (o: object) => parser.associations.get(o as THREE.Object3D) as { nodes?: number; meshes?: number; primitives?: number; materials?: number } | undefined;
    const own = gltf.animations.map((a) => a.name);
    const cur = this.anim?.current ?? null;
    const keepClips = opts.animations === 'current' ? new Set(cur ? [cur] : []) : opts.animations === 'none' ? new Set<string>() : null;

    // parts some exported clip shows (manifest visibility tracks) stay, even when hidden right now; same rules as updateVisibility:
    // LOD > 0 never shows, and a user / part-group choice beats the clip keys unless parts are locked to the clip
    const exported = keepClips ? [...keepClips] : own;
    const clipShown = new Set<THREE.Object3D>();
    for (const p of man.parts) {
      if (p.lod > 0 || (!this.lockPartsToClip && this.userOverride.has(p.node))) continue;
      if (exported.some((n) => man.clips.find((c) => c.name === n)?.visibility?.[p.node]?.some(([, v]) => v !== 0))) {
        const o = this.findNode(p.node);
        if (o) clipShown.add(o);
      }
    }
    const hiddenInView = (o: THREE.Object3D) => {
      for (let p: THREE.Object3D | null = o; p && p !== gltf.scene; p = p.parent) if (!p.visible && !clipShown.has(p)) return true;
      return false;
    };

    const prims: PrimView[] = [];
    gltf.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh || this.isInHeld(m)) return;
      const a = assoc(m);
      if (a?.meshes === undefined || a.primitives === undefined) return; // outline hulls, hologram prepasses: viewer-only
      const pv: PrimView = { mesh: a.meshes, prim: a.primitives, hidden: hiddenInView(m) };
      if (!Array.isArray(m.material)) {
        const mat = m.material as THREE.MeshStandardMaterial;
        pv.material = assoc(mat)?.materials;
        const pal = mat.userData.__palTex as THREE.CanvasTexture[] | undefined;
        // the palette's lit texture is the current baked albedo (mat.map may be a UV-track clone of it, or the raw albedo in Raw mode)
        if (pal?.[0] && this.albedoState.get(mat)?.baked === pal[0]) pv.palette = { lit: pal[0].image as HTMLCanvasElement, shadow: pal[1]?.image as HTMLCanvasElement | undefined };
      }
      prims.push(pv);
    });

    const held: GlbExportInput['held'] = [];
    const h = this.held, bone = h ? assoc(h.obj.parent!)?.nodes : undefined;
    if (h && bone !== undefined) held.push({ url: h.url, bone, position: h.obj.position.toArray(), quaternion: h.obj.quaternion.toArray(), scale: h.obj.scale.toArray() });
    const { exportGlb } = await import('./glbExport');
    return exportGlb({ url, opts, keepClips, prims, held }, onPhase);
  }

  setAlbedoMode(mode: AlbedoMode) { this.albedoMode = mode; this.applyShading(); return this.applyAlbedo().then(() => { if (this.outline.on) this.applyOutline(); }); }

  private pixelCache = new Map<number, ImageData>();
  private maskCache = new Map<number, Uint8ClampedArray>();
  private async texturePixels(idx: number): Promise<ImageData | null> {
    const hit = this.pixelCache.get(idx);
    if (hit) return hit;
    const parser = this.gltf?.parser;
    if (!parser) return null;
    const tex = (await parser.getDependency('texture', idx)) as THREE.Texture;
    const img = tex.image as CanvasImageSource & { width: number; height: number };
    const c = document.createElement('canvas');
    c.width = img.width; c.height = img.height;
    const ctx = c.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(img, 0, 0);
    const data = ctx.getImageData(0, 0, c.width, c.height);
    this.pixelCache.set(idx, data);
    return data;
  }

  /** Recolour colour-table materials (modular NPCs): zone `indices[layer]` (1-based) per layer, lit and shadow textures rebuilt. */
  async setPalette(materialNames: string[], indices: number[]) {
    const gltf = this.gltf;
    if (!gltf) return;
    const mats = new Set<THREE.MeshStandardMaterial>();
    gltf.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      for (const mat of ([] as THREE.Material[]).concat(m.material)) if (materialNames.includes(mat.name.split('@')[0])) mats.add(mat as THREE.MeshStandardMaterial); // `name@variant` = a variant override of the same material
    });
    for (const mat of mats) {
      const pal = mat.userData?.palette as PaletteData | undefined;
      if (!pal) continue;
      const [alb, maskRgb, maskA] = await Promise.all([this.texturePixels(pal.albedo), this.texturePixels(pal.mask), this.texturePixels(pal.maskA)]);
      if (!alb || !maskRgb || !maskA || this.disposed) continue;
      let mask = this.maskCache.get(pal.mask * 100000 + pal.maskA);
      if (!mask) {
        mask = new Uint8ClampedArray(maskRgb.data);
        for (let i = 0; i < mask.length; i += 4) mask[i + 3] = maskA.data[i];
        this.maskCache.set(pal.mask * 100000 + pal.maskA, mask);
      }
      const { lit, shadow } = bakePalette(alb.data, mask, pal, indices);
      let st = this.albedoState.get(mat);
      if (!st) { st = { baked: mat.map, color: mat.color.clone() }; this.albedoState.set(mat, st); }
      const old = st.baked;
      const make = (px: Uint8ClampedArray) => {
        const c = document.createElement('canvas');
        c.width = alb.width; c.height = alb.height;
        c.getContext('2d')!.putImageData(new ImageData(px, alb.width, alb.height), 0, 0);
        const t = new THREE.CanvasTexture(c);
        t.colorSpace = THREE.SRGBColorSpace;
        t.flipY = false;
        if (old) { t.wrapS = old.wrapS; t.wrapT = old.wrapT; t.channel = old.channel; }
        t.anisotropy = old?.anisotropy ?? 1;
        return t;
      };
      const litTex = make(lit);
      const shTex = make(shadow);
      const prev = mat.userData.__palTex as THREE.Texture[] | undefined;
      prev?.forEach((t) => t.dispose());
      mat.userData.__palTex = [litTex, shTex];
      st.baked = litTex;
      if (this.albedoMode === 'baked') mat.map = litTex;
      let cache = this.texCache.get(mat);
      if (!cache) { cache = {}; this.texCache.set(mat, cache); }
      cache.shadowColorTexture = shTex;
      mat.needsUpdate = true;
    }
    this.applyShading();
    void this.applyAlbedo();
  }

  /** Apply Baked / Raw / Off albedo to every material in the scene (including variant materials and the held item). */
  private async applyAlbedo() {
    const gen = ++this.albedoGen;
    const gltf = this.gltf;
    if (!gltf) return;
    const mode = this.albedoMode;
    const mats = new Map<THREE.MeshStandardMaterial, boolean>(); // material -> belongs to the main gltf
    const collect = (root: THREE.Object3D, main: boolean) => root.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh || m.userData.outline) return;
      for (const mat of ([] as THREE.Material[]).concat(m.material)) mats.set(mat as THREE.MeshStandardMaterial, main);
    });
    collect(gltf.scene, true);
    if (this.held) collect(this.held.obj, false); // later: overrides the main flag for the held item's own materials
    for (const [mat, main] of mats) {
      if (!('map' in mat)) continue;
      let st = this.albedoState.get(mat);
      if (!st) { st = { baked: mat.map, color: mat.color.clone() }; this.albedoState.set(mat, st); }
      let target: THREE.Texture | null = st.baked;
      let color = st.color;
      if (mode === 'off') { target = null; color = new THREE.Color(1, 1, 1); }
      else if (mode === 'raw' && main && st.baked) {
        const idx = mat.userData?.rawBaseColorTexture;
        if (typeof idx === 'number') {
          if (st.raw === undefined) {
            try {
              const t = (await gltf.parser.getDependency('texture', idx)) as THREE.Texture;
              t.colorSpace = THREE.SRGBColorSpace;
              t.wrapS = st.baked.wrapS; t.wrapT = st.baked.wrapT;
              t.channel = st.baked.channel;
              t.flipY = st.baked.flipY;
              t.needsUpdate = true;
              st.raw = t;
            } catch (e) { console.warn('raw albedo load failed', e); st.raw = null; }
          }
          if (gen !== this.albedoGen) return;
          if (st.raw) target = st.raw;
        }
      }
      if (gen !== this.albedoGen) return;
      if (mat.map !== target) { if ((mat.map === null) !== (target === null)) mat.needsUpdate = true; mat.map = target; }
      mat.color.copy(color);
      if (mat.userData.__gameUnlit) syncUnlit(mat);
    }
  }

  setShadingMode(mode: ShadingMode) { this.shadingMode = mode; this.applyRig(); this.applyShading(); }
  setEnvironment(env: Environment) {
    this.env = { ...env };
    this.renderer.toneMappingExposure = env.exposure;
    const tm = TONE_MAPS[env.toneMapping] ?? THREE.LinearToneMapping;
    // The Hable curve lives in three's shared tonemapping chunk with its constants baked in, so a parameter
    // change needs new programs: the key goes into every material's customProgramCacheKey (applyShading).
    const hableKey = env.toneMapping === 'hable' ? installHable(env.hable ?? DEFAULT_HABLE) : env.toneMapping === 'unite' ? installCustomTone('unite', UNITE_TONE_GLSL) : '';
    UNITE_K.value = env.toneMapping === 'unite' ? 1 : 0.45; // the curve folds the shaders' 0.45 in (uniteShading.ts)
    if (this.renderer.toneMapping !== tm || hableKey !== this.hableKey) {
      this.renderer.toneMapping = tm;
      this.hableKey = hableKey;
      this.scene.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) for (const mat of ([] as THREE.Material[]).concat(m.material)) mat.needsUpdate = true; });
    }
    this.applyRig();
    this.applyShading();
  }

  private applyRig() {
    const game = this.shadingMode === 'game';
    const e = this.env;
    for (const l of this.pbrLights) l.visible = !game;
    for (const l of this.gameLights) l.visible = game;
    // PBR keeps its own base intensities; the sliders scale them relative to the Game defaults.
    const ak = game ? 1 : 1.2 / DEFAULT_ENV.ambient;
    const kk = game ? 1 : 2.2 / DEFAULT_ENV.key;
    const w = e.warmth;
    // Explicit colour (game-data presets) or colour temperature (Kelvin) first, then the global white-balance shift.
    const tint = (kelvin: number, hex?: string) => {
      const c = hex ? new THREE.Color(hex) : new THREE.Color().setRGB(...kelvinToRGB(kelvin), THREE.LinearSRGBColorSpace);
      const t = new THREE.Color(w >= 0 ? 0xffc27a : 0x9db8ff);
      return c.lerp(t, Math.abs(w) * 0.5);
    };
    const hemi = (game ? this.gameLights[0] : this.pbrLights[0]) as THREE.HemisphereLight;
    hemi.intensity = e.ambient * ak;
    hemi.color.copy(tint(e.skyTemp, e.skyColor));
    if (game) hemi.groundColor.set(e.ground);
    UNITE_ON.value = game && uniteSwitch.enabled && e.toneMapping === 'unite' ? 1 : 0; // UNITE's shader model rides with its own curve in the Game rig (uniteShading.ts); other presets keep the generic Game shading
    UNITE_SCENE.key.value = 1 / DEFAULT_ENV.key; // key colour x intensity relative to the default environment (the game's neutral 1)
    UNITE_SCENE.amb.value.set(hemi.color.r, hemi.color.g, hemi.color.b).multiplyScalar(e.ambient / DEFAULT_ENV.ambient);
    this.key.intensity = e.key * kk;
    this.key.color.copy(tint(e.keyTemp, e.keyColor));
    this.back.intensity = e.back * kk;
    this.back.color.copy(tint(e.backTemp, e.backColor));
    this.back.visible = e.back > 0;
    this.positionKey();
  }

  private positionKey() {
    const e = this.env;
    // PBR keeps its legacy placement: the sliders offset it from the default angles.
    let az = e.azimuth, el = e.elevation;
    if (this.shadingMode !== 'game') {
      az += Math.atan2(0.5, 0.8) * 180 / Math.PI - DEFAULT_ENV.azimuth;
      el += Math.atan2(1.2, Math.hypot(0.5, 0.8)) * 180 / Math.PI - DEFAULT_ENV.elevation;
    }
    this.placeLight(this.key, az, el);
    this.placeLight(this.back, e.backAzimuth, e.backElevation);
  }

  private placeLight(light: THREE.DirectionalLight, az: number, el: number) {
    const { center, radius } = this.lightFrame;
    const a = THREE.MathUtils.degToRad(az), b = THREE.MathUtils.degToRad(Math.min(90, Math.max(0, el)));
    const dir = new THREE.Vector3(Math.cos(b) * Math.sin(a), Math.sin(b), Math.cos(b) * Math.cos(a));
    light.position.copy(center).addScaledVector(dir, radius * 3 * 1.414);
  }

  /** Apply the current shading mode to every material of the model and held item. */
  private applyShading() {
    const seen = new Set<THREE.Material>();
    const each = (root: THREE.Object3D, parser?: GLTF['parser']) => root.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      // The held item is parented inside the trainer's skeleton; its texture
      // indices belong to its own glTF, so never resolve them with the
      // trainer's parser (that painted trainer eye/hair textures onto balls).
      if (root !== this.held?.obj && this.isInHeld(m)) return;
      for (const mat of ([] as THREE.Material[]).concat(m.material)) {
        if (seen.has(mat) || !(mat as THREE.MeshStandardMaterial).isMeshStandardMaterial) continue;
        seen.add(mat);
        applyGameBlend(mat); // variant materials are swapped in after load
        const sm = mat as THREE.MeshStandardMaterial;
        // Overlay order in the transparent pass: a layer-5 overlay is part of the body surface, so it goes before the
        // blended eyes (Cutiefly's face patch covered an eye); a sliding lid goes after its eye.
        if (sm.userData?.layer_overlay) m.renderOrder = Math.min(m.renderOrder, -1);
        else if (sm.userData?.lid_overlay) m.renderOrder = Math.max(m.renderOrder, 1);
        const hk = this.hableKey;
        const gasKey = sm.userData?.gas ? '|gas' : '';
        const ehKey = (sm.userData?.eyeHighlight ? '|eyehl' : '') + ((sm as THREE.MeshPhysicalMaterial).clearcoat > 0 ? '|coat' : '');
        // the matcap patch (matcap.ts) changes the code too: without it in the key, a UNITE metal material reused the
        // unpatched program of a sibling compiled first (Zacian lobby's armour stayed black)
        const mcKey = uniteShadingKey(sm) + (typeof sm.userData?.matcapTexture === 'number' || sm.userData?.metalEnv ? (isMatcapReflect(sm) ? '|mcr' : '|mcg') : '') + uniteKey(sm);
        const fireKey = sm.userData?.fire ? '|fire' : ''; // fireEffect.ts replaces the output: never share a program
        const sphKey = sm.userData?.sphereMap ? '|sph' : ''; // sphereMap.ts samples the map by the view normal
        sm.customProgramCacheKey = () => hk + gasKey + ehKey + mcKey + fireKey + sphKey + '|holo';
        if (sm.userData?.gas || sm.userData?.behindBody) {
          // In game the body is never covered by the gas: it is painted over it (a perfect disc in every shot).
          // Draw the shell first without writing depth so the body wins wherever they overlap on screen.
          m.renderOrder = -10;
          sm.depthWrite = false;
          if (sm.userData?.behindBody && sm.transparent) {
            // Blended effect cards (HOME smoke): three draws `transparent` materials after every opaque one whatever the renderOrder, which
            // would lay the smoke over the face. Keep them in the opaque pass with explicit alpha blending so they really draw first.
            sm.transparent = false;
            sm.blending = THREE.CustomBlending;
            sm.blendSrc = THREE.SrcAlphaFactor;
            sm.blendDst = THREE.OneMinusSrcAlphaFactor;
            sm.blendEquation = THREE.AddEquation;
          }
        }
        this.ensureGas(sm, parser); // before the game patch captures onBeforeCompile, so the two chain
        installHologram(sm); // likewise (inactive until Appearance > Hologram is on)
        installSphereMap(sm); // Stadium 2 env-mapped parts (N64 texture generation), before the game patch chains on
        if (installFire(sm, async (i) => { try { return (await parser!.getDependency('texture', i)) as THREE.Texture; } catch { return null; } })) this.fireMats.add(sm); // UNITE flame materials (fireEffect.ts)
        if (installHomeFire(sm)) this.fireMats.add(sm); // HOME-painted flames (Skeledirge lobby strands)
        installUniteShading(sm); // UNITE lit shading from the decompiled shaders (before the matcap patch, whose terms add onto it)
        if (installMatcap(sm)) { setMatcapTexture(sm, this.getTex(sm, parser, 'matcapTexture', true) ?? (sm.userData.metalEnv ? defaultMatcap() : null)); loadUniteMatcaps(sm, parser); } // UNITE glass / metal / matcap terms (matcap.ts)
        if (installEyeHighlight(sm)) { // SV catchlights (eyeHighlight.ts); the hologram draws its own look
          setEyeHighlightNormal(sm, this.getTex(sm, parser, 'highlightNormalTexture', false));
          setEyeHighlightOn(sm, !this.hologram?.on);
        }
        const coated = installCoatReflection(sm); // clear-coated eyes: the coat reflects the studio map
        if (root !== this.held?.obj) { // the held item (Poké Ball...) stays solid: in game only the trainer is a hologram
          this.holoMats.add(sm);
          setHologram(sm, this.hologram, this.holoOwn(sm));
          updateHologram(sm, this.holoTime);
        }
        if (this.shadingMode === 'game') {
          const tex = {
            shadow: this.albedoMode === 'baked' ? this.getTex(sm, parser, 'shadowColorTexture', true) : null,
            spec: this.getTex(sm, parser, 'specularMaskTexture', false),
            rim: this.getTex(sm, parser, 'rimMaskTexture', false),
          };
          if (isReflective(sm)) this.reflEnv ??= makeReflectionEnv(this.renderer);
          applyGameShading(sm, this.env, tex, isReflective(sm) ? this.reflEnv ?? null : null, (sm.userData as { reflect?: { envIntensity?: number } }).reflect?.envIntensity ?? 1.5);
        } else removeGameShading(sm);
        if (coated) { this.reflEnv ??= makeReflectionEnv(this.renderer); sm.envMap = this.reflEnv; sm.envMapIntensity = COAT_ENV_INTENSITY; }
      }
    });
    if (this.gltf) each(this.gltf.scene, this.gltf.parser);
    if (this.held) each(this.held.obj, this.held.parser);
    if (this.outline.on) this.applyOutline();
  }

  private outline: OutlineState = { ...DEFAULT_OUTLINE };
  private outlineMeshes: THREE.Mesh[] = [];
  setOutline(st: OutlineState) {
    const rebuild = st.on !== this.outline.on || st.color !== this.outline.color || st.tinted !== this.outline.tinted || (st.style ?? 'auto') !== (this.outline.style ?? 'auto');
    this.outline = { ...st };
    if (rebuild) this.applyOutline();
  }

  /** Back-face hull copies of the opaque meshes (see outline.ts), rebuilt when the materials change. Off while the hologram is on. */
  private applyOutline() {
    for (const p of this.outlineMeshes) { p.removeFromParent(); for (const m of ([] as THREE.Material[]).concat(p.material)) m.dispose(); }
    this.outlineMeshes = [];
    if (!this.outline.on || this.outline.style === 'sprite' || this.hologram?.on || !this.gltf) return;
    const meshes: THREE.Mesh[] = [];
    this.gltf.scene.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh && !m.userData.outline && !m.userData.holoPrepass && !this.isInHeld(m)) meshes.push(m); });
    // 'auto' (default) = the game's own outline data when the model carries it (UNITE `extras.unite.outline`), else the generic hull.
    const style = this.outline.style ?? 'auto';
    const game = style === 'auto' && modelHasGameOutline(this.gltf.scene);
    const build = (maps?: Map<number, THREE.Texture>) => {
      if (!this.outline.on || this.outlineGen !== gen) return;
      for (const m of meshes) {
        const p = makeOutlineMesh(m, this.outline.color, this.outline.tinted, game ? { maps: maps ?? new Map() } : undefined);
        if (!p) continue;
        m.add(p);
        this.outlineMeshes.push(p);
      }
    };
    const gen = ++this.outlineGen;
    if (!game) { build(); return; }
    // the `_OutlineMap` textures first (a model built before they load would have no per-texel colour)
    const parser = this.gltf.parser;
    const want = new Set<number>();
    for (const m of meshes) for (const x of ([] as THREE.Material[]).concat(m.material)) { const i = gameOutlineOf(x)?.map?.index; if (i !== undefined) want.add(i); }
    void Promise.all([...want].map(async (i) => {
      let t = this.outlineMapCache.get(i);
      if (!t) { t = (await parser.getDependency('texture', i)) as THREE.Texture; t.colorSpace = THREE.NoColorSpace; t.needsUpdate = true; this.outlineMapCache.set(i, t); }
      return [i, t] as const;
    })).then((pairs) => build(new Map(pairs))).catch((e) => { console.warn('outline maps failed', e); build(); });
  }
  private outlineGen = 0;
  private outlineMapCache = new Map<number, THREE.Texture>();

  /** Turn the hologram effect on/off or change its settings (applies to the model, not the held item). */
  setHologramState(st: HologramState | null) {
    this.hologram = st;
    for (const m of this.holoMats) { setHologram(m, st, this.holoOwn(m)); updateHologram(m, this.holoTime); setEyeHighlightOn(m, !st?.on); }
    this.applyHoloPrepass(!!st?.on);
    this.applyOutline();
  }

  /**
   * Depth-only copies of the model's opaque meshes, drawn in the opaque pass, so the translucent hologram
   * shows only its front-most surface (blended triangles otherwise sort arbitrarily and the back hair shows
   * through the face). Children of their mesh with an identity transform: they follow visibility and pose.
   */
  private holoPrepass: THREE.Mesh[] = [];
  private applyHoloPrepass(on: boolean) {
    for (const p of this.holoPrepass) { p.removeFromParent(); (p.material as THREE.Material).dispose(); }
    this.holoPrepass = [];
    if (!on || !this.gltf) return;
    const meshes: THREE.Mesh[] = [];
    this.gltf.scene.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh && !m.userData.holoPrepass && !m.userData.outline && !this.isInHeld(m)) meshes.push(m); });
    for (const m of meshes) {
      const src = ([] as THREE.Material[]).concat(m.material)[0] as THREE.MeshStandardMaterial;
      const saved = src.userData.__holoSaved as { transparent: boolean } | undefined;
      if (!src || saved?.transparent || src.userData?.gas) continue; // lenses / effects stay blended
      // Pushed back a little so the hologram's own (identical) depth always passes LessEqual.
      const mat = new THREE.MeshBasicMaterial({ colorWrite: false, side: src.side, alphaTest: src.alphaTest, map: src.alphaTest > 0 ? src.map : null, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
      const sk = m as THREE.SkinnedMesh;
      let p: THREE.Mesh;
      if (sk.isSkinnedMesh) { const q = new THREE.SkinnedMesh(sk.geometry, mat); q.bind(sk.skeleton, sk.bindMatrix); q.bindMode = sk.bindMode; p = q; }
      else p = new THREE.Mesh(m.geometry, mat);
      if (m.morphTargetInfluences) p.morphTargetInfluences = m.morphTargetInfluences;
      p.userData.holoPrepass = true;
      p.frustumCulled = false; p.castShadow = false; p.receiveShadow = false;
      p.renderOrder = m.renderOrder - 1;
      p.raycast = () => {};
      m.add(p);
      this.holoPrepass.push(p);
    }
  }

  /** This material's data values for the selected model preset (overlay copies `<mat>~...` use their source's). */
  private holoOwn(mat: THREE.MeshStandardMaterial): HoloParams | undefined {
    const st = this.hologram;
    if (!st) return undefined;
    const ext = (mat.userData?.holocast ?? this.holoBase(mat)?.userData?.holocast) as Record<string, HoloParams> | undefined;
    return ext?.[st.preset];
  }

  private holoBase(mat: THREE.MeshStandardMaterial): THREE.Material | undefined {
    const i = mat.name.indexOf('~');
    if (i < 0) return undefined;
    const name = mat.name.slice(0, i);
    for (const m of this.holoMats) if (m.name === name) return m;
    return undefined;
  }

  /** Gas shells (materials with extras.gas): install the panning shader patch once and fetch the density map. */
  private ensureGas(mat: THREE.MeshStandardMaterial, parser: GLTF['parser'] | undefined) {
    if (!mat.userData?.gas || this.gasMats.has(mat)) return;
    if (!installGas(mat)) return;
    this.gasMats.add(mat);
    updateGas(mat, this.gasTime);
    const idx = mat.userData.gasDensityTexture;
    if (typeof idx !== 'number' || !parser) return;
    void (async () => {
      try {
        const tex = (await parser.getDependency('texture', idx)) as THREE.Texture;
        if (!this.disposed) setGasTexture(mat, tex);
      } catch (e) { console.warn('gas density load failed', e); }
    })();
  }

  /** Cached extras texture (shadow colour / spec mask / rim mask); loads async and re-applies shading when ready. */
  private getTex(mat: THREE.MeshStandardMaterial, parser: GLTF['parser'] | undefined, key: string, srgb: boolean): THREE.Texture | null {
    const idx = mat.userData?.[key];
    if (typeof idx !== 'number' || !parser) return null;
    let cache = this.texCache.get(mat);
    if (!cache) { cache = {}; this.texCache.set(mat, cache); }
    const c = cache[key];
    if (c === 'pending') return null;
    if (c !== undefined) return c;
    cache[key] = 'pending';
    void (async () => {
      let tex: THREE.Texture | null = null;
      try {
        tex = (await parser.getDependency('texture', idx)) as THREE.Texture;
        const base = this.albedoState.get(mat)?.baked ?? mat.map ?? null;
        tex.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
        if (base) { tex.wrapS = base.wrapS; tex.wrapT = base.wrapT; tex.channel = base.channel; tex.flipY = base.flipY; }
        tex.needsUpdate = true;
      } catch (e) { console.warn(key + ' load failed', e); tex = null; }
      cache![key] = tex;
      if (!this.disposed) this.applyShading();
    })();
    return null;
  }

  fitInfo() {
    this.root.updateMatrixWorld(true);
    const box = new THREE.Box3();
    this.gltf?.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      for (let p: THREE.Object3D | null = m; p; p = p.parent) if (!p.visible) return;
      let bb: THREE.Box3 | null = null;
      if ((m as THREE.SkinnedMesh).isSkinnedMesh) { const sm = m as THREE.SkinnedMesh; sm.computeBoundingBox(); bb = sm.boundingBox; }
      if (!bb) { if (!m.geometry.boundingBox) m.geometry.computeBoundingBox(); bb = m.geometry.boundingBox; }
      if (bb) box.union(bb.clone().applyMatrix4(m.matrixWorld));
    });
    if (box.isEmpty()) return null;
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 1e-3);
    const dist = this.fitDistance(radius);
    return { size, center, radius, dist, h: size.y };
  }

  private pendingPreset?: CameraPreset;
  private defaultClipName?: string;

  /** Explicit camera pose (world metres), used by the lighting reference tools and the Reference panel. */
  setCameraPose(fov: number, position: [number, number, number], target: [number, number, number], roll = 0) {
    this.userRoll = roll;
    const damp = this.controls.enableDamping;
    this.controls.enableDamping = false;
    this.camera.fov = fov;
    this.camera.position.set(position[0], position[1], position[2]);
    this.controls.target.set(target[0], target[1], target[2]);
    const fi = this.fitInfo();
    if (fi) {
      // limits follow the pose's own distance, so a restored far / narrow-FOV pose is not clamped by OrbitControls
      const d = this.camera.position.distanceTo(this.controls.target);
      this.applyLimits(fi.radius, d);
      this.camera.near = Math.max(0.001, this.camera.near);
      this.camera.far = Math.max(100, this.camera.far);
    }
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.controls.enableDamping = damp;
  }

  cameraPose(): { fov: number; position: [number, number, number]; target: [number, number, number]; roll?: number } {
    const p = this.camera.position, t = this.controls.target;
    return { fov: this.camera.fov, position: [p.x, p.y, p.z], target: [t.x, t.y, t.z], ...(this.userRoll ? { roll: this.userRoll } : {}) };
  }

  /** Camera roll (deg) the user set; re-applied after OrbitControls every frame (it rebuilds the orientation). */
  private userRoll = 0;
  onCameraChange?: (o: CameraOrbit) => void;
  private lastCamEmit = 0;
  private camKey = '';

  /** Current camera as orbit angles around the target + world position (see CameraOrbit). */
  cameraOrbit(): CameraOrbit {
    const p = this.camera.position, t = this.controls.target;
    const off = p.clone().sub(t);
    const distance = off.length();
    const mx = this.root.scale.x < 0 ? -1 : 1;
    const deg = THREE.MathUtils.radToDeg;
    const yaw = distance > 1e-9 ? deg(Math.atan2(off.x * mx, off.z)) : 0;
    const pitch = distance > 1e-9 ? deg(Math.asin(THREE.MathUtils.clamp(off.y / distance, -1, 1))) : 0;
    return { yaw, pitch, distance, roll: this.userRoll, fov: this.camera.fov, target: [t.x, t.y, t.z], position: [p.x, p.y, p.z] };
  }

  /**
   * Sets the camera from typed values. A patch with `position` moves the camera there (target kept); otherwise
   * yaw / pitch / distance place it on the orbit around the (possibly new) target. Releases a reference lens lock.
   */
  setCameraOrbit(patch: Partial<CameraOrbit>) {
    this.releaseLens();
    const o = { ...this.cameraOrbit(), ...patch };
    let pos = o.position;
    if (!patch.position) {
      const rad = THREE.MathUtils.degToRad;
      const mx = this.root.scale.x < 0 ? -1 : 1;
      const pitch = rad(THREE.MathUtils.clamp(o.pitch, -MAX_PITCH, MAX_PITCH));
      const yaw = rad(o.yaw);
      const d = Math.max(1e-4, o.distance);
      pos = [o.target[0] + d * Math.cos(pitch) * Math.sin(yaw) * mx, o.target[1] + d * Math.sin(pitch), o.target[2] + d * Math.cos(pitch) * Math.cos(yaw)];
    }
    const fov = THREE.MathUtils.clamp(o.fov, 0.5, 170);
    this.setCameraPose(fov, pos, o.target, wrapDeg(o.roll));
    this.emitCamera(true);
  }

  /** Turns yaw / pitch / roll by `deg`, first snapping to the nearest multiple of |deg| so repeated steps land on exact angles. */
  rotateCamera(axis: 'yaw' | 'pitch' | 'roll', deg: number) {
    const o = this.cameraOrbit();
    const step = Math.abs(deg) || 45;
    const snapped = Math.round(o[axis] / step) * step;
    let v = Math.abs(snapped - o[axis]) < 1e-3 ? snapped + deg : (deg > 0 ? Math.ceil(o[axis] / step) : Math.floor(o[axis] / step)) * step;
    if (axis === 'pitch') v = THREE.MathUtils.clamp(v, -MAX_PITCH, MAX_PITCH);
    else v = wrapDeg(v);
    this.setCameraOrbit({ [axis]: v });
  }

  /** Slider spans for the Camera panel, from the model's size (never from the camera's own values, which would make a
   *  slider's range grow while it is dragged): distance from close-up to 3x the Fit distance at the current FOV. */
  // FITTED: 0.3 r .. 3x fit distance, target +-2 r chosen by eye on Lamball (palworld_models), 2026-10-30
  cameraRanges(): { center: [number, number, number]; radius: number; dMin: number; dMax: number } | null {
    const f = this.fitInfo();
    if (!f) return null;
    const dMax = f.dist * 3;
    return { center: [f.center.x, f.center.y, f.center.z], radius: f.radius, dMin: Math.min(f.radius * 0.3, dMax / 10), dMax };
  }

  private applyRoll() {
    if (!this.userRoll || this.lens) return;
    this.camera.rotateZ(THREE.MathUtils.degToRad(-this.userRoll));
    this.camera.updateMatrixWorld();
  }

  private emitCamera(force = false) {
    if (!this.onCameraChange) return;
    const now = performance.now();
    if (!force && now - this.lastCamEmit < 66) return;
    const p = this.camera.position, t = this.controls.target;
    const r = (n: number) => Math.round(n * 1e5);
    const key = [p.x, p.y, p.z, t.x, t.y, t.z, this.camera.fov, this.userRoll, this.root.scale.x].map(r).join(',');
    if (key === this.camKey) return;
    this.camKey = key;
    this.lastCamEmit = now;
    this.onCameraChange(this.cameraOrbit());
  }

  /** Meshes of the model in traversal order (the index is how a SurfaceAnchor finds its mesh again). */
  private anchorMeshes(): THREE.Mesh[] {
    const out: THREE.Mesh[] = [];
    const held = this.held?.obj;
    this.gltf?.scene.traverse((o) => {
      if (!(o as THREE.Mesh).isMesh) return;
      for (let p: THREE.Object3D | null = o; p; p = p.parent) if (p === held) return; // held item: not part of the model
      out.push(o as THREE.Mesh);
    });
    return out;
  }

  /** Model surface point under a canvas position (NDC), as a triangle + barycentric weights so it follows the animation. */
  pickSurface(ndcX: number, ndcY: number): SurfaceAnchor | null {
    if (!this.gltf) return null;
    this.root.updateMatrixWorld(true);
    const meshes = this.anchorMeshes().filter((m) => {
      for (let p: THREE.Object3D | null = m; p; p = p.parent) if (!p.visible) return false;
      return true;
    });
    const all = this.anchorMeshes();
    // SkinnedMesh.raycast culls with its own boundingSphere AND boundingBox, cached from whatever pose they were last
    // computed in: refresh both for the current pose, or posed parts outside the stale box are unclickable.
    for (const m of meshes) {
      const sm = m as THREE.SkinnedMesh;
      if (sm.isSkinnedMesh) { sm.computeBoundingSphere(); sm.computeBoundingBox(); }
      else { m.geometry.computeBoundingSphere(); m.geometry.computeBoundingBox(); }
    }
    this.camera.updateMatrixWorld();
    const rc = new THREE.Raycaster();
    rc.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.camera);
    // Both faces: a mirrored model (negative scale) or inside-out outline shells would otherwise be culled as back faces.
    const sides: [THREE.Material, THREE.Side][] = [];
    for (const m of meshes) for (const mat of ([] as THREE.Material[]).concat(m.material)) { sides.push([mat, mat.side]); mat.side = THREE.DoubleSide; }
    let hits: THREE.Intersection[];
    try { hits = rc.intersectObjects(meshes, false); } finally { for (const [mat, s] of sides) mat.side = s; }
    for (const hit of hits) {
      if (!hit.face) continue;
      const m = hit.object as THREE.Mesh;
      const { a, b, c } = hit.face;
      const [pa, pb, pc] = [a, b, c].map((i) => m.getVertexPosition(i, new THREE.Vector3()).applyMatrix4(m.matrixWorld));
      const bc = THREE.Triangle.getBarycoord(hit.point, pa, pb, pc, new THREE.Vector3());
      if (bc && Number.isFinite(bc.x)) return { mesh: all.indexOf(m), name: m.name, tri: [a, b, c], bary: [bc.x, bc.y, bc.z] };
      return { mesh: all.indexOf(m), name: m.name, tri: [a, a, a], bary: [1, 0, 0] }; // degenerate triangle: its first vertex
    }
    // Ray missed (silhouette edge, thin strands): snap to the front-most vertex projecting within SNAP_PX of the click.
    const SNAP_PX = 30;
    const r = this.renderer.domElement.getBoundingClientRect();
    const cx = (ndcX + 1) / 2 * r.width, cy = (1 - ndcY) / 2 * r.height;
    const v = new THREE.Vector3();
    let best: { m: THREE.Mesh; i: number; score: number } | null = null;
    for (const m of meshes) {
      const n = m.geometry.attributes.position?.count ?? 0;
      for (let i = 0; i < n; i++) {
        m.getVertexPosition(i, v).applyMatrix4(m.matrixWorld).applyMatrix4(this.camera.matrixWorldInverse);
        const depth = -v.z;
        if (depth <= 0) continue;
        v.applyMatrix4(this.camera.projectionMatrix);
        const d = Math.hypot((v.x + 1) / 2 * r.width - cx, (1 - v.y) / 2 * r.height - cy);
        if (d > SNAP_PX) continue;
        const score = depth * (1 + d / 200); // nearest to the camera wins, screen distance as a tie-break
        if (!best || score < best.score) best = { m, i, score };
      }
    }
    if (best) return { mesh: all.indexOf(best.m), name: best.m.name, tri: [best.i, best.i, best.i], bary: [1, 0, 0] };
    console.debug('[pickSurface] miss', { ndcX, ndcY, meshes: meshes.length, hits: hits.length, canvas: [r.width, r.height] });
    return null;
  }

  /** World position of anchors in the current pose (null where the anchor no longer resolves). */
  anchorsWorld(anchors: SurfaceAnchor[]): ([number, number, number] | null)[] {
    const meshes = this.anchorMeshes();
    this.root.updateMatrixWorld(true);
    const v = new THREE.Vector3(), acc = new THREE.Vector3();
    return anchors.map((an) => {
      const m = meshes[an.mesh];
      if (!m || m.name !== an.name) return null;
      const n = m.geometry.attributes.position?.count ?? 0;
      if (an.tri.some((i) => i >= n)) return null;
      acc.set(0, 0, 0);
      an.tri.forEach((i, k) => acc.addScaledVector(m.getVertexPosition(i, v).applyMatrix4(m.matrixWorld), an.bary[k]));
      return [acc.x, acc.y, acc.z];
    });
  }

  /** Client (page) coordinates of world points through the current camera; null when behind it. */
  projectToClient(pts: ([number, number, number] | null)[]): ({ x: number; y: number } | null)[] {
    const r = this.renderer.domElement.getBoundingClientRect();
    const v = new THREE.Vector3();
    this.camera.updateMatrixWorld();
    return pts.map((p) => {
      if (!p) return null;
      v.set(p[0], p[1], p[2]).applyMatrix4(this.camera.matrixWorldInverse);
      if (v.z >= 0) return null;
      v.applyMatrix4(this.camera.projectionMatrix);
      return { x: r.left + (v.x + 1) / 2 * r.width, y: r.top + (1 - v.y) / 2 * r.height };
    });
  }

  canvasRect() { return this.renderer.domElement.getBoundingClientRect(); }

  /** Pivot lock: every frame, pan (camera + orbit target together) so the anchor projects onto `ndc`. Orbiting and
   *  zooming then turn around that model point while it stays glued to its spot on the reference. */
  private pivot?: () => { anchor: SurfaceAnchor; ndc: [number, number] } | null;
  setPivot(fn: (() => { anchor: SurfaceAnchor; ndc: [number, number] } | null) | null) {
    this.pivot = fn ?? undefined;
    const g = fn?.();
    const X = g ? this.anchorsWorld([g.anchor])[0] : null;
    if (!X) return;
    // orbit centre on the view axis at the pin's depth: drags then turn around the pin, not around a far-away point
    const fwd = this.camera.getWorldDirection(new THREE.Vector3());
    const depth = new THREE.Vector3(X[0], X[1], X[2]).sub(this.camera.position).dot(fwd);
    if (depth > 1e-3) { this.controls.target.copy(this.camera.position).addScaledVector(fwd, depth); this.controls.update(); }
  }

  /**
   * Reference lens: the camera solved from pins against a screenshot. Every frame (after OrbitControls) it sets the
   * FOV, shifts the principal point by `shift` canvas px (the screenshot's centre need not be the canvas centre) and
   * rolls the camera `roll` degrees (scene turns counter-clockwise on screen). The reference image is never moved: the
   * model conforms to it. Released by a camera preset / F, which calls onLensCleared.
   */
  private lens?: () => { fov: number; shift: [number, number]; roll: number } | null;
  onLensCleared?: () => void;
  private exporting = false;
  setLens(fn: (() => { fov: number; shift: [number, number]; roll: number } | null) | null) {
    this.lens = fn ?? undefined;
    if (!fn) { this.camera.clearViewOffset(); this.resize(); this.controls.update(); }
  }
  private releaseLens() {
    if (this.exporting || !this.lens) return;
    this.setLens(null);
    this.onLensCleared?.();
  }
  private applyLens() {
    const l = this.lens?.();
    if (!l) return;
    const cam = this.camera, el = this.renderer.domElement;
    const w = el.clientWidth || 1, h = el.clientHeight || 1;
    cam.fov = l.fov;
    cam.setViewOffset(w, h, -l.shift[0], -l.shift[1], w, h); // also updates the projection
    cam.rotateZ(THREE.MathUtils.degToRad(-l.roll));
    cam.updateMatrixWorld();
  }

  heldObject(): THREE.Object3D | null { return this.held?.obj ?? null; }

  /**
   * Small renders for analysis (auto lighting): the drawing buffer is shrunk to W x H (same aspect as the canvas, so
   * the camera and lens are unchanged), background transparent, ground and skeleton hidden. `fn` may render many
   * times; everything is restored before returning, inside the same task, so the live view never shows it.
   */
  probe<T>(W: number, H: number, fn: (render: () => ImageData) => T): T {
    const r = this.renderer, cam = this.camera;
    const ratio = r.getPixelRatio();
    const bg = this.scene.background, groundVis = this.ground.visible, skVis = this.skeletonHelper?.visible;
    const clear = r.getClearColor(new THREE.Color()), clearA = r.getClearAlpha();
    const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    const ctx = cv.getContext('2d', { willReadFrequently: true })!;
    this.ro.disconnect();
    this.exporting = true;
    try {
      r.setPixelRatio(1);
      r.setSize(W, H, false);
      this.scene.background = null;
      this.ground.visible = false;
      if (this.skeletonHelper) this.skeletonHelper.visible = false;
      for (const o of this.outlineMeshes) o.visible = false; // the id / colour passes compare materials, not edges
      for (const e of this.particles) e.mesh.visible = false; // not part of the model's materials
      r.setClearColor(0x000000, 0);
      return fn(() => {
        r.render(this.scene, cam);
        ctx.clearRect(0, 0, W, H);
        ctx.drawImage(r.domElement, 0, 0);
        return ctx.getImageData(0, 0, W, H);
      });
    } finally {
      this.exporting = false;
      this.scene.background = bg;
      this.ground.visible = groundVis;
      if (this.skeletonHelper && skVis !== undefined) this.skeletonHelper.visible = skVis;
      for (const o of this.outlineMeshes) o.visible = true;
      for (const e of this.particles) e.mesh.visible = this.particlesOn;
      r.setClearColor(clear, clearA);
      r.setPixelRatio(ratio);
      this.ro.observe(this.container);
      this.resize();
      this.applyLens();
      r.render(this.scene, cam);
    }
  }

  /** Slower orbit / zoom / pan for fine alignment. */
  setFineControls(on: boolean) {
    this.controls.rotateSpeed = on ? 0.2 : 1;
    this.controls.zoomSpeed = on ? 0.2 : 1;
    this.controls.panSpeed = on ? 0.2 : 1;
  }

  private applyPivot() {
    const g = this.pivot?.();
    if (!g) return;
    const X = this.anchorsWorld([g.anchor])[0];
    if (!X) return;
    const cam = this.camera;
    cam.updateMatrixWorld();
    const v = new THREE.Vector3(X[0], X[1], X[2]).applyMatrix4(cam.matrixWorldInverse);
    const depth = -v.z;
    if (depth <= 1e-4) return;
    v.applyMatrix4(cam.projectionMatrix);
    const t = Math.tan(THREE.MathUtils.degToRad(cam.fov) / 2);
    const ex = v.x - g.ndc[0], ey = v.y - g.ndc[1];
    if (Math.abs(ex) < 1e-6 && Math.abs(ey) < 1e-6) return;
    const right = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(cam.matrixWorld, 1);
    const shift = right.multiplyScalar(ex * depth * t * cam.aspect).add(up.multiplyScalar(ey * depth * t));
    cam.position.add(shift);
    this.controls.target.add(shift);
    cam.updateMatrixWorld();
  }

  /**
   * Full-body idle shot, same framing as tools/render_idle_ortho.mjs for a single model: default clip paused at
   * frame 0, near-orthographic (fov 3), camera on +Z at the model centre x, ground (y = 0) `margin` above the
   * bottom edge, frame height = modelHeight * (1 + 2*margin). If the model is wider than the frame at the
   * current aspect, the frame is widened (height grows, feet keep the same relative margin).
   */
  private idleShot(fov = 3, margin = 0.03) {
    const a = this.anim;
    if (a) {
      const d0 = this.defaultClipName;
      if (d0 && a.has(d0) && a.current !== d0) a.play(d0);
      a.setPaused(true); a.scrubToFrame(0); a.update(0);
      this.root.updateMatrixWorld(true);
    }
    const f = this.fitInfo();
    if (!f) return;
    const top = f.center.y + f.size.y / 2;
    const aspect = this.camera.aspect || 1;
    let frameH = top * (1 + 2 * margin);
    frameH = Math.max(frameH, (f.size.x * (1 + 2 * margin)) / aspect);
    const cy = frameH / 2 - margin * top;
    const d = frameH / 2 / Math.tan(THREE.MathUtils.degToRad(fov) / 2);
    const damp = this.controls.enableDamping;
    this.controls.enableDamping = false;
    this.camera.fov = fov;
    this.camera.position.set(f.center.x, cy, d);
    this.controls.target.set(f.center.x, cy, 0);
    this.controls.minDistance = 0; this.controls.maxDistance = d * 10;
    // generous planes: the old far = d + 2r + 5 hid the model after a ~20% dolly out
    this.camera.near = Math.max(0.01, d * 0.05); this.camera.far = d * 10;
    this.camera.updateProjectionMatrix();
    this.camera.lookAt(f.center.x, cy, 0);
    this.controls.update();
    this.controls.enableDamping = damp;
  }

  /** In-battle view (Showdown sprite angles): fixed yaw/pitch/fov, distance from the bounding sphere and the narrower FOV axis. */
  private battleShot(name: 'battle' | 'battleBack') {
    const f = this.fitInfo();
    if (!f) return;
    const { yaw, pitch } = BATTLE_SHOTS[name];
    const mx = this.root.scale.x < 0 ? -1 : 1;
    const y = THREE.MathUtils.degToRad(yaw), p = THREE.MathUtils.degToRad(pitch);
    // yaw 0 = camera on the model's +Z; positive yaw moves the camera to +X (model turned toward the viewer's left)
    const dir = new THREE.Vector3(Math.sin(y) * mx, Math.sin(p) / Math.cos(p), Math.cos(y)).normalize();
    const vHalf = THREE.MathUtils.degToRad(BATTLE_FOV) / 2;
    const half = Math.atan(Math.tan(vHalf) * Math.min(1, this.camera.aspect || 1));
    const d = (f.radius * BATTLE_FILL) / Math.sin(half);
    this.camera.fov = BATTLE_FOV;
    this.applyLimits(f.radius, d);
    this.camera.position.copy(f.center).addScaledVector(dir, d);
    this.controls.target.copy(f.center);
    this.controls.update();
  }

  setCameraPreset(name: CameraPreset) {
    this.releaseLens();
    this.userRoll = 0;
    if (name === 'fit') { this.recenter(); return; }
    if (!this.gltf) { this.pendingPreset = name; return; }
    if (name === 'idle') { this.idleShot(); return; }
    if (name === 'battle' || name === 'battleBack') { this.battleShot(name); return; }
    this.camera.fov = DEFAULT_FOV;
    this.camera.updateProjectionMatrix();
    if (name === 'vs' && this.anim?.has('advent01_loop') && this.anim.current !== 'advent01_loop') {
      // Pose first so the bounds (and therefore the framing) come from the VS pose.
      this.anim.play('advent01_loop');
      this.anim.update(0);
      this.root.updateMatrixWorld(true);
    }
    const f = this.fitInfo();
    if (!f) return;
    const { center: c, dist, h, radius } = f;
    const mx = this.root.scale.x < 0 ? -1 : 1;
    let target = c.clone();
    let dir: THREE.Vector3;
    let d = dist;
    switch (name) {
      case 'front': dir = new THREE.Vector3(0, 0.1, 1); break;
      case 'side': dir = new THREE.Vector3(1, 0.1, 0); break;
      case 'back': dir = new THREE.Vector3(0, 0.1, -1); break;
      case 'top': dir = new THREE.Vector3(0.01, 1, 0.3); break;
      default:
        // In-game VS framing: head and hands, slightly from below, the
        // character turned a little toward the viewer's left.
        // Trainers (have advent01_loop): tuned against VSGriselle/VSTarragon.
        // Pokémon: a looser low 3/4 close-up of the whole body.
        if (this.anim?.has('advent01_loop')) {
          target = c.clone().add(new THREE.Vector3(0, 0.27 * h, 0));
          dir = new THREE.Vector3(0.4, -0.12, 1);
          d = dist * 0.37;
        } else {
          target = c.clone().add(new THREE.Vector3(0, 0.08 * h, 0));
          dir = new THREE.Vector3(0.45, -0.12, 1);
          d = dist * 0.7;
        }
    }
    dir.x *= mx;
    dir.normalize();
    this.applyLimits(radius, d);
    this.camera.position.copy(target).addScaledVector(dir, d);
    this.controls.target.copy(target);
    this.controls.update();
  }

  setHeldOnlyInClips(on: boolean) { this.heldOnlyInClips = on; this.applyHeldVisibility(); }

  private applyHeldVisibility() {
    if (!this.held) return;
    this.held.obj.visible = !this.heldOnlyInClips || (!!this.anim?.current && this.held.prop.clips.includes(this.anim.current));
  }

  setShadows(on: boolean) {
    this.key.castShadow = on;
    this.ground.visible = on;
  }

  setSkeleton(on: boolean) {
    if (this.skeletonHelper) {
      this.scene.remove(this.skeletonHelper);
      this.skeletonHelper.dispose();
      this.skeletonHelper = undefined;
    }
    if (on && this.gltf) {
      this.skeletonHelper = new THREE.SkeletonHelper(this.gltf.scene);
      this.scene.add(this.skeletonHelper);
    }
  }

  setWireframe(on: boolean) { this.wireframe = on; this.applyWireframe(); }

  private applyWireframe() {
    this.gltf?.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh || m.userData.outline) return;
      for (const mat of ([] as THREE.Material[]).concat(m.material)) (mat as THREE.MeshStandardMaterial).wireframe = this.wireframe;
    });
  }

  private bgColor = new THREE.Color('#1b1d23');
  setBackground(color: string) { this.bgColor.set(color); if (this.scene.background) this.scene.background = this.bgColor; }
  /** false = transparent canvas (a reference image is drawn behind the model by the page). */
  setBackgroundVisible(v: boolean) { this.scene.background = v ? this.bgColor : null; }

  setAutoRotate(speed: number) {
    this.controls.autoRotate = speed > 0;
    this.controls.autoRotateSpeed = speed * 2;
  }

  setMirror(on: boolean) { this.root.scale.x = on ? -1 : 1; }

  /** PNG exports always have a transparent background (the background colour is only for the live view). */
  png(filename: string, opts?: Partial<ExportOptions>) {
    if (opts) { this.captureExport(opts).toBlob((b) => b && download(b, filename), 'image/png'); return; }
    let restore = () => {};
    exportCanvasPng(this.renderer.domElement, filename, () => { restore = this.exportQuality(); this.renderTransparent(); }, () => restore());
  }

  /**
   * Renders the current scene at an explicit size / framing / background and returns the result as a 2D canvas.
   * Camera, controls, canvas size and playback are put back afterwards, so the live view is untouched.
   */
  captureExport(opts: Partial<ExportOptions>): HTMLCanvasElement {
    const o = { ...DEFAULT_EXPORT, ...opts };
    const cap = (n: number) => Math.max(1, Math.min(MAX_EXPORT_PX, Math.round(n)));
    const W = cap(o.width * o.scale), H = cap(o.height * o.scale);
    const ss = o.trim ? 2 : 1; // render the subject larger when it is cropped and rescaled afterwards
    const RW = cap(W * ss), RH = cap(H * ss);

    const cam = this.camera, ctl = this.controls, a = this.anim;
    const saved = {
      pos: cam.position.clone(), quat: cam.quaternion.clone(), fov: cam.fov, near: cam.near, far: cam.far, aspect: cam.aspect,
      target: ctl.target.clone(), minD: ctl.minDistance, maxD: ctl.maxDistance, ratio: this.renderer.getPixelRatio(),
      clip: a?.current ?? null, paused: a?.paused ?? false, frame: a ? a.state().frame : 0,
    };
    this.ro.disconnect();
    this.exporting = true;
    const view = cam.view ? { ...cam.view } : null;
    try {
      if (o.framing === 'fit' || o.framing === 'idle') cam.clearViewOffset();
      this.renderer.setPixelRatio(1);
      this.renderer.setSize(RW, RH, false);
      cam.aspect = RW / RH;
      cam.updateProjectionMatrix();
      if (o.framing === 'fit') this.recenter(o.fov);
      else if (o.framing === 'idle') this.idleShot(o.fov, o.margin);
      this.renderTransparent();

      const src = this.renderer.domElement;
      const out = document.createElement('canvas');
      out.width = W; out.height = H;
      const ctx = out.getContext('2d')!;
      ctx.imageSmoothingQuality = 'high';
      const fill = () => { if (o.background !== 'transparent') { ctx.fillStyle = o.background; ctx.fillRect(0, 0, W, H); } };
      fill();
      const bb = o.trim ? alphaBounds(src, RW, RH) : null;
      if (!o.trim || !bb) { ctx.drawImage(src, 0, 0, W, H); return out; }
      const pad = Math.round(o.margin * Math.min(W, H));
      const s = Math.min((W - 2 * pad) / bb.w, (H - 2 * pad) / bb.h);
      const dw = bb.w * s, dh = bb.h * s;
      ctx.drawImage(src, bb.x, bb.y, bb.w, bb.h, (W - dw) / 2, H - pad - dh, dw, dh);
      return out;
    } finally {
      this.exporting = false;
      if (view?.enabled) cam.setViewOffset(view.fullWidth, view.fullHeight, view.offsetX, view.offsetY, view.width, view.height);
      cam.position.copy(saved.pos); cam.quaternion.copy(saved.quat);
      cam.fov = saved.fov; cam.near = saved.near; cam.far = saved.far; cam.aspect = saved.aspect;
      ctl.target.copy(saved.target); ctl.minDistance = saved.minD; ctl.maxDistance = saved.maxD;
      cam.updateProjectionMatrix();
      this.renderer.setPixelRatio(saved.ratio);
      this.ro.observe(this.container);
      this.resize();
      if (a && o.framing === 'idle') {
        if (saved.clip && a.current !== saved.clip) a.play(saved.clip);
        a.scrubToFrame(saved.frame);
        a.setPaused(saved.paused);
      }
      ctl.update();
    }
  }

  /** In low-GPU mode the live view is reduced; exports are still full definition (2x pixels, 2048 shadow map). Returns the undo. */
  private exportQuality(): () => void {
    if (!this.low) return () => {};
    const sh = this.key.shadow;
    this.renderer.setPixelRatio(Math.max(2, Math.min(window.devicePixelRatio, 2)));
    sh.mapSize.set(2048, 2048); sh.map?.dispose(); sh.map = null;
    return () => {
      this.renderer.setPixelRatio(1);
      sh.mapSize.set(1024, 1024); sh.map?.dispose(); sh.map = null;
    };
  }

  /** Same image as png(), as a data URL (used by tools/render_models.mjs). */
  pngDataUrl(opts?: Partial<ExportOptions>): string {
    if (opts) return this.captureExport(opts).toDataURL('image/png');
    const restore = this.exportQuality();
    this.renderTransparent();
    const url = this.renderer.domElement.toDataURL('image/png');
    restore();
    return url;
  }

  private renderTransparent() {
    const bg = this.scene.background;
    this.scene.background = null;
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.render(this.scene, this.camera);
    this.scene.background = bg;
  }

  private loop = () => {
    if (this.disposed) return;
    this.raf = requestAnimationFrame(this.loop);
    const dt = Math.min(this.clock.getDelta(), 0.1);
    if (this.gifBusy) return; // animatedGif() drives the clip and the renderer meanwhile
    this.anim?.update(dt);
    this.prepareFrame(dt);
    this.applyFly(dt);
    this.controls.update();
    this.applyRoll();
    this.applyLens();
    this.applyPivot();
    this.emitCamera();
    if (this.hover) this.renderHovered();
    else if (this.spriteLive()) this.renderSpriteLive();
    else this.renderer.render(this.scene, this.camera);
    const now = performance.now();
    if (this.anim && this.onFrame && now - this.lastEmit > 66) {
      this.lastEmit = now;
      this.onFrame(this.anim.state());
    }
  };

  /** Moves camera + orbit target together along the view (WASD), world up/down (E/Q); speed scales with the scene. */
  private applyFly(dt: number) {
    if (!this.flyEnabled || !this.flyKeys.size || !this.lightFrame) return;
    const k = this.flyKeys;
    const fwd = new THREE.Vector3();
    this.camera.getWorldDirection(fwd);
    const right = new THREE.Vector3().crossVectors(fwd, this.camera.up).normalize();
    const flat = fwd.clone().setY(0);
    if (flat.lengthSq() > 1e-6) flat.normalize(); else flat.copy(fwd);
    const move = new THREE.Vector3();
    if (k.has('w')) move.add(flat);
    if (k.has('s')) move.sub(flat);
    if (k.has('d')) move.add(right);
    if (k.has('a')) move.sub(right);
    if (k.has('e')) move.y += 1;
    if (k.has('q')) move.y -= 1;
    if (move.lengthSq() < 1e-9) return;
    // FITTED: 0.15 x scene radius per second (Pallet Town ~7 m/s), Shift 4x, chosen by eye, 2026-10-20
    const speed = this.lightFrame.radius * 0.15 * (k.has('shift') ? 4 : 1);
    move.normalize().multiplyScalar(speed * dt);
    this.camera.position.add(move);
    this.controls.target.add(move);
  }

  /** Binary STL of the current pose as the view shows it (model + held item), for 3D printing (stl.ts). */
  stl(opts: Partial<StlOptions> = {}): { blob: Blob; triangles: number } {
    this.prepareFrame(0);
    const st = this.anim?.state();
    const header = `${this.manifest?.id ?? 'model'} ${this.anim?.current ?? 'pose'} frame ${st ? Math.round(st.frame) : 0}`;
    return buildStl(this.root, { ...DEFAULT_STL, ...opts }, header);
  }

  /** Per-frame state the clip drives beyond bones: gas/hologram clocks, mesh visibility, UV tracks, held item. */
  private prepareFrame(dt: number) {
    if (this.gasMats.size) {
      // The game's shader clock runs with the game; here it follows playback so pausing freezes the gas.
      if (!this.anim?.paused) this.gasTime += dt * (this.anim?.timeScale ?? 1);
      for (const m of this.gasMats) updateGas(m, this.gasTime);
    }
    if (this.hologram?.on) {
      // Like the gas: the scroll follows playback, so pausing freezes it (and PNG exports are repeatable).
      if (!this.anim?.paused) this.holoTime += dt * (this.anim?.timeScale ?? 1);
      for (const m of this.holoMats) updateHologram(m, this.holoTime);
    }
    this.updateVisibility();
    this.applyUvOffsets();
    this.applyHeldVisibility();
    this.applyJaw();
    if (this.fireMats.size || this.particles.length) {
      // HOME's flame noise scrolls with playback (pausing freezes it; PNG exports repeat)
      if (!this.anim?.paused && this.fxAnimate) this.fxTime += dt * (this.anim?.timeScale ?? 1);
      for (const m of this.fireMats) {
        const u = m.userData.__fire as FireUniforms | undefined;
        if (u) updateFireNoise(u, m.userData.fire as FireSpec, this.fxTime);
        if (m.userData.__homeFire) updateHomeFire(m, this.fxTime);
      }
      for (const e of this.particles) { e.update(); e.setTime(this.fxTime); }
    }
  }

  // ---- 3DS sprite style (spriteStyle.ts) ----
  private spritePass?: SpritePass;
  private gifBusy = false;
  private spriteLive() { return this.outline.on && this.outline.style === 'sprite' && !this.hologram?.on; }
  private sprite(): SpritePass { return (this.spritePass ??= new SpritePass()); }

  /** Sprite render at `w` x `h` with what a sprite never shows (ground shadow, skeleton) hidden. */
  private renderSprite(w: number, h: number, silhouette: string) {
    const sp = this.sprite();
    sp.silhouette.set(silhouette);
    const groundVis = this.ground.visible, skVis = this.skeletonHelper?.visible;
    this.ground.visible = false;
    if (this.skeletonHelper) this.skeletonHelper.visible = false;
    try { sp.render(this.renderer, this.scene, this.camera, w, h, this.lightFrame.radius); } finally {
      this.ground.visible = groundVis;
      if (this.skeletonHelper && skVis !== undefined) this.skeletonHelper.visible = skVis;
    }
  }

  private spriteBuf = new THREE.Vector2();
  private renderSpriteLive() {
    const px = Math.max(1, this.outline.pixel ?? 3);
    this.renderer.getDrawingBufferSize(this.spriteBuf);
    this.renderSprite(Math.ceil(this.spriteBuf.x / px), Math.ceil(this.spriteBuf.y / px), this.outline.color);
    this.sprite().blit(this.renderer, this.scene.background ? this.bgColor : null);
  }

  /** Render target of the GIF export's normal (non-sprite) frames. */
  private gifRt?: THREE.WebGLRenderTarget;
  private gifLut?: Uint8Array;

  /**
   * One frame of the view as it is drawn live, at w x h, transparent background, as straight-alpha top-row-first
   * RGBA: the 3DS sprite pass when that style is on, else the normal renderer (hull outline, hologram, shading
   * included). What the canvas shows besides the subject (ground shadow, skeleton) is hidden, as in sprite renders.
   */
  private renderGifFrame(w: number, h: number): SpriteFrame {
    if (this.spriteLive()) {
      this.renderSprite(w, h, this.outline.color);
      return this.sprite().read(this.renderer);
    }
    // 2x supersampled, box-filtered below (pixels cannot be read back from a multisampled XR-flagged target).
    const SS = 2, RW = w * SS, RH = h * SS;
    let rt = this.gifRt;
    if (!rt) {
      rt = this.gifRt = new THREE.WebGLRenderTarget(RW, RH, { samples: 0 });
      // Same trick as the sprite pass: three only tone-maps / sRGB-encodes on the screen or an XR target.
      (rt as unknown as { isXRRenderTarget: boolean }).isXRRenderTarget = true;
      rt.texture.colorSpace = THREE.SRGBColorSpace;
    }
    if (rt.width !== RW || rt.height !== RH) rt.setSize(RW, RH);
    const r = this.renderer;
    const prevRt = r.getRenderTarget();
    const clear = r.getClearColor(new THREE.Color()), clearA = r.getClearAlpha();
    const bg = this.scene.background;
    const groundVis = this.ground.visible, skVis = this.skeletonHelper?.visible;
    this.ground.visible = false;
    if (this.skeletonHelper) this.skeletonHelper.visible = false;
    const raw = new Uint8Array(RW * RH * 4);
    try {
      this.scene.background = null;
      r.setClearColor(0x000000, 0);
      r.setRenderTarget(rt);
      r.clear();
      r.render(this.scene, this.camera);
      r.readRenderTargetPixels(rt, 0, 0, RW, RH, raw);
    } finally {
      r.setRenderTarget(prevRt);
      r.setClearColor(clear, clearA);
      this.scene.background = bg;
      this.ground.visible = groundVis;
      if (this.skeletonHelper && skVis !== undefined) this.skeletonHelper.visible = skVis;
    }
    // The shader already wrote sRGB values and the SRGB8 texture encoded them once more on store (the sprite pass
    // undoes that by sampling the texture): decode one step here. Then box-filter SS x SS weighting colour by alpha
    // (straight alpha out), flipping to top row first.
    const lut = (this.gifLut ??= Uint8Array.from({ length: 256 }, (_, v) => {
      const c = v / 255;
      return Math.round(255 * (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)));
    }));
    const data = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let r0 = 0, g0 = 0, b0 = 0, a0 = 0;
      for (let sy = 0; sy < SS; sy++) for (let sx = 0; sx < SS; sx++) {
        const i = ((RH - 1 - (y * SS + sy)) * RW + x * SS + sx) * 4, a = raw[i + 3];
        r0 += lut[raw[i]] * a; g0 += lut[raw[i + 1]] * a; b0 += lut[raw[i + 2]] * a; a0 += a;
      }
      const o = (y * w + x) * 4;
      if (a0 > 0) { data[o] = r0 / a0; data[o + 1] = g0 / a0; data[o + 2] = b0 / a0; }
      data[o + 3] = a0 / (SS * SS);
    }
    return { width: w, height: h, data };
  }

  /**
   * Animated GIF of the current clip as the view draws it (3DS sprite style, hull outline or plain), from the live
   * camera. The render is sized so the subject is `height` px tall on frame 0, sampled at `fps`, cropped to the
   * union of all frames. Playback, camera and canvas are untouched; the live loop pauses while it runs.
   */
  async animatedGif(opts: Partial<GifExportOptions> = {}, onProgress?: (done: number, total: number) => void): Promise<Blob> {
    const o = { ...DEFAULT_GIF_EXPORT, ...opts };
    const a = this.anim;
    if (this.gifBusy) throw new Error('a GIF export is already running');
    this.gifBusy = true;
    const saved = a ? { clip: a.current, paused: a.paused, frame: a.state().frame } : null;
    try {
      this.controls.update();
      this.applyRoll();
      this.applyLens();
      this.applyPivot();
      const timing = a?.clipTiming() ?? null;
      const dur = timing?.duration ?? 0;
      // Loops skip the last frame (it equals the first); one-shots keep it.
      const n = dur > 0 ? Math.max(1, timing!.once ? Math.floor(dur * o.fps + 1e-6) + 1 : Math.round(dur * o.fps)) : 1;
      const setTime = (t: number) => a?.seekTime(t);
      if (a) a.setPaused(true);
      // Size: probe frame 0 at the canvas aspect, then scale so the subject is `height` px tall.
      this.renderer.getDrawingBufferSize(this.spriteBuf);
      const aspect = this.spriteBuf.x / Math.max(1, this.spriteBuf.y);
      const PH = 512;
      setTime(0); this.prepareFrame(0);
      const pb = rowBounds(this.renderGifFrame(Math.round(PH * aspect), PH));
      const subj = pb ? pb.y1 - pb.y0 + 1 : PH;
      const H = Math.max(8, Math.min(4096, Math.round(PH * o.height / Math.max(1, subj))));
      const W = Math.max(8, Math.min(4096, Math.round(H * aspect)));
      const frames: SpriteFrame[] = [];
      const gas0 = this.gasTime, holo0 = this.holoTime, fx0 = this.fxTime;
      for (let i = 0; i < n; i++) {
        setTime(i / o.fps);
        this.prepareFrame(i === 0 ? 0 : 1 / o.fps);
        frames.push(this.renderGifFrame(W, H));
        if (i % 4 === 3) { onProgress?.(i + 1, n); await new Promise((r) => setTimeout(r, 0)); }
      }
      this.gasTime = gas0; this.holoTime = holo0; this.fxTime = fx0;
      onProgress?.(n, n);
      return encodeSpriteGif(frames, { colors: o.colors, delayMs: o.delayMs });
    } finally {
      if (a && saved) {
        if (saved.clip && a.current !== saved.clip) a.play(saved.clip);
        a.scrubToFrame(saved.frame);
        a.setPaused(saved.paused);
      }
      this.prepareFrame(0);
      this.gifBusy = false;
    }
  }

  dispose() {
    window.removeEventListener('keydown', this.onFlyKey);
    window.removeEventListener('keyup', this.onFlyKey);
    window.removeEventListener('blur', this.onFlyBlur);
    this.disposed = true;
    for (const e of this.particles) e.dispose();
    this.particles = [];
    this.setHoverPart(null);
    this.highlight?.dispose();
    this.clearHeld();
    cancelAnimationFrame(this.raf);
    this.spritePass?.dispose();
    this.gifRt?.dispose();
    this.ro.disconnect();
    this.controls.dispose();
    this.anim?.mixer?.stopAllAction?.();
    if (this.skeletonHelper) { this.skeletonHelper.dispose(); this.skeletonHelper = undefined; }
    // Free everything the glTF parser created (scene meshes, variant materials
    // not currently attached, and lazily loaded raw/shadow/mask textures).
    if (this.gltf) {
      disposeTree(this.gltf.scene);
      const assoc = (this.gltf.parser as unknown as { associations?: Map<object, unknown> }).associations;
      if (assoc) for (const k of assoc.keys()) {
        const o = k as { isTexture?: boolean; isMaterial?: boolean; dispose?: () => void };
        if ((o.isTexture || o.isMaterial) && o.dispose) o.dispose();
      }
      this.root.remove(this.gltf.scene);
      this.gltf = undefined;
    }
    this.ground.geometry.dispose();
    (this.ground.material as THREE.Material).dispose();
    this.renderer.domElement.removeEventListener('webglcontextlost', this.onContextLost);
    this.renderer.domElement.removeEventListener('webglcontextrestored', this.onContextRestored);
    this.renderer.dispose();
    this.renderer.forceContextLoss(); // release the GL context now instead of waiting for GC
    this.renderer.domElement.remove();
  }

  /** Called when the browser drops the WebGL context (GPU reset / memory pressure). */
  onContextChange?: (lost: boolean) => void;
  private onContextLost = (e: Event) => {
    e.preventDefault(); // allow the browser to restore it
    cancelAnimationFrame(this.raf);
    this.onContextChange?.(true);
  };
  private onContextRestored = () => {
    this.onContextChange?.(false);
  };
}

export interface GifExportOptions {
  /** Subject height in px on frame 0 (Showdown: Pikachu ~55, Charizard ~100). */
  height: number;
  /** Clip sampling rate. */
  fps: number;
  /** GIF frame delay (10 ms steps; Showdown uses 30 or 40). */
  delayMs: number;
  /** Palette size including transparency (Showdown: 16). */
  colors: number;
}
export const DEFAULT_GIF_EXPORT: GifExportOptions = { height: 96, fps: 30, delayMs: 30, colors: 16 };

/** First and last rows with an opaque pixel. */
function rowBounds(f: SpriteFrame) {
  let y0 = -1, y1 = -1;
  for (let y = 0; y < f.height; y++) {
    for (let x = 0; x < f.width; x++) if (f.data[(y * f.width + x) * 4 + 3] >= 128) { if (y0 < 0) y0 = y; y1 = y; break; }
  }
  return y1 < 0 ? null : { y0, y1 };
}

/** Map the game's blend type (glTF materials[i].extras.blend) onto three.js blending. glTF only has
 *  alphaMode BLEND (straight alpha), which has no additive mode ("Add": glows, energy effects). */
export function applyGameBlend(mat: THREE.Material) {
  if ((mat.userData?.lid_overlay || mat.userData?.layer_overlay) && !mat.userData.__lidDone) {
    // Sliding eyelid layer (importer `_lid_overlay`) or material layer 5 (`_layer5_overlay`: Xerneas' horn glow): the
    // same triangles drawn again on top with the layer texture. Pull it towards the camera so it never z-fights the
    // surface below, and keep that surface's depth.
    mat.userData.__lidDone = true;
    mat.transparent = true;
    mat.depthWrite = false;
    mat.polygonOffset = true;
    mat.polygonOffsetFactor = -1;
    mat.polygonOffsetUnits = -4;
    mat.needsUpdate = true;
  }
  const blend = mat.userData?.blend as string | undefined;
  if (!blend || mat.userData.__blendDone) return;
  mat.userData.__blendDone = true;
  mat.transparent = true;
  if (blend === 'Add') {
    mat.blending = THREE.AdditiveBlending;
    mat.depthWrite = false;
  } else if (blend === 'BlendPreMultiAlpha') {
    // The game shader scales colour by alpha itself, then blends ONE / ONE_MINUS_SRC_ALPHA (Lampent's
    // glass globe, alpha ~0.22). Adding the unscaled colour (blendSrc ONE) blew the globe out to white.
    mat.blending = THREE.NormalBlending;
    mat.premultipliedAlpha = true;
    mat.depthWrite = false;
  } else {
    mat.blending = THREE.NormalBlending; // "Blend": glasses lenses, Munna's mist...
    // UNITE effect cards (flags, glow) never hide what is behind them; solid ones (an opaque body atlas) are ordinary surfaces
    if (mat.userData.effect && !mat.userData.solidEffect) mat.depthWrite = false;
  }
  mat.needsUpdate = true;
}

/** Bounding box of the pixels with alpha > 0 in a (WebGL) canvas, or null when it is empty. */
function alphaBounds(src: HTMLCanvasElement, w: number, h: number) {
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const ctx = tmp.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(src, 0, 0);
  const d = ctx.getImageData(0, 0, w, h).data;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0, i = y * w * 4 + 3; x < w; x++, i += 4) {
      if (d[i] > 0) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; y1 = y; }
    }
  }
  return x1 < 0 ? null : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

function disposeTree(root: THREE.Object3D) {
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    if (!m.isMesh) return;
    m.geometry.dispose();
    for (const mat of ([] as THREE.Material[]).concat(m.material)) {
      for (const v of Object.values(mat)) if (v && (v as THREE.Texture).isTexture) (v as THREE.Texture).dispose();
      mat.dispose();
    }
  });
}
