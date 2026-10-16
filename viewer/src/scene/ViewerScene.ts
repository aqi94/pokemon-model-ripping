import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { Manifest, PropInfo } from '../types';
import { AnimationController, type PlaybackState } from './animation';
import { VariantSwitcher } from './variants';
import { loadGlb } from './loader';
import { exportCanvasPng } from './exporter';
import { DEFAULT_ENV, DEFAULT_HABLE, kelvinToRGB, type Environment, type ToneMap } from './environment';
import { installHable } from './hable';
import { applyGameShading, removeGameShading, syncUnlit } from './gameShading';
import { isReflective, makeReflectionEnv } from './reflection';
import { bakePalette, type PaletteData } from './palette';

export type ShadingMode = 'game' | 'pbr';
export type CameraPreset = 'fit' | 'vs' | 'front' | 'side' | 'back' | 'top';
export type AlbedoMode = 'baked' | 'raw' | 'off';

const TONE_MAPS: Record<ToneMap, THREE.ToneMapping> = {
  linear: THREE.LinearToneMapping, aces: THREE.ACESFilmicToneMapping, agx: THREE.AgXToneMapping, neutral: THREE.NeutralToneMapping,
  hable: THREE.CustomToneMapping, // Hable filmic with the game's parameters, see hable.ts
};

interface AlbedoState { baked: THREE.Texture | null; color: THREE.Color; raw?: THREE.Texture | null }

export class ViewerScene {
  static THREE = THREE; // for page.evaluate tooling (tools/*.mjs) that needs the viewer's own three instance
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(35, 1, 0.01, 1000);
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

  /** `low` = reduced GPU load (no MSAA, pixel ratio 1, 1024 shadow map), used after a lost WebGL context. */
  constructor(private container: HTMLElement, private low = false) {
    this.renderer = new THREE.WebGLRenderer({ antialias: !low, alpha: true }); // alpha: transparent PNG export
    this.renderer.setPixelRatio(low ? 1 : Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.LinearToneMapping; // linear = no curve, but honours toneMappingExposure
    this.renderer.toneMappingExposure = 1;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.addEventListener('webglcontextlost', this.onContextLost, false);
    this.renderer.domElement.addEventListener('webglcontextrestored', this.onContextRestored, false);
    this.scene.background = new THREE.Color('#1b1d23');
    this.scene.add(this.root);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x555566, 1.2);
    this.scene.add(hemi);
    this.key.position.set(3, 6, 4);
    this.key.castShadow = false;
    this.key.shadow.mapSize.set(low ? 1024 : 2048, low ? 1024 : 2048);
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
    this.manifest = manifest;
    const gltf = await loadGlb(glbUrl, onProgress);
    if (this.disposed) return;
    this.gltf = gltf;
    this.root.add(gltf.scene);

    // LOD>0 and hidden-by-default nodes
    for (const p of manifest.parts) {
      const obj = this.findNode(p.node);
      if (obj && (p.lod > 0 || !p.visible)) obj.visible = false;
    }
    gltf.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      m.castShadow = true;
      m.frustumCulled = false;
      for (const mat of ([] as THREE.Material[]).concat(m.material)) {
        const s = mat as THREE.MeshStandardMaterial;
        if ('envMapIntensity' in s) s.envMapIntensity = 0.2;
        applyGameBlend(mat);
      }
    });

    this.variants = new VariantSwitcher(gltf);
    this.variants.onMaterialsChanged = () => { this.applyWireframe(); this.applyShading(); void this.applyAlbedo(); };
    if (this.variants.available) {
      const def = manifest.variantGroups[0]?.options.find((o) => o.default)?.variant ?? 0;
      await this.variants.select(def);
    }

    this.applyShading();
    this.anim = new AnimationController(gltf.scene, gltf.animations, manifest.clips);
    const start = manifest.defaultClip ?? manifest.clips.find((c) => c.default)?.name ?? manifest.clips[0]?.name;
    if (start && this.anim.has(start)) this.anim.play(start);

    this.applyWireframe();
    this.uvMats.clear(); this.uvDriven.clear();
    this.updateVisibility();
    this.applyUvOffsets();
    this.recenter();
    if (this.pendingPreset) { const p = this.pendingPreset; this.pendingPreset = undefined; this.setCameraPreset(p); }
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

  setPartVisible(node: string, v: boolean) {
    this.userOverride.set(node, v);
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
    const tracks = (cur ? man.clips.find((c) => c.name === cur) : undefined)?.uvOffset;
    const time = this.anim ? this.anim.state().time : 0;
    const names = new Set<string>([...this.uvDriven, ...Object.keys(tracks ?? {})]);
    for (const name of names) {
      let mats = this.uvMats.get(name);
      if (!mats) {
        mats = [];
        this.gltf.scene.traverse((o) => {
          const m = o as THREE.Mesh;
          if (!m.isMesh) return;
          for (const mat of ([] as THREE.Material[]).concat(m.material)) if (mat.name === name && (mat as THREE.MeshStandardMaterial).isMeshStandardMaterial && !mats!.includes(mat as THREE.MeshStandardMaterial)) mats!.push(mat as THREE.MeshStandardMaterial);
        });
        this.uvMats.set(name, mats);
      }
      const tr = tracks?.[name];
      let x = 0, y = 0;
      if (tr && tr.times.length) {
        const t = tr.times, v = tr.values;
        let i = 0;
        while (i + 1 < t.length && t[i + 1] <= time) i++;
        if (i + 1 < t.length && t[i + 1] > t[i]) {
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
        const ox = x * this.uvSign.x, oy = y * this.uvSign.y;
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
        const t = tracks?.[p.node];
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

  /** Fit camera to visible meshes (LOD0). */
  recenter() {
    if (!this.gltf) return;
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
    const dist = (radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 0.95;
    this.camera.near = radius / 100;
    this.camera.far = radius * 100;
    this.camera.updateProjectionMatrix();
    const dir = new THREE.Vector3(0.6, 0.25, 1).normalize();
    this.camera.position.copy(center).addScaledVector(dir, dist);
    this.controls.target.copy(center);
    this.controls.minDistance = radius * 0.3;
    this.controls.maxDistance = radius * 20;
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

  setAlbedoMode(mode: AlbedoMode) { this.albedoMode = mode; this.applyShading(); return this.applyAlbedo(); }

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
      for (const mat of ([] as THREE.Material[]).concat(m.material)) if (materialNames.includes(mat.name)) mats.add(mat as THREE.MeshStandardMaterial);
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
      if (!m.isMesh) return;
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
    const hableKey = env.toneMapping === 'hable' ? installHable(env.hable ?? DEFAULT_HABLE) : '';
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
        const hk = this.hableKey;
        sm.customProgramCacheKey = () => hk;
        if (this.shadingMode === 'game') {
          const tex = {
            shadow: this.albedoMode === 'baked' ? this.getTex(sm, parser, 'shadowColorTexture', true) : null,
            spec: this.getTex(sm, parser, 'specularMaskTexture', false),
            rim: this.getTex(sm, parser, 'rimMaskTexture', false),
          };
          if (isReflective(sm)) this.reflEnv ??= makeReflectionEnv(this.renderer);
          applyGameShading(sm, this.env, tex, isReflective(sm) ? this.reflEnv ?? null : null, (sm.userData as { reflect?: { envIntensity?: number } }).reflect?.envIntensity ?? 1.5);
        } else removeGameShading(sm);
      }
    });
    if (this.gltf) each(this.gltf.scene, this.gltf.parser);
    if (this.held) each(this.held.obj, this.held.parser);
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

  private fitInfo() {
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
    const dist = (radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 0.95;
    return { size, center, radius, dist, h: size.y };
  }

  private pendingPreset?: CameraPreset;

  /** Explicit camera pose (world metres), used by the lighting reference tools and the Reference panel. */
  setCameraPose(fov: number, position: [number, number, number], target: [number, number, number]) {
    const damp = this.controls.enableDamping;
    this.controls.enableDamping = false;
    this.camera.fov = fov;
    this.camera.position.set(position[0], position[1], position[2]);
    this.controls.target.set(target[0], target[1], target[2]);
    const fi = this.fitInfo();
    if (fi) { this.camera.near = Math.max(0.001, fi.radius * 0.01); this.camera.far = Math.max(100, fi.radius * 100); }
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.controls.enableDamping = damp;
  }

  setCameraPreset(name: CameraPreset) {
    if (name === 'fit') { this.recenter(); return; }
    if (!this.gltf) { this.pendingPreset = name; return; }
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
    this.camera.near = radius / 100;
    this.camera.far = radius * 100;
    this.camera.updateProjectionMatrix();
    this.camera.position.copy(target).addScaledVector(dir, d);
    this.controls.target.copy(target);
    this.controls.minDistance = radius * 0.3;
    this.controls.maxDistance = radius * 20;
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
      if (!m.isMesh) return;
      for (const mat of ([] as THREE.Material[]).concat(m.material)) (mat as THREE.MeshStandardMaterial).wireframe = this.wireframe;
    });
  }

  setBackground(color: string) { (this.scene.background as THREE.Color).set(color); }

  setAutoRotate(speed: number) {
    this.controls.autoRotate = speed > 0;
    this.controls.autoRotateSpeed = speed * 2;
  }

  setMirror(on: boolean) { this.root.scale.x = on ? -1 : 1; }

  /** PNG exports always have a transparent background (the background colour is only for the live view). */
  png(filename: string) {
    let restore = () => {};
    exportCanvasPng(this.renderer.domElement, filename, () => { restore = this.exportQuality(); this.renderTransparent(); }, () => restore());
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
  pngDataUrl(): string {
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
    this.anim?.update(dt);
    this.updateVisibility();
    this.applyUvOffsets();
    this.applyHeldVisibility();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    const now = performance.now();
    if (this.anim && this.onFrame && now - this.lastEmit > 66) {
      this.lastEmit = now;
      this.onFrame(this.anim.state());
    }
  };

  dispose() {
    this.disposed = true;
    this.clearHeld();
    cancelAnimationFrame(this.raf);
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

/** Map the game's blend type (glTF materials[i].extras.blend) onto three.js blending. glTF only has
 *  alphaMode BLEND (straight alpha), which has no additive mode ("Add": glows, energy effects). */
export function applyGameBlend(mat: THREE.Material) {
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
  }
  mat.needsUpdate = true;
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
