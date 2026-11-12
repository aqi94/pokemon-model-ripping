import * as THREE from 'three';
import { FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';

/**
 * Model Parts hover highlight, after modelviewer.lol (whose look is three's OutlinePass: visible edges 0xFFD700, hidden edges
 * 0xDAA520, strength 4, thickness 3 px, 2 s pulse; other parts at 40 % colour; a hidden part ghosted at 5 % opacity).
 * Drawn as an overlay after the normal screen render, not through EffectComposer: the renderer tone-maps on the screen
 * pass (custom Hable curve included), which a composer's OutputPass cannot reproduce.
 *
 * Mask target: B = other visible model meshes, R = every pixel of the hovered meshes, G = their pixels in front of the rest.
 * Then, on screen: x0.4 where B and not G (dim), and a dilated R edge, gold where the part is in front, goldenrod behind.
 */

const VISIBLE_EDGE = new THREE.Color(0xffd700);
const HIDDEN_EDGE = new THREE.Color(0xdaa520);
const THICKNESS = 3; // px
const PULSE_PERIOD = 2; // s
const DIM = 0.4;

const flat = (r: number, g: number, b: number, depthTest: boolean) => new THREE.MeshBasicMaterial({
  color: new THREE.Color(r, g, b), depthTest, depthWrite: depthTest && b > 0, blending: THREE.AdditiveBlending, transparent: false, side: THREE.DoubleSide,
});

/** `base` cut out by `src`'s texture alpha (eye / mouth cards are quads whose shape is the texture's alpha), colour kept flat. */
function cutout(base: THREE.MeshBasicMaterial, src: THREE.Material): THREE.MeshBasicMaterial {
  const s = src as THREE.MeshStandardMaterial;
  const m = base.clone();
  m.map = s.map ?? null;
  m.alphaMap = s.alphaMap ?? null;
  m.alphaTest = s.alphaTest > 0 ? s.alphaTest : 0.5;
  const flatColor = base.color.clone();
  m.onBeforeCompile = (sh) => {
    sh.uniforms.flatColor = { value: flatColor };
    sh.fragmentShader = 'uniform vec3 flatColor;\n' + sh.fragmentShader.replace('#include <opaque_fragment>', 'outgoingLight = flatColor;\n#include <opaque_fragment>');
  };
  m.customProgramCacheKey = () => 'partHighlightCutout';
  return m;
}
const hasAlpha = (m: THREE.Material) => {
  const s = m as THREE.MeshStandardMaterial;
  return !!(s.map || s.alphaMap) && (s.alphaTest > 0 || s.transparent);
};

const VERT = /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

/** Helpers drawn on top of model meshes (outline / aura hulls, hologram depth prepasses) and non-mesh renderables stay out of the mask. */
const isHelper = (o: THREE.Object3D) => !!(o.userData.outline || o.userData.aura || o.userData.holoPrepass);

export class PartHighlight {
  private rt = new THREE.WebGLRenderTarget(1, 1, { depthBuffer: true });
  private others = flat(0, 0, 1, true);
  private all = flat(1, 0, 0, false);
  private front = flat(0, 1, 0, true);
  private dim = new FullScreenQuad(new THREE.ShaderMaterial({
    uniforms: { tMask: { value: null }, exempt: { value: 1 } },
    vertexShader: VERT,
    fragmentShader: /* glsl */ `uniform sampler2D tMask; varying vec2 vUv;
      uniform float exempt;
      void main() { vec4 m = texture2D(tMask, vUv); gl_FragColor = vec4(vec3(m.b > 0.5 && m.g * exempt < 0.5 ? ${DIM.toFixed(2)} : 1.0), 1.0); }`,
    depthTest: false, depthWrite: false, transparent: true,
    blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.ZeroFactor, blendDst: THREE.SrcColorFactor,
    blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor,
  }));
  private edge = new FullScreenQuad(new THREE.ShaderMaterial({
    uniforms: { tMask: { value: null }, texel: { value: new THREE.Vector2() }, visibleColor: { value: VISIBLE_EDGE }, hiddenColor: { value: HIDDEN_EDGE }, pulse: { value: 1 } },
    vertexShader: VERT,
    fragmentShader: /* glsl */ `uniform sampler2D tMask; uniform vec2 texel; uniform vec3 visibleColor, hiddenColor; uniform float pulse; varying vec2 vUv;
      void main() {
        if (texture2D(tMask, vUv).r > 0.5) discard;
        float best = 1e3; float front = 0.0;
        for (int y = -${THICKNESS}; y <= ${THICKNESS}; y++) for (int x = -${THICKNESS}; x <= ${THICKNESS}; x++) {
          float d = length(vec2(x, y));
          if (d > ${THICKNESS}.5 || d >= best) continue;
          vec4 m = texture2D(tMask, vUv + vec2(x, y) * texel);
          if (m.r > 0.5) { best = d; front = m.g; }
        }
        if (best > ${THICKNESS}.5) discard;
        float a = clamp(1.0 - (best - 1.0) / ${THICKNESS}.0, 0.0, 1.0) * pulse; // solid at the edge, glowing out
        gl_FragColor = vec4(front > 0.5 ? visibleColor : hiddenColor, a);
      }`,
    depthTest: false, depthWrite: false, transparent: true, blending: THREE.NormalBlending,
  }));
  private size = new THREE.Vector2();
  private cutouts = new Map<THREE.MeshBasicMaterial, WeakMap<THREE.Material, THREE.MeshBasicMaterial>>();
  private made: THREE.Material[] = [];

  private maskFor(base: THREE.MeshBasicMaterial, src: THREE.Material): THREE.MeshBasicMaterial {
    if (!hasAlpha(src)) return base;
    let per = this.cutouts.get(base);
    if (!per) this.cutouts.set(base, (per = new WeakMap()));
    let m = per.get(src);
    if (!m) { m = cutout(base, src); per.set(src, m); this.made.push(m); }
    return m;
  }
  private clear = new THREE.Color();

  /**
   * Overlay the highlight on what is on screen. `root` = everything the view counts as the model (held items, rider, mount);
   * `targets` = the hovered part's meshes; `force` = an object made visible for these passes (a hidden part's node).
   */
  render(renderer: THREE.WebGLRenderer, camera: THREE.Camera, root: THREE.Object3D, targets: Set<THREE.Mesh>, force: THREE.Object3D | null, time: number) {
    renderer.getDrawingBufferSize(this.size);
    if (this.rt.width !== this.size.x || this.rt.height !== this.size.y) this.rt.setSize(this.size.x, this.size.y);

    const forceWas = force?.visible ?? true;
    if (force) force.visible = true;
    const meshes: THREE.Mesh[] = [];
    const hidden: THREE.Object3D[] = [];
    root.traverseVisible((o) => {
      const m = o as THREE.Mesh;
      if (m.isMesh && !isHelper(m)) { meshes.push(m); return; }
      if ((o as THREE.Line).isLine || (o as THREE.Points).isPoints || (o as THREE.Sprite).isSprite || isHelper(o)) hidden.push(o);
    });
    const saved = meshes.map((m) => m.material);
    hidden.forEach((o) => { o.visible = false; });

    const wasTarget = renderer.getRenderTarget(), autoClear = renderer.autoClear, alpha = renderer.getClearAlpha();
    renderer.getClearColor(this.clear);
    renderer.setRenderTarget(this.rt);
    renderer.setClearColor(0x000000, 0);
    renderer.clear(true, true, false);
    renderer.autoClear = false;
    const pass = (mat: THREE.MeshBasicMaterial, mine: boolean) => {
      meshes.forEach((m, i) => {
        m.visible = targets.has(m) === mine;
        const src = saved[i];
        m.material = Array.isArray(src) ? src.map((x) => this.maskFor(mat, x)) : this.maskFor(mat, src);
      });
      renderer.render(root, camera);
    };
    // skinned / morphed meshes deform in these flat materials too (three adds the defines per object)
    pass(this.others, false);
    pass(this.all, true);
    pass(this.front, true);
    meshes.forEach((m, i) => { m.visible = true; m.material = saved[i]; });
    hidden.forEach((o) => { o.visible = true; });
    if (force) force.visible = forceWas;

    renderer.setRenderTarget(wasTarget);
    const dm = this.dim.material as THREE.ShaderMaterial, em = this.edge.material as THREE.ShaderMaterial;
    dm.uniforms.tMask.value = this.rt.texture;
    dm.uniforms.exempt.value = force ? 0 : 1; // a ghosted (hidden) part does not hide the dimmed parts behind it
    em.uniforms.tMask.value = this.rt.texture;
    em.uniforms.texel.value.set(1 / this.size.x, 1 / this.size.y);
    // OutlinePass's pulse: between 0.25 and 1 of full strength
    em.uniforms.pulse.value = (1 + 0.25) / 2 + (Math.cos((time * 2 * Math.PI) / PULSE_PERIOD) * (1 - 0.25)) / 2;
    this.dim.render(renderer);
    this.edge.render(renderer);
    renderer.autoClear = autoClear;
    renderer.setClearColor(this.clear, alpha);
  }

  dispose() {
    this.rt.dispose();
    this.others.dispose(); this.all.dispose(); this.front.dispose();
    this.made.forEach((m) => m.dispose());
    this.dim.material.dispose(); this.dim.dispose();
    this.edge.material.dispose(); this.edge.dispose();
  }
}
