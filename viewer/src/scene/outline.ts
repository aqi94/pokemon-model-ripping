import * as THREE from 'three';

/**
 * Cartoon outline (inverted hull): a back-face copy of each opaque mesh whose vertices are pushed outward along
 * the smoothed normal by a constant number of screen pixels. Skinning is applied before the push, so it follows
 * every pose; it needs no post pass, so transparent PNG exports, the lens shift and the probe renders are unchanged.
 */

/** `tinted`: the line is a dark shade of the surface it hugs (its own albedo x TINT_DARKNESS) instead of one flat colour, like the game's outlines. */
/** `style` 'sprite' = the 3DS battle-sprite look (spriteStyle.ts): low-res render + 1-px screen-space lines, `pixel` px per sprite pixel. */
export interface OutlineState { on: boolean; width: number; color: string; tinted: boolean; style?: 'auto' | 'hull' | 'sprite'; pixel?: number }
export const DEFAULT_OUTLINE: OutlineState = { on: false, width: 2, color: '#14121a', tinted: false, style: 'auto', pixel: 3 };
// FITTED: tinted outline darkness, eyeballed against game outlines, 2026-10-19 (engine-rules debt)
const TINT_DARKNESS = 0.3;

/** Reference viewport height: `width` is in pixels at this height and scales with the canvas (exports match the live view). */
const REF_HEIGHT = 800;

/** x = width in drawing-buffer pixels, y/z = drawing-buffer size. Shared by every outline material. */
export const outlineUniform = { value: new THREE.Vector3(2, 1, 1) };

/** Game-data mode: x = line width in drawing-buffer pixels, y/z = buffer size. */
export const gameOutlineUniform = { value: new THREE.Vector3(2, 1, 1) };
/** The game's line width: the OUTLINE vertex programs shift a vertex by 1/216 NDC (0.00462962966 in `m_lob_basepbr_lv3.txt` :42220) x
 *  `_ScreenResolution.y` / `_ScreenParams.y`. `_ScreenResolution` is a global the rip does not hold; taking it as the screen height
 *  gives 1/216 of NDC = H/432 px, so the width scales with the buffer exactly like the hull's px (exports match the live view). */
export const GAME_WIDTH_NDC = 1 / 216;

export function updateOutlineUniform(width: number, bufW: number, bufH: number) {
  outlineUniform.value.set(width * bufH / REF_HEIGHT, bufW, bufH);
  gameOutlineUniform.value.set(GAME_WIDTH_NDC * bufH / 2, bufW, bufH);
}

/** `extras.unite.outline` of a material (importer `outline.py`); `map` is a glTF texture reference. */
export interface GameOutline { color: number[]; lightDir: number[]; offset: number; width_ndc: number; vtex: boolean; vcolor2n?: boolean; map?: { index: number } }
export const gameOutlineOf = (m: THREE.Material): GameOutline | null => (m.userData?.unite?.outline as GameOutline | undefined) ?? null;
/** Does any material of the model carry game outline data (decides the default style)? */
export function modelHasGameOutline(root: THREE.Object3D): boolean {
  let found = false;
  root.traverse((o) => { const m = o as THREE.Mesh; if (!found && m.isMesh) for (const x of ([] as THREE.Material[]).concat(m.material)) if (gameOutlineOf(x)) found = true; });
  return found;
}

/** Unity world -> glTF: UNITE's lobby scenes face +Z in Unity (left-handed) and the importer keeps the model's own axes, so the
 *  game's world-space `_lightDir` x is mirrored. INFERRED (ledger `shader:outline`): check with the render. */
const UNITY_X = -1;

/**
 * Position-welded vertex normals (split vertices at hard edges would open gaps in the hull), stored as the
 * `outlineNormal` attribute of the geometry once.
 */
export function ensureOutlineNormals(geo: THREE.BufferGeometry): boolean {
  if (geo.getAttribute('outlineNormal')) return true;
  const pos = geo.getAttribute('position'), nrm = geo.getAttribute('normal');
  if (!pos || !nrm) return false;
  const n = pos.count;
  const key = (i: number) => `${Math.round(pos.getX(i) * 1e5)},${Math.round(pos.getY(i) * 1e5)},${Math.round(pos.getZ(i) * 1e5)}`;
  const sum = new Map<string, [number, number, number]>();
  for (let i = 0; i < n; i++) {
    const k = key(i);
    let s = sum.get(k);
    if (!s) { s = [0, 0, 0]; sum.set(k, s); }
    s[0] += nrm.getX(i); s[1] += nrm.getY(i); s[2] += nrm.getZ(i);
  }
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const s = sum.get(key(i))!;
    const l = Math.hypot(s[0], s[1], s[2]) || 1;
    out[i * 3] = s[0] / l; out[i * 3 + 1] = s[1] / l; out[i * 3 + 2] = s[2] / l;
  }
  geo.setAttribute('outlineNormal', new THREE.BufferAttribute(out, 3));
  return true;
}

export function makeOutlineMaterial(src: THREE.MeshStandardMaterial, color: string, tinted: boolean): THREE.MeshBasicMaterial {
  const cut = src.alphaTest > 0 && !!src.map;
  const mat = new THREE.MeshBasicMaterial({
    color: tinted ? src.color.clone().multiplyScalar(TINT_DARKNESS) : color, side: THREE.BackSide, alphaTest: cut ? src.alphaTest : 0, map: cut || tinted ? src.map : null,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, toneMapped: false,
  });
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uOutline = outlineUniform;
    shader.vertexShader = shader.vertexShader
      .replace('void main() {', 'attribute vec3 outlineNormal;\nuniform vec3 uOutline;\nvoid main() {')
      .replace('#include <project_vertex>', `#include <project_vertex>
        vec3 oN = outlineNormal;
        #ifdef USE_SKINNING
          mat4 oSkin = skinWeight.x * boneMatX + skinWeight.y * boneMatY + skinWeight.z * boneMatZ + skinWeight.w * boneMatW;
          oSkin = bindMatrixInverse * oSkin * bindMatrix;
          oN = (oSkin * vec4(oN, 0.0)).xyz;
        #endif
        vec2 oD = (projectionMatrix * vec4(normalize(normalMatrix * oN), 0.0)).xy * uOutline.yz;
        if (dot(oD, oD) > 1e-12) gl_Position.xy += normalize(oD) * (2.0 * uOutline.x) / uOutline.yz * gl_Position.w;`);
  };
  mat.customProgramCacheKey = () => 'outline' + (cut ? '|cut' : '');
  return mat;
}

/** Outline copy of `m` (a child with an identity transform: it follows pose, visibility and mirroring), or null if `m` should have none. */
export function makeOutlineMesh(m: THREE.Mesh, color: string, tinted: boolean, game?: { maps: Map<number, THREE.Texture> }): THREE.Mesh | null {
  const srcs = ([] as THREE.Material[]).concat(m.material);
  const ok = (s: THREE.Material): s is THREE.MeshStandardMaterial => {
    const sm = s as THREE.MeshStandardMaterial;
    return !!sm.isMeshStandardMaterial && !sm.transparent && sm.depthWrite && !sm.userData?.gas && !sm.name.includes('~');
  };
  if (!srcs.some(ok) || !ensureOutlineNormals(m.geometry)) return null;
  const mats = srcs.map((s) => {
    if (game) { // game-data mode: only materials whose shader has the pass get a hull
      const go = ok(s) ? gameOutlineOf(s) : null; // ok() narrows to MeshStandardMaterial
      if (go) return makeGameOutlineMaterial(s as THREE.MeshStandardMaterial, go, go.map ? game.maps.get(go.map.index) ?? null : null);
      const hide = new THREE.MeshBasicMaterial(); hide.visible = false; return hide;
    }
    if (ok(s)) return makeOutlineMaterial(s, color, tinted);
    const hide = new THREE.MeshBasicMaterial(); hide.visible = false; return hide;
  });
  const mat = Array.isArray(m.material) ? mats : mats[0]; // an array material needs geometry groups, a single one does not
  const sk = m as THREE.SkinnedMesh;
  let p: THREE.Mesh;
  if (sk.isSkinnedMesh) { const q = new THREE.SkinnedMesh(sk.geometry, mat); q.bind(sk.skeleton, sk.bindMatrix); q.bindMode = sk.bindMode; p = q; }
  else p = new THREE.Mesh(m.geometry, mat);
  if (m.morphTargetInfluences) p.morphTargetInfluences = m.morphTargetInfluences;
  p.userData.outline = true;
  p.frustumCulled = false; p.castShadow = false; p.receiveShadow = false;
  p.renderOrder = m.renderOrder;
  p.raycast = () => {};
  return p;
}


/**
 * Game outline (`shader:outline`): the OUTLINE pass of the lit lobby shaders. Colour = mix(_OutlineColor, 1, clamp(N.L)) x map.rgb with the
 * world normal and `_lightDir` (`m_lob_basepbr_lv3.txt` 42185-42197), a constant screen width (see GAME_WIDTH_NDC) and a push away
 * from the camera along the view ray of length(view shift) x map.a x |_Offset| x 15 (:42198, :42232-42238). Values are display-space
 * (the game is gamma-space), so the colour is decoded to linear for three's sRGB output. Not read: `_VCOLOR2N` (normals from vertex
 * colours, which the importer drops: the welded normal is used), `COLOR_0` when `_VTEX` is off (taken as white).
 */
export function makeGameOutlineMaterial(src: THREE.MeshStandardMaterial, go: GameOutline, map: THREE.Texture | null): THREE.MeshBasicMaterial {
  const cut = src.alphaTest > 0 && !!src.map;
  const mat = new THREE.MeshBasicMaterial({
    color: 0xffffff, side: THREE.BackSide, alphaTest: cut ? src.alphaTest : 0, map: cut ? src.map : null,
    polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, toneMapped: false,
  });
  const ld = new THREE.Vector3(go.lightDir[0] * UNITY_X, go.lightDir[1], go.lightDir[2]).normalize();
  const uniforms = {
    uOutline: gameOutlineUniform,
    uOColor: { value: new THREE.Vector3(go.color[0], go.color[1], go.color[2]) },
    uOLight: { value: ld },
    uOOffset: { value: Math.abs(go.offset) },
    uOMap: { value: map },
  };
  const hasMap = !!map;
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('void main() {', `attribute vec3 outlineNormal;
uniform vec3 uOutline; uniform vec3 uOColor; uniform vec3 uOLight; uniform float uOOffset;
${hasMap ? 'uniform sampler2D uOMap;' : ''}
varying vec3 vOColor;
vec3 oDecode(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
void main() {`)
      .replace('#include <project_vertex>', `#include <project_vertex>
        vec3 oN = outlineNormal; vec3 oL = normal;
        #ifdef USE_SKINNING
          mat4 oSkin = skinWeight.x * boneMatX + skinWeight.y * boneMatY + skinWeight.z * boneMatZ + skinWeight.w * boneMatW;
          oSkin = bindMatrixInverse * oSkin * bindMatrix;
          oN = (oSkin * vec4(oN, 0.0)).xyz; oL = (oSkin * vec4(oL, 0.0)).xyz;
        #endif
        vec3 wN = normalize((modelMatrix * vec4(oL, 0.0)).xyz);
        vec3 oC = mix(uOColor, vec3(1.0), clamp(dot(wN, uOLight), 0.0, 1.0));
        float oA = 1.0;
        ${hasMap ? 'vec4 oM = texture2D(uOMap, uv); oC *= oM.rgb; oA = oM.a;' : ''}
        vOColor = oDecode(oC);
        vec2 oCd = (projectionMatrix * vec4(normalize(normalMatrix * oN), 0.0)).xy;
        if (dot(oCd, oCd) > 1e-12) {
          vec2 oShiftClip = normalize(oCd) * (2.0 * uOutline.x) / uOutline.yz * gl_Position.w;
          vec2 oShift = oShiftClip / vec2(projectionMatrix[0][0], projectionMatrix[1][1]);
          mvPosition.xy += oShift;
          vec3 oDir = normalize(mvPosition.xyz); oDir /= max(abs(oDir.z), 1e-4);
          mvPosition.xyz += oDir * (max(length(oShift), 0.001) * oA * uOOffset * 15.0);
          gl_Position = projectionMatrix * mvPosition;
        }`);
    shader.fragmentShader = shader.fragmentShader
      .replace('void main() {', 'varying vec3 vOColor;\nvoid main() {')
      .replace('#include <opaque_fragment>', 'gl_FragColor = vec4(max(vOColor, vec3(0.0)), 1.0);')
      .replace('#include <color_fragment>', '');
  };
  mat.customProgramCacheKey = () => 'outlineGame' + (cut ? '|cut' : '') + (hasMap ? '|map' : '');
  return mat;
}
