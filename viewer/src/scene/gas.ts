import * as THREE from 'three';

/**
 * Animated gas shells (Gastly's smoke). The importer bakes one frame of the game's effect into the GLB
 * (vertices moved along their normals by DisplacementHeight x (weight x cloud density - retraction), see
 * `importer.gas_displacement`) and ships the cloud-density map as `extras.gasDensityTexture` with
 * `extras.gas = { height, texCoord, speed }` and the per-vertex weight as `_GASWEIGHT`. The game pans that
 * map with time (no per-clip tracks exist for it), so this patch samples it at uv + speed * t and displaces
 * by the *difference* to the baked frame; the retraction term is constant, so the GLB stays exact at t = 0.
 * The shell is opaque, as in game: the face shows through where the shell sinks into the black body.
 */
export interface GasParams { height: number; texCoord?: number; speed?: [number, number] }

interface GasUniforms {
  uGasTex: { value: THREE.Texture | null };
  uGasOff: { value: THREE.Vector2 };
  uGasHeight: { value: number };
  uGasOn: { value: number };
}

export interface GasState { uniforms: GasUniforms; speed: THREE.Vector2 }

const DEFAULT_SPEED: [number, number] = [0.01, 0.07];

const VERT_DECL = `uniform sampler2D uGasTex;
uniform vec2 uGasOff;
uniform float uGasHeight;
uniform float uGasOn;
attribute float _gasweight;
`;

function patch(shader: { vertexShader: string; uniforms: Record<string, unknown> }, u: GasUniforms, texCoord: number) {
  const bv = '#include <begin_vertex>';
  if (!shader.vertexShader.includes(bv)) { console.warn('gas: chunk not found'); return; }
  // three only declares `attribute vec2 uv1` when a material channel uses it (USE_UV1); declare it otherwise.
  const uv1Decl = texCoord === 1 ? '#ifndef USE_UV1\nattribute vec2 uv1;\n#endif\n' : '';
  const uvExpr = texCoord === 1 ? 'uv1' : 'uv';
  // A geometry without `_gasweight` leaves the attribute disabled (reads 0): treat that as weight 1
  // (the importer keeps real weights >= 1e-3).
  shader.vertexShader = VERT_DECL + uv1Decl + shader.vertexShader.replace(bv, `${bv}
	{
		vec2 gUv = ${uvExpr};
		float gW = _gasweight > 0.0 ? _gasweight : 1.0;
		float gD0 = texture2D( uGasTex, gUv ).r;
		float gD1 = texture2D( uGasTex, gUv + uGasOff ).r;
		transformed += normal * ( uGasOn * uGasHeight * gW * ( gD1 - gD0 ) );
	}`);
  Object.assign(shader.uniforms, u);
}

/** Install the gas patch on a material (once). The density texture is attached later with `setGasTexture`. */
export function installGas(mat: THREE.MeshStandardMaterial): GasState | undefined {
  const p = mat.userData?.gas as GasParams | undefined;
  if (!p) return undefined;
  const existing = mat.userData.__gas as GasState | undefined;
  if (existing) return existing;
  const u: GasUniforms = { uGasTex: { value: null }, uGasOff: { value: new THREE.Vector2() }, uGasHeight: { value: p.height }, uGasOn: { value: 0 } };
  const st: GasState = { uniforms: u, speed: new THREE.Vector2(...(p.speed ?? DEFAULT_SPEED)) };
  mat.userData.__gas = st;
  const prev = mat.onBeforeCompile;
  const prevKey = mat.customProgramCacheKey;
  const tc = p.texCoord ?? 0;
  mat.onBeforeCompile = function (this: THREE.Material, shader, renderer) {
    prev?.call(this, shader, renderer);
    patch(shader as never, u, tc);
  };
  mat.customProgramCacheKey = function (this: THREE.Material) { return (prevKey?.call(this) ?? '') + '|gas' + tc; };
  mat.needsUpdate = true;
  return st;
}

export function setGasTexture(mat: THREE.MeshStandardMaterial, tex: THREE.Texture) {
  const st = mat.userData.__gas as GasState | undefined;
  if (!st) return;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.NoColorSpace;
  tex.needsUpdate = true;
  st.uniforms.uGasTex.value = tex;
  st.uniforms.uGasOn.value = 1;
}

/** Advance the pan offset to time `t` (seconds). */
export function updateGas(mat: THREE.MeshStandardMaterial, t: number) {
  const st = mat.userData.__gas as GasState | undefined;
  if (!st) return;
  st.uniforms.uGasOff.value.set(st.speed.x * t, st.speed.y * t);
}
