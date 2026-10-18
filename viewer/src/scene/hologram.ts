import * as THREE from 'three';

/**
 * Holocast hologram effect (Z-A). Trainer `.trmmt` files carry `holo` / `holo_red` material sets whose
 * `.trmtr` turns on `EnableHolocastEffect` with `HolocastColor` (linear HDR), `ScanlineSpeed/Depth/Tiling`
 * and `ScanGradationTiling`, a fixed light from straight above (`LightPitch 90`, no shadows) and stronger AO.
 * The scanlines are procedural (no texture in the rip), so this patch redraws them; the importer exports
 * the values as manifest `holocast` presets and per-material `extras.holocast[set]`.
 *
 * From the data: colour, scanline depth/tiling/speed, gradation tiling, AO strength, top light.
 * Fitted to the screenshots (`samples/hologram_*`), not data: the line pitch (tiling 260 = pi/260 m per line,
 * ~1.2 cm, measured on both samples), the line profile, the gradation band shape, opacity and edge glow.
 * The game draws holograms as `Opaque`, but every shot shows the scene through them.
 */
export interface HoloParams {
  color: [number, number, number]; // linear HDR
  scanlineSpeed: number;
  scanlineDepth: number;
  scanlineTiling: number;
  scanGradationTiling: number;
  occlusionStrength?: number;
  shadingBias?: number;
}

/** Viewer-side state (Appearance > Hologram). `preset` = a manifest set name, a global preset id, or 'custom'. */
export interface HologramState {
  on: boolean;
  preset: string;
  color: [number, number, number]; // hue swatch, 0..1 linear
  intensity: number; // HDR strength (the data's max channel: cyan 3.4, red 6.2)
  depth: number;
  tiling: number;
  speed: number;
  gradation: number;
  gradientStrength: number;
  opacity: number;
  edgeGlow: number;
}

/** Presets read from the Z-A data (most common value per hue), offered for every model. */
export const GLOBAL_HOLO_PRESETS: { id: string; label: string; params: HoloParams }[] = [
  { id: 'g_cyan', label: 'Cyan (data)', params: { color: [0.223, 3.392, 3.392], scanlineSpeed: 2, scanlineDepth: 0.7, scanlineTiling: 260, scanGradationTiling: 1 } },
  { id: 'g_red', label: 'Red (data)', params: { color: [6.191, 0.442, 0.393], scanlineSpeed: 2, scanlineDepth: 0.7, scanlineTiling: 260, scanGradationTiling: 1 } },
  { id: 'g_magenta', label: 'Magenta (Pa Boss)', params: { color: [3.181, 0.843, 3.629], scanlineSpeed: 1, scanlineDepth: 0.9, scanlineTiling: 260, scanGradationTiling: 1 } },
];

export const HOLO_FIT_DEFAULTS = { gradientStrength: 0.3, opacity: 0.8, edgeGlow: 0.8 };

export function stateFromParams(p: HoloParams, preset: string, prev?: HologramState): HologramState {
  const m = Math.max(1e-6, ...p.color);
  return {
    on: prev?.on ?? true,
    preset,
    color: p.color.map((c) => c / m) as [number, number, number],
    intensity: m,
    depth: p.scanlineDepth,
    tiling: p.scanlineTiling,
    speed: p.scanlineSpeed,
    gradation: p.scanGradationTiling,
    gradientStrength: prev?.gradientStrength ?? HOLO_FIT_DEFAULTS.gradientStrength,
    opacity: prev?.opacity ?? HOLO_FIT_DEFAULTS.opacity,
    edgeGlow: prev?.edgeGlow ?? HOLO_FIT_DEFAULTS.edgeGlow,
  };
}

interface HoloUniforms {
  uHoloOn: { value: number };
  uHoloColor: { value: THREE.Vector3 };
  uHoloScan: { value: THREE.Vector4 }; // depth, tiling, phase (speed * t), gradation tiling
  uHoloGrad: { value: THREE.Vector2 }; // strength, phase
  uHoloLook: { value: THREE.Vector4 }; // opacity, edge glow, AO strength, shading bias
}

const VERT_DECL = 'varying float vHoloY;\n';
const FRAG_DECL = `uniform float uHoloOn;
uniform vec3 uHoloColor;
uniform vec4 uHoloScan;
uniform vec2 uHoloGrad;
uniform vec4 uHoloLook;
varying float vHoloY;
`;

function patch(shader: { vertexShader: string; fragmentShader: string; uniforms: Record<string, unknown> }, u: HoloUniforms) {
  const pv = '#include <project_vertex>';
  const of = '#include <opaque_fragment>';
  if (!shader.vertexShader.includes(pv) || !shader.fragmentShader.includes(of)) { console.warn('hologram: chunk not found'); return; }
  shader.vertexShader = VERT_DECL + shader.vertexShader.replace(pv, `${pv}
	vHoloY = ( modelMatrix * vec4( transformed, 1.0 ) ).y;`);
  // Replaces the lit colour (game or PBR) with the holocast colour: it ignores the scene lights, like the
  // game's fixed top light. The albedo is sampled from the map directly because unlit Game materials zero
  // diffuseColor and draw through emissive.
  shader.fragmentShader = FRAG_DECL + shader.fragmentShader.replace(of, `${of}
	if ( uHoloOn > 0.5 ) {
		#ifdef USE_MAP
			vec4 hTex = texture2D( map, vMapUv );
		#else
			vec4 hTex = vec4( 1.0 );
		#endif
		float hLum = dot( hTex.rgb, vec3( 0.2126, 0.7152, 0.0722 ) );
		#ifdef USE_AOMAP
			hLum *= pow( clamp( texture2D( aoMap, vAoMapUv ).r, 0.0, 1.0 ), uHoloLook.z );
		#endif
		vec3 hUp = normalize( ( viewMatrix * vec4( 0.0, 1.0, 0.0, 0.0 ) ).xyz );
		float hLight = mix( 1.0, saturate( dot( normal, hUp ) * 0.5 + 0.5 ), uHoloLook.w );
		float hLine = 0.5 + 0.5 * cos( 2.0 * ( vHoloY * uHoloScan.y - uHoloScan.z ) );
		float hScan = 1.0 - uHoloScan.x * hLine;
		float hGrad = 1.0 - uHoloGrad.x * ( 0.5 + 0.5 * sin( 6.2831853 * ( vHoloY * uHoloScan.w * 0.5 - uHoloGrad.y ) ) );
		float hEdge = pow( 1.0 - saturate( abs( dot( normal, normalize( vViewPosition ) ) ) ), 3.0 ) * uHoloLook.y;
		vec3 hCol = uHoloColor * ( hLum * hLight * hScan * hGrad + hEdge );
		gl_FragColor = vec4( hCol, uHoloLook.x * mix( 1.0, hScan, 0.5 ) * hTex.a );
	}`);
  Object.assign(shader.uniforms, u);
}

interface Saved { transparent: boolean; depthWrite: boolean; alphaTest: number }

/** Install the (inactive) hologram patch once; call before the Game shading patch so the two chain. */
export function installHologram(mat: THREE.MeshStandardMaterial) {
  if (mat.userData.__holo) return;
  const u: HoloUniforms = {
    uHoloOn: { value: 0 }, uHoloColor: { value: new THREE.Vector3() }, uHoloScan: { value: new THREE.Vector4() },
    uHoloGrad: { value: new THREE.Vector2() }, uHoloLook: { value: new THREE.Vector4() },
  };
  mat.userData.__holo = u;
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = function (this: THREE.Material, shader, renderer) {
    prev?.call(this, shader, renderer);
    patch(shader as never, u);
  };
  mat.needsUpdate = true;
}

/**
 * Apply `st` (null = off). `own` = this material's data values for the selected preset (eyes are darker
 * than the body in the data); the user's colour / dials win once they leave the preset ('custom').
 */
export function setHologram(mat: THREE.MeshStandardMaterial, st: HologramState | null, own?: HoloParams) {
  const u = mat.userData.__holo as HoloUniforms | undefined;
  if (!u) return;
  const on = !!st?.on;
  const saved = mat.userData.__holoSaved as Saved | undefined;
  if (on && !saved) {
    mat.userData.__holoSaved = { transparent: mat.transparent, depthWrite: mat.depthWrite, alphaTest: mat.alphaTest } satisfies Saved;
    mat.transparent = true;
    mat.needsUpdate = true;
  } else if (!on && saved) {
    mat.transparent = saved.transparent; mat.depthWrite = saved.depthWrite; mat.alphaTest = saved.alphaTest;
    delete mat.userData.__holoSaved;
    mat.needsUpdate = true;
  }
  u.uHoloOn.value = on ? 1 : 0;
  if (!st || !on) return;
  // Per-material data colour only while a model preset is selected (the colour edits switch to 'custom').
  const col = own && st.preset !== 'custom' ? own.color : st.color.map((c) => c * st.intensity);
  u.uHoloColor.value.set(col[0], col[1], col[2]);
  u.uHoloScan.value.set(st.depth, st.tiling, u.uHoloScan.value.z, st.gradation);
  u.uHoloGrad.value.x = st.gradientStrength;
  u.uHoloLook.value.set(st.opacity, st.edgeGlow, own?.occlusionStrength ?? 1.8, own?.shadingBias ?? 0.6);
  (mat.userData as { __holoSpeed?: number }).__holoSpeed = st.speed;
}

/** Advance the scroll to time `t` (seconds, the playback clock). */
export function updateHologram(mat: THREE.MeshStandardMaterial, t: number) {
  const u = mat.userData.__holo as HoloUniforms | undefined;
  if (!u || !u.uHoloOn.value) return;
  const speed = (mat.userData as { __holoSpeed?: number }).__holoSpeed ?? 2;
  u.uHoloScan.value.z = speed * t;
  u.uHoloGrad.value.y = speed * t * 0.05;
}
