import * as THREE from 'three';

/**
 * SV/LA eye catchlights. ~1430 SV eye materials (`EyeClearCoat`, Pikachu included) have `EnableHighlight` but no
 * highlight mask: the game reflects a dedicated eye light (`PointLightIndex`) off the clear-coat dome with the
 * material's own lobe (`RoughnessHighlight`), glow `EmissionIntensityLayer5 x EmissionColorLayer5`, clamped by the
 * light rig's `EyeHighlightMaxLuminance` (1 in `master`). The importer writes these as `extras.eyeHighlight` and
 * the dome normal (`NormalMap1`, convex; the glTF normal map is the concave iris one) as `extras.highlightNormalTexture`.
 *
 * The eye light's position is not in the files, so its direction (camera space) is fitted to screenshots.
 * Drawn in both shading modes, added to the outgoing light before tone mapping.
 */
export interface EyeHighlightParams { roughness: number; metallic?: number; intensity: number; color?: [number, number, number]; light?: number; normalHeight?: number }

// FITTED: eye light direction in camera space (up-left of the camera) and glow gain (x EmissionIntensityLayer5 x the
// normalised GGX lobe, peak 1 / (pi a^2): the data pairs a wide lobe with a high intensity (Pikachu 0.56 / 10) and a sharp
// one with a low intensity (Chansey 0.3 / 1), which only evens out with the normalisation), by eye on SV Pikachu/Chansey
// without an in-game reference yet, 2026-10-20. The light's position is not in the data.
const EYE_LIGHT_DIR = new THREE.Vector3(-0.35, 0.45, 1).normalize();
const EYE_GAIN = 0.046;
// EyeHighlightMaxLuminance in the SV master rig.
const MAX_LUMINANCE = 1.0;

interface Uniforms { uEhDir: { value: THREE.Vector3 }; uEhNormal: { value: THREE.Texture | null }; uEhHasNormal: { value: number };
  uEhA2: { value: number }; uEhGlow: { value: THREE.Vector3 }; uEhMax: { value: number }; uEhOn: { value: number }; uEhHeight: { value: number } }

const DECL = `uniform vec3 uEhDir;
uniform sampler2D uEhNormal;
uniform float uEhHasNormal;
uniform float uEhA2;
uniform vec3 uEhGlow;
uniform float uEhMax;
uniform float uEhOn;
uniform float uEhHeight;
mat3 ehTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
	vec3 q0 = dFdx( eye_pos.xyz ); vec3 q1 = dFdy( eye_pos.xyz );
	vec2 st0 = dFdx( uv.st ); vec2 st1 = dFdy( uv.st );
	vec3 N = surf_norm;
	vec3 q1perp = cross( q1, N ); vec3 q0perp = cross( N, q0 );
	vec3 T = q1perp * st0.x + q0perp * st1.x;
	vec3 B = q1perp * st0.y + q0perp * st1.y;
	float det = max( dot( T, T ), dot( B, B ) );
	float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
	return mat3( T * scale, B * scale, N );
}
`;

function patch(shader: { fragmentShader: string; uniforms: Record<string, unknown> }, u: Uniforms) {
  const at = '#include <opaque_fragment>';
  if (!shader.fragmentShader.includes(at)) { console.warn('eye highlight: chunk not found'); return; }
  shader.fragmentShader = DECL + shader.fragmentShader.replace(at, `#ifdef USE_MAP
	if ( uEhOn > 0.5 ) {
		vec3 ehN = nonPerturbedNormal;
		if ( uEhHasNormal > 0.5 ) {
			vec3 ehT = texture2D( uEhNormal, vMapUv ).xyz * 2.0 - 1.0;
			ehT.y = - ehT.y;
			ehT.xy *= uEhHeight; // NormalHeight1 // Trinity maps are authored for bottom-left UVs; the frame below follows glTF's flipped V
			ehN = normalize( ehTangentFrame( - vViewPosition, nonPerturbedNormal, vMapUv ) * ehT );
		}
		vec3 ehV = isOrthographic ? vec3( 0.0, 0.0, 1.0 ) : normalize( vViewPosition );
		float ehNH = saturate( dot( ehN, normalize( uEhDir + ehV ) ) );
		// GGX lobe relative to its peak: (a2 / (NH^2 (a2 - 1) + 1))^2; uEhGlow already carries the peak 1 / (pi a2). Clamped (max luminance).
		float ehR = uEhA2 / ( ehNH * ehNH * ( uEhA2 - 1.0 ) + 1.0 );
		outgoingLight += min( uEhGlow * ( ehR * ehR ), vec3( uEhMax ) );
	}
#endif
${at}`);
  Object.assign(shader.uniforms, u);
}

/** Install the catchlight on an eye material (once; before the game-shading patch captures onBeforeCompile). */
export function installEyeHighlight(mat: THREE.MeshStandardMaterial): boolean {
  const p = mat.userData?.eyeHighlight as EyeHighlightParams | undefined;
  if (!p) return false;
  if (mat.userData.__eyeHl) return true;
  const a = p.roughness * p.roughness;
  const c = p.color ?? [1, 1, 1];
  const g = p.intensity * EYE_GAIN / (Math.PI * Math.max(a * a, 1e-4)); // x the lobe's peak
  const u: Uniforms = {
    uEhDir: { value: EYE_LIGHT_DIR.clone() }, uEhNormal: { value: null }, uEhHasNormal: { value: 0 },
    uEhA2: { value: a * a }, uEhGlow: { value: new THREE.Vector3(c[0] * g, c[1] * g, c[2] * g) },
    uEhMax: { value: MAX_LUMINANCE }, uEhOn: { value: 1 }, uEhHeight: { value: p.normalHeight ?? 1 },
  };
  mat.userData.__eyeHl = u;
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = function (this: THREE.Material, shader, renderer) { prev?.call(this, shader, renderer); patch(shader as never, u); };
  mat.needsUpdate = true;
  return true;
}

export function setEyeHighlightNormal(mat: THREE.Material, tex: THREE.Texture | null) {
  const u = mat.userData.__eyeHl as Uniforms | undefined;
  if (!u) return;
  u.uEhNormal.value = tex; u.uEhHasNormal.value = tex ? 1 : 0;
}

/** Off while the hologram is on (it has its own look). Kept in lighting probes: the screenshots show catchlights too. */
export function setEyeHighlightOn(mat: THREE.Material, on: boolean) {
  const u = mat.userData.__eyeHl as Uniforms | undefined;
  if (u) u.uEhOn.value = on ? 1 : 0;
}

/**
 * Clear-coated eyes (glTF KHR_materials_clearcoat, from SV/LA `EyeClearCoat`): they get the studio reflection map so the
 * coat reflects something, but only the coat may use it. The base's environment diffuse and specular are dropped, so a
 * white or coloured iris is not lit by an ambient the body never receives (the body has no environment map).
 */
// FITTED: strength of the studio map in the eye coat. The map was built for BDSP glasses (sky up to 1.85, soft boxes up
// to 7); at 1 the domed coat's Fresnel turned Pikachu/Eevee eyes into chrome balls. By eye on SV Pikachu/Eevee/Chansey, 2026-10-20.
export const COAT_ENV_INTENSITY = 0.2;

export function installCoatReflection(mat: THREE.MeshStandardMaterial): boolean {
  const pm = mat as THREE.MeshPhysicalMaterial;
  if (!pm.isMeshPhysicalMaterial || !(pm.clearcoat > 0)) return false;
  if (mat.userData.__coatEnv) return true;
  mat.userData.__coatEnv = true;
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = function (this: THREE.Material, shader, renderer) {
    prev?.call(this, shader, renderer);
    const at = '#include <lights_fragment_maps>';
    if (!shader.fragmentShader.includes(at)) { console.warn('coat reflection: chunk not found'); return; }
    shader.fragmentShader = shader.fragmentShader.replace(at, `${at}
#if defined( RE_IndirectSpecular )
	iblIrradiance = vec3( 0.0 );
	radiance = vec3( 0.0 );
#endif`);
  };
  mat.needsUpdate = true;
  return true;
}
