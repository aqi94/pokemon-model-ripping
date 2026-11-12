import * as THREE from 'three';

/**
 * The UNITE lit shading model (WP-F of the 2026-11-09 shader survey), from the decompiled shaders (formulas with dump
 * line numbers in `pipeline/importers/unite/shading.py`). A material carrying `extras.unite.shading` has its output
 * replaced by, in game units (linear, before the tone curve):
 *
 *   direct = keyColour x (NdL d + spec)          d = albedo (1 - metal), F0 = mix(0.04, albedo, metal)
 *   spec   = min(D, 100)(0.25 + 0.25 r) F0 NdL _SpecularColor,  D = (a / ((a x)^2 + 1 - x^2))^2, a = r^2,
 *            x = N.V in the lobby shaders (Cloth_lv3.txt:466-546), N.H in the battle shader (actorbase_lv3.txt:346-353)
 *   SH     = d (Atop _SHTopColor + Abot _SHBotColor) _SHScale,  Atop/bot = max(0, .125 +- .25 n + .125 n^2), n = world normal y
 *   AO     = mro B: indirect terms only (lobby), the whole result (battle, `ao.scope` "total")
 *   battle light colour: x 1.2, clamped to 1.2 (actorbase_lv3.txt:355-356)
 *   player shaders: the diffuse light factor is max(NdL clamp(NdL^_ShadowPow + _ShadowOffset), _ShadowScale)
 * then the "UNITE" tone curve (environment.ts, actorbase_lv3.txt:414-423) turns game units into display light.
 *
 * Scene lights: the key light's direction is the game's main light; its colour and the hemisphere ambient steer the
 * strengths RELATIVE to the default environment (key 1.2, ambient 1.9 = the game's neutral 1 / SH 1). The lobby/battle
 * main light colour, direction and the rim globals (`_RimlightScale2`, `_RimLightDir`...) are scripted, not in the rip.
 * The model runs only with the UNITE tone curve (Lighting > Tone mapping "UNITE game curve", preset `unite`): game units
 * need the game's curve, and the other presets were fitted to the generic Game shading (unfitted game units under ACES
 * scored dE 14.6 against 8.6 on the lobby shot). `UNITE_K` is 1 under the curve; elsewhere the matcap terms keep their
 * x0.45 (the factor the curve folds in). Fur sketch bands / anisotropic lobes and the fur rim are not drawn.
 */
export const UNITE_K = { value: 0.45 }; // 1 with the UNITE tone curve, else 0.45 (also read by matcap.ts)
export const UNITE_ON = { value: 1 }; // 0 in the PBR rig or when switched off
/** Tool hook (`window.__uniteShading(false)` renders UNITE materials with the generic Game shading, for before/after comparisons). */
export const uniteSwitch = { enabled: true };
if (typeof window !== 'undefined') (window as unknown as { __uniteShading: (on: boolean) => void }).__uniteShading = (on) => { uniteSwitch.enabled = on; };
export const UNITE_SCENE = { key: { value: 1 }, amb: { value: new THREE.Vector3(1, 1, 1) } };

/** Fragment function for three's CustomToneMapping chunk: the game's post curve (actorbase_lv3.txt:414-423), exposure first. */
export const UNITE_TONE_GLSL = 'vec3 CustomToneMapping( vec3 c ) { c *= toneMappingExposure; vec3 a = c * 0.45; vec3 n = ( 1.1295 * c + 0.03 ) * a; vec3 d = a * ( 1.0935 * c + 0.59 ) + 0.08; return saturate( n / d ); }';

interface Shading {
  form?: string; specTerm?: string; shTop?: number[]; shBot?: number[]; shScale?: number; specColor?: number[];
  roughOffset?: number; metalOffset?: number; lightClamp?: number; shadow?: { offset: number; pow: number; scale: number };
}
type UniteExtras = { shading?: Shading; ao?: { scope?: string; power?: number } };
const v3 = (c: number[] | undefined, d: number) => new THREE.Vector3(c?.[0] ?? d, c?.[1] ?? d, c?.[2] ?? d);

export function uniteShadingData(mat: THREE.Material): Shading | null {
  return (mat.userData as { unite?: UniteExtras } | undefined)?.unite?.shading ?? null;
}
export function uniteShadingKey(mat: THREE.Material): string {
  const s = uniteShadingData(mat);
  return s ? '|us' + (s.specTerm === 'ndh' ? 'h' : 'v') + (s.shadow ? 's' : '') + ((mat.userData as { unite?: UniteExtras }).unite?.ao?.scope === 'total' ? 't' : 'i') : '';
}

const DECL = `#ifndef U_UK_DECL
#define U_UK_DECL
uniform float uUK;
#endif
uniform float uUOn; uniform float uUKey; uniform vec3 uUAmb;
uniform vec3 uUSHTop; uniform vec3 uUSHBot; uniform vec3 uUSpec; uniform vec2 uUOff; uniform float uULClamp; uniform vec3 uUShadow;
`;

const GLSL = (aoTotal: boolean, ndh: boolean, shadow: boolean) => `
if ( uUOn > 0.5 ) {
	vec3 sV = isOrthographic ? vec3( 0.0, 0.0, 1.0 ) : normalize( vViewPosition );
	vec3 sN = normalize( normal );
	float sR = clamp( roughnessFactor + uUOff.x, 0.04, 1.0 );
	float sMt = clamp( metalnessFactor + uUOff.y, 0.0, 1.0 );
	vec3 sD = diffuseColor.rgb * ( 1.0 - sMt );
	vec3 sF0 = mix( vec3( 0.04 ), diffuseColor.rgb, sMt );
	vec3 sL = vec3( 0.0, 1.0, 0.0 );
	vec3 sLc = vec3( 0.0 );
	#if NUM_DIR_LIGHTS > 0
	sL = directionalLights[ 0 ].direction;
	sLc = directionalLights[ 0 ].color * uUKey;
	#endif
	if ( uULClamp > 0.0 ) sLc = min( sLc * uULClamp, vec3( uULClamp ) );
	float sNdL = clamp( dot( sN, sL ), 0.0, 1.0 );
	float sA = sR * sR;
	${ndh ? 'float sX = clamp( dot( sN, normalize( sL + sV ) ), 0.0, 1.0 );' : 'float sX = clamp( abs( dot( sN, sV ) ) + 0.004, 0.0, 1.0 );'}
	float sDd = sA / ( sA * sA * sX * sX + 1.0 - sX * sX );
	float sSp = min( sDd * sDd * ( 0.25 + 0.25 * sR ), 100.0 );
	vec3 sSpec = sSp * sF0 * sNdL * uUSpec;
	float sDiff = sNdL;
	${shadow ? 'sDiff = max( sNdL * clamp( pow( sNdL, uUShadow.y ) + uUShadow.x, 0.0, 1.0 ), uUShadow.z ); sSpec *= max( sNdL, uUShadow.z ) / max( sNdL, 1e-3 );' : ''}
	float sNy = inverseTransformDirection( sN, viewMatrix ).y;
	float sSq = sNy * sNy * 0.125;
	vec2 sAb = max( vec2( sSq + sNy * 0.25 + 0.125, sSq - sNy * 0.25 + 0.125 ), vec2( 0.0 ) );
	vec3 sSH = sD * ( sAb.x * uUSHTop + sAb.y * uUSHBot ) * uUAmb;
	float sAO = 1.0;
	#ifdef USE_AOMAP
	sAO = ambientOcclusion;
	#endif
	vec3 sOut = sLc * ( sDiff * sD + sSpec ) + sSH * ${aoTotal ? '1.0' : 'sAO'};
	${aoTotal ? 'sOut *= sAO;' : ''}
	outgoingLight = sOut * uUK + totalEmissiveRadiance * uUK;
}
`;

function patch(shader: { fragmentShader: string; uniforms: Record<string, unknown> }, u: Record<string, { value: unknown }>, aoTotal: boolean, ndh: boolean, shadow: boolean) {
  const at = '#include <opaque_fragment>';
  if (!shader.fragmentShader.includes(at)) { console.warn('unite shading: chunk not found'); return; }
  shader.fragmentShader = DECL + shader.fragmentShader.replace(at, GLSL(aoTotal, ndh, shadow) + at);
  Object.assign(shader.uniforms, u);
}

/** Install on a material with `extras.unite.shading` (once, before the matcap / game-shading patches capture onBeforeCompile). */
export function installUniteShading(mat: THREE.MeshStandardMaterial): boolean {
  const s = uniteShadingData(mat);
  if (!s) return false;
  if (mat.userData.__uniteShade) return true;
  const ao = (mat.userData as { unite?: UniteExtras }).unite?.ao;
  const u = {
    uUOn: UNITE_ON, uUK: UNITE_K, uUKey: UNITE_SCENE.key, uUAmb: UNITE_SCENE.amb,
    uUSHTop: { value: v3(s.shTop, 2.5).multiplyScalar(s.shScale ?? 1) }, uUSHBot: { value: v3(s.shBot, 2.5).multiplyScalar(s.shScale ?? 1) },
    uUSpec: { value: v3(s.specColor, 1) }, uUOff: { value: new THREE.Vector2(s.roughOffset ?? 0, s.metalOffset ?? 0) },
    uULClamp: { value: s.lightClamp ?? 0 }, uUShadow: { value: new THREE.Vector3(s.shadow?.offset ?? 0, s.shadow?.pow ?? 1, s.shadow?.scale ?? 0) },
  };
  mat.userData.__uniteShade = true;
  const prev = mat.onBeforeCompile;
  // our block must come BEFORE the matcap terms (they add onto outgoingLight): the patch applied first sits first
  mat.onBeforeCompile = (shader, renderer) => {
    patch(shader as never, u, ao?.scope === 'total', s.specTerm === 'ndh', !!s.shadow);
    prev?.call(mat, shader, renderer);
  };
  mat.needsUpdate = true;
  return true;
}
