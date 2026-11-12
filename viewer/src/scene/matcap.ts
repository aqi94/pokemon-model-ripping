import * as THREE from 'three';
import { UNITE_K } from './uniteShading';

/**
 * UNITE matcaps (`extras.matcapTexture`), sampled by the view-space normal. Two uses, set by `extras.matcapMode`:
 * - glass (default; `ghostGlass`: Gengar Space's helmet, Snorlax skin 20's): RGB = the reflection, alpha = coverage
 *   (mostly 8-45 / 255, denser towards the rim). Replaces the lit colour and alpha, drawn as emission. The importer
 *   also writes a flat fallback (mean colour + alpha) as the base colour factor.
 * - reflect (metal): the matcap is a pre-lit reflection sphere (the game's generic `t_mipmap03` chrome room, or the
 *   material's own: Zacian's gold, Ceruledge's grey) added where the material is metallic. A metallic texel has no
 *   diffuse, so without it UNITE metal rendered black. outgoing += metallic x matcap x albedo x gain, gain =
 *   `extras.matcapGain` = 1 / the matcap's mean linear luminance (the lobby JSONs' `_MatcapColor` 4.84 against
 *   ~0.2-mean chrome): the reflection keeps the albedo's brightness. HYPOTHESIS (shader not in the rip), checked on 56
 *   battle/lobby pairs; ledger `material:matcapMetal`. `metalEnv` (no matcap found at all) uses `defaultMatcap`.
 *   LEGACY since 2026-11-09 (GLBs converted before it): the importer now writes `extras.unite.*` instead (below).
 *
 * `extras.unite.{matTex, rampLight, env, glass}` (WP-D of the 2026-11-09 shader survey; formulas and dump line numbers
 * in `pipeline/importers/unite/matcap_terms.py`): the four separate terms the shaders have, drawn after the lit colour.
 * - matTex   (Cloth `_IFMAT`):   += matcap(uv) * (0.25 + 0.75 N.L) * color            additive highlight, any metal
 * - rampLight(Cloth `_IFRAMP`):  += ramp(min(NoV^2, .96) * scale + offset) * (0.25 + 0.75 N.L) * color
 * - env (cloth/pbr/battle/bubble): the matcap standing in for the reflection probe, lod = r (1.7 - 0.7 r) 6
 * - glass (ghostGlass / actorglass): alpha from NoV, colour = matcap x mix(top, bot, thickness)
 * N.L uses the first directional light (the key), no shadow term; the lights' colour is taken as 1. The additive terms
 * carry the shaders' 0.45 exposure (the viewer's diffuse stands for the game's x 0.45); the ACES curve is the viewer's own tone mapping.
 */
const DECL = `uniform sampler2D uMatcap;
uniform float uMatcapOn;
uniform float uMatcapReflect;
uniform float uMatcapGain;
uniform sampler2D uUMat; uniform vec3 uUMatCol; uniform float uUMatOn;
uniform sampler2D uURamp; uniform vec3 uURampCol; uniform vec2 uURampST; uniform float uURampOn;
uniform sampler2D uUEnv; uniform vec3 uUEnvCol; uniform float uUEnvKind;
#ifndef U_UK_DECL
#define U_UK_DECL
uniform float uUK;
#endif
uniform sampler2D uUGlass; uniform vec4 uUGTop; uniform vec4 uUGBot; uniform vec3 uUGP; uniform float uUGAlpha; uniform float uUGKind;
`;

/** `extras.unite` texture reference after the glTF writer: {index, texCoord, name}. */
type Ref = { index?: number };
type UniteData = {
  matTex?: { texture?: Ref; color?: number[] };
  rampLight?: { texture?: Ref; color?: number[]; scale?: number; offset?: number };
  env?: { kind?: string; texture?: Ref; color?: number[] };
  glass?: { kind?: string; texture?: Ref; offset?: number; pow?: number; scale?: number; top?: number[]; bot?: number[]; alpha?: number };
};
const ENV_KIND: Record<string, number> = { cloth: 1, pbr: 2, battle: 3, bubble: 4 };
const t3 = (c: number[] | undefined, d: number[]) => new THREE.Vector3(...((c && c.length >= 3 ? c : d).slice(0, 3) as [number, number, number]));
const t4 = (c: number[] | undefined) => new THREE.Vector4(c?.[0] ?? 1, c?.[1] ?? 1, c?.[2] ?? 1, c?.[3] ?? 1);
export function uniteData(mat: THREE.Material): UniteData | null {
  const d = (mat.userData as { unite?: UniteData } | undefined)?.unite;
  return d && (d.matTex || d.rampLight || d.env || d.glass) ? d : null;
}
/** Program-cache key part: which `extras.unite` terms a material has (they change the compiled code). */
export function uniteKey(mat: THREE.Material): string {
  const d = uniteData(mat);
  if (!d) return '';
  return '|u' + (d.matTex ? 'm' : '') + (d.rampLight ? 'r' : '') + (d.env ? 'e' + (ENV_KIND[d.env.kind ?? ''] ?? 1) : '') + (d.glass ? (d.glass.kind === 'actor' ? 'ga' : 'gg') : '');
}

type Tx = { value: THREE.Texture | null };
type U = {
  uMatcap: Tx; uMatcapOn: { value: number }; uMatcapReflect: { value: number }; uMatcapGain: { value: number };
  uUMat: Tx; uUMatCol: { value: THREE.Vector3 }; uUMatOn: { value: number };
  uURamp: Tx; uURampCol: { value: THREE.Vector3 }; uURampST: { value: THREE.Vector2 }; uURampOn: { value: number };
  uUEnv: Tx; uUEnvCol: { value: THREE.Vector3 }; uUEnvKind: { value: number };
  uUK: { value: number };
  uUGlass: Tx; uUGTop: { value: THREE.Vector4 }; uUGBot: { value: THREE.Vector4 }; uUGP: { value: THREE.Vector3 }; uUGAlpha: { value: number }; uUGKind: { value: number };
};

/** The `extras.unite` terms, added after the lit colour (see the header). Runs in the physical fragment's scope. */
const UNITE_GLSL = `
{
	vec3 uV = isOrthographic ? vec3( 0.0, 0.0, 1.0 ) : normalize( vViewPosition );
	vec3 uN = normalize( normal );
	vec3 uX = normalize( vec3( uV.z, 0.0, - uV.x ) );
	vec3 uY = cross( uV, uX );
	vec2 uUV = vec2( dot( uX, uN ), dot( uY, uN ) ) * 0.5 + 0.5; // Unity matcap coordinates (v up)
	vec2 uST = vec2( uUV.x, 1.0 - uUV.y ); // glTF image rows: top first
	float uNdL = 0.5;
	#if NUM_DIR_LIGHTS > 0
	uNdL = clamp( dot( uN, directionalLights[ 0 ].direction ), 0.0, 1.0 );
	#endif
	float uLt = 0.25 + 0.75 * uNdL;
	float uNoV = abs( dot( uN, uV ) );
	float uAO = 1.0;
	#ifdef USE_AOMAP
	uAO = ambientOcclusion;
	#endif
	if ( uUGKind > 0.5 ) { // glass: replaces colour and alpha
		float gNoV = uUGKind < 1.5 ? min( uNoV + 0.004, 1.0 ) : clamp( dot( uN, uV ), 0.0, 1.0 );
		float gT = clamp( ( gNoV - uUGP.x ) * uUGP.y, 0.0, 1.0 );
		float gK = uUGKind < 1.5 ? clamp( 1.0 - gT + diffuseColor.a, 0.0, 1.0 ) : 1.0 - gT;
		vec4 gR = texture2D( uUGlass, vec2( uUV.x + gK * uUGP.z, 1.0 - ( uUV.y + gK * uUGP.z ) ) );
		vec4 gC = gR * mix( uUGTop, uUGBot, gK );
		if ( uUGKind < 1.5 ) { // ghostGlass: transparent face-on, opaque at grazing angles
			float gA = clamp( gK * uUGBot.a + gC.a, 0.0, 1.0 );
			diffuseColor.a = clamp( uUGAlpha * gA - gNoV, 0.0, 1.0 );
			outgoingLight = gC.rgb + uUGAlpha * diffuseColor.rgb;
		} else { // actorglass
			diffuseColor.a = clamp( ( gK + gC.a ) * uUGAlpha, 0.0, 1.0 );
			outgoingLight = gC.rgb * 1.2;
		}
	} else {
		float uK = uUK; // 0.45 = the factor the game's tone curve folds in (Cloth dump line 9700): our diffuse stands for game diffuse x 0.45; 1 once the viewer runs the UNITE curve (uniteShading.ts), where the curve itself carries it
		if ( uUMatOn > 0.5 ) outgoingLight += uK * texture2D( uUMat, uST ).rgb * uLt * uUMatCol;
		if ( uURampOn > 0.5 ) outgoingLight += uK * texture2D( uURamp, vec2( min( uNoV * uNoV, 0.96 ) * uURampST.x + uURampST.y, 1.0 ) ).rgb * uLt * uURampCol;
		if ( uUEnvKind > 0.5 ) {
			float eR = clamp( roughnessFactor, 0.04, 1.0 );
			float eL = eR * ( 1.7 - 0.7 * eR ) * 6.0;
			vec3 eM = textureLod( uUEnv, uST, eL ).rgb;
			vec3 eF0 = mix( vec3( 0.04 ), diffuseColor.rgb, metalnessFactor );
			if ( uUEnvKind < 1.5 ) outgoingLight += uK * eM * uLt * uUEnvCol * eF0 * uAO; // Cloth _MATCAPENV
			else if ( uUEnvKind < 2.5 ) { // Lengcy / MatcapMask _IF_MATREFL
				float eRef = 0.04 + 0.96 * metalnessFactor;
				float eGraz = clamp( 1.0 - eR + eRef, 0.0, 1.0 );
				float eFr = pow( 1.0 - min( uNoV + 0.004, 1.0 ), 4.0 );
				outgoingLight += uK * eM * uUEnvCol / ( eR * eR * eR * eR + 1.0 ) * mix( eF0, vec3( eGraz ), eFr );
			} else if ( uUEnvKind < 3.5 ) outgoingLight += uK * eM * uUEnvCol * uAO; // battle actorMatcap: matcap x _SpecularColor
			else outgoingLight = diffuseColor.rgb + texture2D( uUEnv, uST ).rgb; // basepbrBubble: unlit albedo + matcap
		}
	}
}
`;

function patch(shader: { fragmentShader: string; uniforms: Record<string, unknown> }, u: U) {
  const at = '#include <opaque_fragment>';
  if (!shader.fragmentShader.includes(at)) { console.warn('matcap: chunk not found'); return; }
  shader.fragmentShader = DECL + shader.fragmentShader.replace(at, `if ( uMatcapOn > 0.5 ) {
	vec3 mcV = isOrthographic ? vec3( 0.0, 0.0, 1.0 ) : normalize( vViewPosition );
	vec3 mcN = uMatcapReflect > 0.5 ? normalize( normal ) : normalize( nonPerturbedNormal );
	vec3 mcX = normalize( vec3( mcV.z, 0.0, - mcV.x ) );
	vec3 mcY = cross( mcV, mcX );
	vec4 mc = texture2D( uMatcap, vec2( dot( mcX, mcN ), - dot( mcY, mcN ) ) * 0.495 + 0.5 ); // glTF image rows: top first
	if ( uMatcapReflect > 0.5 ) {
		outgoingLight += metalnessFactor * mc.rgb * diffuseColor.rgb * uMatcapGain;
	} else {
		outgoingLight = mc.rgb;
		diffuseColor.a = mc.a;
	}
}
${UNITE_GLSL}
${at}`);
  Object.assign(shader.uniforms, u);
}

export function isMatcapReflect(mat: THREE.Material): boolean {
  const ud = mat.userData as { matcapMode?: string; metalEnv?: boolean } | undefined;
  return ud?.matcapMode === 'reflect' || !!ud?.metalEnv;
}

let neutral: THREE.Texture | null = null;
const NEUTRAL_MEAN = 0.746; // mean linear luminance of defaultMatcap's disc (its gain normalises it like the game's matcaps)
/** Stand-in reflection for metal without a matcap: a soft studio sphere (bright sky above, darker floor, one key
 *  highlight upper left). Not game data: the scene's own reflection is not in the rip. */
export function defaultMatcap(): THREE.Texture {
  if (neutral) return neutral;
  const n = 64, data = new Uint8Array(n * n * 4);
  const k = new THREE.Vector3(-0.4, 0.6, 0.7).normalize();
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const u = (x + 0.5) / n * 2 - 1, v = 1 - (y + 0.5) / n * 2; // row 0 = top (glTF convention, flipY false)
    const r2 = Math.min(u * u + v * v, 1), nz = Math.sqrt(1 - r2);
    const c = 0.55 + 0.35 * (v * 0.5 + 0.5) + 0.5 * Math.pow(Math.max(0, u * k.x + v * k.y + nz * k.z), 24); // linear
    const b = Math.round(Math.min(1, c) * 255), i = (y * n + x) * 4;
    data[i] = data[i + 1] = data[i + 2] = b; data[i + 3] = 255;
  }
  neutral = new THREE.DataTexture(data, n, n, THREE.RGBAFormat);
  neutral.colorSpace = THREE.LinearSRGBColorSpace; // values above are linear light (an sRGB tag made metal near black)
  neutral.needsUpdate = true;
  return neutral;
}

/** Install on a material carrying `matcapTexture` (legacy) or `extras.unite` terms (once, before the game-shading patch
 *  captures onBeforeCompile). */
export function installMatcap(mat: THREE.MeshStandardMaterial): boolean {
  const ud = uniteData(mat);
  if (typeof mat.userData?.matcapTexture !== 'number' && !mat.userData?.metalEnv && !ud) return false;
  if (mat.userData.__matcap) return true;
  const reflect = isMatcapReflect(mat);
  const gain = (mat.userData as { matcapGain?: number }).matcapGain ?? (mat.userData.metalEnv ? 1 / NEUTRAL_MEAN : 1);
  const g = ud?.glass;
  const u: U = {
    uMatcap: { value: null }, uMatcapOn: { value: 0 }, uMatcapReflect: { value: reflect ? 1 : 0 }, uMatcapGain: { value: gain },
    uUMat: { value: null }, uUMatCol: { value: t3(ud?.matTex?.color, [2, 2, 2]) }, uUMatOn: { value: 0 },
    uURamp: { value: null }, uURampCol: { value: t3(ud?.rampLight?.color, [1, 1, 1]) },
    uURampST: { value: new THREE.Vector2(ud?.rampLight?.scale ?? 1, ud?.rampLight?.offset ?? 0) }, uURampOn: { value: 0 },
    uUEnv: { value: null }, uUEnvCol: { value: t3(ud?.env?.color, [1, 1, 1]) }, uUEnvKind: { value: 0 },
    uUK: UNITE_K,
    uUGlass: { value: null }, uUGTop: { value: t4(g?.top) }, uUGBot: { value: t4(g?.bot) },
    uUGP: { value: new THREE.Vector3(g?.offset ?? 0, g?.pow ?? 2, g?.scale ?? 1) }, uUGAlpha: { value: g?.alpha ?? 1 }, uUGKind: { value: 0 },
  };
  mat.userData.__matcap = u;
  const prev = mat.onBeforeCompile;
  mat.onBeforeCompile = (shader, renderer) => {
    patch(shader as never, u);
    prev?.call(mat, shader, renderer);
  };
  if (g) { mat.transparent = true; mat.depthWrite = false; } // unite glass: blended by its own alpha
  else if (!reflect && !ud && typeof mat.userData?.matcapTexture === 'number') mat.depthWrite = false; // legacy glass only; metal is an ordinary opaque surface
  mat.needsUpdate = true;
  return true;
}

/** Load the `extras.unite` matcap / ramp textures through the glTF parser and switch their terms on. */
export function loadUniteMatcaps(mat: THREE.MeshStandardMaterial, parser: { getDependency: (type: string, index: number) => Promise<unknown> } | undefined) {
  const u = mat.userData?.__matcap as U | undefined;
  const d = uniteData(mat);
  if (!u || !d || !parser) return;
  const get = async (ref: Ref | undefined, mips: boolean): Promise<THREE.Texture | null> => {
    if (typeof ref?.index !== 'number') return null;
    try {
      const t = (await parser.getDependency('texture', ref.index)) as THREE.Texture;
      t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
      t.flipY = false;
      t.colorSpace = THREE.SRGBColorSpace;
      t.generateMipmaps = mips;
      t.minFilter = mips ? THREE.LinearMipmapLinearFilter : THREE.LinearFilter;
      t.needsUpdate = true;
      return t;
    } catch { return null; }
  };
  void (async () => {
    const [m, r, e, g] = await Promise.all([get(d.matTex?.texture, false), get(d.rampLight?.texture, false), get(d.env?.texture, true), get(d.glass?.texture, false)]);
    if (m) { u.uUMat.value = m; u.uUMatOn.value = 1; }
    if (r) { u.uURamp.value = r; u.uURampOn.value = 1; }
    if (e) { u.uUEnv.value = e; u.uUEnvKind.value = ENV_KIND[d.env?.kind ?? ''] ?? 1; }
    if (g) { u.uUGlass.value = g; u.uUGKind.value = d.glass?.kind === 'actor' ? 2 : 1; }
  })();
}

export function setMatcapTexture(mat: THREE.MeshStandardMaterial, tex: THREE.Texture | null) {
  const u = mat.userData?.__matcap as U | undefined;
  if (!u) return;
  if (tex) { tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping; tex.channel = 0; tex.flipY = false; tex.needsUpdate = true; }
  u.uMatcap.value = tex;
  u.uMatcapOn.value = tex ? 1 : 0;
}
