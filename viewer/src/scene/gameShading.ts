import * as THREE from 'three';
import type { CategoryLight } from './environment';

export interface ShadingParams {
  halfLambertBias: number; shadowStrength: number; shadowColor: [number, number, number]; shadowShift: number;
  rimIntensity: number; rimOffset: number; rimContrast: number; specularIntensity: number;
  occlusionStrength: number; normalHeight: number; unlit: boolean;
  /** SV `NonDirectional` (Gastly smoke): lit by key + ambient colour with no N.L term, no shadow band, no rim/spec. */
  nonDirectional?: boolean;
  specularOffset: number; specularContrast: number; backRimIntensity: number; hairSpecular: boolean;
  /** specular mask G/B carry per-texel offset ((o+1)/2) and contrast (c/2); R is the intensity. */
  specularMaskChannels: boolean;
  fresnel: boolean; fresnelIntensity: number; roughness: number; metallic: number; giGain: number;
}

export interface GameTextures { shadow: THREE.Texture | null; spec: THREE.Texture | null; rim: THREE.Texture | null }

const DEFAULTS: ShadingParams = {
  halfLambertBias: 0.5, shadowStrength: 0.8, shadowColor: [0.6, 0.6, 0.6], shadowShift: 0,
  rimIntensity: 0, rimOffset: 0, rimContrast: 1, specularIntensity: 0.04,
  occlusionStrength: 1, normalHeight: 1, unlit: false,
  specularOffset: 0, specularContrast: 1, backRimIntensity: 0, hairSpecular: false, specularMaskChannels: false,
  fresnel: false, fresnelIntensity: 0, roughness: 1, metallic: 0, giGain: 0.5,
};

// Tunables (compared by eye against the in-game screenshot).
// FITTED: SHADOW_FLOOR / BAND_LO / BAND_HI toon-band shape, compared by eye against the in-game screenshot, 2026-10-19 (engine-rules debt)
const SHADOW_FLOOR = 0.6;
const BAND_LO = 0.30;
const BAND_HI = 0.62;

interface Pbr {
  onBeforeCompile: THREE.Material['onBeforeCompile'];
  customProgramCacheKey: () => string;
  roughness: number; metalness: number; normalScale: THREE.Vector2; aoMapIntensity: number; envMapIntensity: number;
  color?: THREE.Color; emissive?: THREE.Color; emissiveMap?: THREE.Texture | null; emissiveIntensity?: number;
}
type U = { value: unknown };
interface GameUniforms { uWrap: U; uShadowStrength: U; uShadowShift: U; uShadowColor: U; uRim: U; uSpec: U; uShadowAmount: U; uShadowTex: U; uHasShadowTex: U; uRimScale: U; uSaturation: U; uContrast: U;
  uSpecTex: U; uHasSpecTex: U; uRimTex: U; uHasRimTex: U; uSpecP: U; uBackRim: U; uFres: U; uGI: U;
  uCatKey: U; uCatAmb: U; uCatRim: U }

/** Linear RGB multiplier from a scale and an optional sRGB hex colour. */
function catColor(v: THREE.Vector3, scale: number, hex?: string) {
  const c = hex ? new THREE.Color(hex) : new THREE.Color(1, 1, 1);
  v.set(c.r * scale, c.g * scale, c.b * scale);
}

function params(mat: THREE.Material): ShadingParams {
  const s = (mat.userData?.shading ?? {}) as Partial<ShadingParams>;
  return { ...DEFAULTS, ...s };
}

function patch(shader: { fragmentShader: string; uniforms: Record<string, unknown> }, u: GameUniforms) {
  const a = 'float dotNL = saturate( dot( geometryNormal, directLight.direction ) );';
  const b = 'vec3 irradiance = dotNL * directLight.color;';
  const c = 'reflectedLight.directSpecular += irradiance * BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );';
  const d = 'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );';
  const e = 'reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );';
  const head = 'void RE_Direct_Physical(';
  // onBeforeCompile runs before #include resolution, so patch the chunk text and swap it in.
  const inc = '#include <lights_physical_pars_fragment>';
  let f = shader.fragmentShader;
  let f2 = THREE.ShaderChunk.lights_physical_pars_fragment;
  if (!f.includes(inc) || ![a, b, c, d, e, head].every((n) => f2.includes(n))) { console.warn('game shading: chunk not found'); return; }
  // Lights are processed in scene order: the first directional light is the key (toon band, shadow colour on its far side,
  // drives gGameLit for the ambient tint); later ones are back/rim lights that only add light on their lit side.
  f2 = f2.replace(head, `float gGameLit = 1.0;
int gLightN = 0;
vec3 gameShadowCol( const in vec3 diffuse ) {
	vec3 g = diffuse * mix( vec3( 1.0 ), uShadowColor, uShadowStrength ) * 0.6;
	#ifdef USE_MAP
		vec3 gt = texture2D( uShadowTex, vMapUv ).rgb;
		g = mix( g, gt, step( 0.5, uHasShadowTex ) );
	#endif
	return g;
}
${head}`);
  f2 = f2.replace(a, `bool gIsKey = gLightN == 0;
	gLightN++;
	float nlRaw = dot( geometryNormal, directLight.direction );
	float wrapped = saturate( nlRaw * (1.0 - uWrap) + uWrap );
	float lit = smoothstep( uShadowShift * 0.5 + ${BAND_LO.toFixed(2)}, uShadowShift * 0.5 + ${BAND_HI.toFixed(2)}, wrapped );
	if ( gIsKey ) gGameLit = lit;
	vec3 gCatK = gIsKey ? uCatKey : vec3( 1.0 );
	float dotNL = wrapped;`);
  f2 = f2.replace(b, 'vec3 irradiance = directLight.color;');
  f2 = f2.replace(c, `float gNV = saturate( dot( geometryNormal, geometryViewDir ) );
	float gRimM = 1.0;
	float gSpecM = 1.0;
	vec2 gSpecOC = uSpecP.yz;
	#ifdef USE_MAP
		gRimM = mix( 1.0, texture2D( uRimTex, vMapUv ).r, step( 0.5, uHasRimTex ) );
		vec3 gSpecT = texture2D( uSpecTex, vMapUv ).rgb;
		gSpecM = mix( 1.0, gSpecT.r, step( 0.5, uHasSpecTex ) );
		if ( uHasSpecTex > 1.5 ) gSpecOC = vec2( gSpecT.g * 2.0 - 1.0, gSpecT.b * 2.0 );
	#endif
	// Rim is tinted by the albedo: added as white it turned a red Poké Ball pink and dark hair grey.
	vec3 rimF = material.diffuseColor * ( gRimM * uRim.x * uRimScale * pow( saturate( 1.0 - gNV + uRim.y * 0.5 ), max( uRim.z, 1.0 ) ) * 0.25 );
	// SpecularOffset widens the lobe (lower exponent). Adding it to N.H before the power clipped every texel within
	// ~35 deg of the half vector to full intensity: a flat glossy cap over the whole Poké Ball top.
	float gNH = saturate( dot( geometryNormal, normalize( directLight.direction + geometryViewDir ) ) );
	float specF = gSpecM * uSpec * pow( gNH, mix( 96.0, 16.0, saturate( gSpecOC.y * 0.5 ) ) * ( 1.0 - 0.75 * clamp( gSpecOC.x, -1.0, 1.0 ) ) );
	if ( uSpecP.x > 0.5 ) specF = gSpecM * max( uSpec, 0.2 ) * pow( gNH, 24.0 ) * 1.5;
	if ( uFres.x > 0.5 ) specF += 0.5 * uFres.z * pow( gNH, 96.0 );
	reflectedLight.directSpecular += directLight.color * gCatK * ( rimF + specF * lit );`);
  f2 = f2.replace(d, `vec3 gCol = gIsKey ? mix( material.diffuseColor, gameShadowCol( material.diffuseColor ), ( 1.0 - lit ) * uShadowAmount * uShadowStrength / 0.7 )
		: material.diffuseColor * saturate( nlRaw ) * lit;
	reflectedLight.directDiffuse += directLight.color * gCatK * gCol * RECIPROCAL_PI;`);
  // Ambient light is tinted toward the shadow colour everywhere (ShadowingGIGain), more so on the shadow side.
  f2 = f2.replace(e, `vec3 gColI = mix( material.diffuseColor, gameShadowCol( material.diffuseColor ), saturate( max( ( 1.0 - gGameLit ) * uShadowAmount, uGI ) ) );
	reflectedLight.indirectDiffuse += irradiance * uCatAmb * gColI * RECIPROCAL_PI;
	float gNV2 = saturate( dot( geometryNormal, geometryViewDir ) );
	// The category's own rim light (rim_N): a view-dependent edge glow tinted by the albedo like the material rim.
	reflectedLight.indirectDiffuse += material.diffuseColor * uCatRim.rgb * pow( 1.0 - gNV2, max( uCatRim.a, 0.5 ) );
	reflectedLight.indirectDiffuse += material.diffuseColor * ( uBackRim * pow( 1.0 - gNV2, 3.0 ) * uRimScale );
	if ( uFres.x > 0.5 ) reflectedLight.indirectDiffuse += vec3( uFres.y * pow( 1.0 - gNV2, 2.5 ) );`);
  f = f.replace(inc, f2);
  const dith = '#include <dithering_fragment>';
  if (f.includes(dith)) f = f.replace(dith, `{ float gLum = dot( gl_FragColor.rgb, vec3( 0.299, 0.587, 0.114 ) ); gl_FragColor.rgb = mix( vec3( gLum ), gl_FragColor.rgb, uSaturation ); gl_FragColor.rgb = max( ( gl_FragColor.rgb - 0.5 ) * uContrast + 0.5, 0.0 ); }\n${dith}`);
  else console.warn('game shading: chunk not found');
  Object.assign(shader.uniforms, u);
  shader.fragmentShader = `uniform float uWrap;
uniform float uShadowStrength;
uniform float uShadowShift;
uniform vec3 uShadowColor;
uniform vec3 uRim;
uniform float uSpec;
uniform float uShadowAmount;
uniform sampler2D uShadowTex;
uniform float uHasShadowTex;
uniform float uRimScale;
uniform float uSaturation;
uniform float uContrast;
uniform sampler2D uSpecTex;
uniform float uHasSpecTex;
uniform sampler2D uRimTex;
uniform float uHasRimTex;
uniform vec3 uSpecP;
uniform float uBackRim;
uniform vec3 uFres;
uniform float uGI;
uniform vec3 uCatKey;
uniform vec3 uCatAmb;
uniform vec4 uCatRim;
` + f;
}

/** Re-sync emissive state for unlit materials (call after albedo changes mat.map / mat.color). */
export function syncUnlit(mat: THREE.MeshStandardMaterial) {
  const pbr = mat.userData.__pbr as Pbr | undefined;
  if (!pbr || !mat.userData.__gameUnlit) return;
  if (mat.color.getHex() !== 0) pbr.color = mat.color.clone();
  mat.emissive.copy(pbr.color ?? mat.color);
  if (mat.emissiveMap !== mat.map) { mat.emissiveMap = mat.map; mat.needsUpdate = true; }
  mat.color.setRGB(0, 0, 0);
}

export function applyGameShading(mat: THREE.MeshStandardMaterial, env: { detail: number; shadow: number; rim: number; saturation: number; contrast?: number; gi?: number; categories?: Record<string, CategoryLight> }, tex: GameTextures = { shadow: null, spec: null, rim: null }, reflect: THREE.Texture | null = null, reflectIntensity = 1.5) {
  const detail = env.detail;
  if (!mat.userData.__pbr) {
    mat.userData.__pbr = {
      onBeforeCompile: mat.onBeforeCompile, customProgramCacheKey: mat.customProgramCacheKey,
      roughness: mat.roughness, metalness: mat.metalness, normalScale: mat.normalScale.clone(),
      aoMapIntensity: mat.aoMapIntensity, envMapIntensity: mat.envMapIntensity,
      color: mat.color.clone(), emissive: mat.emissive.clone(), emissiveMap: mat.emissiveMap, emissiveIntensity: mat.emissiveIntensity,
    } satisfies Pbr;
  }
  const pbr = mat.userData.__pbr as Pbr;
  const p = params(mat);
  mat.roughness = 1; mat.metalness = 0; mat.envMapIntensity = 0;
  // Per-texel metallic baked from SV `MetallicLayerN` (metallicRoughness texture): keep it, so three's diffuse
  // (albedo x (1 - metallic)) drops the metal parts. SV eyes paint the iris as metal; forced to 0 it was a lit pale disc.
  if (mat.metalnessMap) { mat.metalness = pbr.metalness; mat.roughness = pbr.roughness; }
  if (reflect) { mat.envMap = reflect; mat.envMapIntensity = reflectIntensity; mat.roughness = 0.2; mat.metalness = 0.1; } // _ENVMAP_ON materials (glasses, tubes, metal)
  else if (mat.envMap && mat.userData.__reflEnv) mat.envMap = null;
  mat.userData.__reflEnv = !!reflect;
  const ns = detail * p.normalHeight;
  mat.normalScale.set(ns, Math.sign(pbr.normalScale.y || 1) * ns);
  mat.aoMapIntensity = detail * p.occlusionStrength * 0.5;

  if (p.unlit) {
    if (!mat.userData.__gameUnlit) {
      mat.userData.__gameUnlit = true;
      pbr.color = mat.color.clone();
      mat.emissiveIntensity = 1;
      mat.needsUpdate = true;
    }
    syncUnlit(mat);
    return;
  }

  let u = mat.userData.__gameUniforms as GameUniforms | undefined;
  const first = !u;
  if (!u) {
    u = {
      uWrap: { value: 0 }, uShadowStrength: { value: 0 }, uShadowShift: { value: 0 },
      uShadowColor: { value: new THREE.Vector3() }, uRim: { value: new THREE.Vector3() }, uSpec: { value: 0 },
      uShadowAmount: { value: 0.65 }, uShadowTex: { value: null }, uHasShadowTex: { value: 0 },
      uRimScale: { value: 1 }, uSaturation: { value: 1 }, uContrast: { value: 1 },
      uSpecTex: { value: null }, uHasSpecTex: { value: 0 }, uRimTex: { value: null }, uHasRimTex: { value: 0 },
      uSpecP: { value: new THREE.Vector3() }, uBackRim: { value: 0 }, uFres: { value: new THREE.Vector3() }, uGI: { value: 0.5 },
      uCatKey: { value: new THREE.Vector3(1, 1, 1) }, uCatAmb: { value: new THREE.Vector3(1, 1, 1) }, uCatRim: { value: new THREE.Vector4(0, 0, 0, 3) },
    };
    mat.userData.__gameUniforms = u;
  }
  const nd = !!p.nonDirectional;
  // uWrap 1 makes `wrapped = saturate(nlRaw * (1 - uWrap) + uWrap)` 1 everywhere: the toon band is fully lit, so
  // the colour is albedo x (key + ambient) with no direction, as the game's NonDirectional shader shows in footage.
  u.uWrap.value = nd ? 1 : p.halfLambertBias;
  u.uShadowStrength.value = p.shadowStrength;
  u.uShadowShift.value = nd ? 0 : p.shadowShift;
  (u.uShadowColor.value as THREE.Vector3).set(...p.shadowColor);
  (u.uRim.value as THREE.Vector3).set(nd ? 0 : p.rimIntensity, p.rimOffset, p.rimContrast);
  u.uSpec.value = nd ? 0 : p.specularIntensity;
  u.uShadowAmount.value = env.shadow;
  u.uRimScale.value = env.rim;
  u.uSaturation.value = env.saturation;
  u.uContrast.value = env.contrast ?? 1;
  u.uShadowTex.value = tex.shadow;
  u.uHasShadowTex.value = tex.shadow ? 1 : 0;
  u.uSpecTex.value = tex.spec; u.uHasSpecTex.value = tex.spec ? (p.specularMaskChannels ? 2 : 1) : 0;
  u.uRimTex.value = tex.rim; u.uHasRimTex.value = tex.rim ? 1 : 0;
  (u.uSpecP.value as THREE.Vector3).set(p.hairSpecular ? 1 : 0, p.specularOffset, p.specularContrast);
  u.uBackRim.value = nd ? 0 : p.backRimIntensity;
  u.uGI.value = Math.min(1, (env.gi ?? 1) * p.giGain);
  const cat = env.categories?.[String(mat.userData?.category ?? '')];
  catColor(u.uCatKey.value as THREE.Vector3, cat?.key ?? 1, cat?.keyColor);
  catColor(u.uCatAmb.value as THREE.Vector3, cat?.ambient ?? 1, cat?.ambientColor);
  const rimV = new THREE.Vector3();
  catColor(rimV, nd ? 0 : (cat?.rim ?? 0) * env.rim, cat?.rimColor);
  (u.uCatRim.value as THREE.Vector4).set(rimV.x, rimV.y, rimV.z, cat?.rimPower ?? 3);
  (u.uFres.value as THREE.Vector3).set(p.fresnel ? 1 : 0, p.fresnelIntensity, Math.min(1, Math.max(0, 1.3 - p.roughness + p.metallic)));
  if (first) {
    const uu = u;
    // Chain any patch installed before (gas shells, see gas.ts) so both survive the mode switch.
    const prev = pbr.onBeforeCompile, prevKey = pbr.customProgramCacheKey;
    mat.onBeforeCompile = function (this: THREE.Material, shader, renderer) { prev?.call(this, shader, renderer); patch(shader as never, uu); };
    mat.customProgramCacheKey = function (this: THREE.Material) { return 'game-shading-v4' + (prevKey?.call(this) ?? ''); };
    mat.needsUpdate = true;
  }
}

export function removeGameShading(mat: THREE.MeshStandardMaterial) {
  const pbr = mat.userData.__pbr as Pbr | undefined;
  if (!pbr) return;
  mat.onBeforeCompile = pbr.onBeforeCompile;
  mat.customProgramCacheKey = pbr.customProgramCacheKey;
  mat.roughness = pbr.roughness; mat.metalness = pbr.metalness;
  mat.normalScale.copy(pbr.normalScale);
  mat.aoMapIntensity = pbr.aoMapIntensity; mat.envMapIntensity = pbr.envMapIntensity;
  if (mat.userData.__gameUnlit) {
    mat.userData.__gameUnlit = false;
    if (pbr.color) mat.color.copy(pbr.color);
    mat.emissive.copy(pbr.emissive ?? new THREE.Color(0, 0, 0));
    mat.emissiveMap = pbr.emissiveMap ?? null;
    mat.emissiveIntensity = pbr.emissiveIntensity ?? 1;
  }
  delete mat.userData.__gameUniforms;
  mat.needsUpdate = true;
}
