// John Hable's piecewise power tone curve ("Filmic Worlds", 2017): three segments (toe, linear+gamma middle,
// shoulder) built from six user parameters. Game Freak's Trinity post-process stores exactly these
// (`PfxToeStrength`, `PfxToeLength`, `PfxShoulderStrength`, `PfxShoulderLength`, `PfxShoulderAngle`,
// `PfxTonemapGamma` with `PfxTonemapMode 2`), so the viewer can reproduce the game's curve.
// Port of FilmicToneCurve::CalcDirectParamsFromUser / CreateCurve; constants are baked into GLSL because
// three.js' CustomToneMapping is a plain function in a shared shader chunk (no uniforms).
import * as THREE from 'three';
import type { HableParams } from './environment';

interface Segment { offsetX: number; offsetY: number; scaleX: number; scaleY: number; lnA: number; B: number }

export interface HableCurve { W: number; invW: number; x0: number; x1: number; toe: Segment; mid: Segment; shoulder: Segment }

const sat = (v: number) => Math.min(1, Math.max(0, v));

function evalSeg(s: Segment, x: number): number {
  const x0 = (x - s.offsetX) * s.scaleX;
  const y0 = x0 > 0 ? Math.exp(s.lnA + s.B * Math.log(x0)) : 0;
  return y0 * s.scaleY + s.offsetY;
}

export function evalCurve(c: HableCurve, x: number): number {
  const nx = x * c.invW;
  const s = nx < c.x0 ? c.toe : nx < c.x1 ? c.mid : c.shoulder;
  return evalSeg(s, nx);
}

export function buildHableCurve(u: HableParams): HableCurve {
  const toeLength = sat(u.toeLength), toeStrength = sat(u.toeStrength);
  const shoulderAngle = sat(u.shoulderAngle), shoulderStrength = sat(u.shoulderStrength);
  const shoulderLength = Math.max(1e-5, sat(u.shoulderLength));
  const gamma = u.gamma;
  // direct params
  const x0d = toeLength * 0.5;
  const y0d = (1 - toeStrength) * x0d;
  const remainingY = 1 - y0d;
  const initialW = x0d + remainingY;
  const y1Offset = (1 - shoulderLength) * remainingY;
  const x1d = x0d + y1Offset;
  const y1d = y0d + y1Offset;
  const extraW = Math.pow(2, shoulderStrength) - 1;
  const W = initialW + extraW;
  let overshootX = (W * 2) * shoulderAngle * shoulderStrength;
  let overshootY = 0.5 * shoulderAngle * shoulderStrength;
  // create curve (normalised to W)
  const invW = 1 / W;
  let x0 = x0d / W, x1 = x1d / W, y0 = y0d, y1 = y1d;
  overshootX /= W;
  const m = (y1 - y0) / (x1 - x0), b = y0 - x0 * m;
  const g = gamma;
  const mid: Segment = { offsetX: -(b / m), offsetY: 0, scaleX: 1, scaleY: 1, lnA: g * Math.log(m), B: g };
  const dLinGamma = (x: number) => g * m * Math.pow(m * x + b, g - 1);
  const toeM = dLinGamma(x0), shoulderM = dLinGamma(x1);
  y0 = Math.max(1e-5, Math.pow(y0, g));
  y1 = Math.max(1e-5, Math.pow(y1, g));
  overshootY = Math.pow(1 + overshootY, g) - 1;
  const solveAB = (px: number, py: number, pm: number) => { const B = (pm * px) / py; return { B, lnA: Math.log(py) - B * Math.log(px) }; };
  const t = solveAB(x0, y0, toeM);
  const toe: Segment = { offsetX: 0, offsetY: 0, scaleX: 1, scaleY: 1, lnA: t.lnA, B: t.B };
  const sh = solveAB((1 + overshootX) - x1, (1 + overshootY) - y1, shoulderM);
  const shoulder: Segment = { offsetX: 1 + overshootX, offsetY: 1 + overshootY, scaleX: -1, scaleY: -1, lnA: sh.lnA, B: sh.B };
  const curve: HableCurve = { W, invW, x0, x1, toe, mid, shoulder };
  // normalise so that x = 1 (the white point W in input units) hits 1.0
  const scale = evalCurve(curve, W); // Eval(W): normalised x = 1
  const inv = 1 / scale;
  for (const s of [toe, mid, shoulder]) { s.offsetY *= inv; s.scaleY *= inv; }
  return curve;
}

const f = (v: number) => (Number.isFinite(v) ? v.toExponential(7) : '0.0');

/** GLSL body for three.js' `CustomToneMapping` with this curve's constants baked in. */
export function hableGLSL(c: HableCurve): string {
  const seg = (name: string, s: Segment) =>
    `float hable_${name}( float x ) { float x0 = ( x - ${f(s.offsetX)} ) * ${f(s.scaleX)}; float y0 = x0 > 0.0 ? exp( ${f(s.lnA)} + ${f(s.B)} * log( x0 ) ) : 0.0; return y0 * ${f(s.scaleY)} + ${f(s.offsetY)}; }`;
  return [
    seg('toe', c.toe), seg('mid', c.mid), seg('shoulder', c.shoulder),
    `float hable_eval( float x ) { float nx = x * ${f(c.invW)}; return nx < ${f(c.x0)} ? hable_toe( nx ) : nx < ${f(c.x1)} ? hable_mid( nx ) : hable_shoulder( nx ); }`,
    'vec3 CustomToneMapping( vec3 color ) { color *= toneMappingExposure; return saturate( vec3( hable_eval( color.r ), hable_eval( color.g ), hable_eval( color.b ) ) ); }',
  ].join('\n');
}

const BASE_CHUNK = THREE.ShaderChunk.tonemapping_pars_fragment;
const STUB = 'vec3 CustomToneMapping( vec3 color ) { return color; }';
let installedKey = '';

/** Swap the CustomToneMapping stub in three's shared chunk for this curve. Returns a cache key that changes with
 *  the parameters; materials must expose it through `customProgramCacheKey` so three recompiles their programs. */
export function installHable(params: HableParams): string {
  const key = JSON.stringify(params);
  if (key !== installedKey) {
    THREE.ShaderChunk.tonemapping_pars_fragment = BASE_CHUNK.replace(STUB, hableGLSL(buildHableCurve(params)));
    installedKey = key;
  }
  return key;
}
