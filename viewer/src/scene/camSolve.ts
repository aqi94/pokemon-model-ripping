/**
 * Camera-from-pins solver: finds the perspective camera (position, yaw, pitch, vertical FOV, optional roll about the
 * view axis; up = +Y like OrbitControls) that projects each 3D point onto its 2D target, by Levenberg-Marquardt on the
 * pixel residuals. The viewer camera cannot roll: the caller applies roll by rotating the reference image instead.
 * Multi-start over yaw/pitch so a bad starting angle does not trap it. Pure maths (no three.js) so it can be tested
 * in node.
 */

export type V3 = [number, number, number];

export interface CamPose { fov: number; position: V3; target: V3 }

export interface SolveInput {
  points: V3[];
  /** targets in half-height units around the principal point (x right, y up; the frame's top edge is y = 1), parallel to points */
  targets: [number, number][];
  /** pixels per unit (half the frame height in px): residuals are reported in pixels */
  pxPerUnit: number;
  start: CamPose;
  /** starting roll in degrees */
  startRoll?: number;
  lockFov: boolean;
  lockRoll: boolean;
  /** warm start only (no multi-start): for sweeping many frames from a nearby solution */
  quick?: boolean;
}

/** roll in degrees, counter-clockwise on screen: image = projection rotated by roll */
export interface SolveResult extends CamPose { roll: number; rmsPx: number; perPinPx: number[] }

const DEG = Math.PI / 180;
const BEHIND_PENALTY = 1e4;
/** FOV range the solver may use (degrees). Game cameras sit well inside it; the extremes are where noise sends it. */
const FOV_MIN = 8, FOV_MAX = 60;
/** Free FOV: px of residual per unit of log(tan(fov/2)) away from the start FOV. FOV and distance trade off almost
 *  perfectly for a handful of hand-placed pins (25 px of click noise sent it to 129° or 0.5°), so it needs a prior. */
const FOV_PRIOR_PX = 60;
/** Camera must stay at least this many pin-spread radii from every pin (no solutions inside the head). */
const MIN_DEPTH_RADII = 1.5;
const ROLL_MAX = 45 * DEG;

// p = [px, py, pz, yaw, pitch, logTanHalfFov, roll]; yaw/pitch give the direction the camera LOOKS along.
function dirOf(yaw: number, pitch: number): V3 {
  return [Math.cos(pitch) * Math.sin(yaw), Math.sin(pitch), Math.cos(pitch) * Math.cos(yaw)];
}

function basis(yaw: number, pitch: number) {
  const f = dirOf(yaw, pitch);
  // right = f x up (up = +Y), normalised; then u = r x f
  let r: V3 = [-f[2], 0, f[0]];
  const rl = Math.hypot(r[0], r[2]) || 1e-9;
  r = [r[0] / rl, 0, r[2] / rl];
  const u: V3 = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  return { f, r, u };
}

interface Ctx { logT0: number; minDepth: number }

function residuals(p: number[], inp: SolveInput, out: number[], ctx?: Ctx) {
  const { f, r, u } = basis(p[3], p[4]);
  const t = Math.exp(p[5]);
  const cr = Math.cos(p[6]), sr = Math.sin(p[6]);
  const k = inp.pxPerUnit;
  for (let i = 0; i < inp.points.length; i++) {
    const X = inp.points[i];
    const v0 = X[0] - p[0], v1 = X[1] - p[1], v2 = X[2] - p[2];
    const z = v0 * f[0] + v1 * f[1] + v2 * f[2];
    if (z < 1e-4) { out[2 * i] = BEHIND_PENALTY; out[2 * i + 1] = BEHIND_PENALTY; continue; }
    const x0 = (v0 * r[0] + v1 * r[1] + v2 * r[2]) / (z * t);
    const y0 = (v0 * u[0] + v1 * u[1] + v2 * u[2]) / (z * t);
    out[2 * i] = (x0 * cr - y0 * sr - inp.targets[i][0]) * k;
    out[2 * i + 1] = (x0 * sr + y0 * cr - inp.targets[i][1]) * k;
  }
  if (!ctx) return;
  const n2 = inp.points.length * 2;
  out[n2] = inp.lockFov ? 0 : (p[5] - ctx.logT0) * FOV_PRIOR_PX;
  // soft wall: grows quickly once the camera gets closer than minDepth to any pin
  let near = 0;
  for (const X of inp.points) {
    const z = (X[0] - p[0]) * f[0] + (X[1] - p[1]) * f[1] + (X[2] - p[2]) * f[2];
    if (z < ctx.minDepth) near += ((ctx.minDepth - z) / ctx.minDepth) * 200;
  }
  out[n2 + 1] = near;
}

const sumSq = (r: number[]) => r.reduce((s, v) => s + v * v, 0);

/** Solve A x = b by Gaussian elimination with partial pivoting (n <= 7). */
function solveLinear(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-14) return null;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = c + 1; r < n; r++) {
      const k = M[r][c] / M[c][c];
      for (let j = c; j <= n; j++) M[r][j] -= k * M[c][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let j = r + 1; j < n; j++) s -= M[r][j] * x[j];
    x[r] = s / M[r][r];
  }
  return x;
}

function lm(p0: number[], inp: SolveInput, scale: number, ctx: Ctx, iters = 200): { p: number[]; cost: number } {
  const free = [0, 1, 2, 3, 4, ...(inp.lockFov ? [] : [5]), ...(inp.lockRoll ? [] : [6])];
  const steps = [scale * 1e-5, scale * 1e-5, scale * 1e-5, 1e-6, 1e-6, 1e-6, 1e-6];
  const m = inp.points.length * 2 + 2;
  let p = [...p0];
  const r = new Array(m).fill(0), rt = new Array(m).fill(0);
  residuals(p, inp, r, ctx);
  let cost = sumSq(r);
  let lambda = 1e-3;
  for (let it = 0; it < iters; it++) {
    // numeric Jacobian (central differences)
    const J = free.map((k) => {
      const a = [...p], b = [...p];
      a[k] += steps[k]; b[k] -= steps[k];
      const ra = new Array(m).fill(0), rb = new Array(m).fill(0);
      residuals(a, inp, ra, ctx); residuals(b, inp, rb, ctx);
      return ra.map((v, i) => (v - rb[i]) / (2 * steps[k]));
    });
    const n = free.length;
    const A = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => J[i].reduce((s, v, q) => s + v * J[j][q], 0)));
    const g = J.map((col) => -col.reduce((s, v, q) => s + v * r[q], 0));
    let improved = false;
    for (let tries = 0; tries < 10; tries++) {
      const Ad = A.map((row, i) => row.map((v, j) => (i === j ? v * (1 + lambda) + 1e-12 : v)));
      const dx = solveLinear(Ad, g);
      if (!dx) { lambda *= 10; continue; }
      const pt = [...p];
      free.forEach((k, i) => { pt[k] += dx[i]; });
      pt[4] = Math.max(-89 * DEG, Math.min(89 * DEG, pt[4]));
      pt[5] = Math.max(Math.log(Math.tan(FOV_MIN / 2 * DEG)), Math.min(Math.log(Math.tan(FOV_MAX / 2 * DEG)), pt[5]));
      pt[6] = Math.max(-ROLL_MAX, Math.min(ROLL_MAX, pt[6]));
      residuals(pt, inp, rt, ctx);
      const c = sumSq(rt);
      if (c < cost) {
        const rel = (cost - c) / Math.max(cost, 1e-12);
        p = pt; cost = c; for (let i = 0; i < m; i++) r[i] = rt[i];
        lambda = Math.max(lambda / 10, 1e-9);
        improved = true;
        if (rel < 1e-10) return { p, cost };
        break;
      }
      lambda *= 10;
    }
    if (!improved) break;
  }
  return { p, cost };
}

function centroid(pts: V3[]): V3 {
  const c: V3 = [0, 0, 0];
  for (const p of pts) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
  return [c[0] / pts.length, c[1] / pts.length, c[2] / pts.length];
}

/** Camera looking along (yaw, pitch) placed so the points' centroid lands on the targets' centroid at a matching size. */
function initialFor(yaw: number, pitch: number, logT: number, roll: number, inp: SolveInput, C: V3): number[] {
  const { f, r, u } = basis(yaw, pitch);
  const t = Math.exp(logT);
  const n = inp.points.length;
  // 3D spread perpendicular to the view vs 2D spread of the targets -> distance
  let s3 = 0, s2 = 0;
  const tc = inp.targets.reduce((a, q) => [a[0] + q[0] / n, a[1] + q[1] / n], [0, 0]);
  for (let i = 0; i < n; i++) {
    const d = inp.points[i].map((v, k) => v - C[k]) as V3;
    const dx = d[0] * r[0] + d[1] * r[1] + d[2] * r[2], dy = d[0] * u[0] + d[1] * u[1] + d[2] * u[2];
    s3 += dx * dx + dy * dy;
    const ex = inp.targets[i][0] - tc[0], ey = inp.targets[i][1] - tc[1];
    s2 += ex * ex + ey * ey;
  }
  const dist = Math.sqrt(s3) / Math.max(Math.sqrt(s2) * t, 1e-6);
  // un-roll the targets' centroid into the camera frame
  const c = Math.cos(-roll), s = Math.sin(-roll);
  const ux = tc[0] * c - tc[1] * s, uy = tc[0] * s + tc[1] * c;
  const ax = ux * dist * t, ay = uy * dist * t;
  return [
    C[0] - f[0] * dist - r[0] * ax - u[0] * ay,
    C[1] - f[1] * dist - r[1] * ax - u[1] * ay,
    C[2] - f[2] * dist - r[2] * ax - u[2] * ay,
    yaw, pitch, logT, roll,
  ];
}

export function solveCamera(inp: SolveInput): SolveResult {
  const n = inp.points.length;
  if (n < 2) throw new Error('Need at least 2 pins');
  const C = centroid(inp.points);
  const scale = Math.max(1e-3, Math.sqrt(inp.points.reduce((s, p) => s + (p[0] - C[0]) ** 2 + (p[1] - C[1]) ** 2 + (p[2] - C[2]) ** 2, 0) / n));
  const sp = inp.start.position, st = inp.start.target;
  const d0: V3 = [st[0] - sp[0], st[1] - sp[1], st[2] - sp[2]];
  const l0 = Math.hypot(...d0) || 1;
  const yaw0 = Math.atan2(d0[0], d0[2]);
  const pitch0 = Math.asin(Math.max(-1, Math.min(1, d0[1] / l0)));
  const logT0 = Math.log(Math.tan((Math.max(FOV_MIN, Math.min(FOV_MAX, inp.start.fov)) * DEG) / 2));
  const roll0 = Math.max(-ROLL_MAX, Math.min(ROLL_MAX, (inp.startRoll ?? 0) * DEG));
  const ctx: Ctx = { logT0, minDepth: scale * MIN_DEPTH_RADII };

  const starts: number[][] = [[sp[0], sp[1], sp[2], yaw0, pitch0, logT0, roll0]];
  if (inp.quick) starts.push(initialFor(yaw0, pitch0, logT0, roll0, inp, C));
  else {
    for (const dy of [0, 30, -30, 60, -60, 90, -90, 135, -135, 180]) {
      for (const pitch of [pitch0, 0, -20 * DEG, 20 * DEG]) starts.push(initialFor(yaw0 + dy * DEG, pitch, logT0, roll0, inp, C));
    }
  }
  let best: { p: number[]; cost: number } | null = null;
  for (const s of starts) {
    const res = lm(s, inp, scale, ctx, inp.quick ? 40 : 60);
    if (!best || res.cost < best.cost) best = res;
  }
  if (!inp.quick) best = lm(best!.p, inp, scale, ctx, 300);

  const p = best!.p;
  const r = new Array(2 * n).fill(0);
  residuals(p, inp, r);
  const perPinPx = Array.from({ length: n }, (_, i) => Math.hypot(r[2 * i], r[2 * i + 1]));
  const { f } = basis(p[3], p[4]);
  // orbit target: on the view axis at the depth of the points' centroid
  const depth = Math.max(1e-3, (C[0] - p[0]) * f[0] + (C[1] - p[1]) * f[1] + (C[2] - p[2]) * f[2]);
  return {
    fov: (2 * Math.atan(Math.exp(p[5]))) / DEG,
    roll: p[6] / DEG,
    position: [p[0], p[1], p[2]],
    target: [p[0] + f[0] * depth, p[1] + f[1] * depth, p[2] + f[2] * depth],
    rmsPx: Math.sqrt(sumSq(r) / n),
    perPinPx,
  };
}

/** Project with the same camera model (for tests), in half-height units; roll in degrees. */
export function projectUnits(pose: CamPose, X: V3, roll = 0): [number, number] | null {
  const d: V3 = [pose.target[0] - pose.position[0], pose.target[1] - pose.position[1], pose.target[2] - pose.position[2]];
  const l = Math.hypot(...d) || 1;
  const { f, r, u } = basis(Math.atan2(d[0], d[2]), Math.asin(d[1] / l));
  const v = [X[0] - pose.position[0], X[1] - pose.position[1], X[2] - pose.position[2]];
  const z = v[0] * f[0] + v[1] * f[1] + v[2] * f[2];
  if (z <= 0) return null;
  const t = Math.tan((pose.fov * DEG) / 2);
  const x = (v[0] * r[0] + v[1] * r[1] + v[2] * r[2]) / (z * t), y = (v[0] * u[0] + v[1] * u[1] + v[2] * u[2]) / (z * t);
  const c = Math.cos(roll * DEG), s = Math.sin(roll * DEG);
  return [x * c - y * s, x * s + y * c];
}
