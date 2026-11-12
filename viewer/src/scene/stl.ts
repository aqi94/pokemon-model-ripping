import * as THREE from 'three';

/** STL export settings (Export panel, `window.__stl`). */
export interface StlOptions {
  /** Height of the print in mm (the model's vertical extent); 0 = real size (1 m = 1000 mm). */
  heightMm: number;
  /** Keep effect shells (gas, smoke, additive glows): see-through in the game, rarely printable. */
  includeEffects: boolean;
}
export const DEFAULT_STL: StlOptions = { heightMm: 100, includeEffects: false };

/** Materials a print leaves out: hidden, depth/colour-less helpers, and (unless asked) animated effect shells. */
function skipMaterial(m: THREE.Material | undefined, o: StlOptions) {
  if (!m || !m.visible || m.colorWrite === false) return true;
  if (!o.includeEffects && (m.userData?.__gas || m.userData?.gas || m.userData?.behindBody || m.userData?.effect || m.userData?.glass || m.userData?.blend === 'Add')) return true;
  return false;
}

/**
 * Binary STL of what the view shows under `root`, in the current pose (skinning and morph targets applied), for 3D
 * printing: Z up, millimetres, centred on X/Y, lowest point on z = 0. Viewer-only meshes (outline / aura hulls,
 * hologram prepasses) are skipped. Triangle soup, as STL is; game meshes are not watertight (slicers repair them).
 */
export function buildStl(root: THREE.Object3D, o: StlOptions, header: string): { blob: Blob; triangles: number } {
  root.updateMatrixWorld(true);
  const parts: { pos: Float32Array; idx: number[] }[] = [];
  const v = new THREE.Vector3();
  const min = new THREE.Vector3(Infinity, Infinity, Infinity), max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
  root.traverseVisible((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || mesh.userData.outline || mesh.userData.holoPrepass) return;
    const g = mesh.geometry, P = g.getAttribute('position');
    if (!P) return;
    const mats = ([] as THREE.Material[]).concat(mesh.material);
    const index = g.index;
    const count = index ? index.count : P.count;
    const d0 = Math.max(0, g.drawRange.start), d1 = Math.min(count, d0 + (Number.isFinite(g.drawRange.count) ? g.drawRange.count : count));
    const groups = g.groups.length ? g.groups : [{ start: 0, count, materialIndex: 0 }];
    const flip = mesh.matrixWorld.determinant() < 0; // mirrored: keep the normals pointing out
    const idx: number[] = [];
    for (const gr of groups) {
      if (skipMaterial(Array.isArray(mesh.material) ? mats[gr.materialIndex ?? 0] : mats[0], o)) continue;
      const s = Math.max(d0, gr.start), e = Math.min(d1, gr.start + gr.count);
      for (let i = s; i + 2 < e; i += 3) {
        const a = index ? index.getX(i) : i, b = index ? index.getX(i + 1) : i + 1, c = index ? index.getX(i + 2) : i + 2;
        if (flip) idx.push(a, c, b); else idx.push(a, b, c);
      }
    }
    if (!idx.length) return;
    // Posed positions once per vertex: getVertexPosition applies morph targets and skinning.
    const pos = new Float32Array(P.count * 3);
    for (let i = 0; i < P.count; i++) {
      mesh.getVertexPosition(i, v).applyMatrix4(mesh.matrixWorld);
      // glTF (Y up, metres) -> print (Z up, mm): a +90 deg turn about X, so the winding is unchanged.
      const x = v.x * 1000, y = -v.z * 1000, z = v.y * 1000;
      pos[i * 3] = x; pos[i * 3 + 1] = y; pos[i * 3 + 2] = z;
    }
    for (const i of idx) { min.min(v.fromArray(pos, i * 3)); max.max(v); }
    parts.push({ pos, idx });
  });
  const triangles = parts.reduce((n, p) => n + p.idx.length / 3, 0);
  if (!triangles) throw new Error('nothing visible to export');
  const k = o.heightMm > 0 && max.z > min.z ? o.heightMm / (max.z - min.z) : 1;
  const cx = (min.x + max.x) / 2, cy = (min.y + max.y) / 2, cz = min.z;
  const buf = new ArrayBuffer(84 + triangles * 50), dv = new DataView(buf);
  const head = new TextEncoder().encode(header.slice(0, 80));
  new Uint8Array(buf, 0, 80).set(head.subarray(0, 80));
  dv.setUint32(80, triangles, true);
  let off = 84;
  const A = new THREE.Vector3(), B = new THREE.Vector3(), C = new THREE.Vector3(), n = new THREE.Vector3(), e = new THREE.Vector3();
  const at = (p: Float32Array, i: number, out: THREE.Vector3) => out.set((p[i * 3] - cx) * k, (p[i * 3 + 1] - cy) * k, (p[i * 3 + 2] - cz) * k);
  for (const { pos, idx } of parts) for (let t = 0; t < idx.length; t += 3) {
    at(pos, idx[t], A); at(pos, idx[t + 1], B); at(pos, idx[t + 2], C);
    n.subVectors(B, A).cross(e.subVectors(C, A));
    if (n.lengthSq() > 0) n.normalize();
    for (const w of [n, A, B, C]) { dv.setFloat32(off, w.x, true); dv.setFloat32(off + 4, w.y, true); dv.setFloat32(off + 8, w.z, true); off += 12; }
    dv.setUint16(off, 0, true); off += 2;
  }
  return { blob: new Blob([buf], { type: 'model/stl' }), triangles };
}
