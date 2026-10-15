import * as THREE from 'three';

/** BDSP (Unity) materials the game flags `_ENVMAP_ON`: glasses/lenses, inner tubes, metal fittings. They reflect an environment map. */
export function isReflective(mat: THREE.Material): boolean {
  const kw = (mat.userData as { unity?: { keywords?: string } } | undefined)?.unity?.keywords;
  return typeof kw === 'string' && kw.includes('_ENVMAP_ON');
}

/** A small studio environment (sky gradient + soft boxes) prefiltered for PBR reflections; one per renderer. */
export function makeReflectionEnv(renderer: THREE.WebGLRenderer): THREE.Texture {
  const es = new THREE.Scene();
  const g = new THREE.SphereGeometry(20, 48, 24);
  const pos = g.attributes.position;
  const col: number[] = [];
  for (let i = 0; i < pos.count; i++) {
    const y = pos.getY(i) / 20;
    const c = y > 0 ? 0.55 + 1.3 * y : 0.5 + 0.35 * y;
    col.push(c * 0.95, c * 0.97, c * 1.05);
  }
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  es.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.BackSide })));
  for (const [x, y, z, w, h, k] of [[-7, 5, 8, 6, 5, 7], [8, 3, 6, 4, 6, 5], [0, 14, 2, 9, 3, 4]]) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(k, k, k), side: THREE.DoubleSide }));
    m.position.set(x, y, z);
    m.lookAt(0, 0, 0);
    es.add(m);
  }
  const pm = new THREE.PMREMGenerator(renderer);
  const tex = pm.fromScene(es, 0.015).texture;
  pm.dispose();
  es.traverse((o) => { const m = o as THREE.Mesh; if (m.isMesh) { m.geometry.dispose(); (m.material as THREE.Material).dispose(); } });
  return tex;
}
