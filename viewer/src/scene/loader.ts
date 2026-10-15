import { GLTFLoader, type GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

export function loadGlb(url: string, onProgress?: (frac: number) => void): Promise<GLTF> {
  const loader = new GLTFLoader();
  loader.setMeshoptDecoder(MeshoptDecoder);
  return new Promise((resolve, reject) =>
    loader.load(url, resolve, (e) => { if (e.total) onProgress?.(e.loaded / e.total); }, reject),
  );
}
