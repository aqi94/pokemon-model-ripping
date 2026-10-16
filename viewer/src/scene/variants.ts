import * as THREE from 'three';
import type { GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';

interface Mapping { material: number; variants: number[] }
interface Entry { mesh: THREE.Mesh; defaultMat: THREE.Material | THREE.Material[]; mappings: Mapping[] }

/** Minimal KHR_materials_variants support (three's GLTFLoader does not implement it). */
export class VariantSwitcher {
  private entries: Entry[] = [];
  private materialCache = new Map<number, THREE.Material>();
  readonly names: string[];
  current = -1;
  onMaterialsChanged?: () => void;

  constructor(private gltf: GLTF) {
    const ext = (gltf.parser.json.extensions ?? {}).KHR_materials_variants;
    this.names = (ext?.variants ?? []).map((v: { name: string }) => v.name);
    const parser = gltf.parser;
    gltf.scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      const assoc = parser.associations.get(mesh) as { meshes?: number; primitives?: number } | undefined;
      if (assoc?.meshes === undefined || assoc.primitives === undefined) return;
      const prim = parser.json.meshes[assoc.meshes]?.primitives?.[assoc.primitives];
      const mappings: Mapping[] | undefined = prim?.extensions?.KHR_materials_variants?.mappings;
      if (mappings?.length) this.entries.push({ mesh, defaultMat: mesh.material, mappings });
    });
  }

  get available() { return this.names.length > 0 && this.entries.length > 0; }

  private async material(idx: number): Promise<THREE.Material> {
    let m = this.materialCache.get(idx);
    if (!m) {
      m = (await this.gltf.parser.getDependency('material', idx)) as THREE.Material;
      this.materialCache.set(idx, m);
    }
    return m;
  }

  async select(variantIndex: number) {
    for (const e of this.entries) {
      const map = e.mappings.find((m) => m.variants.includes(variantIndex));
      e.mesh.material = map ? await this.material(map.material) : e.defaultMat;
    }
    this.current = variantIndex;
    this.onMaterialsChanged?.();
  }

  /** Average opaque colour of the largest base-colour texture of a variant (for the swatch). */
  async sampleColor(variantIndex: number): Promise<string | null> {
    try {
      let best: { img: CanvasImageSource | null; area: number; tint: THREE.Color } | null = null;
      for (const e of this.entries) {
        const map = e.mappings.find((m) => m.variants.includes(variantIndex));
        if (!map) continue;
        const mat = (await this.material(map.material)) as THREE.MeshStandardMaterial;
        const img = mat.map?.image as (CanvasImageSource & { width: number; height: number }) | undefined;
        const area = img ? img.width * img.height : 0;
        if (!best || area > best.area) best = { img: img ?? null, area, tint: mat.color };
      }
      if (!best) return null;
      // Variants that only change the material colour factor (skin tones) share one texture: apply the factor.
      if (!best.img) return best.tint.getStyle();
      const c = document.createElement('canvas');
      c.width = c.height = 32;
      const ctx = c.getContext('2d', { willReadFrequently: true })!;
      ctx.drawImage(best.img, 0, 0, 32, 32);
      const d = ctx.getImageData(0, 0, 32, 32).data;
      let r = 0, g = 0, b = 0, n = 0;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i + 3] < 200) continue;
        r += d[i]; g += d[i + 1]; b += d[i + 2]; n++;
      }
      if (!n) return null;
      const avg = new THREE.Color().setRGB(r / n / 255, g / n / 255, b / n / 255, THREE.SRGBColorSpace);
      return avg.multiply(best.tint).getStyle();
    } catch {
      return null;
    }
  }
}
