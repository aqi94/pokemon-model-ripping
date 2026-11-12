/**
 * GLB export: edits the model's original file to match the view (like modelviewer.lol, which uses gltf-transform too),
 * instead of re-exporting the live three.js scene, whose shader patches and runtime textures have no glTF form.
 * Loaded on demand (`import()`), so gltf-transform stays out of the main bundle. Ported from palworld_models.
 */
import { WebIO, type Animation, type Material, type Scene, type Texture, PropertyType } from '@gltf-transform/core';
import { ALL_EXTENSIONS, type MappingList } from '@gltf-transform/extensions';
import { mergeDocuments, prune, unpartition } from '@gltf-transform/functions';
import type { GlbExportOptions } from './exportOptions';
import { decodePixels, encodePng } from './pngEncode';

/** What the view shows, gathered by ViewerScene.glbExport (indices are the original file's). */
export interface GlbExportInput {
  url: string;
  opts: GlbExportOptions;
  /** Animations (by name) to keep; null = all. */
  keepClips: Set<string> | null;
  prims: PrimView[];
  held: { url: string; bone: number; position: number[]; quaternion: number[]; scale: number[] }[];
}

export interface PrimView {
  mesh: number;
  prim: number;
  hidden: boolean;
  /** Material the view shows (an original or variant material of the file). */
  material?: number;
  /** Recoloured (palette) base colour and shadow colour, as canvases. */
  palette?: { lit: HTMLCanvasElement; shadow?: HTMLCanvasElement };
}

type Phase = (label: string) => void;

const io = () => new WebIO().registerExtensions(ALL_EXTENSIONS);

async function fetchBytes(url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GLB download failed: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

const canvasPng = (c: HTMLCanvasElement) => {
  const d = c.getContext('2d')!.getImageData(0, 0, c.width, c.height);
  return encodePng(d.data, c.width, c.height);
};

/**
 * Material extras hold texture *indices* for the viewer's own shaders (raw / shadow colour, specular / rim masks, eye highlight
 * normal, gas density, palette layers, LGPE field layer recipes). gltf-transform renumbers textures on write, so each reference
 * is resolved to its Texture now and renumbered last. A reference is a number under a key ending in "Texture", a palette layer
 * (`palette.albedo/mask/maskA`), or a `{index, texCoord}` object (also inside arrays: the field recipes' layer lists).
 */
type TexRef = { holder: Record<string, unknown>; key: string };
function extrasRefs(extras: Record<string, unknown>): TexRef[] {
  const out: TexRef[] = [];
  const isRef = (r: Record<string, unknown>) => typeof r.index === 'number' && typeof r.texCoord === 'number';
  const visit = (v: unknown, k: string, holder: Record<string, unknown>, palette: boolean) => {
    if (typeof v === 'number' && Number.isInteger(v) && (/Texture$/.test(k) || (palette && /^(albedo|mask|maskA)$/.test(k)))) out.push({ holder, key: k });
    else if (Array.isArray(v)) v.forEach((x) => { if (x && typeof x === 'object' && !Array.isArray(x)) obj(x as Record<string, unknown>, false); });
    else if (v && typeof v === 'object') obj(v as Record<string, unknown>, k === 'palette');
  };
  const obj = (o: Record<string, unknown>, palette: boolean) => {
    if (isRef(o)) { out.push({ holder: o, key: 'index' }); return; }
    for (const [k, v] of Object.entries(o)) visit(v, k, o, palette);
  };
  obj(extras, false);
  return out;
}

export async function exportGlb(input: GlbExportInput, phase: Phase): Promise<{ blob: Blob; bytes: number }> {
  const { opts } = input;
  phase('Loading…');
  const srcBytes = await fetchBytes(input.url);
  const doc = await io().readBinary(srcBytes);
  const srcJson = glbJson(srcBytes).json as GltfJson;
  const root = doc.getRoot();
  phase('Editing…');
  const meshes = root.listMeshes(), nodes = root.listNodes(), materials = root.listMaterials();
  // texture list each material's extras indices refer to (merged items and recolours get theirs)
  const texLists = new Map<Material, (Texture | undefined)[]>();
  const mainTex: (Texture | undefined)[] = root.listTextures();
  for (const m of materials) texLists.set(m, mainTex);
  // the sampler each original texture had (extras-only textures lose theirs: gltf-transform keeps samplers on texture slots)
  const samplerOf = new Map<Texture, Record<string, unknown>>();
  mainTex.forEach((t, i) => { const si = srcJson.textures?.[i]?.sampler; const sm = si === undefined ? undefined : srcJson.samplers?.[si]; if (t && sm) samplerOf.set(t, sm); });

  for (const a of root.listAnimations()) {
    if (opts.animations === 'none' || (input.keepClips && !input.keepClips.has(a.getName()))) disposeAnimation(a);
  }

  // live colours: the material each primitive shows
  if (opts.liveColours) {
    const recoloured = new Map<HTMLCanvasElement, Material>();
    for (const p of input.prims) {
      const prim = meshes[p.mesh]?.listPrimitives()[p.prim];
      if (!prim || p.material === undefined) continue;
      let mat = materials[p.material];
      if (!mat) continue;
      if (p.palette) {
        let r = recoloured.get(p.palette.lit);
        if (!r) {
          r = mat.clone().setName(`${mat.getName()}_recolour`);
          const lit = doc.createTexture(`${mat.getName()}_recolour`).setMimeType('image/png').setImage(await canvasPng(p.palette.lit));
          r.setBaseColorTexture(lit);
          const list = [...(texLists.get(mat) ?? mainTex)];
          const extras = structuredClone(r.getExtras()) as Record<string, unknown>;
          if (p.palette.shadow && typeof extras.shadowColorTexture === 'number') {
            list.push(doc.createTexture(`${mat.getName()}_recolour_shadow`).setMimeType('image/png').setImage(await canvasPng(p.palette.shadow)));
            extras.shadowColorTexture = list.length - 1;
          }
          r.setExtras(extras);
          texLists.set(r, list);
          recoloured.set(p.palette.lit, r);
        }
        mat = r;
      }
      prim.setMaterial(mat);
    }
  }

  if (opts.removeHidden) {
    // resolve first: disposing shifts the later primitives of the same mesh down by one
    const gone = input.prims.filter((p) => p.hidden).map((p) => meshes[p.mesh]?.listPrimitives()[p.prim]);
    for (const prim of gone) prim?.dispose();
    for (const m of meshes) if (!m.isDisposed() && m.listPrimitives().length === 0) m.dispose();
    // a skin on a node without a mesh is invalid glTF (the mesh of a removed part)
    for (const n of root.listNodes()) if (!n.getMesh() && n.getSkin()) n.setSkin(null);
    // the picked options are baked in: the other variants are hidden options too
    // (the mappings are cleared first: while they exist they keep every option's material alive through prune)
    if (opts.liveColours) {
      for (const m of root.listMeshes()) for (const p of m.listPrimitives()) {
        const list = p.getExtension<MappingList>('KHR_materials_variants');
        list?.listMappings().forEach((x) => x.dispose());
        list?.dispose();
      }
      root.listExtensionsUsed().find((e) => e.extensionName === 'KHR_materials_variants')?.dispose();
    }
  }

  // held items, each under its bone with the attach transform the view uses
  let merged = false;
  if (opts.heldItems) {
    for (const h of input.held) {
      const bone = nodes[h.bone];
      if (!bone) continue;
      const src = await io().readBinary(await fetchBytes(h.url));
      for (const a of src.getRoot().listAnimations()) disposeAnimation(a);
      const srcTex: (Texture | undefined)[] = src.getRoot().listTextures();
      const srcMats = src.getRoot().listMaterials();
      const srcScene = src.getRoot().getDefaultScene() ?? src.getRoot().listScenes()[0];
      const map = mergeDocuments(doc, src);
      const tex = srcTex.map((t) => (t ? (map.get(t) as Texture | undefined) : undefined));
      for (const m of srcMats) { const t = map.get(m) as Material | undefined; if (t) texLists.set(t, tex); }
      const scene = map.get(srcScene) as Scene | undefined;
      if (!scene) continue;
      const wrap = doc.createNode(`held_${scene.getName() || 'item'}`)
        .setTranslation(h.position as [number, number, number])
        .setRotation(h.quaternion as [number, number, number, number])
        .setScale(h.scale as [number, number, number]);
      for (const child of scene.listChildren()) { scene.removeChild(child); wrap.addChild(child); }
      bone.addChild(wrap);
      scene.dispose();
      merged = true;
    }
  }
  if (merged) await doc.transform(unpartition());

  // prune everything but textures (extras references are invisible to prune), then the textures nothing uses
  await doc.transform(prune({
    propertyTypes: [PropertyType.NODE, PropertyType.SKIN, PropertyType.MESH, PropertyType.CAMERA, PropertyType.PRIMITIVE, PropertyType.PRIMITIVE_TARGET,
      PropertyType.ANIMATION, PropertyType.MATERIAL, PropertyType.ACCESSOR],
    keepLeaves: true, keepAttributes: true, keepSolidTextures: true, keepExtras: false,
  }));
  const extrasUse = new Map<Material, { refs: TexRef[]; tex: (Texture | undefined)[] }>();
  const used = new Set<Texture>();
  for (const m of root.listMaterials()) {
    const extras = structuredClone(m.getExtras()) as Record<string, unknown>;
    m.setExtras(extras);
    const refs = extrasRefs(extras), list = texLists.get(m) ?? mainTex;
    extrasUse.set(m, { refs, tex: list });
    for (const r of refs) { const t = list[r.holder[r.key] as number]; if (t && !t.isDisposed()) used.add(t); }
  }
  for (const t of root.listTextures()) {
    if (used.has(t)) continue;
    if (t.listParents().some((p) => p !== root)) continue; // a material slot or an extension (clearcoat) uses it
    t.dispose();
  }

  if (opts.maxTexture) {
    phase('Resizing textures…');
    for (const t of root.listTextures()) await capTexture(t, opts.maxTexture);
  }

  // renumber the extras references to the final texture order (= image order on write; fixTextureDefs maps them to textures[])
  const order = root.listTextures();
  for (const [m, { refs, tex }] of extrasUse) {
    if (m.isDisposed()) continue;
    for (const r of refs) {
      const t = tex[r.holder[r.key] as number];
      const i = t && !t.isDisposed() ? order.indexOf(t) : -1;
      if (i >= 0) r.holder[r.key] = i;
      else delete r.holder[r.key];
    }
  }

  phase('Writing…');
  const bytes = fixTextureDefs(await io().writeBinary(doc), order.map((t) => samplerOf.get(t)));
  return { blob: new Blob([bytes as BlobPart], { type: 'model/gltf-binary' }), bytes: bytes.byteLength };
}

/** An animation with its channels and samplers: a disposed animation leaves its samplers alive, and they keep every
 *  keyframe accessor out of prune (a one-clip Pikachu export kept the other 125 clips' 50k accessors in its JSON). */
function disposeAnimation(a: Animation) {
  for (const c of a.listChannels()) c.dispose();
  for (const sm of a.listSamplers()) sm.dispose();
  a.dispose();
}

/** Shrinks a texture so its longest side is at most `max` (aspect kept), re-encoded as PNG with its channels untouched. */
async function capTexture(t: Texture, max: number) {
  const size = t.getSize(), img = t.getImage();
  if (!size || !img || Math.max(size[0], size[1]) <= max) return;
  const k = max / Math.max(size[0], size[1]);
  const w = Math.max(1, Math.round(size[0] * k)), h = Math.max(1, Math.round(size[1] * k));
  const px = await decodePixels(img, t.getMimeType(), w, h);
  t.setImage(await encodePng(px.data, px.width, px.height)).setMimeType('image/png');
}

interface GltfJson {
  textures?: { source?: number; sampler?: number }[];
  samplers?: Record<string, unknown>[];
  images?: unknown[];
  materials?: { extras?: Record<string, unknown> }[];
}

/** JSON chunk of a GLB, plus where the chunks after it start. */
function glbJson(b: Uint8Array): { json: unknown; end: number } {
  const v = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const len = v.getUint32(12, true);
  return { json: JSON.parse(new TextDecoder().decode(b.subarray(20, 20 + len))), end: 20 + len };
}

/**
 * gltf-transform writes a `textures[]` entry only for textures a material slot uses, so the viewer's extras-only textures
 * (raw albedo, shadow colour, masks, palette layers) keep their images but get no texture: their (image-order) indices would
 * point past `textures[]`. Gives each one a texture (reusing one with the same image and sampler) and points the ref at it.
 * `samplers[i]` = the original sampler of the texture written as image i.
 */
function fixTextureDefs(bytes: Uint8Array, samplers: (Record<string, unknown> | undefined)[]): Uint8Array {
  const { json: j, end } = glbJson(bytes) as { json: GltfJson; end: number };
  const textures = (j.textures ??= []);
  const sameSampler = (a: Record<string, unknown> | undefined, b: Record<string, unknown> | undefined) => JSON.stringify(a ?? {}) === JSON.stringify(b ?? {});
  const samplerIndex = (sm: Record<string, unknown>) => {
    const list = (j.samplers ??= []);
    let i = list.findIndex((x) => sameSampler(x, sm));
    if (i < 0) { list.push(sm); i = list.length - 1; }
    return i;
  };
  const textureFor = (image: number) => {
    const sm = samplers[image];
    let i = textures.findIndex((t) => t.source === image && sameSampler(t.sampler === undefined ? undefined : j.samplers?.[t.sampler], sm));
    if (i < 0) { textures.push({ source: image, ...(sm ? { sampler: samplerIndex(sm) } : {}) }); i = textures.length - 1; }
    return i;
  };
  let changed = false;
  for (const m of j.materials ?? []) {
    if (!m.extras) continue;
    for (const r of extrasRefs(m.extras)) {
      const image = r.holder[r.key] as number;
      if (image >= (j.images?.length ?? 0)) { delete r.holder[r.key]; changed = true; continue; }
      const t = textureFor(image);
      if (t !== image || textures[image]?.source !== image) changed = true;
      r.holder[r.key] = t;
    }
  }
  if (!changed) return bytes;
  // repack: new JSON chunk (space-padded to 4 bytes), the other chunks as they were
  const text = new TextEncoder().encode(JSON.stringify(j));
  const pad = (4 - (text.length % 4)) % 4;
  const rest = bytes.subarray(end);
  const out = new Uint8Array(20 + text.length + pad + rest.length);
  const v = new DataView(out.buffer);
  out.set(bytes.subarray(0, 12));
  v.setUint32(8, out.length, true);
  v.setUint32(12, text.length + pad, true);
  v.setUint32(16, 0x4e4f534a, true); // 'JSON'
  out.set(text, 20);
  out.fill(0x20, 20 + text.length, 20 + text.length + pad);
  out.set(rest, 20 + text.length + pad);
  return out;
}
