export interface IndexModel {
  game: string;
  id: string;
  kind: 'pokemon' | 'trainer' | 'object' | string;
  displayName: string;
  clips: number;
  variantGroups: string[];
  path: string;
  dex?: number | null;
  form?: string | null;
  thumbnail?: string | null; // relative to out/ (served at /models/)
  item?: string | null;
}
export interface ModelIndex {
  indexVersion: number;
  games: Record<string, string>;
  models: IndexModel[];
}
export interface VariantOption { variant: number; name: string; label: string; default: boolean }
export interface VariantGroup { name: string; label: string; options: VariantOption[] }
export interface ClipInfo {
  name: string; index: number; duration: number; frameRate: number; frameCount: number;
  loop: boolean; category: string; default: boolean; visibility?: VisibilityTracks; uvOffset?: UvOffsetTracks;
}
/** Per-material texture offset animation (BDSP eye gaze): material name -> keys in seconds, offsets in UV units (x right, y up). */
export type UvOffsetTracks = Record<string, { times: number[]; values: [number, number][] }>;
export type VisibilityTracks = Record<string, [number, number][]>;
export interface PartInfo { node: string; label: string; lod: number; visible: boolean; group?: string; option?: string; accessory?: boolean }
export interface PaletteEntry { label: string; materials: string[]; divide: number; indices: number[]; layers: number[]; zones: string[] }
export interface PartOption { id: string; label: string; default: boolean; nodes: string[]; palettes?: PaletteEntry[] }
export interface PartGroup { name: string; label: string; options: PartOption[] }
export interface PropInfo { bone: string; side: string; clips: string[] }
export interface Manifest {
  manifestVersion: number;
  game: string;
  id: string;
  kind: string;
  displayName: string;
  files: { model: string };
  variantGroups: VariantGroup[];
  clips: ClipInfo[];
  defaultClip?: string;
  parts: PartInfo[];
  partGroups?: PartGroup[];
  props?: PropInfo[];
  stats: { joints: number; meshes: number; triangles: number; textures: number; clips: number; glbBytes: number };
  extras?: { dex?: number; form?: string; species?: string };
}
