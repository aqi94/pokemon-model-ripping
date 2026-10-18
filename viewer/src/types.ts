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
export interface GameIndex {
  indexVersion: number;
  game: string;
  name: string;
  models: IndexModel[];
}
export interface GameSummary { name: string; count: number; kinds: Record<string, number> }
export interface VariantOption { variant: number; name: string; label: string; default: boolean }
export interface VariantGroup { name: string; label: string; options: VariantOption[] }
export interface ClipInfo {
  name: string; index: number; duration: number; frameRate: number; frameCount: number;
  loop: boolean; category: string; default: boolean; visibility?: VisibilityTracks; uvOffset?: UvOffsetTracks; uvOffsetMode?: 'gltf_step';
}
/** Per-material texture offset animation (BDSP eye gaze): material name -> keys in seconds, offsets in UV units (x right, y up). */
export type UvOffsetTracks = Record<string, { times: number[]; values: [number, number][] }>;
export type VisibilityTracks = Record<string, [number, number][]>;
export interface PartInfo { node: string; label: string; lod: number; visible: boolean; group?: string; option?: string; accessory?: boolean }
export interface PaletteEntry { label: string; materials: string[]; divide: number; indices: number[]; layers: number[]; zones: string[];
  /** Colours the model did not ship with (eyes of the Common NPC faces): the UI labels the row as custom. */
  custom?: boolean;
  /** One swatch per (layer 2, layer 3) zone pair, zones 2j+2 / 2j+3 for pair j (labels in `pairLabels`). */
  pairs?: boolean; pairLabels?: string[];
  /** Single-layer custom rows: label per zone (zone 1 = the model's own base layer, not offered). */
  zoneLabels?: string[];
  /** The palette's layer 1 is the skin: the viewer drives its zone from the selected skin tone variant. */
  skinLink?: boolean;
  /** Offered zones in order (0 = same zone as the skin, labelled `noneLabel`); default = every zone but the first. */
  offer?: number[]; noneLabel?: string }
export interface PartOption { id: string; label: string; default: boolean; nodes: string[]; palettes?: PaletteEntry[];
  /** Hidden-by-default beard meshes of this face: the Appearance section offers a Facial hair toggle. */
  facialHair?: string[] }
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
  /** Z-A Holocast material sets (`holo`, `holo_red`): hologram presets from the data */
  holocast?: HolocastPreset[];
  stats: { joints: number; meshes: number; triangles: number; textures: number; clips: number; glbBytes: number };
  extras?: { dex?: number; form?: string; species?: string };
}

/** A calibrated gameplay screenshot (refs/<game>/index.json, written by tools/lightrefs/build_presets.py). */
export interface RefEntry {
  scene: string; sceneLabel: string; preset: string; slug: string; model: string; name: string; context: string;
  clip: string; frame: number; camera: { fov: number; position: [number, number, number]; target: [number, number, number] };
  image: string; width: number; height: number; dE: number | null; video?: string | null; timecode?: string | null;
}
export interface RefIndex { game: string; builtAt: string; refs: RefEntry[] }

export interface HolocastPreset {
  name: string;
  label: string;
  color: [number, number, number];
  scanlineSpeed: number;
  scanlineDepth: number;
  scanlineTiling: number;
  scanGradationTiling: number;
}
