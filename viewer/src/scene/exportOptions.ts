/** PNG export settings shared by the Export panel and ViewerScene.png / pngDataUrl (tools pass the same fields). */
export interface ExportOptions {
  width: number;
  height: number;
  /** Pixel multiplier: output is (width*scale) x (height*scale). */
  scale: number;
  /** current = the live camera; fit = fit the whole model; idle = paused default clip, near-orthographic full body. */
  framing: 'current' | 'fit' | 'idle';
  /** Per-side padding as a fraction of the subject height (idle framing) / of the canvas (trim). */
  margin: number;
  /** Vertical FOV in degrees for fit/idle framing. */
  fov: number;
  /** 'transparent' or a #rrggbb colour. */
  background: string;
  /** Crop to the alpha bounding box, then scale to fit width x height with `margin` padding, feet-aligned. */
  trim: boolean;
}

export const DEFAULT_EXPORT: ExportOptions = {
  width: 1024, height: 1024, scale: 1, framing: 'current', margin: 0.03, fov: 3, background: 'transparent', trim: false,
};

export const EXPORT_PRESETS: { label: string; opts: Partial<ExportOptions> }[] = [
  { label: 'Square 1024', opts: { width: 1024, height: 1024, framing: 'fit', fov: 30, margin: 0.04, trim: true } },
  { label: 'Idle 640x1024', opts: { width: 640, height: 1024, framing: 'idle', fov: 3, margin: 0.03, trim: false } },
  { label: 'Thumb 512', opts: { width: 512, height: 512, framing: 'fit', fov: 30, margin: 0.04, trim: true } },
  { label: 'Screen', opts: { framing: 'current', trim: false, scale: 1 } },
];

const KEY = 'exportOptions';

/** null = the user never configured an export, so the PNG button keeps exporting the live canvas as before. */
export function loadExportOptions(): ExportOptions | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...DEFAULT_EXPORT, ...JSON.parse(raw) };
  } catch { /* storage unavailable */ }
  return null;
}

export function saveExportOptions(o: ExportOptions) {
  try { localStorage.setItem(KEY, JSON.stringify(o)); } catch { /* storage unavailable */ }
}

export const MAX_EXPORT_PX = 8192;
