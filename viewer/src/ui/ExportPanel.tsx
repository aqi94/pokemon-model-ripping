import { EXPORT_PRESETS, MAX_EXPORT_PX, type ExportFormat, type ExportOptions, type GlbExportOptions } from '../scene/exportOptions';
import type { GifExportOptions } from '../scene/ViewerScene';
import type { StlOptions } from '../scene/stl';

interface Props {
  /** Which format's options the popover shows (one toolbox button each). */
  format: ExportFormat;
  value: ExportOptions;
  onChange: (patch: Partial<ExportOptions>) => void;
  onExport: () => void;
  onClose: () => void;
  top: number;
  gif: GifExportOptions;
  onGifChange: (patch: Partial<GifExportOptions>) => void;
  onGif: () => void;
  /** null = idle, else a progress label. */
  gifProgress: string | null;
  stl: StlOptions;
  onStlChange: (patch: Partial<StlOptions>) => void;
  onStl: () => void;
  /** Result of the last STL export (triangle count or error). */
  stlInfo: string | null;
  glb: GlbExportOptions;
  onGlbChange: (patch: Partial<GlbExportOptions>) => void;
  onGlb: () => void;
  /** Phase while running, else the last result (file size or error). */
  glbInfo: string | null;
  glbBusy: boolean;
  /** The model has clips / an attached held item: options that would do nothing are disabled. */
  glbHas: { clips: boolean; held: boolean };
}

const num = (v: string, min: number, max: number, fallback: number) => {
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
};

const TITLES: Record<ExportFormat, string> = { png: 'Export PNG', gif: 'Export animated GIF', stl: 'Export STL (3D print)', glb: 'Export GLB' };

/** Popover next to the toolbox export buttons: the options of one format. */
export function ExportPanel(p: Props) {
  const { format, onClose, top } = p;
  return (
    <div className="export-panel" style={{ top, maxHeight: `calc(100vh - ${top + 8}px)` }} role="dialog" aria-label={TITLES[format]}>
      <div className="export-head"><b>{TITLES[format]}</b><button className="tool-btn" title="Close" onClick={onClose}>×</button></div>
      <div className="export-col">
        {format === 'png' && <PngOptions {...p} />}
        {format === 'gif' && <GifOptions {...p} />}
        {format === 'stl' && <StlOptionsView {...p} />}
        {format === 'glb' && <GlbOptions {...p} />}
      </div>
    </div>
  );
}

function PngOptions({ value: o, onChange, onExport }: Props) {
  const solid = o.background !== 'transparent';
  const framed = o.framing !== 'current';
  return <>
    <label>Preset
      <select value="" onChange={(e) => { const p = EXPORT_PRESETS.find((x) => x.label === e.target.value); if (p) onChange(p.opts); }}>
        <option value="">Choose…</option>
        {EXPORT_PRESETS.map((p) => <option key={p.label} value={p.label}>{p.label}</option>)}
      </select>
    </label>
    <label>Size (px)
      <span className="export-row">
        <input type="number" min={1} max={MAX_EXPORT_PX} value={o.width} onChange={(e) => onChange({ width: num(e.target.value, 1, MAX_EXPORT_PX, o.width) })} />
        ×
        <input type="number" min={1} max={MAX_EXPORT_PX} value={o.height} onChange={(e) => onChange({ height: num(e.target.value, 1, MAX_EXPORT_PX, o.height) })} />
      </span>
    </label>
    <label>Scale
      <select value={o.scale} onChange={(e) => onChange({ scale: Number(e.target.value) })}>
        {[1, 2, 3, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
      </select>
    </label>
    <label>Framing
      <select value={o.framing} onChange={(e) => onChange({ framing: e.target.value as ExportOptions['framing'] })}>
        <option value="current">Current view</option>
        <option value="fit">Fit model</option>
        <option value="idle">Idle ortho (frame 0)</option>
      </select>
    </label>
    <label>FOV {o.fov}°
      <input type="range" min={1} max={60} step={1} value={o.fov} disabled={!framed} onChange={(e) => onChange({ fov: Number(e.target.value) })} />
    </label>
    <label title="Idle: fraction of model height above/below. Trim: fraction of the smaller canvas side.">Padding {(o.margin * 100).toFixed(1)}%
      <input type="range" min={0} max={0.3} step={0.005} value={o.margin} disabled={o.framing !== 'idle' && !o.trim} onChange={(e) => onChange({ margin: Number(e.target.value) })} />
    </label>
    <label className="export-check"><input type="checkbox" checked={o.trim} onChange={(e) => onChange({ trim: e.target.checked })} />
      Trim to subject (scale to fit, feet-aligned)</label>
    <label className="export-check"><input type="checkbox" checked={solid} onChange={(e) => onChange({ background: e.target.checked ? '#808080' : 'transparent' })} />
      Solid background
      {solid && <input type="color" value={o.background} onChange={(e) => onChange({ background: e.target.value })} />}
    </label>
    <div className="export-foot">
      <span>{Math.min(MAX_EXPORT_PX, o.width * o.scale)} × {Math.min(MAX_EXPORT_PX, o.height * o.scale)} px</span>
      <button className="export-go" onClick={onExport}>Export</button>
    </div>
  </>;
}

function GifOptions({ gif: g, onGifChange, onGif, gifProgress }: Props) {
  return <>
    <span className="export-note">Current clip from the current camera, drawn as the view shows it (outline style included), transparent background.</span>
    <label title="Subject height on the first frame (Showdown sprites: Pikachu ~55, Charizard ~100)">Subject height (px)
      <input type="number" min={16} max={1024} value={g.height} onChange={(e) => onGifChange({ height: num(e.target.value, 16, 1024, g.height) })} />
    </label>
    <label>Frame rate
      <span className="export-row">
        <input type="number" min={5} max={60} value={g.fps} onChange={(e) => onGifChange({ fps: num(e.target.value, 5, 60, g.fps) })} /> fps,
        <input type="number" min={10} max={200} step={10} value={g.delayMs} onChange={(e) => onGifChange({ delayMs: num(e.target.value, 10, 200, g.delayMs) })} /> ms/frame
      </span>
    </label>
    <label title="Palette size including transparency (Showdown: 16)">Colours {g.colors}
      <input type="range" min={4} max={256} step={1} value={g.colors} onChange={(e) => onGifChange({ colors: Number(e.target.value) })} />
    </label>
    <div className="export-foot">
      <span>{gifProgress !== null ? `Rendering ${gifProgress}` : `${(1000 / g.delayMs).toFixed(0)} fps playback`}</span>
      <button className="export-go" disabled={gifProgress !== null} onClick={onGif}>Export GIF</button>
    </div>
  </>;
}

function StlOptionsView({ stl: s, onStlChange, onStl, stlInfo }: Props) {
  return <>
    <span className="export-note">Current pose and frame, untextured, Z up, feet on the bed. Game meshes are not watertight; slicers repair them on import.</span>
    <label>Height (mm)
      <input type="number" min={1} max={2000} value={s.heightMm || ''} disabled={!s.heightMm} placeholder="real size"
        onChange={(e) => onStlChange({ heightMm: num(e.target.value, 1, 2000, s.heightMm) })} />
    </label>
    <label className="export-check"><input type="checkbox" checked={!s.heightMm} onChange={(e) => onStlChange({ heightMm: e.target.checked ? 0 : 100 })} />
      Real size (1 m = 1000 mm)</label>
    <label className="export-check"><input type="checkbox" checked={s.includeEffects} onChange={(e) => onStlChange({ includeEffects: e.target.checked })} />
      Include effect shells (gas, smoke, glows)</label>
    <div className="export-foot">
      <span>{stlInfo ?? ''}</span>
      <button className="export-go" onClick={onStl}>Export STL</button>
    </div>
  </>;
}

function GlbOptions({ glb: o, onGlbChange, onGlb, glbInfo, glbBusy, glbHas }: Props) {
  const anim = (v: GlbExportOptions['animations'], label: string, title: string) => (
    <label className="export-check" title={title}>
      <input type="radio" name="glb-anim" checked={o.animations === v} disabled={!glbHas.clips} onChange={() => onGlbChange({ animations: v })} />{label}
    </label>
  );
  return <>
    <span className="export-note">The model file, edited to match the view. Skinned, in the bind pose; materials are glTF PBR (viewer-only effects are kept as extras).</span>
    <fieldset className="export-group"><legend>Animations</legend>
      {anim('all', 'All clips', "Every clip of the model")}
      {anim('current', 'Current clip only', 'Smallest file that still animates')}
      {anim('none', 'None', 'Mesh, skeleton and materials only')}
    </fieldset>
    <label className="export-check"><input type="checkbox" checked={o.removeHidden} onChange={(e) => onGlbChange({ removeHidden: e.target.checked })} />
      Remove hidden parts</label>
    <label className="export-check" title="The held item shown in the view (Poké Ball), attached to its bone">
      <input type="checkbox" checked={o.heldItems} disabled={!glbHas.held} onChange={(e) => onGlbChange({ heldItems: e.target.checked })} />
      Include held item</label>
    <label className="export-check" title="Variants (shiny, colourways, skin tones) and palette colours picked in the view">
      <input type="checkbox" checked={o.liveColours} onChange={(e) => onGlbChange({ liveColours: e.target.checked })} />
      Keep current colours</label>
    <label>Texture size
      <select value={o.maxTexture} onChange={(e) => onGlbChange({ maxTexture: Number(e.target.value) as GlbExportOptions['maxTexture'] })}>
        <option value={0}>Full</option>
        <option value={2048}>2048 px max</option>
        <option value={1024}>1024 px max</option>
        <option value={512}>512 px max</option>
      </select>
    </label>
    <div className="export-foot">
      <span>{glbInfo ?? ''}</span>
      <button className="export-go" disabled={glbBusy} onClick={onGlb}>Export GLB</button>
    </div>
  </>;
}
