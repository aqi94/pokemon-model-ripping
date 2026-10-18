import { EXPORT_PRESETS, MAX_EXPORT_PX, type ExportOptions } from '../scene/exportOptions';

interface Props {
  value: ExportOptions;
  onChange: (patch: Partial<ExportOptions>) => void;
  onExport: () => void;
  onClose: () => void;
  top: number;
}

const num = (v: string, min: number, max: number, fallback: number) => {
  const n = Number(v);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
};

/** Popover next to the toolbox PNG button: canvas size, scale, framing, padding, background, trim. */
export function ExportPanel({ value: o, onChange, onExport, onClose, top }: Props) {
  const solid = o.background !== 'transparent';
  const framed = o.framing !== 'current';
  return (
    <div className="export-panel" style={{ top }} role="dialog" aria-label="PNG export">
      <div className="export-head"><b>Export PNG</b><button className="tool-btn" title="Close" onClick={onClose}>×</button></div>
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
    </div>
  );
}
