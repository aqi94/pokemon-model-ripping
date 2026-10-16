import type { Options } from './OptionsSection';

export type CameraPreset = 'fit' | 'vs' | 'front' | 'side' | 'back' | 'top';

interface Props {
  options: Options;
  onChange: (patch: Partial<Options>) => void;
  onCamera: (p: CameraPreset) => void;
  onRecenter: () => void;
  onFullscreen: () => void;
  onPng: () => void;
  onGlb: () => void;
  disabled: boolean;
}

const VIEWS: [CameraPreset, string, string][] = [
  ['fit', '◎', 'Fit view (F)'], ['front', 'F', 'Front (1)'], ['side', 'S', 'Side (2)'], ['back', 'B', 'Back (3)'], ['top', 'T', 'Top (4)'], ['vs', 'VS', 'VS shot (5)'],
];

/** Photoshop-style vertical toolbox: view presets, display toggles, export. */
export function Toolbox({ options, onChange, onCamera, onRecenter, onFullscreen, onPng, onGlb, disabled }: Props) {
  const tog = (key: 'skeleton' | 'wireframe' | 'shadows' | 'mirror', glyph: string, title: string) => (
    <button className={'tool-btn' + (options[key] ? ' on' : '')} title={title} aria-pressed={options[key]} onClick={() => onChange({ [key]: !options[key] })}>{glyph}</button>
  );
  return (
    <nav className="toolbox" aria-label="Tools">
      {VIEWS.map(([p, g, t]) => <button key={p} className="tool-btn" title={t} disabled={disabled} onClick={() => onCamera(p)}>{g}</button>)}
      <button className="tool-btn" title="Recenter camera" onClick={onRecenter} disabled={disabled}>⌖</button>
      <hr />
      {tog('mirror', '⇋', 'Mirror model')}
      {tog('skeleton', '☠', 'Skeleton')}
      {tog('wireframe', '▦', 'Wireframe')}
      {tog('shadows', '◐', 'Shadows')}
      <hr />
      <button className="tool-btn" title="Full screen" onClick={onFullscreen}>⛶</button>
      <button className="tool-btn txt" title="Export PNG (transparent)" onClick={onPng} disabled={disabled}>PNG</button>
      <button className="tool-btn txt" title="Export GLB" onClick={onGlb} disabled={disabled}>GLB</button>
    </nav>
  );
}
