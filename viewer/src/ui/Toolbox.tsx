import type { Options } from './OptionsSection';

export type CameraPreset = 'fit' | 'vs' | 'front' | 'side' | 'back' | 'top' | 'idle';

interface Props {
  options: Options;
  onChange: (patch: Partial<Options>) => void;
  onCamera: (p: CameraPreset) => void;
  onRecenter: () => void;
  onFullscreen: () => void;
  onPngOptions: (top: number) => void;
  onGlb: () => void;
  disabled: boolean;
  refImage: boolean;
  refVisible: boolean;
  refTransforming: boolean;
  onRefVisible: () => void;
  onRefTransform: () => void;
}

const VIEWS: [CameraPreset, string, string][] = [
  ['fit', '◎', 'Fit view (F)'], ['front', 'F', 'Front (1)'], ['side', 'S', 'Side (2)'], ['back', 'B', 'Back (3)'], ['top', 'T', 'Top (4)'], ['vs', 'VS', 'VS shot (5)'], ['idle', 'I', 'Idle ortho shot (6)'],
];

/** Photoshop-style vertical toolbox: view presets, display toggles, export. */
export function Toolbox({ options, onChange, onCamera, onRecenter, onFullscreen, onPngOptions, onGlb, disabled,
  refImage, refVisible, refTransforming, onRefVisible, onRefTransform }: Props) {
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
      <button className={'tool-btn' + (refImage && refVisible ? ' on' : '')} title={refImage ? 'Show reference image' : 'Reference image: load one in the Reference tab or drop it on the canvas'}
        aria-pressed={refImage && refVisible} disabled={!refImage} onClick={onRefVisible}>🖼</button>
      <button className={'tool-btn' + (refTransforming ? ' on' : '')} title="Move / scale / rotate the reference image (R)"
        aria-pressed={refTransforming} disabled={!refImage} onClick={onRefTransform}>⤧</button>
      <hr />
      <button className="tool-btn" title="Full screen" onClick={onFullscreen}>⛶</button>
      <button className="tool-btn txt" title="Export PNG (opens options)" disabled={disabled}
        onClick={(e) => onPngOptions(e.currentTarget.getBoundingClientRect().top)}>PNG</button>
      {!import.meta.env.VITE_GLB_BASE && <button className="tool-btn txt" title="Export GLB" onClick={onGlb} disabled={disabled}>GLB</button>}
    </nav>
  );
}
