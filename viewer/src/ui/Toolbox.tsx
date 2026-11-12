import type { ExportFormat } from '../scene/exportOptions';
import type { Options } from './OptionsSection';
import { PUBLIC_BUILD } from '../data';

export type CameraPreset = 'fit' | 'vs' | 'front' | 'side' | 'back' | 'top' | 'idle' | 'battle' | 'battleBack';

interface Props {
  options: Options;
  onChange: (patch: Partial<Options>) => void;
  /** Opens / closes the Camera flyout; `top` = the button's top edge. */
  onCameraPanel: (top: number) => void;
  cameraOpen: boolean;
  onFullscreen: () => void;
  /** Opens / closes / switches the export popover of one format; `top` = the button's top edge. */
  onExport: (format: ExportFormat, top: number) => void;
  /** Export popover that is open. */
  exportOpen: ExportFormat | null;
  disabled: boolean;
  refImage: boolean;
  refVisible: boolean;
  refTransforming: boolean;
  onRefVisible: () => void;
  onRefTransform: () => void;
}

// GLB is left out of the public build (VITE_GLB_BASE set): user decision.
const EXPORTS: [ExportFormat, string][] = [
  ['png', 'Export PNG image'], ['gif', 'Export animated GIF of the current clip'],
  ['stl', 'Export STL for 3D printing'], ['glb', 'Export GLB model (edited to match the view)'],
];

/** Photoshop-style vertical toolbox: camera panel, display toggles, export. */
export function Toolbox({ options, onChange, onCameraPanel, cameraOpen, onFullscreen, onExport, exportOpen, disabled,
  refImage, refVisible, refTransforming, onRefVisible, onRefTransform }: Props) {
  const tog = (key: 'skeleton' | 'wireframe' | 'shadows' | 'mirror', glyph: string, title: string) => (
    <button className={'tool-btn' + (options[key] ? ' on' : '')} title={title} aria-pressed={options[key]} onClick={() => onChange({ [key]: !options[key] })}>{glyph}</button>
  );
  return (
    <nav className="toolbox" aria-label="Tools">
      <button className={'tool-btn' + (cameraOpen ? ' on' : '')} title="Camera panel: angles, distance, FOV, roll" aria-expanded={cameraOpen} disabled={disabled}
        onClick={(e) => onCameraPanel(e.currentTarget.getBoundingClientRect().top)}>
        <svg className="ico" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" aria-hidden="true"><path d="M2.5 6.5h3l1.5-2h6l1.5 2h3v9h-15z" /><circle cx="10" cy="10.8" r="3" /></svg>
      </button>
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
      {EXPORTS.filter(([f]) => f !== 'glb' || !PUBLIC_BUILD).map(([f, t]) => (
        <button key={f} className={'tool-btn txt' + (exportOpen === f ? ' on' : '')} title={t} aria-expanded={exportOpen === f} disabled={disabled}
          onClick={(e) => onExport(f, e.currentTarget.getBoundingClientRect().top)}>{f.toUpperCase()}</button>
      ))}
    </nav>
  );
}
