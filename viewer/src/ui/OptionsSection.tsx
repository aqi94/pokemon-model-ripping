import { Accordion } from './Accordion';
import { DEFAULT_ENV, LIGHTING_PRESETS, matchPreset, type Environment, type ToneMap } from '../scene/environment';

export interface Options {
  shadows: boolean;
  skeleton: boolean;
  wireframe: boolean;
  background: string;
  autoRotate: number;
  mirror: boolean;
  albedo: 'baked' | 'raw' | 'off';
  shading: 'game' | 'pbr';
  env: Environment;
}

interface Props {
  options: Options;
  onChange: (patch: Partial<Options>) => void;
  onFullscreen: () => void;
  onRecenter: () => void;
  onCamera: (p: 'fit' | 'vs' | 'front' | 'side' | 'back' | 'top') => void;
}

type NumKey = { [K in keyof Environment]: Environment[K] extends number ? K : never }[keyof Environment];

const ENV_SLIDERS: { key: NumKey; label: string; min: number; max: number; step: number; digits: number; gameOnly?: boolean }[] = [
  { key: 'ambient', label: 'Ambient', min: 0, max: 4, step: 0.05, digits: 2 },
  { key: 'key', label: 'Key light', min: 0, max: 4, step: 0.05, digits: 2 },
  { key: 'azimuth', label: 'Key azimuth', min: -180, max: 180, step: 1, digits: 0 },
  { key: 'elevation', label: 'Key elevation', min: 0, max: 90, step: 1, digits: 0 },
  { key: 'keyTemp', label: 'Key temp K', min: 1500, max: 12000, step: 100, digits: 0 },
  { key: 'back', label: 'Back light', min: 0, max: 4, step: 0.05, digits: 2 },
  { key: 'backAzimuth', label: 'Back azimuth', min: -180, max: 180, step: 1, digits: 0 },
  { key: 'backElevation', label: 'Back elevation', min: 0, max: 90, step: 1, digits: 0 },
  { key: 'backTemp', label: 'Back temp K', min: 1500, max: 12000, step: 100, digits: 0 },
  { key: 'skyTemp', label: 'Sky temp K', min: 1500, max: 12000, step: 100, digits: 0 },
  { key: 'warmth', label: 'White balance', min: -1, max: 1, step: 0.05, digits: 2 },
  { key: 'shadow', label: 'Shadow', min: 0, max: 1, step: 0.05, digits: 2, gameOnly: true },
  { key: 'rim', label: 'Rim', min: 0, max: 2, step: 0.05, digits: 2, gameOnly: true },
  { key: 'detail', label: 'Detail', min: 0, max: 1, step: 0.05, digits: 2, gameOnly: true },
  { key: 'exposure', label: 'Exposure', min: 0.5, max: 2, step: 0.05, digits: 2 },
  { key: 'contrast', label: 'Contrast', min: 0.5, max: 1.5, step: 0.05, digits: 2, gameOnly: true },
  { key: 'saturation', label: 'Saturation', min: 0, max: 2, step: 0.05, digits: 2, gameOnly: true },
  { key: 'gi', label: 'Ambient tint', min: 0, max: 2, step: 0.05, digits: 2, gameOnly: true },
];

export function OptionsSection({ options, onChange, onFullscreen, onRecenter, onCamera }: Props) {
  return (
    <Accordion title="Options">
      <label className="check"><input type="checkbox" checked={options.shadows} onChange={(e) => onChange({ shadows: e.target.checked })} /><span>Shadows</span></label>
      <label className="check"><input type="checkbox" checked={options.skeleton} onChange={(e) => onChange({ skeleton: e.target.checked })} /><span>Skeleton</span></label>
      <label className="check"><input type="checkbox" checked={options.wireframe} onChange={(e) => onChange({ wireframe: e.target.checked })} /><span>Wireframe</span></label>
      <label className="field">
        <span>Shading</span>
        <select className="text" value={options.shading} onChange={(e) => onChange({ shading: e.target.value as Options['shading'] })}>
          <option value="game">Game</option>
          <option value="pbr">PBR</option>
        </select>
      </label>
      <label className="field">
        <span>Albedo</span>
        <select className="text" value={options.albedo} onChange={(e) => onChange({ albedo: e.target.value as Options['albedo'] })}>
          <option value="baked">Baked</option>
          <option value="raw">Raw albedo</option>
          <option value="off">Off</option>
        </select>
      </label>
      <label className="field"><span>Background</span><span className="row"><input type="color" value={options.background} onChange={(e) => onChange({ background: e.target.value })} /><button className="btn small" onClick={() => onChange({ background: '#f4f4f4' })}>White</button></span></label>
      <label className="field">
        <span>Auto-rotate {options.autoRotate ? options.autoRotate.toFixed(1) : 'off'}</span>
        <input type="range" min={0} max={5} step={0.1} value={options.autoRotate} onChange={(e) => onChange({ autoRotate: Number(e.target.value) })} />
      </label>
      <div className="field"><span><b>Environment</b></span><button className="btn small" onClick={() => onChange({ env: { ...DEFAULT_ENV } })}>Reset environment</button></div>
      <label className="field">
        <span>Lighting</span>
        <select className="text" value={matchPreset(options.env) ?? ''} onChange={(e) => { const pr = LIGHTING_PRESETS.find((x) => x.id === e.target.value); if (pr) onChange({ env: { ...pr.env } }); }}>
          {matchPreset(options.env) === null && <option value="">Custom</option>}
          {LIGHTING_PRESETS.map((pr) => <option key={pr.id} value={pr.id}>{pr.label}</option>)}
        </select>
      </label>
      <label className="field">
        <span>Tone mapping</span>
        <select className="text" value={options.env.toneMapping} onChange={(e) => onChange({ env: { ...options.env, toneMapping: e.target.value as ToneMap } })}>
          <option value="linear">Linear</option>
          <option value="aces">ACES filmic</option>
          <option value="agx">AgX</option>
          <option value="neutral">Khronos neutral</option>
        </select>
      </label>
      <label className="field"><span>Ground bounce</span><input type="color" value={options.env.ground} onChange={(e) => onChange({ env: { ...options.env, ground: e.target.value } })} /></label>
      {ENV_SLIDERS.map((sl) => {
        const dis = sl.gameOnly && options.shading !== 'game';
        return (
          <label className="field" key={sl.key}>
            <span>{sl.label} {options.env[sl.key].toFixed(sl.digits)}</span>
            <input type="range" min={sl.min} max={sl.max} step={sl.step} disabled={dis} value={options.env[sl.key]}
              onChange={(e) => onChange({ env: { ...options.env, [sl.key]: Number(e.target.value) } })} />
          </label>
        );
      })}
      <div className="field"><span>Camera</span></div>
      <div className="row wrap">
        {(['fit', 'vs', 'front', 'side', 'back', 'top'] as const).map((p) => (
          <button key={p} className="btn small" onClick={() => onCamera(p)}>{p === 'vs' ? 'VS' : p[0].toUpperCase() + p.slice(1)}</button>
        ))}
      </div>
      <div className="row wrap">
        <button className="btn small" onClick={onFullscreen}>Full screen</button>
        <button className="btn small" onClick={() => onChange({ mirror: !options.mirror })}>Mirror model{options.mirror ? ' (on)' : ''}</button>
        <button className="btn small" onClick={onRecenter}>Recenter camera</button>
      </div>
    </Accordion>
  );
}
