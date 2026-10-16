import { useState } from 'react';
import { DEFAULT_ENV, DEFAULT_HABLE, matchPreset, type Environment, type ToneMap } from '../scene/environment';
import { USER_PREFIX, deleteUserPreset, matchAnyPreset, saveUserPreset } from '../scene/userPresets';
import { LightingSelect } from './LightingSelect';

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
}

// Kelvin sliders drop the explicit colour a game-data preset may carry (the colour would otherwise win).
const CLEARS: Partial<Record<keyof Environment, keyof Environment>> = { keyTemp: 'keyColor', skyTemp: 'skyColor', backTemp: 'backColor' };

type NumKey = { [K in keyof Environment]-?: Environment[K] extends number ? K : never }[keyof Environment];

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

export function RenderPanel({ options, onChange }: Props) {
  return (
    <div className="pane">
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
    </div>
  );
}

const icon = { width: 14, height: 14, viewBox: '0 0 16 16', fill: 'none', stroke: 'currentColor', strokeWidth: 1.4, strokeLinejoin: 'round', strokeLinecap: 'round', 'aria-hidden': true } as const;
const FloppyIcon = () => <svg {...icon}><path d="M2.5 2.5h9l2 2v9h-11z" /><path d="M5 2.5v3.5h5V2.5M5 13.5V9h6v4.5" /></svg>;
const TrashIcon = () => <svg {...icon}><path d="M2.5 4h11M6 4V2.5h4V4M4 4l.7 9.5h6.6L12 4M6.8 6.5v4.5M9.2 6.5v4.5" /></svg>;

export function LightingPanel({ options, onChange }: Props) {
  const setEnv = (key: keyof Environment, value: number) => {
    const env = { ...options.env, [key]: value };
    const clear = CLEARS[key];
    if (clear) delete env[clear];
    onChange({ env });
  };
  const [name, setName] = useState('');
  const current = matchAnyPreset(options.env, matchPreset);
  const save = () => { if (!name.trim()) return; saveUserPreset(name, options.env); setName(''); };
  return (
    <div className="pane">
      <div className="field"><span><b>Environment</b></span><button className="btn small" onClick={() => onChange({ env: { ...DEFAULT_ENV } })}>Reset environment</button></div>
      <label className="field">
        <span>Lighting</span>
        <LightingSelect env={options.env} onPick={(env) => onChange({ env })} />
      </label>
      <div className="field save-preset">
        <input className="text" placeholder="Save current as..." value={name} maxLength={40} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') save(); }} />
        <button className="btn small" disabled={!name.trim()} onClick={save} title="Save" aria-label="Save current lighting as a preset (same name overwrites)"><FloppyIcon /></button>
        {current?.startsWith(USER_PREFIX) && <button className="btn small" onClick={() => deleteUserPreset(current)} title="Delete" aria-label="Delete the selected saved preset"><TrashIcon /></button>}
      </div>
      <label className="field">
        <span>Tone mapping</span>
        <select className="text" value={options.env.toneMapping} onChange={(e) => {
          const toneMapping = e.target.value as ToneMap;
          const env = { ...options.env, toneMapping };
          if (toneMapping === 'hable') env.hable ??= { ...DEFAULT_HABLE }; else delete env.hable;
          onChange({ env });
        }}>
          <option value="linear">Linear</option>
          <option value="aces">ACES filmic</option>
          <option value="agx">AgX</option>
          <option value="neutral">Khronos neutral</option>
          <option value="hable">Hable filmic (game curve)</option>
        </select>
      </label>
      {(options.env.keyColor || options.env.skyColor || options.env.backColor) && (
        <div className="field"><span className="muted">Light colours come from the game data; moving a temperature slider replaces them.</span></div>
      )}
      <label className="field"><span>Ground bounce</span><input type="color" value={options.env.ground} onChange={(e) => onChange({ env: { ...options.env, ground: e.target.value } })} /></label>
      {ENV_SLIDERS.map((sl) => {
        const dis = sl.gameOnly && options.shading !== 'game';
        return (
          <label className="field" key={sl.key}>
            <span>{sl.label} {options.env[sl.key].toFixed(sl.digits)}</span>
            <input type="range" min={sl.min} max={sl.max} step={sl.step} disabled={dis} value={options.env[sl.key]}
              onChange={(e) => setEnv(sl.key, Number(e.target.value))} />
          </label>
        );
      })}
    </div>
  );
}
