import { useState } from 'react';
import { CATEGORY_LABELS, DEFAULT_ENV, DEFAULT_HABLE, matchPreset, type CategoryLight, type Environment, type ToneMap } from '../scene/environment';
import { USER_PREFIX, deleteUserPreset, matchAnyPreset, saveUserPreset } from '../scene/userPresets';
import { LightingSelect } from './LightingSelect';

export interface Options {
  shadows: boolean;
  /** UNITE idle particle stand-ins (tail flame, gas); undefined = on. */
  particles?: boolean;
  animateEffects?: boolean;
  skeleton: boolean;
  wireframe: boolean;
  outline: boolean;
  outlineWidth: number;
  outlineColor: string;
  outlineTinted: boolean;
  /** hull = cartoon inverted hull; sprite = 3DS battle-sprite look (low-res render + 1-px lines). */
  outlineStyle?: 'auto' | 'hull' | 'sprite';
  /** Sprite style: screen px per sprite pixel. */
  outlinePixel?: number;
  outlineSpriteColor?: string;
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
      <label className="check" title="Stand-ins for in-game particle effects the rip lacks (tail flames, gas)"><input type="checkbox" checked={options.particles ?? true} onChange={(e) => onChange({ particles: e.target.checked })} /><span>Particles</span></label>
      <label className="check" title="Scroll flame noise at Pokémon HOME's speeds (UNITE flames)"><input type="checkbox" checked={options.animateEffects ?? true} onChange={(e) => onChange({ animateEffects: e.target.checked })} /><span>Animate effects</span></label>
      <label className="check"><input type="checkbox" checked={options.skeleton} onChange={(e) => onChange({ skeleton: e.target.checked })} /><span>Skeleton</span></label>
      <label className="check"><input type="checkbox" checked={options.wireframe} onChange={(e) => onChange({ wireframe: e.target.checked })} /><span>Wireframe</span></label>
      <label className="check"><input type="checkbox" checked={options.outline} onChange={(e) => onChange({ outline: e.target.checked })} /><span>Outline</span></label>
      {options.outline && (
        <label className="field">
          <span>Outline style</span>
          <select className="text" value={options.outlineStyle ?? 'auto'} onChange={(e) => onChange({ outlineStyle: e.target.value as Options['outlineStyle'] })}>
            <option value="auto">Game data (else hull)</option>
            <option value="hull">Cartoon hull</option>
            <option value="sprite">3DS sprite</option>
          </select>
        </label>
      )}
      {options.outline && options.outlineStyle === 'sprite' && (
        <>
          <label className="field">
            <span>Pixel size {options.outlinePixel ?? 3} px</span>
            <input type="range" min={1} max={8} step={1} value={options.outlinePixel ?? 3} onChange={(e) => onChange({ outlinePixel: Number(e.target.value) })} />
          </label>
          <label className="field"><span>Silhouette colour</span><input type="color" value={options.outlineSpriteColor ?? '#272726'} onChange={(e) => onChange({ outlineSpriteColor: e.target.value })} /></label>
        </>
      )}
      {options.outline && options.outlineStyle !== 'sprite' && (
        <>
          <label className="field">
            <span>Outline width {options.outlineWidth.toFixed(1)} px</span>
            <input type="range" min={0.5} max={6} step={0.5} value={options.outlineWidth} onChange={(e) => onChange({ outlineWidth: Number(e.target.value) })} />
          </label>
          <label className="check"><input type="checkbox" checked={options.outlineTinted} onChange={(e) => onChange({ outlineTinted: e.target.checked })} /><span>Tint to surface colour</span></label>
          {!options.outlineTinted && <label className="field"><span>Outline colour</span><input type="color" value={options.outlineColor} onChange={(e) => onChange({ outlineColor: e.target.value })} /></label>}
        </>
      )}
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
          <option value="unite">UNITE game curve</option>
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
      {options.shading === 'game' && <CategoryLights env={options.env} onChange={(env) => onChange({ env })} />}
    </div>
  );
}

const CAT_SLIDERS: { key: 'key' | 'ambient' | 'rim' | 'rimPower'; color?: 'keyColor' | 'ambientColor' | 'rimColor'; label: string; max: number; def: number }[] = [
  { key: 'key', color: 'keyColor', label: 'Key', max: 3, def: 1 },
  { key: 'ambient', color: 'ambientColor', label: 'Ambient', max: 3, def: 1 },
  { key: 'rim', color: 'rimColor', label: 'Rim light', max: 2, def: 0 },
  { key: 'rimPower', label: 'Rim power', max: 8, def: 3 },
];

/** The game lights each material category (CategoryLabel) with its own key / ambient / rim set (dir_N, GILightCommon.ColorN, rim_N). */
function CategoryLights({ env, onChange }: { env: Environment; onChange: (env: Environment) => void }) {
  const [cat, setCat] = useState('4');
  const c: CategoryLight = env.categories?.[cat] ?? {};
  const set = (patch: Partial<CategoryLight>) => {
    const next = { ...c, ...patch };
    const categories = { ...env.categories, [cat]: next };
    onChange({ ...env, categories });
  };
  const reset = () => {
    const categories = { ...env.categories };
    delete categories[cat];
    const { categories: _old, ...rest } = env;
    onChange(Object.keys(categories).length ? { ...rest, categories } : rest);
  };
  return (
    <>
      <div className="field"><span><b>Material categories</b></span><button className="btn small" disabled={!env.categories?.[cat]} onClick={reset}>Reset</button></div>
      <label className="field">
        <span>Category</span>
        <select className="text" value={cat} onChange={(e) => setCat(e.target.value)}>
          {Object.entries(CATEGORY_LABELS).map(([k, v]) => <option key={k} value={k}>{v}{env.categories?.[k] ? ' *' : ''}</option>)}
        </select>
      </label>
      {CAT_SLIDERS.map((sl) => {
        const v = c[sl.key] ?? sl.def;
        return (
          <label className="field" key={sl.key}>
            <span className="row">{sl.label} {v.toFixed(2)}{sl.color && <input type="color" value={c[sl.color] ?? '#ffffff'} onChange={(e) => set({ [sl.color!]: e.target.value })} />}</span>
            <input type="range" min={0} max={sl.max} step={0.05} value={v} onChange={(e) => set({ [sl.key]: Number(e.target.value) })} />
          </label>
        );
      })}
    </>
  );
}
