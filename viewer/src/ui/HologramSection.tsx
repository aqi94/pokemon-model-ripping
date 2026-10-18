import * as THREE from 'three';
import type { HolocastPreset } from '../types';
import { Accordion } from './Accordion';
import { GLOBAL_HOLO_PRESETS, HOLO_FIT_DEFAULTS, stateFromParams, type HoloParams, type HologramState } from '../scene/hologram';

interface Props {
  /** this model's data sets (manifest `holocast`), listed first */
  presets: HolocastPreset[];
  state: HologramState;
  onChange: (s: HologramState) => void;
}

export function presetParams(id: string, presets: HolocastPreset[]): HoloParams | undefined {
  return presets.find((p) => p.name === id) ?? GLOBAL_HOLO_PRESETS.find((p) => p.id === id)?.params;
}

/** Default state for a model: its first data set, else the data's cyan. Keeps on/opacity/glow from `prev`. */
export function defaultHologram(presets: HolocastPreset[], prev?: HologramState): HologramState {
  const id = presets[0]?.name ?? GLOBAL_HOLO_PRESETS[0].id;
  return stateFromParams(presetParams(id, presets)!, id, prev ? { ...prev } : { on: false } as HologramState);
}

function toHex(c: [number, number, number]) {
  return '#' + new THREE.Color().setRGB(c[0], c[1], c[2]).getHexString(); // linear -> sRGB hex
}

function fromHex(hex: string): [number, number, number] {
  const c = new THREE.Color(hex); // sRGB hex -> linear
  const m = Math.max(1e-6, c.r, c.g, c.b);
  return [c.r / m, c.g / m, c.b / m];
}

const DIALS: { key: keyof HologramState; label: string; min: number; max: number; step: number; digits: number; title: string }[] = [
  { key: 'intensity', label: 'Intensity', min: 0, max: 8, step: 0.05, digits: 2, title: 'HDR strength of the hologram colour (data: cyan 3.4, red 6.2)' },
  { key: 'tiling', label: 'Scanline density', min: 20, max: 600, step: 1, digits: 0, title: 'ScanlineTiling (data 260 = one line every ~1.2 cm)' },
  { key: 'depth', label: 'Scanline depth', min: 0, max: 1, step: 0.01, digits: 2, title: 'ScanlineDepth: how dark the lines are (data 0.7)' },
  { key: 'speed', label: 'Scroll speed', min: -6, max: 6, step: 0.1, digits: 1, title: 'ScanlineSpeed (data 2; pauses with playback)' },
  { key: 'gradation', label: 'Gradient bands', min: 0, max: 8, step: 0.1, digits: 1, title: 'ScanGradationTiling: slow brightness bands over the body (data 1)' },
  { key: 'gradientStrength', label: 'Gradient strength', min: 0, max: 1, step: 0.01, digits: 2, title: 'How strong the slow bands are (fitted to screenshots, not data)' },
  { key: 'opacity', label: 'Opacity', min: 0.05, max: 1, step: 0.01, digits: 2, title: 'The game marks holograms Opaque, but every screenshot shows the scene through them' },
  { key: 'edgeGlow', label: 'Edge glow', min: 0, max: 2, step: 0.01, digits: 2, title: 'Fresnel rim glow (fitted to screenshots, not data)' },
];

/** Appearance > Hologram: the Z-A Holocast effect (any model; data presets first). */
export function HologramSection({ presets, state, onChange }: Props) {
  const set = (p: Partial<HologramState>, custom = false) => onChange({ ...state, ...p, ...(custom ? { preset: 'custom' } : {}) });
  const pick = (id: string) => {
    const params = presetParams(id, presets);
    if (params) onChange(stateFromParams(params, id, state));
    else set({ preset: 'custom' });
  };
  const reset = () => {
    const params = presetParams(state.preset, presets);
    onChange(params ? { ...stateFromParams(params, state.preset, state), ...HOLO_FIT_DEFAULTS } : defaultHologram(presets, { ...state, ...HOLO_FIT_DEFAULTS }));
  };
  return (
    <Accordion title="Hologram">
      <label className="check" title="Z-A Holocast effect (trainer .trmmt 'holo' material sets)">
        <input type="checkbox" checked={state.on} onChange={(e) => set({ on: e.target.checked })} /><span>Hologram</span>
      </label>
      {state.on && <>
        <label className="field">
          <span>Preset</span>
          <select className="text" value={state.preset} onChange={(e) => pick(e.target.value)} aria-label="Hologram preset">
            {presets.length > 0 && <optgroup label="This model (game data)">
              {presets.map((p) => <option key={p.name} value={p.name}>{p.label}</option>)}
            </optgroup>}
            <optgroup label="Z-A data">
              {GLOBAL_HOLO_PRESETS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
            </optgroup>
            <option value="custom">Custom</option>
          </select>
        </label>
        <label className="field">
          <span>Colour</span>
          <span className="row">
            <input type="color" value={toHex(state.color)} onChange={(e) => set({ color: fromHex(e.target.value) }, true)} aria-label="Hologram colour" />
            <button className="btn small" onClick={reset} title="Restore the preset's values">Reset</button>
          </span>
        </label>
        {DIALS.map((d) => (
          <label className="field" key={d.key} title={d.title}>
            <span>{d.label} {(state[d.key] as number).toFixed(d.digits)}</span>
            <input type="range" min={d.min} max={d.max} step={d.step} value={state[d.key] as number}
              onChange={(e) => set({ [d.key]: Number(e.target.value) } as Partial<HologramState>, d.key === 'intensity')} />
          </label>
        ))}
      </>}
    </Accordion>
  );
}
