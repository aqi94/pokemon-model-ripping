import { useEffect, useState } from 'react';
import { MAX_PITCH, type CameraOrbit, type CameraPreset } from '../scene/ViewerScene';
import { Accordion } from './Accordion';

interface Props {
  orbit: CameraOrbit | null;
  onChange: (patch: Partial<CameraOrbit>) => void;
  onRotate: (axis: 'yaw' | 'pitch' | 'roll', deg: number) => void;
  onPreset: (p: CameraPreset) => void;
  ranges: () => { center: [number, number, number]; radius: number; dMin: number; dMax: number } | null;
}

type Axis = 'yaw' | 'pitch' | 'roll';

/** Typed number field: shows the live value, keeps what the user is typing until Enter / blur commits it. */
function NumField({ value, digits, step, onCommit }: { value: number; digits: number; step: number; onCommit: (v: number) => void }) {
  const shown = Number(value.toFixed(digits)).toString();
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    if (draft === null) return;
    const v = Number(draft);
    setDraft(null);
    if (draft.trim() !== '' && Number.isFinite(v)) onCommit(v);
  };
  return (
    <input className="text num" type="number" step={step} value={draft ?? shown}
      onFocus={(e) => e.currentTarget.select()}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => { if (e.key === 'Enter') { commit(); e.currentTarget.blur(); } else if (e.key === 'Escape') { setDraft(null); e.currentTarget.blur(); } }} />
  );
}

/** Slider + exact number field on one row. The number field accepts any float, outside the slider range too. */
function Row({ label, unit, value, min, max, step, digits, numStep, onCommit, extra, snap, log }: {
  label: string; unit: string; value: number; min: number; max: number; step: number; digits: number; numStep: number;
  onCommit: (v: number) => void; extra?: React.ReactNode;
  /** slider only: round to multiples of this (then clamp), e.g. 45° */
  snap?: number;
  /** logarithmic slider (min > 0): equal travel = equal ratio, so it is as fine close up as far away */
  log?: boolean;
}) {
  const fromSlider = (v: number) => onCommit(snap ? Math.min(max, Math.max(min, Math.round(v / snap) * snap)) : v);
  const clamped = Math.min(max, Math.max(min, value));
  const lr = Math.log(max / min);
  const slider = log
    ? <input type="range" min={0} max={1000} step={1} value={Math.round((Math.log(clamped / min) / lr) * 1000)} onChange={(e) => onCommit(min * Math.exp((Number(e.target.value) / 1000) * lr))} />
    : <input type="range" min={min} max={max} step={step} value={clamped} onChange={(e) => fromSlider(Number(e.target.value))} />;
  return (
    <div className="cam-row">
      <span className="cam-label">{label}</span>
      {slider}
      <NumField value={value} digits={digits} step={numStep} onCommit={onCommit} />
      <span className="cam-unit">{unit}</span>
      {extra ?? <span />}
    </div>
  );
}

/** Framed camera angles, with their keyboard shortcut (handled in Viewer.tsx). */
export const CAMERA_VIEWS: [CameraPreset, string, string][] = [
  ['fit', 'Fit', 'F'], ['front', 'Front', '1'], ['side', 'Side', '2'], ['back', 'Back', '3'], ['top', 'Top', '4'],
  ['vs', 'VS shot', '5'], ['idle', 'Idle', '6'], ['battle', 'Battle front', '7'], ['battleBack', 'Battle back', '8'],
];
const VIEW_TIPS: Partial<Record<CameraPreset, string>> = { fit: 'Fit the model and reset roll / lens', idle: 'Idle ortho shot' };

/** Camera flyout: framed angles, then orbit angles / distance / FOV; roll, target, position and copy/paste under Advanced. */
export function CameraPanel({ orbit, onChange, onRotate, onPreset, ranges }: Props) {
  const [snap, setSnap] = useState(() => { try { return localStorage.getItem('camSnap45') === '1'; } catch { return false; } });
  useEffect(() => { try { localStorage.setItem('camSnap45', snap ? '1' : '0'); } catch { /* ignore */ } }, [snap]);
  const [msg, setMsg] = useState('');
  if (!orbit) return <div className="pane"><span className="muted">Load a model to edit the camera.</span></div>;

  const snapTo = snap ? 45 : undefined;
  const steps = (axis: Axis) => (
    <span className="cam-steps">
      <button className="btn small" title={`${axis} -45°`} onClick={() => onRotate(axis, -45)}>-45</button>
      <button className="btn small" title={`${axis} +45°`} onClick={() => onRotate(axis, 45)}>+45</button>
    </span>
  );
  const vec = (key: 'target' | 'position', i: number, v: number) => {
    const a: [number, number, number] = [...orbit[key]];
    a[i] = v;
    onChange({ [key]: a });
  };
  // slider span follows the scene size so small Pals and big ones are both usable; typed values are unbounded
  const rg = ranges() ?? { center: orbit.target, radius: 1, dMin: 0.1, dMax: 10 };

  const copy = async () => {
    const pose = { yaw: +orbit.yaw.toFixed(4), pitch: +orbit.pitch.toFixed(4), distance: +orbit.distance.toFixed(5), roll: +orbit.roll.toFixed(4),
      fov: +orbit.fov.toFixed(4), target: orbit.target.map((n) => +n.toFixed(5)) };
    try { await navigator.clipboard.writeText(JSON.stringify(pose)); setMsg('Copied'); } catch { setMsg('Clipboard unavailable'); }
  };
  const paste = async () => {
    try {
      const j = JSON.parse(await navigator.clipboard.readText());
      const patch: Partial<CameraOrbit> = {};
      for (const k of ['yaw', 'pitch', 'distance', 'roll', 'fov'] as const) if (typeof j[k] === 'number') patch[k] = j[k];
      const isVec = (v: unknown): v is [number, number, number] => Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === 'number');
      if (isVec(j.target)) patch.target = j.target;
      if (isVec(j.position) && !('yaw' in patch)) patch.position = j.position;
      if (!Object.keys(patch).length) throw new Error('no camera fields');
      onChange(patch); setMsg('Pasted');
    } catch { setMsg('Clipboard has no camera pose'); }
  };

  return (
    <div className="cam-pane">
      <div className="cam-views">
        {CAMERA_VIEWS.map(([p, label, key]) => (
          <button key={p} className="btn small" title={`${VIEW_TIPS[p] ?? label} (${key})`} onClick={() => onPreset(p)}>
            {label}<kbd>{key}</kbd>
          </button>
        ))}
      </div>
      <div className="field cam-head"><span><b>Orbit</b></span>
        <label className="check inline"><input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} /><span>Snap 45°</span></label>
      </div>
      <Row label="Yaw" unit="°" value={orbit.yaw} min={-180} max={180} step={0.1} snap={snapTo} digits={2} numStep={0.1} onCommit={(v) => onChange({ yaw: v })} extra={steps('yaw')} />
      <Row label="Pitch" unit="°" value={orbit.pitch} min={-MAX_PITCH} max={MAX_PITCH} step={0.1} snap={snapTo} digits={2} numStep={0.1}
        onCommit={(v) => onChange({ pitch: Math.max(-MAX_PITCH, Math.min(MAX_PITCH, v)) })} extra={steps('pitch')} />
      <Row label="Distance" unit="m" value={orbit.distance} min={rg.dMin} max={rg.dMax} step={0.001} log digits={3} numStep={0.01}
        onCommit={(v) => v > 0 && onChange({ distance: v })} />
      <Row label="FOV" unit="°" value={orbit.fov} min={1} max={120} step={0.1} digits={2} numStep={0.5} onCommit={(v) => onChange({ fov: v })} />

      <Accordion title="Advanced" id="cam-adv" defaultOpen={false} className="cam-adv">
        <Row label="Roll" unit="°" value={orbit.roll} min={-180} max={180} step={0.1} snap={snapTo} digits={2} numStep={0.1} onCommit={(v) => onChange({ roll: v })} extra={steps('roll')} />
        <div className="field cam-head"><span><b>Target</b></span>
          <button className="btn small" onClick={() => { const c = ranges()?.center; if (c) onChange({ target: c }); }}>Model centre</button>
        </div>
        {(['X', 'Y', 'Z'] as const).map((n, i) => (
          <Row key={'t' + n} label={n} unit="m" value={orbit.target[i]} min={rg.center[i] - rg.radius * 2} max={rg.center[i] + rg.radius * 2} step={0.001} digits={3} numStep={0.01} onCommit={(v) => vec('target', i, v)} />
        ))}
        <div className="field cam-head"><span><b>Position</b></span></div>
        {(['X', 'Y', 'Z'] as const).map((n, i) => (
          <Row key={'p' + n} label={n} unit="m" value={orbit.position[i]} min={rg.center[i] - rg.dMax} max={rg.center[i] + rg.dMax} step={0.001} digits={3} numStep={0.01} onCommit={(v) => vec('position', i, v)} />
        ))}
        <div className="field cam-footer">
          <button className="btn small" onClick={copy} title="Copy the camera as JSON">Copy</button>
          <button className="btn small" onClick={paste} title="Apply a camera JSON from the clipboard">Paste</button>
          {msg && <span className="muted">{msg}</span>}
        </div>
        <div className="field cam-head"><span className="muted">Shift + ← / → turns 45°, Shift + ↑ / ↓ tilts 45°.</span></div>
      </Accordion>
    </div>
  );
}
