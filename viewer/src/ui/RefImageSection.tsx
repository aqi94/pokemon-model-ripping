import { useEffect, useRef, useState } from 'react';
import { DEFAULT_TRANSFORM, listSamples, sampleUrl, type RefImage, type RefPin, type RefSettings, type RefTransform, type SampleFile } from '../scene/refImage';
import { pinColor } from './RefImageLayer';

export interface SolveSummary { rmsPx: number; perPinPx: Record<number, number>; fov: number; roll: number }

export interface AlignProps {
  pins: RefPin[];
  placing: boolean;
  lockFov: boolean;
  /** last solve: RMS + per-pin error in px (null = not solved since the pins changed) */
  result: SolveSummary | null;
  error: string | null;
  onPlacing: (v: boolean) => void;
  onLockFov: (v: boolean) => void;
  onRemove: (id: number) => void;
  onClear: () => void;
  onSolve: () => void;
  /** the reference is an uncropped screenshot: solve against it directly and place it (centred, canvas height) */
  fullFrame: boolean;
  onFullFrame: (v: boolean) => void;
  solveRoll: boolean;
  onSolveRoll: (v: boolean) => void;
  pivotId: number | null;
  onPivot: (id: number) => void;
  fine: boolean;
  onFine: (v: boolean) => void;
  /** frame search progress (null = idle) */
  search: string | null;
  onFindFrame: (allClips: boolean) => void;
  onCancelSearch: () => void;
  /** the camera follows the solved lens (and the reference image if it is moved) */
  locked: boolean;
  onRelease: () => void;
}

interface Props {
  image: RefImage | null;
  settings: RefSettings;
  transforming: boolean;
  onSettings: (patch: Partial<RefSettings>) => void;
  onTransform: (t: RefTransform) => void;
  onTransforming: (v: boolean) => void;
  onPickSample: (path: string) => void;
  onFile: (f: File) => void;
  onClear: () => void;
  /** scale that makes the image's height fill the stage (null before the stage is measured) */
  fillHeightScale: () => number | null;
  align: AlignProps;
  light: AutoLightProps;
}

export interface AutoLightProps {
  busy: string | null;
  result: { presetLabel: string | null; before: number; after: number; dEBefore: number; dEAfter: number; shift: [number, number]; seconds: number } | null;
  error: string | null;
  scanPresets: boolean;
  register: boolean;
  excluding: boolean;
  excludeCount: number;
  canUndo: boolean;
  onRun: () => void;
  onCancel: () => void;
  onUndo: () => void;
  onScanPresets: (v: boolean) => void;
  onRegister: (v: boolean) => void;
  onExcluding: (v: boolean) => void;
  onClearExcludes: () => void;
}

const num = (v: number, d = 3) => Number(v.toFixed(d));

/** Reference image controls: load (samples folder when running locally, upload / drag-drop always), layer, opacity, transform. */
export function RefImageSection({ image, settings, transforming, onSettings, onTransform, onTransforming, onPickSample, onFile, onClear, fillHeightScale, align, light }: Props) {
  const [samples, setSamples] = useState<SampleFile[] | null | undefined>(undefined);
  const [filter, setFilter] = useState('');
  const [showList, setShowList] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const refresh = () => { void listSamples().then(setSamples); };
  useEffect(refresh, []);
  const t = settings.transform;
  const setT = (patch: Partial<RefTransform>) => onTransform({ ...t, ...patch });
  const list = (samples ?? []).filter((s) => s.path.toLowerCase().includes(filter.toLowerCase()));

  return (
    <div className="pane refimg-section">
      <div className="field">
        <span><b>Reference image</b>{image ? <span className="muted"> {image.name}</span> : null}</span>
        {image && <button className="btn small" onClick={onClear}>Clear</button>}
      </div>
      <div className="row wrap">
        {samples && <button className={'btn small' + (showList ? ' primary' : '')} onClick={() => setShowList((v) => !v)}>Samples folder…</button>}
        <button className="btn small" onClick={() => fileInput.current?.click()}>Upload…</button>
        <input ref={fileInput} type="file" accept="image/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ''; }} />
      </div>
      <div className="muted hint">{samples === null ? 'Upload or drop an image onto the canvas (kept for this session only).' : 'Or drop an image onto the canvas.'}</div>

      {samples && showList && (
        <div className="refimg-samples">
          <div className="row">
            <input className="text" placeholder={`Filter ${samples.length} images`} value={filter} onChange={(e) => setFilter(e.target.value)} />
            <button className="btn small" style={{ flex: 'none' }} title="Rescan the folder" onClick={refresh}>↻</button>
          </div>
          <div className="reflist">
            {list.map((s) => (
              <button key={s.path} className={'refrow' + (image?.samplePath === s.path ? ' active' : '')} onClick={() => onPickSample(s.path)} title={s.path}>
                <img src={sampleUrl(s.path)} alt="" loading="lazy" />
                <span className="reftext">{s.path}</span>
              </button>
            ))}
            {!list.length && <div className="muted">No images match.</div>}
          </div>
        </div>
      )}

      {image && (
        <>
          <label className="field"><span>Show</span><input type="checkbox" checked={settings.visible} onChange={(e) => onSettings({ visible: e.target.checked })} /></label>
          <div className="field">
            <span>Layer</span>
            <span className="row" style={{ marginTop: 0 }}>
              <button className={'btn small' + (settings.layer === 'background' ? ' primary' : '')} onClick={() => onSettings({ layer: 'background' })}>Behind model</button>
              <button className={'btn small' + (settings.layer === 'overlay' ? ' primary' : '')} onClick={() => onSettings({ layer: 'overlay' })}>Over model</button>
            </span>
          </div>
          <label className="field"><span>Opacity {settings.opacity.toFixed(2)}</span>
            <input type="range" min={0} max={1} step={0.01} value={settings.opacity} onChange={(e) => onSettings({ opacity: Number(e.target.value) })} /></label>
          <div className="field">
            <span>Transform <span className="muted">(R)</span></span>
            <button className={'btn small' + (transforming ? ' primary' : '')} onClick={() => onTransforming(!transforming)}>{transforming ? 'Done' : 'Edit'}</button>
          </div>
          <div className="refimg-grid">
            <label>X <input type="number" step={0.01} value={num(t.x)} onChange={(e) => setT({ x: Number(e.target.value) })} /></label>
            <label>Y <input type="number" step={0.01} value={num(t.y)} onChange={(e) => setT({ y: Number(e.target.value) })} /></label>
            <label>Scale <input type="number" step={0.01} min={0.02} value={num(t.scale)} onChange={(e) => setT({ scale: Math.max(0.02, Number(e.target.value) || 1) })} /></label>
            <label>Rotate° <input type="number" step={1} value={num(t.rotation, 1)} onChange={(e) => setT({ rotation: Number(e.target.value) })} /></label>
          </div>
          <div className="row wrap">
            <button className="btn small" onClick={() => setT({ x: 0.5, y: 0.5, scale: 1 })} title="Fit inside the canvas, centred">Fit</button>
            <button className="btn small" onClick={() => { const s = fillHeightScale(); if (s) setT({ y: 0.5, scale: s }); }} title="Image height = canvas height">Fill height</button>
            <button className={'btn small' + (t.flipX ? ' primary' : '')} onClick={() => setT({ flipX: !t.flipX })}>Flip H</button>
            <button className="btn small" onClick={() => onTransform({ ...DEFAULT_TRANSFORM })}>Reset</button>
          </div>
          {transforming && <div className="muted hint">The camera is locked while transforming. Arrow keys nudge (Shift = 10 px).</div>}
          <AlignBlock {...align} />
          <AutoLightBlock {...light} />
        </>
      )}
    </div>
  );
}

/** Camera auto-align: pins pair a point on the reference with a point on the model; Solve finds the camera. */
function AlignBlock(a: AlignProps) {
  const { pins, placing, lockFov, result, error, onPlacing, onLockFov, onRemove, onClear, onSolve } = a;
  const done = pins.filter((p) => p.anchor).length;
  const fovLocked = lockFov || done < 4;
  const busy = a.search !== null;
  return (
    <div className="refimg-align">
      <div className="field">
        <span><b>Align camera</b> <span className="muted">(A)</span></span>
        <button className={'btn small' + (placing ? ' primary' : '')} onClick={() => onPlacing(!placing)}>{placing ? 'Done' : 'Place pins'}</button>
      </div>
      <div className="muted hint">1. Pick the clip that matches the image (playback pauses while pinning). 2. Click a feature on the image, then the same spot on the model; 5+ pins on clear points (eye corners, nose, chin, ear, collar) spread over the subject. 3. Solve, or Find best frame if the pose is not exact.</div>
      {pins.length > 0 && (
        <div className="pinlist">
          {pins.map((p, i) => (
            <div className="pinrow" key={p.id}>
              <span className="dot" style={{ background: pinColor(i) }} />
              <span className="grow">Pin {i + 1}{!p.anchor && <span className="muted"> (click the model)</span>}</span>
              {result && result.perPinPx[p.id] !== undefined && <span className="muted">{result.perPinPx[p.id].toFixed(1)} px</span>}
              {p.anchor && (
                <button className={'btn small' + (a.pivotId === p.id ? ' primary' : '')} style={{ flex: 'none' }}
                  title="Pivot: orbit and zoom around this pin while it stays on its spot in the image" onClick={() => a.onPivot(p.id)}>⌖</button>
              )}
              <button className="btn small" style={{ flex: 'none' }} title="Remove pin" onClick={() => onRemove(p.id)}>×</button>
            </div>
          ))}
        </div>
      )}
      <label className="field" title="The image is a whole, uncropped screenshot: its centre is the lens centre, so the solve is relative to the image wherever it sits (the camera gets a matching lens shift and follows the image if you move it later). Untick for crops: the solve is then relative to the canvas.">
        <span>Uncropped screenshot</span><input type="checkbox" checked={a.fullFrame} onChange={(e) => a.onFullFrame(e.target.checked)} /></label>
      <label className="field" title="Camera tilt about the view axis (dutch angle), applied to the 3D camera.">
        <span>Solve roll</span><input type="checkbox" checked={a.solveRoll} onChange={(e) => a.onSolveRoll(e.target.checked)} /></label>
      <label className="field" title="Unlocked, the solver also adjusts the FOV (kept between 8° and 60°, pulled toward the current one). Needs well-placed pins."><span>Lock FOV{done < 4 ? <span className="muted"> (4+ pins to unlock)</span> : null}</span>
        <input type="checkbox" checked={fovLocked} disabled={done < 4} onChange={(e) => onLockFov(e.target.checked)} /></label>
      <label className="field" title="Slower orbit / zoom / pan for hand adjustment. Combine with a pivot pin (⌖) so the view turns around that point.">
        <span>Fine camera controls</span><input type="checkbox" checked={a.fine} onChange={(e) => a.onFine(e.target.checked)} /></label>
      <div className="row wrap">
        <button className="btn small primary" disabled={done < 3 || busy} onClick={onSolve} title="Find the camera that lines the pins up (Enter while placing)">Solve camera</button>
        <button className="btn small" disabled={done < 3 || busy} onClick={() => a.onFindFrame(false)} title="Solve at every frame of the current clip and keep the frame that fits best">Find best frame</button>
        <button className="btn small" disabled={done < 3 || busy} onClick={() => a.onFindFrame(true)} title="Same, over every clip (sampled), then refined">All clips</button>
        {pins.length > 0 && <button className="btn small" disabled={busy} onClick={onClear}>Clear pins</button>}
      </div>
      {busy && <div className="row"><span className="muted hint" style={{ flex: 1 }}>Searching {a.search}…</span><button className="btn small" style={{ flex: 'none' }} onClick={a.onCancelSearch}>Cancel</button></div>}
      {a.locked && (
        <div className="row"><span className="muted hint" style={{ flex: 1 }}>Camera lens locked to the reference (follows it if moved). Orbit/zoom still work; F or a view preset releases it.</span>
          <button className="btn small" style={{ flex: 'none' }} onClick={a.onRelease}>Release</button></div>
      )}
      {result && <div className="muted hint">RMS error {result.rmsPx.toFixed(1)} px, FOV {result.fov.toFixed(1)}°, roll {result.roll.toFixed(1)}°. A pin far above the rest is misplaced or sits on a part whose pose differs (hands, hair): remove it and solve again.</div>}
      {error && <div className="muted hint">{error}</div>}
    </div>
  );
}

/** Fit the lighting so the aligned model matches the reference's colours (scene/autoLight.ts). */
function AutoLightBlock(l: AutoLightProps) {
  const busy = l.busy !== null;
  return (
    <div className="refimg-align">
      <div className="field">
        <span><b>Auto lighting</b></span>
        <button className="btn small primary" disabled={busy} onClick={l.onRun} title="Align the camera first (Solve). Compares the model's colours with the reference per material and fits the lighting (exposure and tone mapping stay the preset's).">Match lighting</button>
      </div>
      <label className="field" title="Score every lighting preset first and start from the closest one. Off: start from the current lighting.">
        <span>Start from best preset</span><input type="checkbox" checked={l.scanPresets} onChange={(e) => l.onScanPresets(e.target.checked)} /></label>
      <label className="field" title="Slide the reference up to a few px to absorb what the camera solve could not.">
        <span>Correct small misalignment</span><input type="checkbox" checked={l.register} onChange={(e) => l.onRegister(e.target.checked)} /></label>
      <div className="field">
        <span>Ignore areas{l.excludeCount ? <span className="muted"> ({l.excludeCount})</span> : null}</span>
        <span className="row" style={{ marginTop: 0 }}>
          <button className={'btn small' + (l.excluding ? ' primary' : '')} onClick={() => l.onExcluding(!l.excluding)} title="Drag boxes on the canvas over dialogue boxes, other characters, effects...">{l.excluding ? 'Done' : 'Draw'}</button>
          {l.excludeCount > 0 && <button className="btn small" onClick={l.onClearExcludes}>Clear</button>}
        </span>
      </div>
      {busy && <div className="row"><span className="muted hint" style={{ flex: 1 }}>{l.busy}</span><button className="btn small" style={{ flex: 'none' }} onClick={l.onCancel}>Cancel</button></div>}
      {l.result && (
        <div className="row">
          <span className="muted hint" style={{ flex: 1 }}>
            {l.result.presetLabel ? <>Started from <b>{l.result.presetLabel}</b>. </> : 'Started from the current lighting. '}
            Colour error {l.result.dEBefore.toFixed(1)} → {l.result.dEAfter.toFixed(1)} (fit score {l.result.before.toFixed(1)} → {l.result.after.toFixed(1)}){l.result.shift[0] || l.result.shift[1] ? `, reference nudged ${l.result.shift[0]},${l.result.shift[1]} px for the comparison` : ''}; {l.result.seconds.toFixed(1)} s.
          </span>
          {l.canUndo && <button className="btn small" style={{ flex: 'none' }} onClick={l.onUndo}>Revert</button>}
        </div>
      )}
      {l.error && <div className="muted hint">{l.error}</div>}
    </div>
  );
}
