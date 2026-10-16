import type { RefEntry } from '../types';

export type RefMode = 'off' | 'overlay' | 'split';

interface Props {
  refs: RefEntry[];
  modelId: string;
  active: RefEntry | null;
  mode: RefMode;
  opacity: number;
  showAll: boolean;
  onPick: (r: RefEntry) => void;
  onMode: (m: RefMode) => void;
  onOpacity: (v: number) => void;
  onShowAll: (v: boolean) => void;
  onReapply: () => void;
}

/** Calibrated gameplay screenshots (refs/<game>/index.json): pick one to pose the model like it and show it beside / over the render. */
export function ReferencePanel({ refs, modelId, active, mode, opacity, showAll, onPick, onMode, onOpacity, onShowAll, onReapply }: Props) {
  const list = showAll ? refs : refs.filter((r) => r.model === modelId);
  return (
    <div className="pane">
      <div className="field">
        <span><b>Reference</b> {active ? `(${active.sceneLabel})` : ''}</span>
        <label className="muted"><input type="checkbox" checked={showAll} onChange={(e) => onShowAll(e.target.checked)} /> all models</label>
      </div>
      <div className="field">
        <span>Show</span>
        <span className="row">
          {(['off', 'overlay', 'split'] as RefMode[]).map((m) => (
            <button key={m} className={'btn small' + (mode === m ? ' primary' : '')} disabled={!active && m !== 'off'} onClick={() => onMode(m)}>
              {m === 'off' ? 'Off' : m === 'overlay' ? 'Overlay' : 'Side by side'}
            </button>
          ))}
        </span>
      </div>
      {mode === 'overlay' && (
        <label className="field"><span>Opacity {opacity.toFixed(2)}</span>
          <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => onOpacity(Number(e.target.value))} /></label>
      )}
      {active && <div className="field"><span className="muted">{active.clip} @ {active.frame}, preset {active.preset}{active.dE != null ? `, dE ${active.dE}` : ''}</span>
        <button className="btn small" onClick={onReapply}>Re-apply pose</button></div>}
      {!list.length && <div className="muted">No calibrated screenshots for this model{showAll ? '' : ' (tick "all models")'}.</div>}
      <div className="reflist">
        {list.map((r) => (
          <button key={`${r.scene}/${r.slug}`} className={'refrow' + (active && active.slug === r.slug && active.scene === r.scene ? ' active' : '')} onClick={() => onPick(r)} title={r.video ? `${r.video} @ ${r.timecode}` : r.slug}>
            <img src={r.image} alt="" loading="lazy" />
            <span className="reftext"><b>{r.sceneLabel}</b><br />{r.name} ({r.model}) · {r.context}{r.dE != null ? ` · dE ${r.dE}` : ''}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
