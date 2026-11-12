import { useEffect, useState } from 'react';
import type { LooksEntry } from '../scene/savedLooks';
import { Accordion } from './Accordion';

interface Props {
  entry: LooksEntry;
  /** where saves go, shown as a hint (`out/za/cc_dm/configs.json` or "this browser") */
  where: string;
  /** name of the look applied last (selected in the dropdown) */
  current: string | null;
  canRevert: boolean;
  onApply: (name: string) => void;
  onSave: (name: string, withLighting: boolean) => void;
  onDelete: (name: string) => void;
  onDefault: (name: string | null) => void;
  onRevert: () => void;
}

const icon = { width: 14, height: 14, viewBox: '0 0 16 16', fill: 'none', stroke: 'currentColor', strokeWidth: 1.4, strokeLinejoin: 'round', strokeLinecap: 'round', 'aria-hidden': true } as const;
const FloppyIcon = () => <svg {...icon}><path d="M2.5 2.5h9l2 2v9h-11z" /><path d="M5 2.5v3.5h5V2.5M5 13.5V9h6v4.5" /></svg>;
const TrashIcon = () => <svg {...icon}><path d="M2.5 4h11M6 4V2.5h4V4M4 4l.7 9.5h6.6L12 4M6.8 6.5v4.5M9.2 6.5v4.5" /></svg>;
const StarIcon = ({ on }: { on: boolean }) => <svg {...icon} fill={on ? 'currentColor' : 'none'}><path d="M8 1.8l1.9 3.9 4.3.6-3.1 3 .7 4.3L8 11.6l-3.8 2 .7-4.3-3.1-3 4.3-.6z" /></svg>;

/** Appearance tab > Saved looks: named appearance + held item + animation + camera per model; the starred one auto-loads. */
export function SavedLooksSection({ entry, where, current, canRevert, onApply, onSave, onDelete, onDefault, onRevert }: Props) {
  const [name, setName] = useState('');
  const sel = entry.looks.find((l) => l.name === current) ?? null;
  const [withLight, setWithLight] = useState(false);
  useEffect(() => { setWithLight(!!sel?.lighting); }, [sel?.name, sel?.lighting]); // eslint-disable-line react-hooks/exhaustive-deps
  const save = () => { if (name.trim()) { onSave(name.trim(), withLight); setName(''); } };
  return (
    <Accordion title="Saved looks">
    <div className="var-group saved-looks">
      <div className="save-preset" style={{ marginBottom: 6 }}>
        <select className="text" value={sel?.name ?? ''} aria-label="Saved look" disabled={!entry.looks.length}
          onChange={(e) => { if (e.target.value) onApply(e.target.value); }}>
          <option value="">{entry.looks.length ? '(choose a look)' : '(none saved)'}</option>
          {entry.looks.map((l) => <option key={l.name} value={l.name}>{l.default ? '★ ' : ''}{l.name}</option>)}
        </select>
        {sel && <button className="btn small" onClick={() => onDefault(sel.default ? null : sel.name)} aria-pressed={!!sel.default}
          title={sel.default ? 'Default look (auto-loads when the model opens): click to unset' : 'Make this the default look (auto-loads when the model opens)'}><StarIcon on={!!sel.default} /></button>}
        {sel && <button className="btn small" onClick={() => onSave(sel.name, withLight)} title={`Overwrite "${sel.name}" with the current state`}><FloppyIcon /></button>}
        {sel && <button className="btn small" onClick={() => onDelete(sel.name)} title={`Delete "${sel.name}"`}><TrashIcon /></button>}
      </div>
      <div className="save-preset" style={{ marginBottom: 6 }}>
        <input className="text" placeholder="Save current as..." value={name} maxLength={60} onChange={(e) => setName(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') save(); }} aria-label="New look name" />
        <button className="btn small" disabled={!name.trim()} onClick={save} title="Save the appearance, held item, animation and camera (same name overwrites)"><FloppyIcon /></button>
        <button className="btn small" disabled={!canRevert} onClick={onRevert} title="Undo the last applied look">Revert</button>
      </div>
      <label className="check" title="Also store the current lighting in the look (applied as the environment when the look loads)">
        <input type="checkbox" checked={withLight} onChange={(e) => setWithLight(e.target.checked)} /> Include lighting
      </label>
      <div className="hint dim" style={{ fontSize: 11 }}>Saved to {where}</div>
    </div>
    </Accordion>
  );
}
