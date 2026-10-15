import { useEffect, useMemo, useRef, useState } from 'react';
import type { IndexModel, ModelIndex } from '../types';
import { Accordion } from './Accordion';

const KINDS = [['all', 'All'], ['pokemon', 'Pokémon'], ['trainer', 'Trainers'], ['object', 'Items']] as const;
const KIND_LABEL: Record<string, string> = { pokemon: 'Pokémon', trainer: 'Trainers', object: 'Items' };

interface Props {
  index: ModelIndex | null;
  game?: string;
  id: string;
  onSelect: (m: IndexModel) => void;
}

/** Sidebar model switcher: search + kind filter + list; prev/next step through the filtered list. */
export function ModelPicker({ index, game, id, onSelect }: Props) {
  const [q, setQ] = useState('');
  const [kind, setKind] = useState<string>('all');
  const listRef = useRef<HTMLDivElement>(null);

  const models = useMemo(() => {
    const f = q.trim().toLowerCase();
    return (index?.models ?? []).filter(
      (m) => (kind === 'all' || m.kind === kind)
        && (!f || m.displayName.toLowerCase().includes(f) || m.id.includes(f) || String(m.dex ?? '').includes(f)),
    );
  }, [index, q, kind]);

  const isCurrent = (m: IndexModel) => m.id === id && (!game || m.game === game);
  const cur = models.findIndex(isCurrent);

  // Keep the current model in view when it changes (e.g. via prev/next or browser back).
  useEffect(() => {
    listRef.current?.querySelector('.model-item.active')?.scrollIntoView({ block: 'nearest' });
  }, [id, game, models]);

  const step = (d: number) => {
    if (!models.length) return;
    const i = cur < 0 ? (d > 0 ? 0 : models.length - 1) : (cur + d + models.length) % models.length;
    onSelect(models[i]);
  };

  let lastKind = '';
  return (
    <Accordion title="Model">
      <input className="text" placeholder="Search name, id or dex #..." value={q} onChange={(e) => setQ(e.target.value)} />
      <div className="row wrap" style={{ marginTop: 8 }}>
        {KINDS.map(([k, label]) => (
          <button key={k} className={'btn small' + (kind === k ? ' primary' : '')} onClick={() => setKind(k)}>{label}</button>
        ))}
      </div>
      <div className="row" style={{ marginTop: 8 }}>
        <button className="btn small" onClick={() => step(-1)} title="Previous model">&#9664; Prev</button>
        <span className="dim" style={{ flex: 1, textAlign: 'center' }}>{cur >= 0 ? `${cur + 1} / ${models.length}` : `${models.length} models`}</span>
        <button className="btn small" onClick={() => step(1)} title="Next model">Next &#9654;</button>
      </div>
      <div className="anim-list model-list" ref={listRef}>
        {models.map((m) => {
          const head = kind === 'all' && m.kind !== lastKind ? (lastKind = m.kind) : null;
          return (
            <div key={m.game + '/' + m.id}>
              {head && <div className="anim-cat">{KIND_LABEL[head] ?? head}</div>}
              <button className={'anim-item model-item' + (isCurrent(m) ? ' active' : '')} onClick={() => onSelect(m)}>
                <span className="model-thumb">{m.thumbnail ? <img src={`/models/${m.thumbnail}`} alt="" loading="lazy" /> : null}</span>
                <span className="model-name">{m.displayName}</span>
                <span className="dim">{m.id}</span>
              </button>
            </div>
          );
        })}
        {index && models.length === 0 && <div className="dim pad">No models match.</div>}
      </div>
    </Accordion>
  );
}
