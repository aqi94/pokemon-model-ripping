import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type { GameIndex, IndexModel } from '../types';
import { modelAsset } from '../data';

const KINDS = [['all', 'All'], ['pokemon', 'Pokémon'], ['trainer', 'Trainers'], ['object', 'Items']] as const;
const ROW = 34;
const OVERSCAN = 6;

interface Props {
  index: GameIndex | null;
  id: string;
  onSelect: (m: IndexModel) => void;
}

/** Model switcher for the current game: search + kind filter + windowed list (only visible rows are mounted). */
export function ModelPicker({ index, id, onSelect }: Props) {
  const [q, setQ] = useState('');
  const [kind, setKind] = useState<string>('all');
  const listRef = useRef<HTMLDivElement>(null);
  const [top, setTop] = useState(0);
  const [height, setHeight] = useState(400);

  const models = useMemo(() => {
    const f = q.trim().toLowerCase();
    return (index?.models ?? []).filter(
      (m) => (kind === 'all' || m.kind === kind)
        && (!f || m.displayName.toLowerCase().includes(f) || m.id.includes(f) || String(m.dex ?? '').includes(f)),
    );
  }, [index, q, kind]);

  const cur = models.findIndex((m) => m.id === id);

  useLayoutEffect(() => {
    const el = listRef.current; if (!el) return;
    const ro = new ResizeObserver(() => setHeight(el.clientHeight));
    ro.observe(el); setHeight(el.clientHeight);
    return () => ro.disconnect();
  }, []);

  // Keep the current model in view when it changes (prev/next, back button, first load).
  useEffect(() => {
    const el = listRef.current; if (!el || cur < 0) return;
    const y = cur * ROW;
    if (y < el.scrollTop || y + ROW > el.scrollTop + el.clientHeight) el.scrollTop = Math.max(0, y - el.clientHeight / 2);
  }, [id, cur === -1, index]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = (d: number) => {
    if (!models.length) return;
    onSelect(models[cur < 0 ? (d > 0 ? 0 : models.length - 1) : (cur + d + models.length) % models.length]);
  };

  const first = Math.max(0, Math.floor(top / ROW) - OVERSCAN);
  const last = Math.min(models.length, Math.ceil((top + height) / ROW) + OVERSCAN);

  return (
    <div className="picker">
      <input className="text" placeholder="Search name, id or dex #..." value={q} onChange={(e) => setQ(e.target.value)} />
      <div className="row wrap" style={{ marginTop: 6 }}>
        {KINDS.map(([k, label]) => (
          <button key={k} className={'btn small' + (kind === k ? ' primary' : '')} onClick={() => setKind(k)}>{label}</button>
        ))}
      </div>
      <div className="row" style={{ marginTop: 6 }}>
        <button className="btn small" onClick={() => step(-1)} title="Previous model">&#9664;</button>
        <span className="dim" style={{ flex: 1, textAlign: 'center' }}>{cur >= 0 ? `${cur + 1} / ${models.length}` : `${models.length} models`}</span>
        <button className="btn small" onClick={() => step(1)} title="Next model">&#9654;</button>
      </div>
      <div className="anim-list model-list" ref={listRef} onScroll={(e) => setTop(e.currentTarget.scrollTop)}>
        <div style={{ height: models.length * ROW, position: 'relative' }}>
          {models.slice(first, last).map((m, i) => (
            <button key={m.id} style={{ position: 'absolute', top: (first + i) * ROW, height: ROW, left: 0, right: 0 }}
              className={'anim-item model-item' + (m.id === id ? ' active' : '')} onClick={() => onSelect(m)}>
              <span className="model-thumb">{m.thumbnail ? <img src={modelAsset(m.thumbnail)} alt="" loading="lazy" decoding="async" /> : null}</span>
              <span className="model-name">{m.displayName}</span>
              <span className="dim">{m.kind === 'pokemon' ? '' : m.kind}</span>
            </button>
          ))}
        </div>
        {index && models.length === 0 && <div className="dim pad">No models match.</div>}
      </div>
    </div>
  );
}
