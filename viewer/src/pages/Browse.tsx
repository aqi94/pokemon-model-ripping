import { useEffect, useMemo, useRef, useState } from 'react';
import { loadGameIndex, modelAsset } from '../data';
import type { GameIndex } from '../types';

const KINDS = [['all', 'All'], ['pokemon', 'Pokémon'], ['trainer', 'Trainers'], ['object', 'Items']] as const;
const CHUNK = 120;

/** One game's thumbnail grid. Tiles render in chunks so a 900-model game does not mount every <img> at once. */
export function Browse({ game }: { game: string }) {
  const [index, setIndex] = useState<GameIndex | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState('');
  const [kind, setKind] = useState<string>('all');
  const [shown, setShown] = useState(CHUNK);
  const sentinel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadGameIndex(game).then((i) => { setIndex(i); document.title = `${i.name} - Pokémon Model Viewer`; }).catch((e) => setErr(String(e instanceof Error ? e.message : e)));
  }, [game]);

  const models = useMemo(() => {
    const f = q.trim().toLowerCase();
    return (index?.models ?? []).filter(
      (m) => (kind === 'all' ? m.kind !== 'object' : m.kind === kind) && (!f || m.displayName.toLowerCase().includes(f) || m.id.includes(f) || String(m.dex ?? '').includes(f)),
    );
  }, [index, q, kind]);

  useEffect(() => setShown(CHUNK), [q, kind, game]);
  useEffect(() => {
    const el = sentinel.current;
    if (!el) return;
    const io = new IntersectionObserver((e) => { if (e[0].isIntersecting) setShown((n) => n + CHUNK); }, { rootMargin: '600px' });
    io.observe(el);
    return () => io.disconnect();
  }, [models]);

  return (
    <div className="browse">
      <header>
        <a className="btn small" href={import.meta.env.BASE_URL}>&#9664; Games</a>
        <h1>{index?.name ?? game}</h1>
        <input className="text" placeholder="Search by name, id or dex #..." value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="tabs">
          {KINDS.map(([k, label]) => (
            <button key={k} className={'btn small' + (kind === k ? ' primary' : '')} onClick={() => setKind(k)}>{label}</button>
          ))}
        </div>
      </header>
      {err && <div className="overlay err">{err}</div>}
      <div className="grid">
        {models.slice(0, shown).map((m) => (
          <a key={m.id} className="tile" href={`${import.meta.env.BASE_URL}${m.game}/${encodeURIComponent(m.id)}`}>
            <div className="thumb">
              {m.thumbnail ? (
                <img src={modelAsset(m.thumbnail)} alt={m.displayName} loading="lazy" decoding="async" />
              ) : (
                <span className="ph">{m.dex != null ? '#' + String(m.dex).padStart(4, '0') : m.kind}</span>
              )}
            </div>
            <div className="meta">
              <strong>{m.displayName}</strong>
              <span className="dim">{m.id} · {m.clips} clips</span>
            </div>
          </a>
        ))}
        {index && models.length === 0 && <div className="dim pad">No models match.</div>}
      </div>
      <div ref={sentinel} style={{ height: 1 }} />
    </div>
  );
}
