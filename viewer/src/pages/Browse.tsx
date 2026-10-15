import { useEffect, useMemo, useState } from 'react';
import type { ModelIndex } from '../types';

const KINDS = [['all', 'All'], ['pokemon', 'Pokémon'], ['trainer', 'Trainers'], ['object', 'Items']] as const;

export function Browse() {
  const [index, setIndex] = useState<ModelIndex | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [q, setQ] = useState('');
  const [kind, setKind] = useState<string>('all');
  const [game, setGame] = useState<string>('all');

  useEffect(() => {
    document.title = 'Pokémon Model Viewer';
    fetch('/models/index.json').then((r) => r.json()).then(setIndex).catch((e) => setErr(String(e)));
  }, []);

  const models = useMemo(() => {
    const f = q.trim().toLowerCase();
    return (index?.models ?? []).filter(
      (m) => (kind === 'all' ? m.kind !== 'object' : m.kind === kind) && (game === 'all' || m.game === game) && (!f || m.displayName.toLowerCase().includes(f) || m.id.includes(f) || String(m.dex ?? '').includes(f)),
    );
  }, [index, q, kind, game]);

  return (
    <div className="browse">
      <header>
        <h1>Pokémon Model Viewer</h1>
        <input className="text" placeholder="Search by name, id or dex #..." value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="tabs">
          {KINDS.map(([k, label]) => (
            <button key={k} className={'btn small' + (kind === k ? ' primary' : '')} onClick={() => setKind(k)}>{label}</button>
          ))}
        </div>
        {index && Object.keys(index.games).length > 1 && (
          <div className="tabs">
            {([['all', 'All games'], ...Object.entries(index.games)] as [string, string][]).map(([g, label]) => (
              <button key={g} className={'btn small' + (game === g ? ' primary' : '')} onClick={() => setGame(g)}>{label}</button>
            ))}
          </div>
        )}
      </header>
      {err && <div className="overlay err">{err}</div>}
      <div className="grid">
        {models.map((m) => (
          <a key={m.game + m.id} className="tile" href={`/model-viewer?id=${m.game}/${m.id}`}>
            <div className="thumb">
              {m.thumbnail ? (
                <img src={`/models/${m.thumbnail}`} alt={m.displayName} loading="lazy" />
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
    </div>
  );
}
