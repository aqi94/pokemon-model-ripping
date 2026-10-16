import { useCallback, useEffect, useState } from 'react';
import { Browse } from './pages/Browse';
import { Games } from './pages/Games';
import { Viewer } from './pages/Viewer';
import type { IndexModel } from './types';

// Each game is a domain: `/` lists games, `/<game>` is that game's grid, `/<game>/<id>` the viewer.
// The viewer stays mounted and switches models in place (history.pushState). Legacy URLs
// (`/model-viewer?id=game/id`, `/pokemon/<id>`, `/browse`) still resolve; tools/*.mjs use them.
type Route =
  | { view: 'games' }
  | { view: 'browse'; game: string }
  | { view: 'viewer'; game: string; id: string };

const RESERVED = new Set(['browse', 'model-viewer', 'pokemon']);
const LAST = (game: string) => `lastModel:${game}`;

function parse(): Route {
  const root = import.meta.env.BASE_URL.replace(/\/$/, '');
  const path = window.location.pathname.startsWith(root) ? window.location.pathname.slice(root.length) : window.location.pathname;
  const parts = path.split('/').filter(Boolean).map(decodeURIComponent);
  const params = new URLSearchParams(window.location.search);
  const [a, b] = parts;
  if (!a || a === 'browse') return { view: 'games' };
  if (a === 'model-viewer') {
    const raw = params.get('id') ?? '';
    const [g, id] = raw.includes('/') ? raw.split('/') : ['za', raw];
    return id ? { view: 'viewer', game: g, id } : { view: 'games' };
  }
  if (a === 'pokemon' && b) return { view: 'viewer', game: 'za', id: b };
  if (RESERVED.has(a)) return { view: 'games' };
  return b ? { view: 'viewer', game: a, id: b } : { view: 'browse', game: a };
}

export const modelUrl = (game: string, id: string) => `${import.meta.env.BASE_URL}${game}/${encodeURIComponent(id)}`;

export function App() {
  const [route, setRoute] = useState<Route>(parse);

  useEffect(() => {
    const onPop = () => setRoute(parse());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => {
    if (route.view !== 'viewer') return;
    try { localStorage.setItem(LAST(route.game), route.id); } catch { /* ignore */ }
  }, [route]);

  const select = useCallback((m: IndexModel) => {
    const url = modelUrl(m.game, m.id);
    if (window.location.pathname !== url) window.history.pushState(null, '', url);
    setRoute({ view: 'viewer', game: m.game, id: m.id });
  }, []);

  if (route.view === 'games') return <Games />;
  if (route.view === 'browse') return <Browse game={route.game} />;
  return <Viewer key={route.game} game={route.game} id={route.id} onSelect={select} />;
}
