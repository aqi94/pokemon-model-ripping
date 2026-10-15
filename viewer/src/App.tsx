import { useCallback, useEffect, useState } from 'react';
import { Browse } from './pages/Browse';
import { Viewer } from './pages/Viewer';
import type { IndexModel } from './types';

// Single page: the viewer stays mounted and switches models in place (history.pushState,
// so back/forward and links still work). "/" opens the last viewed model; the grid lives at /browse.
type Route = { view: 'browse' } | { view: 'viewer'; game?: string; id: string };

const LAST = 'lastModel';
const DEFAULT_MODEL = 'za/tr0013_00';

function parse(): Route {
  const path = window.location.pathname.replace(/\/+$/, '') || '/';
  const params = new URLSearchParams(window.location.search);
  if (path === '/browse') return { view: 'browse' };
  const m = path.match(/^\/pokemon\/([^/]+)$/);
  if (m) return { view: 'viewer', id: decodeURIComponent(m[1]) };
  let raw = params.get('id') ?? '';
  if (!raw) { try { raw = localStorage.getItem(LAST) || DEFAULT_MODEL; } catch { raw = DEFAULT_MODEL; } }
  const [a, b] = raw.includes('/') ? raw.split('/') : [undefined, raw];
  return { view: 'viewer', game: a, id: b ?? '' };
}

export function App() {
  const [route, setRoute] = useState<Route>(parse);

  useEffect(() => {
    const onPop = () => setRoute(parse());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => {
    if (route.view !== 'viewer') return;
    try { localStorage.setItem(LAST, `${route.game ? route.game + '/' : ''}${route.id}`); } catch { /* ignore */ }
  }, [route]);

  const select = useCallback((m: IndexModel) => {
    const url = `/model-viewer?id=${m.game}/${m.id}`;
    if (window.location.pathname + window.location.search !== url) window.history.pushState(null, '', url);
    setRoute({ view: 'viewer', game: m.game, id: m.id });
  }, []);

  if (route.view === 'browse') return <Browse />;
  return <Viewer game={route.game} id={route.id} onSelect={select} />;
}
