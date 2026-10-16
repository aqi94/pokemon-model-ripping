import { useState, type ReactNode } from 'react';

export interface TabDef { id: string; label: string; content: ReactNode; hidden?: boolean }

/** Tabbed panel group (right dock). The selected tab is remembered per `storageKey`. */
export function Tabs({ tabs, storageKey }: { tabs: TabDef[]; storageKey: string }) {
  const [sel, setSel] = useState(() => { try { return localStorage.getItem(storageKey) ?? ''; } catch { return ''; } });
  const vis = tabs.filter((t) => !t.hidden);
  const active = vis.find((t) => t.id === sel) ?? vis[0];
  return (
    <div className="tabgroup">
      <div className="tabbar" role="tablist">
        {vis.map((t) => (
          <button key={t.id} role="tab" aria-selected={t === active} className={'tab' + (t === active ? ' active' : '')}
            onClick={() => { setSel(t.id); try { localStorage.setItem(storageKey, t.id); } catch { /* ignore */ } }}>{t.label}</button>
        ))}
      </div>
      <div className="tabpane" role="tabpanel">{active?.content}</div>
    </div>
  );
}
