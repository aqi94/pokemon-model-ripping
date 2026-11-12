import { useEffect, type ReactNode } from 'react';

/** Panel that opens beside a toolbox button (fixed, so the toolbox's overflow does not clip it).
 *  Stays open while the user orbits the model (live values); closes with ×, Escape or its toolbox button. */
export function Flyout({ title, top, onClose, children, className }: {
  title: string; top: number; onClose: () => void; children: ReactNode; className?: string;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT')) return;
      onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return (
    <div className={'export-panel flyout' + (className ? ' ' + className : '')} style={{ top, maxHeight: `calc(100vh - ${top + 8}px)` }} role="dialog" aria-label={title}>
      <div className="export-head"><b>{title}</b><button className="tool-btn" title="Close (Esc)" onClick={onClose}>×</button></div>
      {children}
    </div>
  );
}
