import { useState, type ReactNode } from 'react';

/** Vertical split: `top` fills the rest and scrolls, `bottom` has a height set by dragging the divider (remembered per `storageKey`). */
export function SplitPane({ top, bottom, storageKey, initial = 220 }: { top: ReactNode; bottom: ReactNode; storageKey: string; initial?: number }) {
  const [h, setH] = useState(() => {
    try { const v = Number(localStorage.getItem(storageKey)); return v >= 60 ? v : initial; } catch { return initial; }
  });
  const startDrag = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    const startY = e.clientY, startH = h;
    const max = (e.currentTarget.parentElement?.clientHeight ?? 600) - 80;
    let cur = h;
    const move = (ev: PointerEvent) => { cur = Math.max(60, Math.min(max, startH + startY - ev.clientY)); setH(cur); };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      try { localStorage.setItem(storageKey, String(Math.round(cur))); } catch { /* ignore */ }
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };
  return (
    <div className="split">
      <div className="split-top">{top}</div>
      <div className="vhandle" onPointerDown={startDrag} role="separator" aria-orientation="horizontal" />
      <div className="split-bottom" style={{ height: h }}>{bottom}</div>
    </div>
  );
}
