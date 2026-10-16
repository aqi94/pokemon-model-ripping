import { createPortal } from 'react-dom';
import { useEffect, useRef, useState } from 'react';
import type { PartGroup, PaletteEntry } from '../types';
import { Accordion } from './Accordion';

interface Props {
  groups: PartGroup[];
  selected: Record<string, string>;
  /** current palette zone per layer, keyed by the palette entry's first material */
  palettes: Record<string, number[]>;
  onSelect: (group: string, option: string) => void;
  onPalette: (entry: PaletteEntry, layer: number, zone: number) => void;
}

/** One colour layer: a single square showing the current zone; click to expand the full swatch grid. */
function PaletteRow({ entry: p, layer, current, onPalette }: {
  entry: PaletteEntry; layer: number; current: number; onPalette: Props['onPalette'];
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const popRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; right: number } | null>(null);
  const name = `${p.label}${p.layers.length > 1 ? ` colour ${layer}` : ' colour'}`;
  const toggle = () => {
    if (!open && btnRef.current) {
      const r = btnRef.current.getBoundingClientRect();
      setPos({ top: r.bottom + 4, right: window.innerWidth - r.right });
    }
    setOpen(!open);
  };
  useEffect(() => {
    if (!open) return;
    const down = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!ref.current?.contains(t) && !popRef.current?.contains(t)) setOpen(false);
    };
    const key = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    const close = () => setOpen(false);
    document.addEventListener('pointerdown', down);
    document.addEventListener('keydown', key);
    window.addEventListener('resize', close);
    document.addEventListener('scroll', close, true);
    return () => {
      document.removeEventListener('pointerdown', down); document.removeEventListener('keydown', key);
      window.removeEventListener('resize', close); document.removeEventListener('scroll', close, true);
    };
  }, [open]);
  return (
    <div className="pal" ref={ref}>
      <div className="pal-head">
        <div className="label">{name}</div>
        <button
          className="zone current"
          style={{ background: p.zones[current - 1] ?? 'var(--bg3)' }}
          title={`${name}: palette ${current} (click to change)`}
          aria-label={`${name}: current palette ${current}`}
          aria-haspopup="true"
          aria-expanded={open}
          ref={btnRef}
          onClick={toggle}
        />
      </div>
      {open && pos && createPortal(
        <div className="zones popover" role="dialog" aria-label={name} ref={popRef} style={{ top: pos.top, right: pos.right }}>
          {p.zones.map((hex, z) => (
            <button
              key={z}
              className={'zone' + (current === z + 1 ? ' active' : '')}
              style={{ background: hex }}
              title={`Palette colour ${z + 1}`}
              aria-label={`${name}: palette ${z + 1}`}
              onClick={() => { onPalette(p, layer, z + 1); setOpen(false); }}
            />
          ))}
        </div>,
        document.body,
      )}
    </div>
  );
}

/** Modular models (generic NPCs): exactly one option per group is shown; its colour layers pick palette zones. */
export function PartGroupsSection({ groups, selected, palettes, onSelect, onPalette }: Props) {
  return (
    <Accordion title="Appearance">
      {groups.map((g) => {
        const i = Math.max(0, g.options.findIndex((o) => o.id === selected[g.name]));
        const opt = g.options[i];
        const step = (d: number) => onSelect(g.name, g.options[(i + d + g.options.length) % g.options.length].id);
        return (
          <div className="var-group" key={g.name}>
            <div className="label">{g.label} <span className="dim">({i + 1}/{g.options.length})</span></div>
            <div className="row">
              <button className="btn small" onClick={() => step(-1)} aria-label={`Previous ${g.label}`}>&#9664;</button>
              <select className="text" value={opt.id} onChange={(e) => onSelect(g.name, e.target.value)} aria-label={g.label}>
                {g.options.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
              </select>
              <button className="btn small" onClick={() => step(1)} aria-label={`Next ${g.label}`}>&#9654;</button>
            </div>
            {(opt.palettes ?? []).filter((p) => p.divide > 1).map((p) => {
              const cur = palettes[p.materials[0]] ?? p.indices;
              return p.layers.map((layer) => (
                <PaletteRow key={`${p.materials[0]}:${layer}`} entry={p} layer={layer} current={cur[layer - 1]} onPalette={onPalette} />
              ));
            })}
          </div>
        );
      })}
    </Accordion>
  );
}
