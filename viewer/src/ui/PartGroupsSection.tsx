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
  /** selected skin tone (1-based zone), for palettes whose layer 1 is the skin */
  skin: number;
  /** node visibility (Model Parts state) and its setter, for the Facial hair toggle */
  partsVis: Record<string, boolean>;
  onTogglePart: (node: string, v: boolean) => void;
}

/** Colour tables repeat colours (schemes side by side, black padding cells): one swatch per distinct colour, first zone wins. */
function uniqueZones(zones: string[]): { hex: string; zones: number[] }[] {
  const out = new Map<string, number[]>();
  zones.forEach((hex, z) => {
    const k = hex.toLowerCase();
    const list = out.get(k);
    if (list) list.push(z + 1);
    else out.set(k, [z + 1]);
  });
  return [...out.entries()].map(([hex, zs]) => ({ hex, zones: zs }));
}

/** One colour layer: a single square showing the current zone; click to expand the full swatch grid. */
function PaletteRow({ entry: p, layer, current, skin, onPalette }: {
  entry: PaletteEntry; layer: number; current: number; skin: number; onPalette: Props['onPalette'];
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const popRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; right: number } | null>(null);
  const pairs = !!p.pairs;
  const offer = p.offer;
  const zoneName = (z: number) => (z === 0 ? p.noneLabel ?? 'None' : p.zoneLabels?.[z - 1] ?? `palette ${z}`);
  // Zone 0 shows as the skin colour (zone `skin` of the same table; Tone 3 until the skin is known).
  const zoneHex = (z: number) => p.zones[(z === 0 ? skin : z) - 1] ?? 'var(--bg3)';
  const name = `${p.label}${p.layers.length > 1 ? ` colour ${layer}` : ' colour'}`;
  // A pair is two zones (iris, inner iris) shown as one split swatch
  const pairBg = (z: number) => `linear-gradient(135deg, ${p.zones[z - 1]} 50%, ${p.zones[z] ?? p.zones[z - 1]} 50%)`;
  const pairIdx = (z: number) => (z - 2) / 2;
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
    // Scrolling the long swatch list itself must not close it (only the page scrolling away from the button does).
    const scroll = (e: Event) => { if (!popRef.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener('pointerdown', down);
    document.addEventListener('keydown', key);
    window.addEventListener('resize', close);
    document.addEventListener('scroll', scroll, true);
    return () => {
      document.removeEventListener('pointerdown', down); document.removeEventListener('keydown', key);
      window.removeEventListener('resize', close); document.removeEventListener('scroll', scroll, true);
    };
  }, [open]);
  return (
    <div className="pal" ref={ref}>
      <div className="pal-head">
        <div className="label">{name}{p.custom && <span className="dim custom-tag" title="Not a colour the model shipped with: borrowed from other NPC faces"> (custom)</span>}</div>
        <button
          className="zone current"
          style={{ background: pairs ? pairBg(current) : offer ? zoneHex(current) : (p.zones[current - 1] ?? 'var(--bg3)') }}
          title={offer ? `${name}: ${zoneName(current)}${current === 0 || current === offer[1] ? '' : ' (custom, not from this model)'} (click to change)` : pairs ? `${name}: ${p.pairLabels?.[pairIdx(current)] ?? 'custom'}${pairIdx(current) > 0 ? ' (custom, not from this model)' : ''} (click to change)` : p.zoneLabels ? `${name}: ${p.zoneLabels[current - 1]}${current > 2 ? ' (custom, not from this model)' : ''} (click to change)` : `${name}: palette ${current} (click to change)`}
          aria-label={`${name}: current palette ${current}`}
          aria-haspopup="true"
          aria-expanded={open}
          ref={btnRef}
          onClick={toggle}
        />
      </div>
      {open && pos && createPortal(
        <div className="zones popover" role="dialog" aria-label={name} ref={popRef} style={{ top: pos.top, right: pos.right }}>
          {offer ? offer.map((z) => (
            <button
              key={z}
              className={'zone' + (z === current ? ' active' : '')}
              style={{ background: zoneHex(z) }}
              title={z === 0 || z === offer[1] ? zoneName(z) : `Custom: ${zoneName(z)} (not from this model)`}
              aria-label={`${name}: ${zoneName(z)}`}
              onClick={() => { onPalette(p, layer, z); setOpen(false); }}
            />
          )) : pairs ? Array.from({ length: Math.floor((p.zones.length - 1) / 2) }, (_, j) => 2 * j + 2).map((z) => (
            <button
              key={z}
              className={'zone' + (z === current ? ' active' : '')}
              style={{ background: pairBg(z) }}
              title={z === 2 ? 'Original eye colour of this model' : `Custom: ${p.pairLabels?.[pairIdx(z)] ?? 'eye colour'} (not from this model)`}
              aria-label={`${name}: ${p.pairLabels?.[pairIdx(z)] ?? z}`}
              onClick={() => { onPalette(p, layer, z); setOpen(false); }}
            />
          )) : p.custom && p.zoneLabels ? p.zones.map((_, i) => i + 1).slice(1).map((z) => (
            <button
              key={z}
              className={'zone' + (z === current ? ' active' : '')}
              style={{ background: p.zones[z - 1] }}
              title={z === 2 ? 'Original colour of this model' : `Custom: ${p.zoneLabels![z - 1]} (not from this model)`}
              aria-label={`${name}: ${p.zoneLabels![z - 1]}`}
              onClick={() => { onPalette(p, layer, z); setOpen(false); }}
            />
          )) : uniqueZones(p.zones).map(({ hex, zones }) => (
            <button
              key={zones[0]}
              className={'zone' + (hex === p.zones[current - 1]?.toLowerCase() ? ' active' : '')}
              style={{ background: hex }}
              title={p.own && zones[0] > p.own ? `Custom: ${p.zoneLabels?.[zones[0] - 1] ?? `colour ${zones[0]}`} (not from this model)` : `Palette colour ${zones[0]}${zones.length > 1 ? ` (also ${zones.slice(1).join(', ')})` : ''}`}
              aria-label={`${name}: palette ${zones[0]}`}
              onClick={() => { onPalette(p, layer, zones[0]); setOpen(false); }}
            />
          ))}
        </div>,
        document.body,
      )}
    </div>
  );
}

/** Modular models (generic NPCs): exactly one option per group is shown; its colour layers pick palette zones. */
export function PartGroupsSection({ groups, selected, palettes, onSelect, onPalette, skin, partsVis, onTogglePart }: Props) {
  return (
    <Accordion title="Appearance">
      {groups.map((g) => {
        const i = Math.max(0, g.options.findIndex((o) => o.id === selected[g.name]));
        const opt = g.options[i];
        const step = (d: number) => onSelect(g.name, g.options[(i + d + g.options.length) % g.options.length].id);
        return (
          <div className="var-group" key={g.name}>
            <div className="label">{g.label} <span className="dim">({i + 1}/{g.options.length})</span></div>
            {g.options.length > 1 && <div className="row">
              <button className="btn small" onClick={() => step(-1)} aria-label={`Previous ${g.label}`}>&#9664;</button>
              <select className="text" value={opt.id} onChange={(e) => onSelect(g.name, e.target.value)} aria-label={g.label}>
                {g.options.map((o) => <option key={o.id} value={o.id}>{o.label}</option>)}
              </select>
              <button className="btn small" onClick={() => step(1)} aria-label={`Next ${g.label}`}>&#9654;</button>
            </div>}
            {opt.facialHair && (
              <label className="check">
                <input type="checkbox" checked={opt.facialHair.every((n) => !!partsVis[n])} onChange={(e) => opt.facialHair!.forEach((n) => onTogglePart(n, e.target.checked))} />
                <span>Facial hair</span>
              </label>
            )}
            {(opt.palettes ?? []).filter((p) => p.divide > 1).map((p) => {
              const cur = palettes[p.materials[0]] ?? p.indices;
              return p.layers.map((layer) => (
                <PaletteRow key={`${p.materials[0]}:${layer}`} entry={p} layer={layer} current={cur[layer - 1]} skin={skin} onPalette={onPalette} />
              ));
            })}
          </div>
        );
      })}
    </Accordion>
  );
}
