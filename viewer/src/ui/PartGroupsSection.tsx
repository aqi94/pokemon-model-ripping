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
                <div className="pal" key={`${p.materials[0]}:${layer}`}>
                  <div className="label">{p.label}{p.layers.length > 1 ? ` colour ${layer}` : ' colour'}</div>
                  <div className="zones">
                    {p.zones.map((hex, z) => (
                      <button
                        key={z}
                        className={'zone' + (cur[layer - 1] === z + 1 ? ' active' : '')}
                        style={{ background: hex }}
                        title={`Palette colour ${z + 1}`}
                        aria-label={`${p.label} colour ${layer}: palette ${z + 1}`}
                        onClick={() => onPalette(p, layer, z + 1)}
                      />
                    ))}
                  </div>
                </div>
              ));
            })}
          </div>
        );
      })}
    </Accordion>
  );
}
