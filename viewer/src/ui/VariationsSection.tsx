import type { VariantGroup } from '../types';
import { Accordion } from './Accordion';

interface Props {
  groups: VariantGroup[];
  selected: Record<string, number>;
  colors: Record<number, string | null>;
  onSelect: (group: string, variant: number) => void;
}

export function VariationsSection({ groups, selected, colors, onSelect }: Props) {
  if (!groups.length) return null;
  return (
    <Accordion title="Variations">
      {groups.map((g) => (
        <div key={g.name} className="var-group">
          <div className="label">{g.label}</div>
          <div className="swatches">
            {g.options.map((o) => (
              <button
                key={o.variant}
                className={'swatch' + (selected[g.name] === o.variant ? ' active' : '')}
                onClick={() => onSelect(g.name, o.variant)}
                title={o.label}
                aria-label={o.label}
                aria-pressed={selected[g.name] === o.variant}
              >
                <span className="dot" style={{ background: colors[o.variant] ?? 'var(--bg3)' }} />
                <span>{o.label}</span>
              </button>
            ))}
          </div>
        </div>
      ))}
    </Accordion>
  );
}
