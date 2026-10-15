import type { PartInfo } from '../types';
import { Accordion } from './Accordion';

interface Props {
  parts: PartInfo[];
  visible: Record<string, boolean>;
  driven?: string[];
  onToggle: (node: string, v: boolean) => void;
  onAll: (v: boolean) => void;
}

export function PartsSection({ parts, visible, driven = [], onToggle, onAll }: Props) {
  return (
    <Accordion title="Model Parts">
      <div className="row">
        <button className="btn small" onClick={() => onAll(true)}>Show all</button>
        <button className="btn small" onClick={() => onAll(false)}>Hide all</button>
      </div>
      <div className="parts">
        {parts.map((p) => (
          <label key={p.node} className="check">
            <input type="checkbox" checked={!!visible[p.node]} disabled={driven.includes(p.node)} onChange={(e) => onToggle(p.node, e.target.checked)} />
            <span>{p.label}{driven.includes(p.node) ? ' (clip)' : ''}</span>
          </label>
        ))}
      </div>
    </Accordion>
  );
}
