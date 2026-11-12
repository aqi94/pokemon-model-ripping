import { useEffect, useRef, type ReactNode } from 'react';
import type { PartInfo } from '../types';
import { Accordion } from './Accordion';

interface Props {
  parts: PartInfo[];
  visible: Record<string, boolean>;
  driven?: string[];
  onToggle: (node: string, v: boolean) => void;
  onAll: (v: boolean) => void;
  /** hovering a checkbox row highlights that part in the view (null = none) */
  onHover?: (node: string | null) => void;
  lockToClip: boolean;
  onLockChange: (v: boolean) => void;
  /** extra controls shown at the top of the section (held item) */
  children?: ReactNode;
}

export function PartsSection({ parts, visible, driven = [], onToggle, onAll, onHover, lockToClip, onLockChange, children }: Props) {
  const hover = useRef(onHover);
  hover.current = onHover;
  useEffect(() => () => hover.current?.(null), []);
  return (
    <Accordion title="Model Parts">
      {children}
      <div className="row">
        <button className="btn small" onClick={() => onAll(true)}>Show all</button>
        <button className="btn small" onClick={() => onAll(false)}>Hide all</button>
      </div>
      <label className="check">
        <input type="checkbox" checked={lockToClip} onChange={(e) => onLockChange(e.target.checked)} />
        <span>Lock parts to clip</span>
      </label>
      <div className="parts" onMouseLeave={() => onHover?.(null)}>
        {parts.map((p) => (
          <label key={p.node} className="check part-row" onMouseEnter={() => onHover?.(p.node)} onMouseLeave={() => onHover?.(null)}>
            <input type="checkbox" checked={!!visible[p.node]} disabled={driven.includes(p.node)} onChange={(e) => onToggle(p.node, e.target.checked)}
              onFocus={() => onHover?.(p.node)} onBlur={() => onHover?.(null)} />
            <span>{p.label}{driven.includes(p.node) ? ' (clip)' : ''}</span>
          </label>
        ))}
      </div>
    </Accordion>
  );
}
