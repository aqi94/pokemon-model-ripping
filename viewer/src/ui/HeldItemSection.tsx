import type { IndexModel, PropInfo } from '../types';
import { Accordion } from './Accordion';

interface Props {
  items: IndexModel[];
  props: PropInfo[];
  item: string;
  bone: string;
  onlyInClips: boolean;
  onItem: (id: string) => void;
  onBone: (bone: string) => void;
  onOnlyInClips: (v: boolean) => void;
}

const sideLabel = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) + ' hand' : 'Hand');

export function HeldItemSection({ items, props, item, bone, onlyInClips, onItem, onBone, onOnlyInClips }: Props) {
  return (
    <Accordion title="Held Item">
      <div className="var-group">
        <select className="text" value={item} onChange={(e) => onItem(e.target.value)} aria-label="Held item">
          <option value="">None</option>
          {items.map((m) => <option key={m.id} value={m.id}>{m.displayName}</option>)}
        </select>
      </div>
      {props.length > 1 && (
        <div className="var-group">
          <select className="text" value={bone} onChange={(e) => onBone(e.target.value)} aria-label="Hand">
            {props.map((p) => <option key={p.bone} value={p.bone}>{sideLabel(p.side)}</option>)}
          </select>
        </div>
      )}
      <label className="check">
        <input type="checkbox" checked={onlyInClips} onChange={(e) => onOnlyInClips(e.target.checked)} /> Only in animations that use it
      </label>
    </Accordion>
  );
}
