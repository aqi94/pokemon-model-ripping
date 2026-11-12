import { Accordion } from './Accordion';

interface Props {
  /** 0 = the jaw as the clip poses it, 1 = the widest pose any clip reaches */
  value: number;
  onChange: (v: number) => void;
}

/** Appearance > Jaw: extra mouth opening on top of the clip (UNITE: mouths are bone-driven, no blend shapes). */
export function JawSection({ value, onChange }: Props) {
  return (
    <Accordion title="Jaw">
      <label className="field" title="Adds a share of the widest jaw opening found in the model's clips on top of the playing clip">
        <span>Open {Math.round(value * 100)}%</span>
        <input type="range" min={0} max={1} step={0.01} value={value} onChange={(e) => onChange(Number(e.target.value))} aria-label="Jaw opening" />
      </label>
    </Accordion>
  );
}
