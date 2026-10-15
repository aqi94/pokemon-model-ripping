import { Accordion } from './Accordion';

export function ExportSection({ onPng, onGlb }: { onPng: () => void; onGlb: () => void }) {
  return (
    <Accordion title="Export">
      <div className="row">
        <button className="btn" onClick={onPng}>Export .png</button>
        <button className="btn" onClick={onGlb}>Export .glb</button>
      </div>
    </Accordion>
  );
}
