import { useEffect, useMemo, useRef, useState } from 'react';
import type { ClipInfo } from '../types';
import type { PlaybackState } from '../scene/animation';
import { Accordion } from './Accordion';

const ORDER = ['idle', 'move', 'battle', 'emote', 'other'];

interface Props {
  clips: ClipInfo[];
  state: PlaybackState;
  onSelect: (name: string) => void;
  onPrev: () => void;
  onNext: () => void;
  onTogglePause: () => void;
  onScrub: (frame: number) => void;
  onCycleScale: () => void;
}

export function AnimationsSection({ clips, state, onSelect, onPrev, onNext, onTogglePause, onScrub, onCycleScale }: Props) {
  const [q, setQ] = useState('');
  const activeRef = useRef<HTMLButtonElement>(null);
  const groups = useMemo(() => {
    const f = q.trim().toLowerCase();
    const filtered = clips.filter((c) => !f || c.name.toLowerCase().includes(f));
    const m = new Map<string, ClipInfo[]>();
    for (const c of filtered) {
      const k = ORDER.includes(c.category) ? c.category : 'other';
      if (!m.has(k)) m.set(k, []);
      m.get(k)!.push(c);
    }
    return ORDER.filter((k) => m.has(k)).map((k) => [k, m.get(k)!] as const);
  }, [clips, q]);

  useEffect(() => { activeRef.current?.scrollIntoView({ block: 'nearest' }); }, [state.clip]);

  return (
    <Accordion title="Animations">
      <input className="text" placeholder="Filter animations..." value={q} onChange={(e) => setQ(e.target.value)} />
      <div className="anim-list" role="listbox">
        {groups.map(([cat, list]) => (
          <div key={cat}>
            <div className="anim-cat">{cat} <span>({list.length})</span></div>
            {list.map((c) => (
              <button
                key={c.name}
                ref={c.name === state.clip ? activeRef : undefined}
                className={'anim-item' + (c.name === state.clip ? ' active' : '')}
                onClick={() => onSelect(c.name)}
                role="option"
                aria-selected={c.name === state.clip}
              >
                <span>{c.name}</span>
                <span className="dim">{c.loop ? '↻' : ''}</span>
              </button>
            ))}
          </div>
        ))}
        {groups.length === 0 && <div className="dim pad">No matches</div>}
      </div>
      <div className="row">
        <button className="btn" onClick={onPrev} title="Previous (Left)">&#9664; Prev</button>
        <button className="btn primary" onClick={onTogglePause} title="Play/Pause (P)">{state.paused ? '▶ Play' : '❚❚ Pause'}</button>
        <button className="btn" onClick={onNext} title="Next (Right)">Next &#9654;</button>
      </div>
      <div className="row timeline">
        <input
          type="range"
          min={0}
          max={Math.max(0, state.frameCount - 1)}
          step={1}
          value={state.frame}
          onChange={(e) => onScrub(Number(e.target.value))}
          aria-label="Timeline"
        />
      </div>
      <div className="row between">
        <span className="dim">Frame {state.frame} / {Math.max(0, state.frameCount - 1)}</span>
        <button className="btn small" onClick={onCycleScale} title="Time scale (T)">{state.timeScale}x</button>
      </div>
      <div className="dim hint">Keys: &larr; &rarr; clip, P pause, T speed</div>
    </Accordion>
  );
}
