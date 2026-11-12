import type { ReactNode } from 'react';
import type { PlaybackState } from '../scene/animation';

interface Props {
  state: PlaybackState;
  onPrev: () => void;
  onNext: () => void;
  onTogglePause: () => void;
  onScrub: (frame: number) => void;
  onCycleScale: () => void;
  onToggleLoop: () => void;
}

/** One 24-unit grid for every transport icon so no glyph looks smaller than its neighbours. */
function Icon({ children }: { children: ReactNode }) {
  return <svg className="ico" viewBox="0 0 24 24" aria-hidden="true">{children}</svg>;
}
const Prev = () => <Icon><path d="M6 5h2v14H6zM20 5v14L9.5 12z" fill="currentColor" /></Icon>;
const Next = () => <Icon><path d="M16 5h2v14h-2zM4 5v14l10.5-7z" fill="currentColor" /></Icon>;
const Play = () => <Icon><path d="M8 5v14l11-7z" fill="currentColor" /></Icon>;
const Pause = () => <Icon><path d="M7 5h4v14H7zM13 5h4v14h-4z" fill="currentColor" /></Icon>;
const Repeat = () => (
  <Icon><path d="M17 2l4 4-4 4M3 11V9a3 3 0 0 1 3-3h15M7 22l-4-4 4-4M21 13v2a3 3 0 0 1-3 3H3"
    fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></Icon>
);

/** Bottom bar, like Photoshop's timeline: transport, scrubber, frame readout, speed. */
export function Timeline({ state, onPrev, onNext, onTogglePause, onScrub, onCycleScale, onToggleLoop }: Props) {
  const last = Math.max(0, state.frameCount - 1);
  return (
    <div className="timeline-bar">
      <button className="tool-btn" onClick={onPrev} title="Previous clip (Left)" aria-label="Previous clip"><Prev /></button>
      <button className="tool-btn" onClick={onTogglePause} title={state.paused ? 'Play (P)' : 'Pause (P)'} aria-label="Play/Pause">{state.paused ? <Play /> : <Pause />}</button>
      <button className="tool-btn" onClick={onNext} title="Next clip (Right)" aria-label="Next clip"><Next /></button>
      <button className={'tool-btn' + (state.loop ? ' on' : '')} onClick={onToggleLoop} aria-pressed={state.loop} aria-label="Loop"
        title={state.loop ? 'Loop: on, clip repeats (L)' : 'Loop: off, clip plays once (L)'}><Repeat /></button>
      <input className="scrub" type="range" min={0} max={last} step={1} value={state.frame} onChange={(e) => onScrub(Number(e.target.value))} aria-label="Timeline" />
      <span className="dim mono">{state.frame} / {last}</span>
      <button className="btn small" onClick={onCycleScale} title="Time scale (T)">{state.timeScale}x</button>
      <span className="clip-name" title={state.clip ?? ''}>{state.clip ?? 'no clip'}</span>
    </div>
  );
}
