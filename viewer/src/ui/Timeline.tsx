import type { PlaybackState } from '../scene/animation';

interface Props {
  state: PlaybackState;
  onPrev: () => void;
  onNext: () => void;
  onTogglePause: () => void;
  onScrub: (frame: number) => void;
  onCycleScale: () => void;
}

/** Bottom bar, like Photoshop's timeline: transport, scrubber, frame readout, speed. */
export function Timeline({ state, onPrev, onNext, onTogglePause, onScrub, onCycleScale }: Props) {
  const last = Math.max(0, state.frameCount - 1);
  return (
    <div className="timeline-bar">
      <button className="tool-btn" onClick={onPrev} title="Previous clip (Left)">&#9664;&#9664;</button>
      <button className="tool-btn" onClick={onTogglePause} title="Play/Pause (P)">{state.paused ? '▶' : '❚❚'}</button>
      <button className="tool-btn" onClick={onNext} title="Next clip (Right)">&#9654;&#9654;</button>
      <input className="scrub" type="range" min={0} max={last} step={1} value={state.frame} onChange={(e) => onScrub(Number(e.target.value))} aria-label="Timeline" />
      <span className="dim mono">{state.frame} / {last}</span>
      <button className="btn small" onClick={onCycleScale} title="Time scale (T)">{state.timeScale}x</button>
      <span className="clip-name" title={state.clip ?? ''}>{state.clip ?? 'no clip'}</span>
    </div>
  );
}
