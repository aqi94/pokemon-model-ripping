import { useState } from 'react';
import { LightingPanel, type Options } from './OptionsSection';
import { LightingSelect } from './LightingSelect';

interface Props {
  game: string;
  gameName: string;
  title: string;
  options: Options;
  onChange: (patch: Partial<Options>) => void;
  panelsHidden: boolean;
  onTogglePanels: () => void;
}

/** Photoshop-style options bar across the top: domain switcher, current model, and the settings tinkered with most (lighting, shading, albedo, background). */
export function OptionsBar({ game, gameName, title, options, onChange, panelsHidden, onTogglePanels }: Props) {
  const [lightOpen, setLightOpen] = useState(() => { try { return localStorage.getItem('lightOpen') === '1'; } catch { return false; } });
  const toggleLight = () => setLightOpen((o) => { const n = !o; try { localStorage.setItem('lightOpen', n ? '1' : '0'); } catch { /* ignore */ } return n; });
  return (
    <header className="optionsbar">
      <a className="btn small" href={import.meta.env.BASE_URL} title="All games">&#9664; Games</a>
      <a className="btn small" href={`${import.meta.env.BASE_URL}${game}`} title="Thumbnail grid for this game">&#9638; {gameName || game}</a>
      <strong className="ob-title">{title}</strong>
      <span className="spacer" />
      <div className="light-menu">
        <button className={'btn small' + (lightOpen ? ' primary' : '')} onClick={toggleLight} title="Light settings" aria-expanded={lightOpen}>&#9728;</button>
        {lightOpen && <div className="light-pop"><LightingPanel options={options} onChange={onChange} /></div>}
      </div>
      <label className="ob-field">Lighting
        <LightingSelect env={options.env} onPick={(env) => onChange({ env })} />
      </label>
      <label className="ob-field">Shading
        <select className="text" value={options.shading} onChange={(e) => onChange({ shading: e.target.value as Options['shading'] })}>
          <option value="game">Game</option><option value="pbr">PBR</option>
        </select>
      </label>
      <label className="ob-field">Albedo
        <select className="text" value={options.albedo} onChange={(e) => onChange({ albedo: e.target.value as Options['albedo'] })}>
          <option value="baked">Baked</option><option value="raw">Raw</option><option value="off">Off</option>
        </select>
      </label>
      <label className="ob-field">BG
        <input type="color" value={options.background} onChange={(e) => onChange({ background: e.target.value })} />
      </label>
      <button className={'btn small' + (panelsHidden ? ' primary' : '')} onClick={onTogglePanels} title="Hide / show panels (Tab)">Panels</button>
    </header>
  );
}
