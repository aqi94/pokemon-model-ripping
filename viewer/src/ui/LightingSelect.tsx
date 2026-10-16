import { LIGHTING_PRESETS, matchPreset, type Environment } from '../scene/environment';
import { USER_PREFIX, findPreset, matchAnyPreset, useUserPresets } from '../scene/userPresets';

/** Lighting preset dropdown: built-in presets first, then the user's saved ones (starred, tinted gold). */
export function LightingSelect({ env, onPick }: { env: Environment; onPick: (env: Environment) => void }) {
  const user = useUserPresets();
  const current = matchAnyPreset(env, matchPreset);
  return (
    <select className={'text' + (current === null || current.startsWith(USER_PREFIX) ? ' user-preset' : '')} value={current ?? ''} onChange={(e) => { const pr = findPreset(e.target.value); if (pr) onPick({ ...pr.env }); }}>
      {current === null && <option value="">Custom</option>}
      {LIGHTING_PRESETS.map((pr) => <option key={pr.id} value={pr.id}>{pr.label}</option>)}
      {user.map((pr) => <option key={pr.id} value={pr.id} className="user-preset">{pr.label + ' ★'}</option>)}
    </select>
  );
}
