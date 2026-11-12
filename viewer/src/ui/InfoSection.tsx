import type { Manifest } from '../types';
import { Accordion } from './Accordion';

export function InfoSection({ manifest, gameName }: { manifest: Manifest; gameName: string }) {
  const x = manifest.extras ?? {};
  const rows: [string, string][] = [
    ['Name', manifest.displayName],
    ['Game', gameName],
    ['ID', manifest.id],
  ];
  if (x.dex != null) rows.push(['Dex #', String(x.dex).padStart(4, '0')]);
  if (x.form) rows.push(['Form', x.form]);
  rows.push(
    ['Triangles', manifest.stats.triangles.toLocaleString()],
    ['Joints', String(manifest.stats.joints)],
    ['Clips', String(manifest.stats.clips ?? manifest.clips.length)],
  );
  // what the rip does not contain (UNITE importer): particle effects (only their empty emitter nodes were exported) and
  // parts whose texture is missing (imported hidden as `~notex`, switchable in Model Parts)
  const particles = x.particlesNotInRip;
  if (particles?.length) rows.push(['Not in rip', `particle effects: ${particles.join(', ')}`]);
  const notex = manifest.parts?.filter((p) => p.node.endsWith('~notex')).length ?? 0;
  if (notex) rows.push(['Untextured', `${notex} part${notex > 1 ? 's' : ''} without texture in the rip (hidden; Model Parts)`]);
  return (
    <Accordion title="Information">
      <dl className="info">
        {rows.map(([k, v]) => (
          <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
        ))}
      </dl>
    </Accordion>
  );
}
