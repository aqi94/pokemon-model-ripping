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
