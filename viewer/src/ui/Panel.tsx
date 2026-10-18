import type { ReactNode } from 'react';

/** Photoshop-style docked panel: title bar + body. `grow` panels share the remaining column height and scroll inside. */
export function Panel({ title, children, grow = false, actions, flex }: { title: string; children: ReactNode; grow?: boolean; actions?: ReactNode; flex?: number }) {
  return (
    <section className={'panel' + (grow ? ' grow' : '')} style={flex === undefined ? undefined : { flexGrow: flex }}>
      <header className="panel-head"><span>{title}</span>{actions}</header>
      <div className="panel-body">{children}</div>
    </section>
  );
}
