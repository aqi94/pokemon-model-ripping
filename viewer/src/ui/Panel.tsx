import type { ReactNode } from 'react';

/** Photoshop-style docked panel: title bar + body. `grow` panels share the remaining column height and scroll inside. */
export function Panel({ title, children, grow = false, actions }: { title: string; children: ReactNode; grow?: boolean; actions?: ReactNode }) {
  return (
    <section className={'panel' + (grow ? ' grow' : '')}>
      <header className="panel-head"><span>{title}</span>{actions}</header>
      <div className="panel-body">{children}</div>
    </section>
  );
}
