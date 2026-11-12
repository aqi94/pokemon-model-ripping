import { useState, type ReactNode } from 'react';

const readOpen = (id: string | undefined, def: boolean) => {
  if (!id) return def;
  try { const v = localStorage.getItem('acc:' + id); return v === null ? def : v === '1'; } catch { return def; }
};

/** Collapsible section. With an `id` the open / closed state is remembered across reloads. */
export function Accordion({ title, children, defaultOpen = true, id, className }: {
  title: ReactNode; children: ReactNode; defaultOpen?: boolean; id?: string; className?: string;
}) {
  if (!id && typeof title === 'string') id = title.toLowerCase().replace(/\W+/g, '-'); // remembered per section title
  const [open, setOpen] = useState(() => readOpen(id, defaultOpen));
  const toggle = () => {
    const n = !open;
    setOpen(n);
    if (id) try { localStorage.setItem('acc:' + id, n ? '1' : '0'); } catch { /* ignore */ }
  };
  return (
    <section className={'acc' + (className ? ' ' + className : '')}>
      <button className="acc-head" onClick={toggle} aria-expanded={open}>
        <span>{title}</span>
        <span className={'chev' + (open ? ' open' : '')}>&#9662;</span>
      </button>
      {open && <div className="acc-body">{children}</div>}
    </section>
  );
}
