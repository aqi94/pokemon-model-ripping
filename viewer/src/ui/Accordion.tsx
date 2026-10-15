import { useState, type ReactNode } from 'react';

export function Accordion({ title, children, defaultOpen = true }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="acc">
      <button className="acc-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span>{title}</span>
        <span className={'chev' + (open ? ' open' : '')}>&#9662;</span>
      </button>
      {open && <div className="acc-body">{children}</div>}
    </section>
  );
}
