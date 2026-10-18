import { useEffect, useRef, useState } from 'react';
import { fitSize, imageToStage, stageToImage, type RefBox, type RefImage, type RefPin, type RefSettings, type RefTransform, type SurfaceAnchor } from '../scene/refImage';

export interface PinApi {
  pins: RefPin[];
  /** placing pins: click = image point, next click on the model = its surface point */
  active: boolean;
  onPins: (pins: RefPin[]) => void;
  /** surface anchor under a client (page) position, null when the click misses the model */
  pick: (clientX: number, clientY: number) => SurfaceAnchor | null;
  /** client positions of the pins' model points in the current pose and camera */
  project: (pins: RefPin[]) => ({ x: number; y: number } | null)[];
}

interface Props {
  image: RefImage;
  settings: RefSettings;
  transforming: boolean;
  onTransform: (t: RefTransform) => void;
  pin: PinApi;
  /** areas ignored by Auto lighting; `drawing` = drag on the canvas adds one */
  excl: { boxes: RefBox[]; drawing: boolean; onBoxes: (b: RefBox[]) => void };
}

const PIN_COLORS = ['#ff4d6d', '#ffd23f', '#3ddc97', '#4cc9f0', '#b388ff', '#ff9f1c', '#f15bb5', '#9ef01a'];
export const pinColor = (i: number) => PIN_COLORS[i % PIN_COLORS.length];

type Drag =
  | { kind: 'move'; px: number; py: number; t: RefTransform }
  | { kind: 'scale'; d0: number; t: RefTransform }
  | { kind: 'rotate'; a0: number; t: RefTransform };

/** Screen-space reference image (does not follow the 3D camera). Behind the canvas (layer "background", the scene
 *  background is made transparent by the page) or above it ("overlay"). In transform mode a full-stage frame on top
 *  captures the pointer: drag = move, corners = scale, top knob = rotate (Shift snaps 15°), wheel = zoom about the
 *  cursor, Alt+wheel = rotate. The 3D camera does not receive input while transforming. */
export function RefImageLayer({ image, settings, transforming, onTransform, pin, excl }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const frame = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 1, h: 1 });
  const drag = useRef<Drag | null>(null);
  const live = useRef({ settings, size, onTransform, pin, image });
  live.current = { settings, size, onTransform, pin, image };
  const [modelPts, setModelPts] = useState<({ x: number; y: number } | null)[]>([]);
  const [missed, setMissed] = useState(false);
  const pinDrag = useRef<{ id: number } | null>(null);

  // Model-side pin positions follow the animation and the camera: re-project every frame while pins exist.
  const hasPins = pin.pins.length > 0;
  useEffect(() => {
    if (!hasPins) { setModelPts([]); return; }
    let raf = 0, last = '';
    const tick = () => {
      const { pin: p } = live.current;
      const r = host.current?.getBoundingClientRect();
      if (r) {
        const pts = p.project(p.pins).map((q) => (q ? { x: q.x - r.left, y: q.y - r.top } : null));
        const key = pts.map((q) => (q ? `${q.x.toFixed(1)},${q.y.toFixed(1)}` : '-')).join(' ');
        if (key !== last) { last = key; setModelPts(pts); }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [hasPins]);

  // Pin placement: plain clicks on the stage (bubble phase, so the canvas still orbits/zooms on drag and wheel).
  useEffect(() => {
    if (!pin.active) return;
    const stageEl = host.current?.parentElement;
    if (!stageEl) return;
    let down: { x: number; y: number; id: number } | null = null;
    const onDown = (e: PointerEvent) => { if (e.button === 0) down = { x: e.clientX, y: e.clientY, id: e.pointerId }; };
    const onUp = (e: PointerEvent) => {
      const d = down; down = null;
      if (!d || e.button !== 0 || d.id !== e.pointerId || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) return;
      const { pin: p, size: sz, settings: s, image: img } = live.current;
      const pending = p.pins.find((q) => !q.anchor);
      if (pending) {
        const a = p.pick(e.clientX, e.clientY);
        if (!a) { setMissed(true); return; }
        setMissed(false);
        p.onPins(p.pins.map((q) => (q.id === pending.id ? { ...q, anchor: a } : q)));
      } else {
        const r = host.current!.getBoundingClientRect();
        const { u, v } = stageToImage(e.clientX - r.left, e.clientY - r.top, img, s.transform, sz.w, sz.h);
        const id = p.pins.reduce((m, q) => Math.max(m, q.id), 0) + 1;
        setMissed(false);
        p.onPins([...p.pins, { id, u, v, anchor: null }]);
      }
    };
    stageEl.addEventListener('pointerdown', onDown);
    stageEl.addEventListener('pointerup', onUp);
    return () => { stageEl.removeEventListener('pointerdown', onDown); stageEl.removeEventListener('pointerup', onUp); };
  }, [pin.active]);

  useEffect(() => {
    const el = host.current!;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth || 1, h: el.clientHeight || 1 }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // native wheel listener: React's is passive and cannot preventDefault
  useEffect(() => {
    const el = frame.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const { settings: s, size: sz, onTransform: cb } = live.current;
      const t = s.transform;
      if (e.altKey) {
        cb({ ...t, rotation: t.rotation + (e.deltaY > 0 ? 1 : -1) * (e.shiftKey ? 15 : 2) });
        return;
      }
      const r = el.getBoundingClientRect();
      const px = (e.clientX - r.left) / sz.w, py = (e.clientY - r.top) / sz.h;
      const f = Math.exp(-e.deltaY * (e.shiftKey ? 0.0003 : 0.0015));
      const scale = Math.min(50, Math.max(0.02, t.scale * f));
      const k = scale / t.scale;
      cb({ ...t, scale, x: px + (t.x - px) * k, y: py + (t.y - py) * k });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [transforming]);

  const t = settings.transform;
  const fit = fitSize(image, size.w, size.h);
  const cx = t.x * size.w, cy = t.y * size.h;

  const local = (e: React.PointerEvent) => {
    const r = host.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  const begin = (e: React.PointerEvent, kind: Drag['kind']) => {
    e.preventDefault(); e.stopPropagation();
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    const p = local(e);
    const t0 = live.current.settings.transform;
    if (kind === 'move') drag.current = { kind, px: p.x, py: p.y, t: t0 };
    else if (kind === 'scale') drag.current = { kind, d0: Math.hypot(p.x - cx, p.y - cy) || 1, t: t0 };
    else drag.current = { kind, a0: Math.atan2(p.y - cy, p.x - cx), t: t0 };
  };
  const move = (e: React.PointerEvent) => {
    const d = drag.current; if (!d) return;
    const p = local(e);
    if (d.kind === 'move') onTransform({ ...d.t, x: d.t.x + (p.x - d.px) / size.w, y: d.t.y + (p.y - d.py) / size.h });
    else if (d.kind === 'scale') onTransform({ ...d.t, scale: Math.min(50, Math.max(0.02, d.t.scale * Math.hypot(p.x - cx, p.y - cy) / d.d0)) });
    else {
      let rot = d.t.rotation + (Math.atan2(p.y - cy, p.x - cx) - d.a0) * 180 / Math.PI;
      if (e.shiftKey) rot = Math.round(rot / 15) * 15;
      onTransform({ ...d.t, rotation: rot });
    }
  };
  const end = () => { drag.current = null; };

  const pinDown = (e: React.PointerEvent, id: number) => {
    e.preventDefault(); e.stopPropagation();
    if (e.button === 2 || e.altKey) { pin.onPins(pin.pins.filter((q) => q.id !== id)); return; }
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    pinDrag.current = { id };
  };
  const pinMove = (e: React.PointerEvent) => {
    const d = pinDrag.current; if (!d) return;
    const p = local(e);
    const { u, v } = stageToImage(p.x, p.y, image, t, size.w, size.h);
    pin.onPins(pin.pins.map((q) => (q.id === d.id ? { ...q, u, v } : q)));
  };
  const pinUp = (e: React.PointerEvent) => { if (pinDrag.current) { e.stopPropagation(); pinDrag.current = null; } };
  const pending = pin.pins.find((q) => !q.anchor);
  // exclude areas: drag a rectangle (screen-aligned), stored as the image-space box of its corners
  const [draft, setDraft] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const exDown = (e: React.PointerEvent) => {
    e.preventDefault(); (e.currentTarget as Element).setPointerCapture(e.pointerId);
    const p = local(e); setDraft({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
  };
  const exMove = (e: React.PointerEvent) => { if (draft) { const p = local(e); setDraft({ ...draft, x1: p.x, y1: p.y }); } };
  const exUp = () => {
    const d = draft; setDraft(null);
    if (!d || Math.abs(d.x1 - d.x0) < 4 || Math.abs(d.y1 - d.y0) < 4) return;
    const corners = [[d.x0, d.y0], [d.x1, d.y0], [d.x0, d.y1], [d.x1, d.y1]].map(([x, y]) => stageToImage(x, y, image, t, size.w, size.h));
    const us = corners.map((c) => c.u), vs = corners.map((c) => c.v);
    excl.onBoxes([...excl.boxes, { u0: Math.min(...us), v0: Math.min(...vs), u1: Math.max(...us), v1: Math.max(...vs) }]);
  };
  const boxPoly = (b: RefBox) => [[b.u0, b.v0], [b.u1, b.v0], [b.u1, b.v1], [b.u0, b.v1]]
    .map(([u, v]) => imageToStage(u, v, image, t, size.w, size.h)).map((p) => `${p.x},${p.y}`).join(' ');
  const showPins = pin.active || (hasPins && settings.visible);

  const boxStyle = { left: cx, top: cy, width: fit.w * t.scale, height: fit.h * t.scale, transform: `translate(-50%, -50%) rotate(${t.rotation}deg)` };
  return (
    <>
      <div ref={host} className={`refimg-layer ${settings.layer}`}>
        {settings.visible && (
          <img className="refimg" src={image.src} alt="" draggable={false}
            style={{ ...boxStyle, opacity: settings.opacity, transform: `${boxStyle.transform}${t.flipX ? ' scaleX(-1)' : ''}` }} />
        )}
      </div>
      {(excl.drawing || (excl.boxes.length > 0 && settings.visible)) && (
        <svg className="refpins" width={size.w} height={size.h}>
          {excl.boxes.map((b, i) => (
            <polygon key={i} points={boxPoly(b)} className="refexcl" onContextMenu={(e) => e.preventDefault()}
              onPointerDown={(e) => { if (e.button === 2 || e.altKey) { e.stopPropagation(); excl.onBoxes(excl.boxes.filter((_, j) => j !== i)); } }} />
          ))}
        </svg>
      )}
      {excl.drawing && (
        <div className="refimg-frame excl" onPointerDown={exDown} onPointerMove={exMove} onPointerUp={exUp} onPointerCancel={() => setDraft(null)}>
          {draft && <div className="refexcl-draft" style={{ left: Math.min(draft.x0, draft.x1), top: Math.min(draft.y0, draft.y1), width: Math.abs(draft.x1 - draft.x0), height: Math.abs(draft.y1 - draft.y0) }} />}
          <div className="refimg-hint">Drag boxes over things that are not the model's lighting (dialogue box, other characters, effects). Right-click a box removes it.</div>
        </div>
      )}
      {showPins && (
        <svg className="refpins" width={size.w} height={size.h}>
          {pin.pins.map((q, i) => {
            const a = imageToStage(q.u, q.v, image, t, size.w, size.h);
            const m = modelPts[i];
            const col = pinColor(i);
            return (
              <g key={q.id}>
                {m && <line x1={a.x} y1={a.y} x2={m.x} y2={m.y} stroke={col} strokeWidth={1.5} strokeDasharray="4 3" />}
                {m && <circle cx={m.x} cy={m.y} r={5} fill="none" stroke={col} strokeWidth={2} />}
                {m && <circle cx={m.x} cy={m.y} r={1.5} fill={col} />}
                <circle className="refpin" cx={a.x} cy={a.y} r={8} fill={col} stroke="#000" strokeWidth={1.5}
                  onPointerDown={(e) => pinDown(e, q.id)} onPointerMove={pinMove} onPointerUp={pinUp} onContextMenu={(e) => e.preventDefault()} />
                <text x={a.x} y={a.y + 3.5} textAnchor="middle" className="refpin-num">{i + 1}</text>
              </g>
            );
          })}
        </svg>
      )}
      {pin.active && (
        <div className="refimg-hint" style={{ zIndex: 5 }}>
          {pending ? <>Pin {pin.pins.indexOf(pending) + 1}: now click the same spot on the <b>model</b>{missed ? ' (nothing of the model under that click, try a bit closer to it)' : ''}. Esc cancels.</>
            : <>Click a feature on the <b>reference image</b> (eye, chin, fingertip…), then the same spot on the model. Drag a dot to adjust, right-click removes. Drag still orbits.</>}
        </div>
      )}
      {transforming && (
        <div ref={frame} className="refimg-frame" onPointerDown={(e) => begin(e, 'move')} onPointerMove={move} onPointerUp={end} onPointerCancel={end}>
          <div className="refimg-box" style={boxStyle}>
            {(['nw', 'ne', 'sw', 'se'] as const).map((c) => (
              <span key={c} className={`refimg-handle ${c}`} onPointerDown={(e) => begin(e, 'scale')} onPointerMove={move} onPointerUp={end} onPointerCancel={end} />
            ))}
            <span className="refimg-rot" title="Rotate (Shift snaps 15°)" onPointerDown={(e) => begin(e, 'rotate')} onPointerMove={move} onPointerUp={end} onPointerCancel={end} />
          </div>
          <div className="refimg-hint">Transforming reference: drag to move, corners scale, knob rotates, wheel zooms (Alt+wheel rotates). R or Done to finish.</div>
        </div>
      )}
    </>
  );
}
