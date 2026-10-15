import { useCallback, useEffect, useRef, useState } from 'react';
import type { IndexModel, Manifest, ModelIndex, PaletteEntry } from '../types';
import { DEFAULT_ENV } from '../scene/environment';
import { ViewerScene } from '../scene/ViewerScene';
import { TIME_SCALES, type PlaybackState } from '../scene/animation';
import { exportGlb } from '../scene/exporter';
import { InfoSection } from '../ui/InfoSection';
import { AnimationsSection } from '../ui/AnimationsSection';
import { VariationsSection } from '../ui/VariationsSection';
import { HeldItemSection } from '../ui/HeldItemSection';
import { PartsSection } from '../ui/PartsSection';
import { PartGroupsSection } from '../ui/PartGroupsSection';
import { OptionsSection, type Options } from '../ui/OptionsSection';
import { ExportSection } from '../ui/ExportSection';
import { ModelPicker } from '../ui/ModelPicker';

const EMPTY: PlaybackState = { clip: null, time: 0, duration: 0, frame: 0, frameCount: 0, paused: false, timeScale: 1 };
const DEFAULT_OPTIONS: Options = { shadows: false, skeleton: false, wireframe: false, background: '#1b1d23', autoRotate: 0, mirror: false, albedo: 'baked', shading: 'game', env: { ...DEFAULT_ENV } };

/** Stays mounted while the model changes (single page): options persist, only the scene is rebuilt. */
export function Viewer({ game, id, onSelect }: { game?: string; id: string; onSelect: (m: IndexModel) => void }) {
  const canvasHost = useRef<HTMLDivElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<ViewerScene | null>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [gameName, setGameName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<number | null>(0);
  const [pb, setPb] = useState<PlaybackState>(EMPTY);
  const [partsDriven, setPartsDriven] = useState<string[]>([]);
  const [partsVis, setPartsVis] = useState<Record<string, boolean>>({});
  const [variantSel, setVariantSel] = useState<Record<string, number>>({});
  const [groupSel, setGroupSel] = useState<Record<string, string>>({});
  const [palSel, setPalSel] = useState<Record<string, number[]>>({});
  const [variantColors, setVariantColors] = useState<Record<number, string | null>>({});
  const [options, setOptions] = useState<Options>(DEFAULT_OPTIONS);
  const optionsRef = useRef<Options>(DEFAULT_OPTIONS);
  const [index, setIndex] = useState<ModelIndex | null>(null);
  const [sidebarW, setSidebarW] = useState(() => {
    try { return Number(localStorage.getItem('sidebarW')) || 340; } catch { return 340; }
  });
  const modelUrl = useRef('');
  const [items, setItems] = useState<IndexModel[]>([]);
  const [heldItem, setHeldItem] = useState('');
  const [heldBone, setHeldBone] = useState('');
  const [heldOnly, setHeldOnly] = useState(true);
  const itemsRef = useRef<IndexModel[]>([]);
  const heldOnlyRef = useRef(true);

  useEffect(() => {
    fetch('/models/index.json').then((r) => r.json()).then(setIndex).catch(() => { /* the load effect reports errors */ });
  }, []);

  // load model
  useEffect(() => {
    let cancelled = false;
    const host = canvasHost.current!;
    const vs = new ViewerScene(host);
    sceneRef.current = vs;
    vs.onFrame = setPb;
    vs.onPartsState = (v, d) => { setPartsVis(v); setPartsDriven(d); };
    // GPU context lost: three.js cannot rebuild a loaded scene in place, so
    // show a message and reload once the browser restores the context.
    vs.onContextChange = (lost) => {
      if (lost) setError('The graphics context was lost (GPU reset or out of memory). Reloading when the browser allows it…');
      else window.location.reload();
    };
    (window as any).__viewer = vs; // handy for debugging
    // Carry the current options over to the new scene (the rest are applied after load).
    const o = optionsRef.current;
    vs.setShadingMode(o.shading); vs.setEnvironment(o.env); vs.setBackground(o.background);
    vs.setAutoRotate(o.autoRotate); vs.setMirror(o.mirror); vs.setShadows(o.shadows); vs.setWireframe(o.wireframe);
    setManifest(null); setError(null); setProgress(0); setHeldItem(''); setPb(EMPTY); setVariantColors({});
    (async () => {
      try {
        const index: ModelIndex = await (await fetch('/models/index.json')).json();
        const entry = index.models.find((m) => m.id === id && (!game || m.game === game));
        if (!entry) throw new Error(`Model not found in index: ${game ? game + '/' : ''}${id}`);
        const objs = index.models.filter((m) => m.kind === 'object').sort((a, b) => a.id.localeCompare(b.id));
        itemsRef.current = objs; setItems(objs);
        setGameName(index.games[entry.game] ?? entry.game);
        const base = `/models/${entry.path}`;
        const man: Manifest = await (await fetch(base + 'manifest.json')).json();
        // manifests can contain duplicate clip names; the glTF mixer can only address one per name
        const seen = new Set<string>();
        man.clips = man.clips.filter((c) => !seen.has(c.name) && !!seen.add(c.name));
        modelUrl.current = base + man.files.model;
        await vs.load(man, modelUrl.current, (f) => setProgress(f));
        if (cancelled) return;
        if (optionsRef.current.skeleton) vs.setSkeleton(true);
        if (optionsRef.current.albedo !== 'baked') void vs.setAlbedoMode(optionsRef.current.albedo);
        const pv: Record<string, boolean> = {};
        for (const p of man.parts) if (p.lod === 0) pv[p.node] = vs.findNode(p.node)?.visible ?? false;
        setPartsVis(pv);
        const sel: Record<string, number> = {};
        for (const g of man.variantGroups) sel[g.name] = g.options.find((o) => o.default)?.variant ?? g.options[0]?.variant ?? 0;
        setVariantSel(sel);
        const gsel: Record<string, string> = {};
        for (const g of man.partGroups ?? []) gsel[g.name] = (g.options.find((o) => o.default) ?? g.options[0]).id;
        setGroupSel(gsel);
        setPalSel({});
        setHeldBone(man.props?.find((p) => p.side === 'right')?.bone ?? man.props?.[0]?.bone ?? '');
        vs.setHeldOnlyInClips(heldOnlyRef.current);
        setManifest(man);
        setProgress(null);
        document.title = `${man.displayName} - Pokémon Model Viewer`;
        // swatches
        const cols: Record<number, string | null> = {};
        const current = vs.variants?.current ?? 0;
        for (const g of man.variantGroups) for (const o of g.options) cols[o.variant] = (await vs.variants?.sampleColor(o.variant)) ?? null;
        if (!cancelled) setVariantColors(cols);
        void current;
      } catch (e) {
        if (!cancelled) { setError(String(e instanceof Error ? e.message : e)); setProgress(null); }
      }
    })();
    return () => { cancelled = true; vs.dispose(); sceneRef.current = null; };
  }, [game, id]);

  const clips = manifest?.clips ?? [];
  const playClip = useCallback((name: string) => {
    sceneRef.current?.anim?.play(name);
    const s = sceneRef.current?.anim?.state(); if (s) setPb(s);
  }, []);
  const step = useCallback((d: number) => {
    const a = sceneRef.current?.anim;
    if (!a || !clips.length) return;
    const i = clips.findIndex((c) => c.name === a.current);
    playClip(clips[(i + d + clips.length) % clips.length].name);
  }, [clips, playClip]);
  const togglePause = useCallback(() => {
    const a = sceneRef.current?.anim; if (!a) return;
    a.setPaused(!a.paused); setPb(a.state());
  }, []);
  const cycleScale = useCallback(() => {
    const a = sceneRef.current?.anim; if (!a) return;
    a.setTimeScale(TIME_SCALES[(TIME_SCALES.indexOf(a.timeScale) + 1) % TIME_SCALES.length]); setPb(a.state());
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && (t.tagName === 'INPUT' && (t as HTMLInputElement).type === 'text')) return;
      if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
      else if (e.key === 'p' || e.key === 'P') togglePause();
      else if (e.key === 't' || e.key === 'T') cycleScale();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step, togglePause, cycleScale]);

  const patchOptions = (patch: Partial<Options>) => {
    optionsRef.current = { ...optionsRef.current, ...patch };
    setOptions(optionsRef.current);
    const vs = sceneRef.current; if (!vs) return;
    if (patch.shadows !== undefined) vs.setShadows(patch.shadows);
    if (patch.skeleton !== undefined) vs.setSkeleton(patch.skeleton);
    if (patch.wireframe !== undefined) vs.setWireframe(patch.wireframe);
    if (patch.background !== undefined) vs.setBackground(patch.background);
    if (patch.autoRotate !== undefined) vs.setAutoRotate(patch.autoRotate);
    if (patch.mirror !== undefined) vs.setMirror(patch.mirror);
    if (patch.shading !== undefined) vs.setShadingMode(patch.shading);
    if (patch.env !== undefined) vs.setEnvironment(patch.env);
    if (patch.albedo !== undefined) void vs.setAlbedoMode(patch.albedo);
  };

  const applyHeld = (itemId: string, bone: string) => {
    const it = itemsRef.current.find((m) => m.id === itemId);
    const prop = manifest?.props?.find((p) => p.bone === bone);
    void sceneRef.current?.setHeldItem(it ? `/models/${it.path}model.glb` : null, prop);
  };

  const togglePart = (node: string, v: boolean) => {
    sceneRef.current?.setPartVisible(node, v);
  };
  /** Modular models: show only the chosen option's meshes of a group (accessory meshes such as glasses stay off until ticked). */
  const selectPartOption = (group: string, option: string) => {
    const g = manifest?.partGroups?.find((x) => x.name === group);
    if (!g) return;
    setGroupSel((s) => ({ ...s, [group]: option }));
    for (const o of g.options) {
      for (const node of o.nodes) {
        const acc = manifest?.parts.find((p) => p.node === node)?.accessory;
        sceneRef.current?.setPartVisible(node, o.id === option && !acc);
      }
    }
  };
  /** Pick palette zone `zone` (1-based) for colour layer `layer` of a recolourable material group (outfit, hair, eyes...). */
  const selectPalette = (entry: PaletteEntry, layer: number, zone: number) => {
    const key = entry.materials[0];
    const next = [...(palSel[key] ?? entry.indices)];
    next[layer - 1] = zone;
    setPalSel((s) => ({ ...s, [key]: next }));
    void sceneRef.current?.setPalette(entry.materials, next);
  };
  const allParts = (v: boolean) => {
    for (const p of manifest?.parts ?? []) if (p.lod === 0 && (!p.group || groupSel[p.group] === p.option)) sceneRef.current?.setPartVisible(p.node, v);
  };

  const startDrag = (e: React.PointerEvent) => {
    e.preventDefault();
    const move = (ev: PointerEvent) => {
      const w = Math.min(Math.max(window.innerWidth - ev.clientX, 240), Math.min(700, window.innerWidth - 200));
      setSidebarW(w);
      try { localStorage.setItem('sidebarW', String(w)); } catch { /* ignore */ }
    };
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  return (
    <div className="viewer" style={{ ['--sidebar-w' as string]: sidebarW + 'px' }}>
      <div className="stage" ref={stage}>
        <div className="canvas-host" ref={canvasHost} />
        <a className="back" href="/browse">&#9638; Grid</a>
        {manifest && <div className="title-chip">{manifest.displayName}</div>}
        {progress !== null && !error && <div className="overlay">Loading model{progress ? ` ${Math.round(progress * 100)}%` : '...'}</div>}
        {error && <div className="overlay err">{error}</div>}
      </div>
      <div className="handle" onPointerDown={startDrag} role="separator" aria-orientation="vertical" />
      <aside className="sidebar">
        <ModelPicker index={index} game={game} id={id} onSelect={onSelect} />
        {manifest && (
          <>
            <InfoSection manifest={manifest} gameName={gameName} />
            <AnimationsSection
              clips={clips}
              state={pb}
              onSelect={playClip}
              onPrev={() => step(-1)}
              onNext={() => step(1)}
              onTogglePause={togglePause}
              onScrub={(f) => { sceneRef.current?.anim?.scrubToFrame(f); const s = sceneRef.current?.anim?.state(); if (s) setPb(s); }}
              onCycleScale={cycleScale}
            />
            <VariationsSection
              groups={manifest.variantGroups}
              selected={variantSel}
              colors={variantColors}
              onSelect={(g, v) => { setVariantSel((s) => ({ ...s, [g]: v })); void sceneRef.current?.variants?.select(v); }}
            />
            {(manifest.props?.length ?? 0) > 0 && (
              <HeldItemSection
                items={items}
                props={manifest.props!}
                item={heldItem}
                bone={heldBone}
                onlyInClips={heldOnly}
                onItem={(i) => { setHeldItem(i); applyHeld(i, heldBone); }}
                onBone={(b) => { setHeldBone(b); applyHeld(heldItem, b); }}
                onOnlyInClips={(v) => { setHeldOnly(v); heldOnlyRef.current = v; sceneRef.current?.setHeldOnlyInClips(v); }}
              />
            )}
            {(manifest.partGroups?.length ?? 0) > 0 && (
              <PartGroupsSection groups={manifest.partGroups!} selected={groupSel} palettes={palSel} onSelect={selectPartOption} onPalette={selectPalette} />
            )}
            <PartsSection parts={manifest.parts.filter((p) => p.lod === 0 && (!p.group || groupSel[p.group] === p.option))} visible={partsVis} driven={partsDriven} onToggle={togglePart} onAll={allParts} />
            <OptionsSection
              options={options}
              onChange={patchOptions}
              onFullscreen={() => { stage.current?.requestFullscreen?.(); }}
              onRecenter={() => sceneRef.current?.recenter()}
              onCamera={(p) => sceneRef.current?.setCameraPreset(p)}
            />
            <ExportSection
              onPng={() => sceneRef.current?.png(`${manifest.id}.png`)}
              onGlb={() => void exportGlb(modelUrl.current, `${manifest.id}.glb`)}
            />
          </>
        )}
      </aside>
    </div>
  );
}
