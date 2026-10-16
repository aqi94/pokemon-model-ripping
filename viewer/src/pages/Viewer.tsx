import { useCallback, useEffect, useRef, useState } from 'react';
import type { GameIndex, IndexModel, Manifest, PaletteEntry, RefEntry } from '../types';
import { loadGameIndex, loadRefIndex, modelAsset } from '../data';
import { DEFAULT_ENV, LIGHTING_PRESETS } from '../scene/environment';
import { ReferencePanel, type RefMode } from '../ui/ReferencePanel';
import { loadLastEnv, storeLastEnv } from '../scene/userPresets';
import { ViewerScene } from '../scene/ViewerScene';
import { TIME_SCALES, type PlaybackState } from '../scene/animation';
import { exportGlb } from '../scene/exporter';
import { InfoSection } from '../ui/InfoSection';
import { AnimationsSection } from '../ui/AnimationsSection';
import { VariationsSection } from '../ui/VariationsSection';
import { HeldItemSection } from '../ui/HeldItemSection';
import { PartsSection } from '../ui/PartsSection';
import { PartGroupsSection } from '../ui/PartGroupsSection';
import { RenderPanel, type Options } from '../ui/OptionsSection';
import { Accordion } from '../ui/Accordion';
import { SplitPane } from '../ui/SplitPane';
import { ModelPicker } from '../ui/ModelPicker';
import { Panel } from '../ui/Panel';
import { Tabs } from '../ui/Tabs';
import { Timeline } from '../ui/Timeline';
import { Toolbox } from '../ui/Toolbox';
import { OptionsBar } from '../ui/OptionsBar';

const EMPTY: PlaybackState = { clip: null, time: 0, duration: 0, frame: 0, frameCount: 0, paused: false, timeScale: 1 };
const DEFAULT_OPTIONS: Options = { shadows: false, skeleton: false, wireframe: false, background: '#1b1d23', autoRotate: 0, mirror: false, albedo: 'baked', shading: 'game', env: loadLastEnv() };

/** Everything the user changed on one model, captured when the GPU context is lost and re-applied after the rebuild. */
interface Snapshot {
  id: string; variantSel: Record<string, number>; groupSel: Record<string, string>; partsVis: Record<string, boolean>;
  palettes: Record<string, { materials: string[]; indices: number[] }>; heldItem: string; heldBone: string;
  clip: string | null; frame: number; paused: boolean; timeScale: number;
  cam: { fov: number; position: [number, number, number]; target: [number, number, number] };
}
const lowGpu = () => { try { return sessionStorage.getItem('lowGpu') === '1'; } catch { return false; } };

/** Stays mounted while the model changes (single page): options persist, only the scene is rebuilt. */
export function Viewer({ game, id, onSelect }: { game: string; id: string; onSelect: (m: IndexModel) => void }) {
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
  const [index, setIndex] = useState<GameIndex | null>(null);
  const [sidebarW, setSidebarW] = useState(() => {
    try { return Number(localStorage.getItem('sidebarW')) || 340; } catch { return 340; }
  });
  const modelUrl = useRef('');
  const [items, setItems] = useState<IndexModel[]>([]);
  const [heldItem, setHeldItem] = useState('');
  const [heldBone, setHeldBone] = useState('');
  const [heldOnly, setHeldOnly] = useState(true);
  const itemsRef = useRef<IndexModel[]>([]);
  // GPU context-loss recovery: the scene is rebuilt in place and the per-model state below is re-applied.
  const [reloadKey, setReloadKey] = useState(0);
  const [gpuFailed, setGpuFailed] = useState(false);
  const snapRef = useRef<Snapshot | null>(null);
  const lossTimes = useRef<number[]>([]);
  const palApplied = useRef<Record<string, { materials: string[]; indices: number[] }>>({});
  const live = useRef({ variantSel, groupSel, partsVis, heldItem, heldBone });
  live.current = { variantSel, groupSel, partsVis, heldItem, heldBone };
  const heldOnlyRef = useRef(true);

  useEffect(() => {
    loadGameIndex(game).then(setIndex).catch(() => { /* the load effect reports errors */ });
  }, [game]);
  const [panelsHidden, setPanelsHidden] = useState(false);

  // Reference panel: calibrated gameplay screenshots (refs/<game>/index.json, served at /refs/)
  const [refs, setRefs] = useState<RefEntry[]>([]);
  const [refActive, setRefActive] = useState<RefEntry | null>(null);
  const [refMode, setRefMode] = useState<RefMode>('off');
  const [refOpacity, setRefOpacity] = useState(0.5);
  const [refShowAll, setRefShowAll] = useState(false);
  useEffect(() => { loadRefIndex(game).then((ix) => setRefs(ix?.refs ?? [])); }, [game]);
  useEffect(() => { setRefActive(null); setRefMode('off'); }, [id]);
  const applyRef = useCallback((r: RefEntry) => {
    const vs = sceneRef.current; if (!vs?.anim) return;
    vs.anim.play(r.clip); vs.anim.setPaused(true); vs.anim.scrubToFrame(r.frame);
    const s = vs.anim.state(); if (s) setPb(s);
    vs.setCameraPose(r.camera.fov, r.camera.position, r.camera.target);
  }, []);
  const pickRef = (r: RefEntry) => {
    if (r.model !== id) { const m = index?.models.find((x) => x.id === r.model); if (m) onSelect(m); return; }
    setRefActive(r);
    if (refMode === 'off') setRefMode('split');
    const pr = LIGHTING_PRESETS.find((p) => p.id === r.preset);
    if (pr) patchOptions({ env: { ...pr.env } });
    // the camera pose depends on the canvas aspect, which the split/overlay layout changes: apply after the layout settles
    requestAnimationFrame(() => requestAnimationFrame(() => applyRef(r)));
  };

  /** Re-apply the user's per-model choices on a freshly rebuilt scene (options come from optionsRef). */
  const restoreSnapshot = async (vs: ViewerScene, man: Manifest, snap: Snapshot) => {
    for (const g of man.variantGroups) {
      const v = snap.variantSel[g.name];
      if (v !== undefined && g.options.some((o) => o.variant === v)) await vs.variants?.select(v);
    }
    setVariantSel((s) => ({ ...s, ...snap.variantSel }));
    for (const g of man.partGroups ?? []) {
      const opt = snap.groupSel[g.name];
      if (!opt) continue;
      for (const o of g.options) for (const node of o.nodes) {
        const acc = man.parts.find((p) => p.node === node)?.accessory;
        vs.setPartVisible(node, o.id === opt && !acc);
      }
    }
    setGroupSel((s) => ({ ...s, ...snap.groupSel }));
    for (const pal of Object.values(snap.palettes)) await vs.setPalette(pal.materials, pal.indices);
    palApplied.current = snap.palettes;
    setPalSel(Object.fromEntries(Object.entries(snap.palettes).map(([k, v]) => [k, v.indices])));
    for (const [node, v] of Object.entries(snap.partsVis)) vs.setPartVisible(node, v);
    setPartsVis((cur) => ({ ...cur, ...snap.partsVis }));
    if (snap.clip && vs.anim) {
      vs.anim.play(snap.clip); vs.anim.setTimeScale(snap.timeScale); vs.anim.scrubToFrame(snap.frame); vs.anim.setPaused(snap.paused);
      setPb(vs.anim.state());
    }
    const it = itemsRef.current.find((m) => m.id === snap.heldItem);
    setHeldBone(snap.heldBone); setHeldItem(snap.heldItem);
    if (it) await vs.setHeldItem(modelAsset(`${it.path}model.glb`), man.props?.find((p) => p.bone === snap.heldBone));
    vs.setCameraPose(snap.cam.fov, snap.cam.position, snap.cam.target);
  };

  // load model
  useEffect(() => {
    let cancelled = false;
    const host = canvasHost.current!;
    const vs = new ViewerScene(host, lowGpu());
    sceneRef.current = vs;
    vs.onFrame = setPb;
    vs.onPartsState = (v, d) => { setPartsVis(v); setPartsDriven(d); };
    // GPU context lost: three.js cannot rebuild a loaded scene in place, so snapshot the user's state, then
    // rebuild the scene on a fresh canvas (reloadKey) and re-apply it. After a loss we render at reduced
    // quality, and give up (no reload loop) after 2 losses within a minute.
    let recovering = false;
    vs.onContextChange = (lost) => {
      if (!lost || recovering) return;
      recovering = true;
      const now = Date.now();
      lossTimes.current = [...lossTimes.current.filter((t) => now - t < 60000), now];
      try { sessionStorage.setItem('lowGpu', '1'); } catch { /* ignore */ }
      if (lossTimes.current.length > 2) { setGpuFailed(true); setError('The graphics context keeps getting lost (GPU reset or out of memory). Close other tabs or restart the browser, then retry.'); return; }
      const st = vs.anim?.state();
      const l = live.current;
      snapRef.current = {
        id, variantSel: l.variantSel, groupSel: l.groupSel, partsVis: l.partsVis, palettes: palApplied.current,
        heldItem: l.heldItem, heldBone: l.heldBone,
        clip: st?.clip ?? null, frame: st?.frame ?? 0, paused: st?.paused ?? false, timeScale: st?.timeScale ?? 1,
        cam: { fov: vs.camera.fov, position: vs.camera.position.toArray() as [number, number, number], target: vs.controls.target.toArray() as [number, number, number] },
      };
      setError('The graphics context was lost. Restoring the model and your settings…');
      setTimeout(() => setReloadKey((k) => k + 1), 500);
    };
    (window as any).__viewer = vs; // handy for debugging
    // Carry the current options over to the new scene (the rest are applied after load).
    const o = optionsRef.current;
    vs.setShadingMode(o.shading); vs.setEnvironment(o.env); vs.setBackground(o.background);
    vs.setAutoRotate(o.autoRotate); vs.setMirror(o.mirror); vs.setShadows(o.shadows); vs.setWireframe(o.wireframe);
    setManifest(null); setError(null); setGpuFailed(false); setProgress(0); setHeldItem(''); setPb(EMPTY); setVariantColors({});
    const snap = snapRef.current?.id === id ? snapRef.current : null;
    snapRef.current = null;
    if (!snap) { palApplied.current = {}; lossTimes.current = []; }
    (async () => {
      try {
        const index = await loadGameIndex(game);
        const entry = index.models.find((m) => m.id === id);
        if (!entry) throw new Error(`Model not found in index: ${game}/${id}`);
        const objs = index.models.filter((m) => m.kind === 'object').sort((a, b) => a.id.localeCompare(b.id));
        itemsRef.current = objs; setItems(objs);
        setGameName(index.name);
        const man: Manifest = await (await fetch(modelAsset(entry.path + 'manifest.json'))).json();
        // manifests can contain duplicate clip names; the glTF mixer can only address one per name
        const seen = new Set<string>();
        man.clips = man.clips.filter((c) => !seen.has(c.name) && !!seen.add(c.name));
        modelUrl.current = modelAsset(entry.path + man.files.model);
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
        setPalSel({}); if (!snap) palApplied.current = {};
        setHeldBone(man.props?.find((p) => p.side === 'right')?.bone ?? man.props?.[0]?.bone ?? '');
        vs.setHeldOnlyInClips(heldOnlyRef.current);
        if (snap) await restoreSnapshot(vs, man, snap);
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
  }, [game, id, reloadKey]);

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
      else if (e.key === 'Tab') { setPanelsHidden((h) => !h); e.preventDefault(); }
      else if (e.key === 'f' || e.key === 'F') sceneRef.current?.setCameraPreset('fit');
      else if (e.key >= '1' && e.key <= '5') sceneRef.current?.setCameraPreset((['front', 'side', 'back', 'top', 'vs'] as const)[Number(e.key) - 1]);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step, togglePause, cycleScale]);

  const patchOptions = (patch: Partial<Options>) => {
    optionsRef.current = { ...optionsRef.current, ...patch };
    setOptions(optionsRef.current);
    if (patch.env) storeLastEnv(patch.env);
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
    void sceneRef.current?.setHeldItem(it ? modelAsset(`${it.path}model.glb`) : null, prop);
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
    palApplied.current = { ...palApplied.current, [key]: { materials: entry.materials, indices: next } };
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

  const visibleParts = manifest ? manifest.parts.filter((p) => p.lod === 0 && (!p.group || groupSel[p.group] === p.option)) : [];

  return (
    <div className={'viewer' + (panelsHidden ? ' no-panels' : '')} style={{ ['--sidebar-w' as string]: sidebarW + 'px' }}>
      <OptionsBar game={game} gameName={gameName} title={manifest?.displayName ?? id} options={options} onChange={patchOptions}
        panelsHidden={panelsHidden} onTogglePanels={() => setPanelsHidden((h) => !h)} />
      <div className="workspace">
        <Toolbox
          options={options}
          onChange={patchOptions}
          disabled={!manifest}
          onCamera={(p) => sceneRef.current?.setCameraPreset(p)}
          onRecenter={() => sceneRef.current?.recenter()}
          onFullscreen={() => { stage.current?.requestFullscreen?.(); }}
          onPng={() => manifest && sceneRef.current?.png(`${manifest.id}.png`)}
          onGlb={() => manifest && void exportGlb(modelUrl.current, `${manifest.id}.glb`)}
        />
        <aside className="leftrail">
          <Panel title="Models" grow><ModelPicker index={index} id={id} onSelect={onSelect} /></Panel>
          <Panel title={`Animations${clips.length ? ` (${clips.length})` : ''}`} grow>
            <AnimationsSection clips={clips} state={pb} onSelect={playClip} />
          </Panel>
        </aside>
        <div className="center">
          <div className={'stage' + (refActive && refMode !== 'off' ? ` ref-${refMode}` : '')} ref={stage}
            style={refActive ? { ['--ref-ar' as string]: `${refActive.width} / ${refActive.height}` } : undefined}>
            <div className="canvas-host" ref={canvasHost} />
            {refActive && refMode === 'split' && <img className="ref-side-img" src={refActive.image} alt="" />}
            {refActive && refMode === 'overlay' && <img className="ref-img" src={refActive.image} alt="" style={{ opacity: refOpacity }} />}
            {progress !== null && !error && <div className="overlay">Loading model{progress ? ` ${Math.round(progress * 100)}%` : '...'}</div>}
            {error && <div className="overlay err">{error}{gpuFailed && <> <button className="btn small" onClick={() => { lossTimes.current = []; setReloadKey((k) => k + 1); }}>Retry</button></>}</div>}
          </div>
          <Timeline
            state={pb}
            onPrev={() => step(-1)}
            onNext={() => step(1)}
            onTogglePause={togglePause}
            onScrub={(f) => { sceneRef.current?.anim?.scrubToFrame(f); const s = sceneRef.current?.anim?.state(); if (s) setPb(s); }}
            onCycleScale={cycleScale}
          />
        </div>
        <div className="handle" onPointerDown={startDrag} role="separator" aria-orientation="vertical" />
        <aside className="sidebar">
          {manifest && (
            <Tabs storageKey="rightTab" tabs={[
              {
                id: 'look', label: 'Appearance', content: (
                  <SplitPane storageKey="partsPaneH" bottom={
                    <PartsSection parts={visibleParts} visible={partsVis} driven={partsDriven} onToggle={togglePart} onAll={allParts}>
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
                    </PartsSection>
                  } top={
                  <>
                    <VariationsSection
                      groups={manifest.variantGroups}
                      selected={variantSel}
                      colors={variantColors}
                      onSelect={(g, v) => { setVariantSel((s) => ({ ...s, [g]: v })); void sceneRef.current?.variants?.select(v); }}
                    />
                    {(manifest.partGroups?.length ?? 0) > 0 && (
                      <PartGroupsSection groups={manifest.partGroups!} selected={groupSel} palettes={palSel} onSelect={selectPartOption} onPalette={selectPalette} />
                    )}
                  </>
                  } />
                ),
              },
              {
                id: 'ref', label: 'Reference', hidden: !refs.length, content: (
                  <ReferencePanel refs={refs} modelId={id} active={refActive} mode={refMode} opacity={refOpacity} showAll={refShowAll}
                    onPick={pickRef} onMode={(m) => { setRefMode(m); if (refActive) requestAnimationFrame(() => requestAnimationFrame(() => applyRef(refActive))); }}
                    onOpacity={setRefOpacity} onShowAll={setRefShowAll} onReapply={() => refActive && applyRef(refActive)} />
                ),
              },
              {
                id: 'render', label: 'Render & Info', content: (
                  <>
                    <Accordion title="Render"><RenderPanel options={options} onChange={patchOptions} /></Accordion>
                    <InfoSection manifest={manifest} gameName={gameName} />
                  </>
                ),
              },
            ]} />
          )}
        </aside>
      </div>
    </div>
  );
}
