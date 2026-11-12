import { useCallback, useEffect, useRef, useState } from 'react';
import type { GameIndex, IndexModel, Manifest, PaletteEntry, RefEntry } from '../types';
import { loadGameIndex, loadRefIndex, modelAsset } from '../data';
import { DEFAULT_ENV, LIGHTING_PRESETS, type Environment } from '../scene/environment';
import { ReferencePanel, type RefMode } from '../ui/ReferencePanel';
import { RefImageLayer } from '../ui/RefImageLayer';
import { RefImageSection } from '../ui/RefImageSection';
import { DEFAULT_SETTINGS as REF_DEFAULTS, DEFAULT_TRANSFORM, fitSize, imageToStage, loadImageSize, loadLast, loadPins, sampleUrl, storeLast,
  storePins, transformFor, type RefImage, type RefPin, type RefSettings, type RefTransform } from '../scene/refImage';
import { solveCamera, type SolveResult } from '../scene/camSolve';
import { autoLight } from '../scene/autoLight';
import { loadExcludes, storeExcludes, type RefBox } from '../scene/refImage';
import type { SolveSummary } from '../ui/RefImageSection';
import { loadLastEnv, storeLastEnv } from '../scene/userPresets';
import { ViewerScene, type CameraOrbit } from '../scene/ViewerScene';
import { Flyout } from '../ui/Flyout';
import { CameraPanel } from '../ui/CameraPanel';
import { TIME_SCALES, type PlaybackState } from '../scene/animation';
import { DEFAULT_EXPORT, loadExportOptions, saveExportOptions, loadGlbOptions, saveGlbOptions, type ExportFormat, type ExportOptions, type GlbExportOptions } from '../scene/exportOptions';
import { ExportPanel } from '../ui/ExportPanel';
import { SPRITE_SILHOUETTE } from '../scene/spriteStyle';
import { DEFAULT_GIF_EXPORT, type GifExportOptions } from '../scene/ViewerScene';
import { download } from '../scene/exporter';
import { DEFAULT_STL, type StlOptions } from '../scene/stl';
import type { OutlineState } from '../scene/outline';

/** Render-tab outline options -> ViewerScene state (the sprite style keeps its own silhouette colour). */
function outlineState(o: Options): OutlineState {
  const sprite = o.outlineStyle === 'sprite';
  return { on: o.outline, width: o.outlineWidth, color: sprite ? (o.outlineSpriteColor ?? SPRITE_SILHOUETTE) : o.outlineColor, tinted: o.outlineTinted,
    style: sprite ? 'sprite' : (o.outlineStyle ?? 'auto'), pixel: o.outlinePixel ?? 3 };
}

const GIF_KEY = 'spriteGif';
const formatBytes = (n: number) => (n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);
/** Popover heights, to keep each export popover on screen. */
const EXPORT_H: Record<ExportFormat, number> = { png: 470, gif: 300, stl: 250, glb: 380 };

const STL_KEY = 'stlExport';
function loadStlOptions(): StlOptions {
  try { const raw = localStorage.getItem(STL_KEY); if (raw) return { ...DEFAULT_STL, ...JSON.parse(raw) }; } catch { /* storage unavailable */ }
  return { ...DEFAULT_STL };
}

function loadGifOptions(): GifExportOptions {
  try { const raw = localStorage.getItem(GIF_KEY); if (raw) return { ...DEFAULT_GIF_EXPORT, ...JSON.parse(raw) }; } catch { /* storage unavailable */ }
  return { ...DEFAULT_GIF_EXPORT };
}
import { InfoSection } from '../ui/InfoSection';
import { AnimationsSection } from '../ui/AnimationsSection';
import { VariationsSection } from '../ui/VariationsSection';
import { HeldItemSection } from '../ui/HeldItemSection';
import { PartsSection } from '../ui/PartsSection';
import { PartGroupsSection } from '../ui/PartGroupsSection';
import { HologramSection, defaultHologram } from '../ui/HologramSection';
import { JawSection } from '../ui/JawSection';
import type { HologramState } from '../scene/hologram';
import { RenderPanel, type Options } from '../ui/OptionsSection';
import { Accordion } from '../ui/Accordion';
import { SplitPane } from '../ui/SplitPane';
import { ModelPicker } from '../ui/ModelPicker';
import { Panel } from '../ui/Panel';
import { Tabs } from '../ui/Tabs';
import { Timeline } from '../ui/Timeline';
import { Toolbox } from '../ui/Toolbox';
import { OptionsBar } from '../ui/OptionsBar';
import { SavedLooksSection } from '../ui/SavedLooksSection';
import { deleteLook, loadLooks, saveLook, setDefaultLook, startupLook, useLooks, type Look } from '../scene/savedLooks';

const EMPTY: PlaybackState = { clip: null, time: 0, duration: 0, frame: 0, frameCount: 0, paused: false, timeScale: 1, loop: true };
const DEFAULT_OPTIONS: Options = { shadows: false, skeleton: false, wireframe: false, outline: false, outlineWidth: 2, outlineColor: '#14121a', outlineTinted: false, outlineStyle: 'auto', outlinePixel: 3, outlineSpriteColor: SPRITE_SILHOUETTE, background: '#1b1d23', autoRotate: 0, mirror: false, albedo: 'baked', shading: 'game', env: loadLastEnv() };

/** Everything the user changed on one model (same shape as a saved look), captured when the GPU context is lost and
 *  re-applied after the rebuild of the same model. */
interface Snapshot { id: string; look: Look }
/** `?look=<name>`: the saved look to apply on open instead of the default one (tools/*.mjs use it). */
const urlLook = () => { try { return new URLSearchParams(window.location.search).get('look'); } catch { return null; } };
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
  const [lockParts, setLockParts] = useState(false);
  const [partsVis, setPartsVis] = useState<Record<string, boolean>>({});
  const [variantSel, setVariantSel] = useState<Record<string, number>>({});
  const [groupSel, setGroupSel] = useState<Record<string, string>>({});
  const [palSel, setPalSel] = useState<Record<string, number[]>>({});
  const [variantColors, setVariantColors] = useState<Record<number, string | null>>({});
  // Appearance > Hologram. On/off carries over model switches; the look dials (opacity, glow, gradient) persist.
  const [hologram, setHologramS] = useState<HologramState>(() => {
    let look = {};
    try { look = JSON.parse(localStorage.getItem('hologram.look') ?? '{}'); } catch { /* none */ }
    return { ...defaultHologram([]), ...look, on: false };
  });
  const hologramRef = useRef(hologram);
  const setHologram = (st: HologramState) => {
    hologramRef.current = st; setHologramS(st);
    sceneRef.current?.setHologramState(st);
    try { localStorage.setItem('hologram.look', JSON.stringify({ opacity: st.opacity, edgeGlow: st.edgeGlow, gradientStrength: st.gradientStrength })); } catch { /* ignore */ }
  };
  const [jaw, setJawS] = useState(0);
  const setJaw = (v: number) => { setJawS(v); sceneRef.current?.setJawOpen(v); };
  const [options, setOptions] = useState<Options>(DEFAULT_OPTIONS);
  const optionsRef = useRef<Options>(DEFAULT_OPTIONS);
  const [index, setIndex] = useState<GameIndex | null>(null);
  const [railSplit, setRailSplit] = useState(() => {
    try { const f = Number(localStorage.getItem('railSplit')); return f >= 0.1 && f <= 0.9 ? f : 0.5; } catch { return 0.5; }
  });
  const [sidebarW, setSidebarW] = useState(() => {
    try { return Number(localStorage.getItem('sidebarW')) || 340; } catch { return 340; }
  });
  const modelUrl = useRef('');
  // null until the user configures an export: the PNG button then exports the live canvas as before.
  const [exportOpts, setExportOpts] = useState<ExportOptions | null>(loadExportOptions);
  const [camOrbit, setCamOrbit] = useState<CameraOrbit | null>(null);
  const [camPanelTop, setCamPanelTop] = useState<number | null>(null);
  const [exportFly, setExportFly] = useState<{ format: ExportFormat; top: number } | null>(null);
  const patchExport = (patch: Partial<ExportOptions>) => {
    const next = { ...(exportOpts ?? DEFAULT_EXPORT), ...patch };
    setExportOpts(next); saveExportOptions(next);
  };
  const [gifOpts, setGifOpts] = useState<GifExportOptions>(loadGifOptions);
  const [stlOpts, setStlOpts] = useState<StlOptions>(loadStlOptions);
  const [stlInfo, setStlInfo] = useState<string | null>(null);
  const patchStl = (patch: Partial<StlOptions>) => {
    setStlOpts((o) => { const next = { ...o, ...patch }; try { localStorage.setItem(STL_KEY, JSON.stringify(next)); } catch { /* ignore */ } return next; });
  };
  const [glbOpts, setGlbOpts] = useState<GlbExportOptions>(loadGlbOptions);
  const [glbInfo, setGlbInfo] = useState<string | null>(null);
  const [glbBusy, setGlbBusy] = useState(false);
  const patchGlb = (patch: Partial<GlbExportOptions>) => {
    setGlbOpts((g) => { const next = { ...g, ...patch }; saveGlbOptions(next); return next; });
  };
  const [gifProgress, setGifProgress] = useState<string | null>(null);
  const patchGif = (patch: Partial<GifExportOptions>) => {
    setGifOpts((g) => { const next = { ...g, ...patch }; try { localStorage.setItem(GIF_KEY, JSON.stringify(next)); } catch { /* ignore */ } return next; });
  };
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
  // Saved looks (Appearance tab): the look applied last, and the state before it (Revert)
  const looks = useLooks(game, id);
  const [currentLook, setCurrentLook] = useState<string | null>(null);
  const [revertLook, setRevertLook] = useState<Look | null>(null);
  useEffect(() => { setCurrentLook(null); setRevertLook(null); }, [game, id]);

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

  // Free reference image (samples/ folder, upload or drop): screen space, independent of the camera
  const [refImg, setRefImg] = useState<RefImage | null>(null);
  const [refSet, setRefSet] = useState<RefSettings>(REF_DEFAULTS);
  const [refXf, setRefXf] = useState(false);
  const [dropping, setDropping] = useState(false);
  const refBgHidden = !!refImg && refSet.visible && refSet.layer === 'background';
  const refBgHiddenRef = useRef(refBgHidden);
  refBgHiddenRef.current = refBgHidden;
  useEffect(() => { sceneRef.current?.setBackgroundVisible(!refBgHidden); }, [refBgHidden]);
  useEffect(() => {
    const last = loadLast(); if (!last) return;
    const src = sampleUrl(last.samplePath);
    loadImageSize(src).then(({ w, h }) => {
      setRefImg((cur) => cur ?? { src, name: last.samplePath.split('/').pop()!, samplePath: last.samplePath, w, h });
      setRefSet(last.settings);
    }).catch(() => { /* sample gone */ });
  }, []);
  useEffect(() => { if (refImg) storeLast(refImg.samplePath, refSet); }, [refImg, refSet]);
  const showRefImage = async (src: string, name: string, samplePath: string | null) => {
    try {
      const { w, h } = await loadImageSize(src);
      setRefImg((old) => { if (old && !old.samplePath && old.src !== src) URL.revokeObjectURL(old.src); return { src, name, samplePath, w, h }; });
      setRefSet((s) => ({ ...s, visible: true, transform: (samplePath && transformFor(samplePath)) || { ...DEFAULT_TRANSFORM } }));
      setRefMode('off'); // the calibrated overlay reshapes the canvas: one reference at a time
    } catch { if (!samplePath) URL.revokeObjectURL(src); }
  };
  const refFile = (f: File) => { if (f.type.startsWith('image/')) void showRefImage(URL.createObjectURL(f), f.name, null); };
  const clearRefImage = () => {
    if (refImg && !refImg.samplePath) URL.revokeObjectURL(refImg.src);
    setRefImg(null); setRefXf(false); setPlacing(false); storeLast(null, refSet);
  };
  // Camera auto-align pins (image point <-> model surface point), per sample image + model
  const [pins, setPinsState] = useState<RefPin[]>([]);
  const [placing, setPlacing] = useState(false);
  const [lockFov, setLockFov] = useState(true);
  const [solveRes, setSolveRes] = useState<SolveSummary | null>(null);
  const [fullFrame, setFullFrame] = useState(true);
  const [solveRoll, setSolveRoll] = useState(true);
  const [pivotId, setPivotId] = useState<number | null>(null);
  const [fine, setFine] = useState(false);
  const [search, setSearch] = useState<string | null>(null);
  // Auto lighting (scene/autoLight.ts)
  const [lightBusy, setLightBusy] = useState<string | null>(null);
  const [lightRes, setLightRes] = useState<{ presetLabel: string | null; before: number; after: number; dEBefore: number; dEAfter: number; shift: [number, number]; seconds: number } | null>(null);
  const [lightErr, setLightErr] = useState<string | null>(null);
  const [lightUndo, setLightUndo] = useState<Environment | null>(null);
  const [lightScan, setLightScan] = useState(true);
  const [lightReg, setLightReg] = useState(true);
  const [excluding, setExcluding] = useState(false);
  const [excludes, setExcludesState] = useState<RefBox[]>([]);
  const lightCancel = useRef(false);
  useEffect(() => { setExcludesState(refImg?.samplePath ? loadExcludes(refImg.samplePath) : []); setLightRes(null); setLightErr(null); }, [refImg?.src]); // eslint-disable-line react-hooks/exhaustive-deps
  const setExcludes = (b: RefBox[]) => { setExcludesState(b); if (refImg?.samplePath) storeExcludes(refImg.samplePath, b); };
  /** Draws the reference as it sits over the canvas into a probe-size context; ignored areas are cleared (alpha 0). */
  const loadedRef = useRef<{ src: string; el: HTMLImageElement } | null>(null);
  const runAutoLight = async () => {
    const vs = sceneRef.current, host = stage.current, img = refImg;
    if (!vs || !host || !img || lightBusy) return;
    freezePose(); setPlacing(false); setExcluding(false); setRefXf(false);
    setLightErr(null); setLightRes(null);
    if (loadedRef.current?.src !== img.src) {
      const el = new Image(); el.src = img.src; await el.decode();
      loadedRef.current = { src: img.src, el };
    }
    const el = loadedRef.current.el, t = refSet.transform, boxes = excludes;
    const draw = (ctx: CanvasRenderingContext2D, W: number) => {
      const sr = host.getBoundingClientRect(), cr = vs.canvasRect();
      const k = W / cr.width;
      const f = fitSize(img, sr.width, sr.height), bw = f.w * t.scale, bh = f.h * t.scale;
      ctx.setTransform(k, 0, 0, k, 0, 0);
      ctx.translate(sr.left - cr.left + t.x * sr.width, sr.top - cr.top + t.y * sr.height);
      ctx.rotate(t.rotation * Math.PI / 180);
      if (t.flipX) ctx.scale(-1, 1);
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(el, -bw / 2, -bh / 2, bw, bh);
      for (const b of boxes) ctx.clearRect(-bw / 2 + b.u0 * bw, -bh / 2 + b.v0 * bh, (b.u1 - b.u0) * bw, (b.v1 - b.v0) * bh);
      ctx.setTransform(1, 0, 0, 1, 0, 0);
    };
    const start = optionsRef.current.env;
    lightCancel.current = false;
    setLightBusy('Starting…');
    try {
      const r = await autoLight(vs, draw, {
        presets: LIGHTING_PRESETS, start, scanPresets: lightScan, register: lightReg,
        onProgress: setLightBusy, cancelled: () => lightCancel.current,
      });
      setLightUndo(start);
      patchOptions({ env: r.env });
      setLightRes({ presetLabel: r.presetId ? LIGHTING_PRESETS.find((p) => p.id === r.presetId)?.label ?? r.presetId : null,
        before: r.before, after: r.after, dEBefore: r.dEBefore, dEAfter: r.dEAfter, shift: r.shift, seconds: r.ms / 1000 });
    } catch (e) {
      vs.setEnvironment(start);
      const msg = String((e as Error).message ?? e);
      setLightErr(msg === 'cancelled' ? 'Cancelled.' : msg);
    } finally { setLightBusy(null); }
  };
  /** Solved camera lens, relative to the reference image ('image') or the canvas ('canvas'); null = free camera. */
  const [lensLock, setLensLock] = useState<{ kind: 'image' | 'canvas'; fov: number; roll: number } | null>(null);
  const lensLockRef = useRef(lensLock); lensLockRef.current = lensLock;
  const refLive = useRef({ refImg, refSet }); refLive.current = { refImg, refSet };
  const searchCancel = useRef(false);
  const [solveErr, setSolveErr] = useState<string | null>(null);
  const pinKey = refImg?.samplePath ? `${game}/${id}` : null;
  useEffect(() => {
    setPinsState(refImg?.samplePath ? loadPins(refImg.samplePath, `${game}/${id}`) : []);
    setSolveRes(null); setSolveErr(null);
  }, [refImg?.src, game, id]); // eslint-disable-line react-hooks/exhaustive-deps
  const setPins = useCallback((p: RefPin[]) => {
    setPinsState(p); setSolveRes(null); setSolveErr(null);
    if (refImg?.samplePath && pinKey) storePins(refImg.samplePath, pinKey, p.filter((q) => q.anchor));
  }, [refImg?.samplePath, pinKey]);
  const pinsRef = useRef(pins); pinsRef.current = pins;
  const placingRef = useRef(placing); placingRef.current = placing;
  /** Pins only line up on a still pose: freeze playback on the current frame. */
  const freezePose = () => {
    const a = sceneRef.current?.anim; if (!a) return;
    if (!a.state()?.paused) { a.setPaused(true); const st = a.state(); if (st) setPb(st); }
  };
  const startPlacing = (v: boolean) => {
    setPlacing(v);
    if (v) setExcluding(false);
    if (v) freezePose();
    if (v) { setRefXf(false); setRefSet((s) => ({ ...s, visible: true })); }
    else if (pinsRef.current.some((q) => !q.anchor)) setPins(pinsRef.current.filter((q) => q.anchor));
  };
  type Use = { q: RefPin; w: [number, number, number] };
  /** Pins that resolve on the model in the current pose. */
  const usablePins = (): Use[] | null => {
    const vs = sceneRef.current; if (!vs) return null;
    const ready = pinsRef.current.filter((q) => q.anchor);
    const world = vs.anchorsWorld(ready.map((q) => q.anchor!));
    if (world.some((w) => !w)) return null;
    return ready.map((q, i) => ({ q, w: world[i]! }));
  };
  /**
   * Pin targets in half-height units around the principal point, plus the starting FOV/roll in the same frame.
   * Uncropped: the screenshot's own frame (its centre is the lens centre; FOV relative to its height; roll relative to
   * its edges), so the result does not depend on where the reference sits. Otherwise: the canvas frame.
   */
  const pinTargets = (use: Use[]) => {
    const vs = sceneRef.current!, host = stage.current!, img = refImg!, t = refSet.transform;
    const cr = vs.canvasRect(), sr = host.getBoundingClientRect();
    const L = lensLockRef.current;
    const tanHalf = (deg: number) => Math.tan(deg * Math.PI / 360);
    const camFov = vs.cameraPose().fov;
    if (fullFrame) {
      const aspect = img.w / img.h;
      const Hd = fitSize(img, sr.width, sr.height).h * t.scale; // displayed image height, px
      const targets = use.map(({ q }) => [((t.flipX ? 1 - q.u : q.u) - 0.5) * 2 * aspect, (0.5 - q.v) * 2] as [number, number]);
      const startFov = L?.kind === 'image' ? L.fov : Math.atan(tanHalf(camFov) * Hd / cr.height) * 360 / Math.PI;
      return { targets, pxPerUnit: Hd / 2, startRoll: L?.kind === 'image' ? L.roll : t.rotation, startFov };
    }
    const targets = use.map(({ q }) => {
      const p = imageToStage(q.u, q.v, img, t, sr.width, sr.height);
      const x = sr.left + p.x - cr.left, y = sr.top + p.y - cr.top;
      return [(x - cr.width / 2) / (cr.height / 2), (cr.height / 2 - y) / (cr.height / 2)] as [number, number];
    });
    return { targets, pxPerUnit: cr.height / 2, startRoll: L?.kind === 'canvas' ? L.roll : 0, startFov: L?.kind === 'canvas' ? L.fov : camFov };
  };
  /** The model conforms to the reference: the solve becomes the camera + a lens lock (FOV, lens shift, roll) that
   *  follows the reference image if it is moved later. The image itself is never changed. */
  const applySolve = (r: SolveResult, use: Use[]) => {
    const vs = sceneRef.current!;
    vs.setCameraPose(vs.cameraPose().fov, r.position, r.target);
    setLensLock({ kind: fullFrame ? 'image' : 'canvas', fov: r.fov, roll: r.roll });
    setSolveRes({ rmsPx: r.rmsPx, fov: r.fov, roll: r.roll, perPinPx: Object.fromEntries(use.map(({ q }, i) => [q.id, r.perPinPx[i]])) });
    setSolveErr(null);
  };
  const solveOpts = (use: Use[]) => ({ lockFov: lockFov || use.length < 4, lockRoll: !solveRoll });
  const solvePins = () => {
    const vs = sceneRef.current;
    if (!vs || !stage.current || !refImg) return;
    freezePose();
    const use = usablePins();
    if (!use || use.length < 3) { setSolveErr('Need at least 3 pins that resolve on this model.'); return; }
    try {
      const { targets, pxPerUnit, startRoll, startFov } = pinTargets(use);
      const r = solveCamera({ points: use.map((x) => x.w), targets, pxPerUnit, start: { ...vs.cameraPose(), fov: startFov }, startRoll, ...solveOpts(use) });
      applySolve(r, use);
    } catch (e) { setSolveErr(String((e as Error).message ?? e)); }
  };
  /** Solve the camera at every frame of the current clip (or a sample of every clip) and keep the best pose + camera. */
  const findBestFrame = async (allClips: boolean) => {
    const vs = sceneRef.current, a = vs?.anim;
    if (!vs || !a || !stage.current || !refImg || search) return;
    freezePose();
    const use0 = usablePins();
    if (!use0 || use0.length < 3) { setSolveErr('Need at least 3 pins that resolve on this model.'); return; }
    const { targets, pxPerUnit, startRoll, startFov } = pinTargets(use0);
    const opts = solveOpts(use0);
    const startClip = a.current, startFrame = a.state().frame;
    const names = allClips ? clips.map((c) => c.name).filter((n) => a.has(n)) : startClip ? [startClip] : [];
    const seed = solveCamera({ points: use0.map((x) => x.w), targets, pxPerUnit, start: { ...vs.cameraPose(), fov: startFov }, startRoll, ...opts });
    const found: { clip: string; frame: number; rms: number }[] = [];
    searchCancel.current = false;
    let lastYield = performance.now();
    for (let ci = 0; ci < names.length && !searchCancel.current; ci++) {
      if (a.current !== names[ci]) a.play(names[ci]);
      a.setPaused(true);
      const fc = Math.max(1, a.state().frameCount);
      const step = allClips ? Math.max(1, Math.round(fc / 30)) : 1;
      let warm: SolveResult = seed;
      for (let f = 0; f < fc && !searchCancel.current; f += step) {
        a.scrubToFrame(f);
        const w = vs.anchorsWorld(use0.map((x) => x.q.anchor!));
        if (w.some((x) => !x)) continue;
        const pts = w as [number, number, number][];
        // warm start from the previous frame, plus the seed camera (clip changes can jump)
        let r = solveCamera({ points: pts, targets, pxPerUnit, start: warm, startRoll: warm.roll, quick: true, ...opts });
        const r2 = solveCamera({ points: pts, targets, pxPerUnit, start: seed, startRoll: seed.roll, quick: true, ...opts });
        if (r2.rmsPx < r.rmsPx) r = r2;
        warm = r;
        found.push({ clip: names[ci], frame: f, rms: r.rmsPx });
        if (performance.now() - lastYield > 40) {
          setSearch(`${allClips ? `${names[ci]} (${ci + 1}/${names.length})` : 'frame'} ${f}/${fc - 1}`);
          await new Promise((res) => setTimeout(res, 0));
          lastYield = performance.now();
        }
      }
    }
    // refine the best few with the full multi-start solve
    found.sort((x, y) => x.rms - y.rms);
    let best: { clip: string; frame: number; r: SolveResult } | null = null;
    for (const c of searchCancel.current ? [] : found.slice(0, 6)) {
      if (a.current !== c.clip) a.play(c.clip);
      a.setPaused(true); a.scrubToFrame(c.frame);
      const w = vs.anchorsWorld(use0.map((x) => x.q.anchor!)) as [number, number, number][];
      const r = solveCamera({ points: w, targets, pxPerUnit, start: seed, startRoll: seed.roll, ...opts });
      if (!best || r.rmsPx < best.r.rmsPx) best = { clip: c.clip, frame: c.frame, r };
    }
    const target = best ?? (startClip ? { clip: startClip, frame: startFrame, r: null } : null);
    if (target) { if (a.current !== target.clip) a.play(target.clip); a.setPaused(true); a.scrubToFrame(target.frame); }
    const st = a.state(); setPb(st);
    setSearch(null);
    if (best) applySolve(best.r, use0);
    if (best) setSolveErr(`Best: ${best.clip} frame ${best.frame} (${best.r.rmsPx.toFixed(1)} px).`);
  };
  const solveRef = useRef(solvePins); solveRef.current = solvePins;
  const startPlacingRef = useRef(startPlacing); startPlacingRef.current = startPlacing;
  const setPinsRef = useRef(setPins); setPinsRef.current = setPins;
  // pivot lock: the chosen pin's model point stays on its image point while orbiting / zooming
  const pivotLive = useRef({ pivotId, refImg, refSet }); pivotLive.current = { pivotId, refImg, refSet };
  useEffect(() => { if (pivotId !== null && !pins.some((q) => q.id === pivotId && q.anchor)) setPivotId(null); }, [pins, pivotId]);
  useEffect(() => {
    const vs = sceneRef.current; if (!vs) return;
    if (pivotId === null) { vs.setPivot(null); return; }
    vs.setPivot(() => {
      const { pivotId: pid, refImg: img, refSet: rs } = pivotLive.current;
      const q = pinsRef.current.find((x) => x.id === pid);
      const host = stage.current, scene = sceneRef.current;
      if (!q?.anchor || !img || !host || !scene || !rs.visible) return null;
      const sr = host.getBoundingClientRect(), cr = scene.canvasRect();
      const p = imageToStage(q.u, q.v, img, rs.transform, sr.width, sr.height);
      const x = sr.left + p.x - cr.left, y = sr.top + p.y - cr.top;
      return { anchor: q.anchor, ndc: [x / cr.width * 2 - 1, 1 - y / cr.height * 2] };
    });
    return () => { sceneRef.current?.setPivot(null); };
  }, [pivotId, manifest]);
  useEffect(() => { sceneRef.current?.setFineControls(fine); }, [fine, manifest]);
  // lens lock -> camera every frame, recomputed from the reference's current placement
  useEffect(() => {
    const vs = sceneRef.current; if (!vs) return;
    vs.onLensCleared = () => setLensLock(null);
    vs.onCameraChange = setCamOrbit;
    if (!lensLock) { vs.setLens(null); return; }
    vs.setLens(() => {
      const L = lensLockRef.current, { refImg: img, refSet: rs } = pivotLive.current;
      const host = stage.current, scene = sceneRef.current;
      if (!L || !host || !scene) return null;
      if (L.kind === 'canvas' || !img) return { fov: L.fov, shift: [0, 0], roll: L.roll };
      const sr = host.getBoundingClientRect(), cr = scene.canvasRect(), t = rs.transform;
      const Hd = fitSize(img, sr.width, sr.height).h * t.scale;
      const fov = Math.atan(Math.tan(L.fov * Math.PI / 360) * cr.height / Hd) * 360 / Math.PI;
      const cx = sr.left + t.x * sr.width - cr.left, cy = sr.top + t.y * sr.height - cr.top;
      // CSS rotation is clockwise on screen; the lens roll is counter-clockwise relative to the image
      return { fov, shift: [cx - cr.width / 2, cy - cr.height / 2], roll: L.roll - t.rotation };
    });
  }, [lensLock, manifest]);
  useEffect(() => { setLensLock(null); }, [refImg?.src, id]);
  const setRefTransform = useCallback((t: RefTransform) => setRefSet((s) => ({ ...s, transform: t })), []);
  const stageFillHeight = () => {
    const el = stage.current; if (!el || !refImg) return null;
    return el.clientHeight / fitSize(refImg, el.clientWidth, el.clientHeight).h;
  };
  const applyRef = useCallback((r: RefEntry) => {
    const vs = sceneRef.current; if (!vs?.anim) return;
    vs.setLens(null); setLensLock(null);
    vs.anim.play(r.clip); vs.anim.setPaused(true); vs.anim.scrubToFrame(r.frame);
    const s = vs.anim.state(); if (s) setPb(s);
    vs.setCameraPose(r.camera.fov, r.camera.position, r.camera.target);
  }, []);
  const pickRef = (r: RefEntry) => {
    if (r.model !== id) { const m = index?.models.find((x) => x.id === r.model); if (m) onSelect(m); return; }
    setRefActive(r);
    setRefSet((s) => ({ ...s, visible: false })); setRefXf(false);
    if (refMode === 'off') setRefMode('split');
    const pr = LIGHTING_PRESETS.find((p) => p.id === r.preset);
    if (pr) patchOptions({ env: { ...pr.env } });
    // the camera pose depends on the canvas aspect, which the split/overlay layout changes: apply after the layout settles
    requestAnimationFrame(() => requestAnimationFrame(() => applyRef(r)));
  };

  /** The current per-model state as a look (appearance, held item, animation, camera + lens lock). */
  const captureLook = (vs: ViewerScene, name: string): Look => {
    const st = vs.anim?.state();
    const l = live.current, L = lensLockRef.current, r = refLive.current;
    // part overrides minus what the selected group options already imply (keeps configs.json small and readable)
    const implied = new Map<string, boolean>();
    for (const g of vs.manifest?.partGroups ?? []) for (const o of g.options) for (const node of o.nodes)
      implied.set(node, o.id === l.groupSel[g.name] && !vs.manifest!.parts.find((p) => p.node === node)?.accessory);
    const partsVis = Object.fromEntries(Object.entries(vs.partOverrides()).filter(([n, v]) => implied.get(n) !== v));
    return {
      name,
      appearance: { variantSel: { ...l.variantSel }, groupSel: { ...l.groupSel }, palettes: { ...palApplied.current }, partsVis,
        hologram: { ...hologramRef.current }, albedo: optionsRef.current.albedo },
      held: { item: l.heldItem, bone: l.heldBone, only: heldOnlyRef.current },
      anim: { clip: st?.clip ?? null, frame: st?.frame ?? 0, paused: st?.paused ?? false, timeScale: st?.timeScale ?? 1 },
      camera: { ...vs.cameraPose(), ...(L ? { lens: { ...L, samplePath: r.refImg?.samplePath ?? null, transform: { ...r.refSet.transform } } } : {}) },
    };
  };

  /**
   * Applies a look (saved look, `?look=`, or the context-loss snapshot) to the loaded scene. Order matters: variants ->
   * part groups -> palettes (awaited) -> part overrides -> hologram -> held item -> animation -> lighting -> camera (raw
   * pose, never the VS preset, which forces advent01_loop) -> lens lock on a later frame (after the lens-reset effect).
   * `reset` first returns parts/palettes to the model defaults (switching looks on an open model); missing sections are
   * left as they are. Options that are not in the look persist; `albedo` is a global option, so applying it carries over.
   */
  const applyLook = async (vs: ViewerScene, man: Manifest, look: Look, reset: boolean) => {
    const ap = look.appearance ?? {};
    const vsel: Record<string, number> = {};
    for (const g of man.variantGroups) {
      const v = ap.variantSel?.[g.name];
      if (v !== undefined && g.options.some((o) => o.variant === v)) { await vs.variants?.select(v, g.options.map((o) => o.variant)); vsel[g.name] = v; }
    }
    setVariantSel((s) => ({ ...s, ...vsel }));
    const groups = man.partGroups ?? [];
    if (reset) vs.clearPartOverrides();
    const gsel: Record<string, string> = {};
    for (const g of groups) {
      const want = ap.groupSel?.[g.name];
      const def = (g.options.find((o) => o.default) ?? g.options[0]).id;
      const cur = reset ? def : live.current.groupSel[g.name] ?? def;
      gsel[g.name] = want && g.options.some((o) => o.id === want) ? want : cur;
      if (!reset && gsel[g.name] === cur) continue;
      for (const o of g.options) for (const node of o.nodes) {
        const acc = man.parts.find((p) => p.node === node)?.accessory;
        vs.setPartVisible(node, o.id === gsel[g.name] && !acc);
      }
    }
    setGroupSel((s) => ({ ...s, ...gsel }));
    const pals = ap.palettes ?? {};
    if (reset) {
      const entries = groups.flatMap((g) => g.options).flatMap((o) => o.palettes ?? []);
      for (const key of Object.keys(palApplied.current)) {
        if (pals[key]) continue;
        const e = entries.find((p) => p.materials[0] === key);
        if (e) await vs.setPalette(e.materials, e.indices);
      }
    }
    for (const pal of Object.values(pals)) await vs.setPalette(pal.materials, pal.indices);
    palApplied.current = reset ? { ...pals } : { ...palApplied.current, ...pals };
    setPalSel(Object.fromEntries(Object.entries(palApplied.current).map(([k, v]) => [k, v.indices])));
    for (const [node, v] of Object.entries(ap.partsVis ?? {})) vs.setPartVisible(node, v);
    if (ap.hologram) setHologram({ ...hologramRef.current, ...ap.hologram });
    if (ap.albedo && ap.albedo !== optionsRef.current.albedo) patchOptions({ albedo: ap.albedo });
    if (look.held) {
      const h = look.held;
      const prop = man.props?.find((p) => p.bone === h.bone) ?? man.props?.find((p) => p.side === 'right') ?? man.props?.[0];
      const it = itemsRef.current.find((m) => m.id === h.item);
      setHeldOnly(h.only); heldOnlyRef.current = h.only; vs.setHeldOnlyInClips(h.only);
      if (prop) setHeldBone(prop.bone);
      setHeldItem(it ? it.id : '');
      if (prop) await vs.setHeldItem(it ? modelAsset(`${it.path}model.glb`) : null, prop);
    }
    const an = look.anim;
    if (an?.clip && vs.anim?.has(an.clip)) {
      vs.anim.play(an.clip); vs.anim.setTimeScale(an.timeScale); vs.anim.scrubToFrame(an.frame); vs.anim.setPaused(an.paused);
      setPb(vs.anim.state());
    }
    if (look.lighting) patchOptions({ env: { ...DEFAULT_ENV, ...look.lighting } });
    const cam = look.camera;
    if (cam) {
      vs.setCameraPose(cam.fov, cam.position, cam.target, cam.roll ?? 0);
      const L = cam.lens;
      if (L) {
        // an image lock only makes sense over the same reference sample; otherwise keep the effective lens relative to the canvas
        const img = refLive.current.refImg;
        const same = L.kind === 'image' && !!L.samplePath && img?.samplePath === L.samplePath;
        const lock = same ? { kind: 'image' as const, fov: L.fov, roll: L.roll }
          : { kind: 'canvas' as const, fov: L.kind === 'canvas' ? L.fov : cam.fov, roll: L.kind === 'canvas' ? L.roll : L.roll - (L.transform?.rotation ?? 0) };
        if (same && L.transform) setRefSet((st) => ({ ...st, transform: { ...L.transform! } }));
        requestAnimationFrame(() => requestAnimationFrame(() => { if (sceneRef.current === vs) setLensLock(lock); }));
      }
    }
  };

  const lookWhere = looks.backend === 'server' ? `out/${game}/${id}/configs.json` : 'this browser (localStorage)';
  const pickLook = async (name: string) => {
    const vs = sceneRef.current, lk = looks.looks.find((l) => l.name === name);
    if (!vs || !manifest || !lk) return;
    setRevertLook(captureLook(vs, '(before)'));
    setCurrentLook(name);
    await applyLook(vs, manifest, lk, true);
  };
  const storeLook = async (name: string, withLighting: boolean) => {
    const vs = sceneRef.current; if (!vs) return;
    const lk = captureLook(vs, name);
    const old = looks.looks.find((l) => l.name.toLowerCase() === name.toLowerCase());
    if (old?.default) lk.default = true;
    if (withLighting) lk.lighting = structuredClone(optionsRef.current.env);
    await saveLook(game, id, lk);
    setCurrentLook(old?.name ?? name);
  };

  // load model
  useEffect(() => {
    let cancelled = false;
    const host = canvasHost.current!;
    let vs: ViewerScene;
    try { vs = new ViewerScene(host, lowGpu()); } catch (e) {
      setManifest(null); setProgress(null);
      setError('WebGL is unavailable in this browser (' + String(e instanceof Error ? e.message : e) + '). Enable hardware acceleration / WebGL, update your graphics driver, or try another browser.');
      return;
    }
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
      snapRef.current = { id, look: captureLook(vs, '') };
      setError('The graphics context was lost. Restoring the model and your settings…');
      setTimeout(() => setReloadKey((k) => k + 1), 500);
    };
    (window as any).__viewer = vs; // handy for debugging
    // tools/*.mjs: GIF of the current clip as the view draws it, as a data URL (__sprite = old name).
    (window as any).__gif = (window as any).__sprite = async (opts: Partial<GifExportOptions> = {}) => {
      const b = await vs.animatedGif(opts);
      return await new Promise<string>((res) => { const r = new FileReader(); r.onload = () => res(r.result as string); r.readAsDataURL(b); });
    };
    (window as any).__stl = async (opts: Partial<StlOptions> = {}) => {
      const { blob } = vs.stl(opts);
      return await new Promise<string>((res) => { const r = new FileReader(); r.onload = () => res(r.result as string); r.readAsDataURL(blob); });
    };
    // tools: GLB as the view shows it (scene/glbExport.ts), as a data URL
    (window as any).__glb = async (opts: Partial<GlbExportOptions> = {}) => {
      const { blob } = await vs.glbExport(modelUrl.current, { ...loadGlbOptions(), ...opts }, () => {});
      return await new Promise<string>((res) => { const r = new FileReader(); r.onload = () => res(r.result as string); r.readAsDataURL(blob); });
    };
    // tools: capture / apply a look without the UI (render scripts that want a saved look's state)
    (window as any).__looks = { capture: (name = '') => captureLook(vs, name), apply: (look: Look) => (vs.manifest ? applyLook(vs, vs.manifest, look, true) : Promise.resolve()) };
    // Carry the current options over to the new scene (the rest are applied after load).
    const o = optionsRef.current;
    vs.setShadingMode(o.shading); vs.setEnvironment(o.env); vs.setBackground(o.background);
    vs.setAutoRotate(o.autoRotate); vs.setMirror(o.mirror); vs.setShadows(o.shadows); vs.setParticles(o.particles ?? true); vs.setEffectsAnimated(o.animateEffects ?? true); vs.setWireframe(o.wireframe);
    vs.setOutline(outlineState(o));
    vs.setBackgroundVisible(!refBgHiddenRef.current);
    setManifest(null); setError(null); setGpuFailed(false); setProgress(0); setHeldItem(''); setPb(EMPTY); setVariantColors({});
    const snap = snapRef.current?.id === id ? snapRef.current : null;
    snapRef.current = null;
    if (!snap) { palApplied.current = {}; lossTimes.current = []; }
    (async () => {
      const looksP = loadLooks(game, id).catch(() => null);
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
        setJawS(0); vs.setJawOpen(0);
        { const h = defaultHologram(man.holocast ?? [], hologramRef.current); hologramRef.current = h; setHologramS(h); vs.setHologramState(h); }
        // context-loss snapshot of this model, else `?look=<name>`, else the model's default saved look
        const start = snap?.look ?? startupLook((await looksP)?.looks ?? [], urlLook());
        if (cancelled) return;
        if (start) { await applyLook(vs, man, start, false); if (!snap) setCurrentLook(start.name); }
        setManifest(man);
        setProgress(null);
        document.title = `${man.displayName} - Pokémon Model Viewer`;
        // swatches
        const cols: Record<number, string | null> = {};
        const current = vs.variants?.current ?? 0;
        for (const g of man.variantGroups) for (const o of g.options) cols[o.variant] = (await vs.variants?.sampleColor(o.variant, g.options.map((x) => x.variant))) ?? null;
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
  const toggleLoop = useCallback(() => {
    const a = sceneRef.current?.anim; if (!a) return;
    a.setLoop(!a.state().loop); setPb(a.state());
  }, []);
  const cycleScale = useCallback(() => {
    const a = sceneRef.current?.anim; if (!a) return;
    a.setTimeScale(TIME_SCALES[(TIME_SCALES.indexOf(a.timeScale) + 1) % TIME_SCALES.length]); setPb(a.state());
  }, []);

  const refImgRef = useRef(refImg); refImgRef.current = refImg;
  const refXfRef = useRef(refXf); refXfRef.current = refXf;
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t && ((t.tagName === 'INPUT' && ['text', 'number'].includes((t as HTMLInputElement).type)) || t.tagName === 'TEXTAREA')) return;
      if ((e.key === 'r' || e.key === 'R') && !e.ctrlKey && !e.metaKey) { if (refImgRef.current) { setPlacing(false); setRefXf((v) => !v); } return; }
      if ((e.key === 'a' || e.key === 'A') && !e.ctrlKey && !e.metaKey) { if (refImgRef.current) startPlacingRef.current(!placingRef.current); return; }
      if (placingRef.current) {
        if (e.key === 'Escape') {
          const ps = pinsRef.current;
          if (ps.some((q) => !q.anchor)) setPinsRef.current(ps.filter((q) => q.anchor)); else startPlacingRef.current(false);
          return;
        }
        if (e.key === 'Enter') { solveRef.current(); return; }
      }
      if (refXfRef.current) {
        if (e.key === 'Escape' || e.key === 'Enter') { setRefXf(false); return; }
        const el = stage.current;
        const d = e.shiftKey ? 10 : 1;
        const dx = e.key === 'ArrowLeft' ? -d : e.key === 'ArrowRight' ? d : 0;
        const dy = e.key === 'ArrowUp' ? -d : e.key === 'ArrowDown' ? d : 0;
        if ((dx || dy) && el) {
          e.preventDefault();
          setRefSet((s) => ({ ...s, transform: { ...s.transform, x: s.transform.x + dx / el.clientWidth, y: s.transform.y + dy / el.clientHeight } }));
          return;
        }
      }
      if (e.shiftKey && e.key.startsWith('Arrow') && !e.ctrlKey && !e.metaKey) {
        // Shift+arrows: 45° camera steps (Shift+←/→ turn, Shift+↑/↓ tilt)
        const r = { ArrowLeft: ['yaw', -45], ArrowRight: ['yaw', 45], ArrowUp: ['pitch', 45], ArrowDown: ['pitch', -45] } as const;
        const hit = r[e.key as keyof typeof r];
        if (hit) { sceneRef.current?.rotateCamera(hit[0], hit[1]); e.preventDefault(); return; }
      }
      if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
      else if (e.key === 'p' || e.key === 'P') togglePause();
      else if (e.key === 'l' || e.key === 'L') toggleLoop();
      else if (e.key === 't' || e.key === 'T') cycleScale();
      else if (e.key === 'Tab') { setPanelsHidden((h) => !h); e.preventDefault(); }
      else if (e.key === 'f' || e.key === 'F') sceneRef.current?.setCameraPreset('fit');
      else if (e.key >= '1' && e.key <= '8') sceneRef.current?.setCameraPreset((['front', 'side', 'back', 'top', 'vs', 'idle', 'battle', 'battleBack'] as const)[Number(e.key) - 1]);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [step, togglePause, cycleScale, toggleLoop]);

  const patchOptions = (patch: Partial<Options>) => {
    optionsRef.current = { ...optionsRef.current, ...patch };
    setOptions(optionsRef.current);
    if (patch.env) storeLastEnv(patch.env);
    const vs = sceneRef.current; if (!vs) return;
    if (patch.shadows !== undefined) vs.setShadows(patch.shadows);
    if (patch.particles !== undefined) vs.setParticles(patch.particles);
    if (patch.animateEffects !== undefined) vs.setEffectsAnimated(patch.animateEffects);
    if (patch.skeleton !== undefined) vs.setSkeleton(patch.skeleton);
    if (patch.wireframe !== undefined) vs.setWireframe(patch.wireframe);
    if (patch.outline !== undefined || patch.outlineWidth !== undefined || patch.outlineColor !== undefined || patch.outlineTinted !== undefined
      || patch.outlineStyle !== undefined || patch.outlinePixel !== undefined || patch.outlineSpriteColor !== undefined) {
      const o = { ...optionsRef.current, ...patch };
      vs.setOutline(outlineState(o));
    }
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

  const changeLock = (v: boolean) => {
    setLockParts(v);
    sceneRef.current?.setLockPartsToClip(v);
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
  /** Selected skin tone as a palette zone (1-based; the `skin` variant group's `toneN` options, default Tone 3). */
  const skinTone = (sel: Record<string, number> = variantSel): number => {
    const g = manifest?.variantGroups.find((x) => x.name === 'skin');
    const o = g?.options.find((x) => x.variant === sel.skin) ?? g?.options.find((x) => x.default);
    return Number(/^tone(\d+)/.exec(o?.name ?? '')?.[1] ?? 3);
  };
  /** Switching variants swaps the skin materials back to the pre-baked tone: palettes whose layer 1 is the skin are re-baked with the new tone. */
  const selectVariant = async (group: string, variant: number) => {
    const opts = manifest?.variantGroups.find((x) => x.name === group)?.options.map((o) => o.variant);
    const applied = await sceneRef.current?.variants?.select(variant, opts);
    if (applied === false || group !== 'skin') return; // overtaken by a newer pick: its own call re-bakes
    const tone = skinTone({ ...variantSel, skin: variant });
    for (const [key, ap] of Object.entries(palApplied.current)) {
      const entry = manifest?.partGroups?.flatMap((g) => g.options).flatMap((o) => o.palettes ?? []).find((p) => p.materials[0] === key);
      if (!entry?.skinLink) continue;
      const indices = [tone, ...ap.indices.slice(1).map((z, i) => (entry.skinLayers?.includes(i + 2) ? tone : z))];
      palApplied.current = { ...palApplied.current, [key]: { ...ap, indices } };
      setPalSel((s) => ({ ...s, [key]: indices }));
      await sceneRef.current?.setPalette(ap.materials, indices);
    }
  };
  /** Pick palette zone `zone` (1-based) for colour layer `layer` of a recolourable material group (outfit, hair, eyes...). */
  const selectPalette = (entry: PaletteEntry, layer: number, zone: number) => {
    const key = entry.materials[0];
    const next = [...(palSel[key] ?? entry.indices)];
    next[layer - 1] = zone;
    if (entry.skinLink) {
      next[0] = skinTone();
      for (const l of entry.skinLayers ?? []) next[l - 1] = skinTone();
    }
    if (entry.pairs) next[layer] = zone + 1; // a pair row sets iris (layer 2) and inner iris (layer 3) together
    setPalSel((s) => ({ ...s, [key]: next }));
    palApplied.current = { ...palApplied.current, [key]: { materials: entry.materials, indices: next } };
    void sceneRef.current?.setPalette(entry.materials, next);
  };
  const allParts = (v: boolean) => {
    for (const p of manifest?.parts ?? []) if (p.lod === 0 && (!p.group || groupSel[p.group] === p.option)) sceneRef.current?.setPartVisible(p.node, v);
  };

  const startRailDrag = (e: React.PointerEvent) => {
    e.preventDefault();
    const rail = (e.currentTarget as HTMLElement).parentElement!;
    const move = (ev: PointerEvent) => {
      const r = rail.getBoundingClientRect();
      const f = Math.min(Math.max((ev.clientY - r.top) / r.height, 0.1), 0.9);
      setRailSplit(f);
      try { localStorage.setItem('railSplit', String(f)); } catch { /* ignore */ }
    };
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
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
          cameraOpen={camPanelTop !== null}
          onCameraPanel={(top) => { setExportFly(null); setCamPanelTop((t) => (t === null ? top : null)); }}
          onFullscreen={() => { stage.current?.requestFullscreen?.(); }}
          exportOpen={exportFly?.format ?? null}
          onExport={(format, top) => { setCamPanelTop(null); setExportFly((f) => (f?.format === format ? null : { format, top })); }}
          refImage={!!refImg}
          refVisible={refSet.visible}
          refTransforming={refXf}
          onRefVisible={() => setRefSet((s) => ({ ...s, visible: !s.visible }))}
          onRefTransform={() => { setPlacing(false); setRefXf((v) => !v); }}
        />
        {camPanelTop !== null && (
          <Flyout title="Camera" top={Math.max(48, Math.min(camPanelTop, window.innerHeight - 330))} onClose={() => setCamPanelTop(null)} className="cam-fly">
            <CameraPanel orbit={manifest ? camOrbit : null}
              onChange={(p) => sceneRef.current?.setCameraOrbit(p)}
              onRotate={(axis, d) => sceneRef.current?.rotateCamera(axis, d)}
              onPreset={(p) => sceneRef.current?.setCameraPreset(p)}
              ranges={() => sceneRef.current?.cameraRanges() ?? null} />
          </Flyout>
        )}
        {exportFly !== null && (
          <ExportPanel format={exportFly.format} top={Math.max(48, Math.min(exportFly.top, window.innerHeight - EXPORT_H[exportFly.format]))} value={exportOpts ?? DEFAULT_EXPORT}
            onChange={patchExport} onClose={() => setExportFly(null)}
            onExport={() => { if (manifest) sceneRef.current?.png(`${manifest.id}.png`, exportOpts ?? DEFAULT_EXPORT); }}
            stl={stlOpts} onStlChange={patchStl} stlInfo={stlInfo}
            onStl={() => {
              const vs = sceneRef.current;
              if (!vs || !manifest) return;
              try {
                const { blob, triangles } = vs.stl(stlOpts);
                download(blob, `${manifest.id}.stl`);
                setStlInfo(`${triangles.toLocaleString()} triangles`);
              } catch (e) { console.error('[stl]', e); setStlInfo(String((e as Error).message ?? e)); }
            }}
            glb={glbOpts} onGlbChange={patchGlb} glbInfo={glbInfo} glbBusy={glbBusy}
            glbHas={{ clips: clips.length > 0, held: !!sceneRef.current?.hasHeld() }}
            onGlb={() => {
              const vs = sceneRef.current;
              if (!vs || !manifest || glbBusy) return;
              setGlbBusy(true); setGlbInfo('Loading…');
              const clip = glbOpts.animations === 'current' && vs.anim?.current ? `_${vs.anim.current.replace(/[^\w.-]+/g, '_')}` : '';
              vs.glbExport(modelUrl.current, glbOpts, (phase) => setGlbInfo(phase))
                .then(({ blob, bytes }) => { download(blob, `${manifest.id}${clip}.glb`); setGlbInfo(formatBytes(bytes)); })
                .catch((e) => { console.error('[glb]', e); setGlbInfo(String((e as Error).message ?? e)); })
                .finally(() => setGlbBusy(false));
            }}
            gif={gifOpts} onGifChange={patchGif} gifProgress={gifProgress}
            onGif={() => {
              const vs = sceneRef.current;
              if (!vs || !manifest || gifProgress !== null) return;
              const clip = vs.anim?.current ?? 'pose';
              setGifProgress('0%');
              vs.animatedGif(gifOpts, (d, t) => setGifProgress(`${Math.round((100 * d) / t)}%`))
                .then((b) => download(b, `${manifest.id}_${clip.replace(/[^\w.-]+/g, '_')}.gif`))
                .catch((e) => console.error('[gif]', e))
                .finally(() => setGifProgress(null));
            }} />
        )}
        <aside className="leftrail">
          <Panel title="Models" grow flex={railSplit}><ModelPicker index={index} id={id} onSelect={onSelect} /></Panel>
          <div className="rail-split" onPointerDown={startRailDrag} title="Drag to resize" />
          <Panel title={`Animations${clips.length ? ` (${clips.length})` : ''}`} grow flex={1 - railSplit}>
            <AnimationsSection clips={clips} state={pb} onSelect={playClip} />
          </Panel>
        </aside>
        <div className="center">
          <div className={'stage' + (refActive && refMode !== 'off' ? ` ref-${refMode}` : '') + (dropping ? ' dropping' : '')} ref={stage}
            style={{ ...(refActive ? { ['--ref-ar' as string]: `${refActive.width} / ${refActive.height}` } : {}), ...(refBgHidden ? { background: options.background } : {}) }}
            onDragOver={(e) => { if (e.dataTransfer.types.includes('Files')) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; setDropping(true); } }}
            onDragLeave={(e) => { if (!stage.current?.contains(e.relatedTarget as Node)) setDropping(false); }}
            onDrop={(e) => {
              e.preventDefault(); setDropping(false);
              const f = Array.from(e.dataTransfer.files).find((x) => x.type.startsWith('image/'));
              if (f) refFile(f);
            }}>
            <div className="canvas-host" ref={canvasHost} />
            {refImg && <RefImageLayer image={refImg} settings={refSet} transforming={refXf} onTransform={setRefTransform}
              excl={{ boxes: excludes, drawing: excluding, onBoxes: setExcludes }}
              pin={{
                pins, active: placing, onPins: setPins,
                pick: (x, y) => {
                  const vs = sceneRef.current; if (!vs) return null;
                  const r = vs.canvasRect();
                  return vs.pickSurface((x - r.left) / r.width * 2 - 1, 1 - (y - r.top) / r.height * 2);
                },
                project: (ps) => {
                  const vs = sceneRef.current; if (!vs) return ps.map(() => null);
                  const w = vs.anchorsWorld(ps.map((q) => q.anchor ?? { mesh: -1, name: '', tri: [0, 0, 0], bary: [0, 0, 0] }));
                  return vs.projectToClient(w);
                },
              }} />}
            {dropping && <div className="overlay drop-hint">Drop image to use as reference</div>}
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
            onToggleLoop={toggleLoop}
          />
        </div>
        <div className="handle" onPointerDown={startDrag} role="separator" aria-orientation="vertical" />
        <aside className="sidebar">
          {manifest && (
            <Tabs storageKey="rightTab" tabs={[
              {
                id: 'look', label: 'Appearance', content: (
                  <SplitPane storageKey="partsPaneH" bottom={
                    <PartsSection parts={visibleParts} visible={partsVis} driven={partsDriven} onToggle={togglePart} onAll={allParts} onHover={(n) => sceneRef.current?.setHoverPart(n)} lockToClip={lockParts} onLockChange={changeLock}>
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
                    <SavedLooksSection entry={looks} where={lookWhere} current={currentLook} canRevert={!!revertLook}
                      onApply={(n) => void pickLook(n)} onSave={(n, wl) => void storeLook(n, wl)}
                      onDelete={(n) => { void deleteLook(game, id, n); setCurrentLook(null); }}
                      onDefault={(n) => void setDefaultLook(game, id, n)}
                      onRevert={() => { const vs = sceneRef.current; if (vs && manifest && revertLook) void applyLook(vs, manifest, revertLook, true); setRevertLook(null); setCurrentLook(null); }} />
                    <VariationsSection
                      groups={manifest.variantGroups}
                      selected={variantSel}
                      colors={variantColors}
                      onSelect={(g, v) => { setVariantSel((s) => ({ ...s, [g]: v })); void selectVariant(g, v); }}
                    />
                    {(manifest.partGroups?.length ?? 0) > 0 && (
                      <PartGroupsSection groups={manifest.partGroups!} selected={groupSel} palettes={palSel} onSelect={selectPartOption} onPalette={selectPalette} skin={skinTone()} partsVis={partsVis} onTogglePart={(n, v) => { togglePart(n, v); setPartsVis((c) => ({ ...c, [n]: v })); }} />
                    )}
                    {(manifest.extras?.jaws?.length ?? 0) > 0 && <JawSection value={jaw} onChange={setJaw} />}
                    <HologramSection presets={manifest.holocast ?? []} state={hologram} onChange={setHologram} />
                  </>
                  } />
                ),
              },
              {
                id: 'ref', label: 'Reference', content: (
                  <>
                    <RefImageSection image={refImg} settings={refSet} transforming={refXf}
                      onSettings={(p) => setRefSet((s) => ({ ...s, ...p }))} onTransform={setRefTransform} onTransforming={setRefXf}
                      onPickSample={(p) => void showRefImage(sampleUrl(p), p.split('/').pop()!, p)} onFile={refFile} onClear={clearRefImage}
                      fillHeightScale={stageFillHeight}
                      align={{ pins, placing, lockFov, result: solveRes, error: solveErr, onPlacing: startPlacing, onLockFov: setLockFov,
                        onRemove: (pid) => setPins(pins.filter((q) => q.id !== pid)), onClear: () => setPins([]), onSolve: solvePins,
                        fullFrame, onFullFrame: setFullFrame, solveRoll, onSolveRoll: setSolveRoll,
                        pivotId, onPivot: (pid) => setPivotId((cur) => (cur === pid ? null : pid)), fine, onFine: setFine,
                        search, onFindFrame: (all) => void findBestFrame(all), onCancelSearch: () => { searchCancel.current = true; },
                        locked: !!lensLock, onRelease: () => setLensLock(null) }}
                      light={{ busy: lightBusy, result: lightRes, error: lightErr, scanPresets: lightScan, register: lightReg,
                        excluding, excludeCount: excludes.length, canUndo: !!lightUndo,
                        onRun: () => void runAutoLight(), onCancel: () => { lightCancel.current = true; },
                        onUndo: () => { if (lightUndo) { patchOptions({ env: lightUndo }); setLightUndo(null); setLightRes(null); } },
                        onScanPresets: setLightScan, onRegister: setLightReg,
                        onExcluding: (v) => { setExcluding(v); if (v) { setPlacing(false); setRefXf(false); setRefSet((st) => ({ ...st, visible: true })); } },
                        onClearExcludes: () => setExcludes([]) }} />
                    {refs.length > 0 && (
                      <ReferencePanel refs={refs} modelId={id} active={refActive} mode={refMode} opacity={refOpacity} showAll={refShowAll}
                        onPick={pickRef}
                        onMode={(m) => {
                          setRefMode(m);
                          if (m !== 'off') { setRefSet((s) => ({ ...s, visible: false })); setRefXf(false); }
                          if (refActive) requestAnimationFrame(() => requestAnimationFrame(() => applyRef(refActive)));
                        }}
                        onOpacity={setRefOpacity} onShowAll={setRefShowAll} onReapply={() => refActive && applyRef(refActive)} />
                    )}
                  </>
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
