// Print world positions of bones whose name matches a regex, for a model in the running viewer (debug helper).
//   node tools/probe_bones.mjs <game/id> [regex] [clip] [frame]
import { chromium } from 'playwright';

const [id, rx = 'Item|Hand|Light|Eff|Head', clip = 'advent02_b', frame = '0'] = process.argv.slice(2);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 700, height: 500 } });
await page.goto(`http://localhost:5173/model-viewer?id=${encodeURIComponent(id)}`);
await page.waitForFunction(() => window.__viewer?.anim && window.__viewer.manifest, null, { timeout: 120000 });
const out = await page.evaluate(({ rx, clip, frame }) => {
  const vs = window.__viewer; const a = vs.anim;
  a.play(a.has(clip) ? clip : 'wait_b'); a.setPaused(true);
  const fc = Math.max(1, (a.state().frameCount || 1) - 1); a.scrubToFrame(Math.round(Number(frame) * fc));
  vs.root.updateMatrixWorld(true);
  const res = []; const re = new RegExp(rx);
  vs.gltf.scene.traverse((o) => { if (re.test(o.name)) { const p = o.getWorldPosition(o.position.clone()); res.push([o.name, +p.x.toFixed(3), +p.y.toFixed(3), +p.z.toFixed(3)]); } });
  const urls = performance.getEntriesByType('resource').map((r) => r.name).filter((n) => /three/.test(n)).slice(0, 4);
  return { res, urls };
}, { rx, clip, frame });
console.log(JSON.stringify(out, null, 0));
await browser.close();
