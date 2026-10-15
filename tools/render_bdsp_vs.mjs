// Square transparent renders of BDSP battle trainers in their `advent02_b` (VS intro) pose, falling back to advent_b then wait_b through the running viewer dev server.
//
//   node tools/render_bdsp_vs.mjs --out <dir> [--size 1024] [--clip advent02_b] [--frames 0.5,1] [--camera front] [--preset bdspVs]
//                                 [--ball RItem1|LItem1] [--fx police|psychic] [--tag _x] <game/id> ...
// Writes <id>_f<frame>.png per requested frame fraction; crop/fit to the final canvas afterwards (tools/fit_square.py).
import { chromium } from 'playwright';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf('--' + n); return i < 0 ? d : args[i + 1]; };
const ids = args.filter((a, i) => !a.startsWith('--') && !(i > 0 && args[i - 1].startsWith('--')));
const out = opt('out');
if (!out || !ids.length) { console.error('usage: node tools/render_bdsp_vs.mjs --out <dir> [--size N] [--clip c] [--frames f,f] <game/id> ...'); process.exit(1); }
fs.mkdirSync(out, { recursive: true });
const S = Number(opt('size', '1024'));
const frames = opt('frames', '1').split(',').map(Number);
const clip = opt('clip', 'advent02_b');
const base = opt('url', 'http://localhost:5173');
// --ball <bone>: parent the Poké Ball (bdsp/ob0204_00) to that attach bone (RItem1 / LItem1 ...); --tag names the output suffix.
const ball = opt('ball', '');
const tag = opt('tag', '');
const ballScale = Number(opt('ballscale', '1'));
// --fx: the game's light effects are particle systems that are not in the rip; approximate them with additive glow sprites.
//   police: lens glow on the `Light` bone + baton glow on RItem1.  psychic: ring of beads through LHand -> RItem1 (the orb above the head) -> RHand.
const fx = opt('fx', '');

const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--use-angle=d3d11', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: S + 400, height: S + 40 }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.error('page error:', e.message));

for (const id of ids) {
  await page.goto(`${base}/model-viewer?id=${encodeURIComponent(id)}`);
  try {
    await page.waitForFunction(() => window.__viewer?.anim && window.__viewer.manifest, null, { timeout: 120000 });
  } catch { console.error(id, 'failed to load'); continue; }
  await page.evaluate(async ({ S, preset }) => {
    const st = document.createElement('style');
    st.textContent = `.sidebar,.handle,.back,.title-chip{display:none!important}.viewer{display:block!important}
      .stage{position:fixed!important;left:0;top:0;width:${S}px!important;height:${S}px!important}`;
    document.head.appendChild(st);
    const vs = window.__viewer;
    const m = await import('/src/scene/environment.ts');
    vs.renderer.setPixelRatio(1);
    vs.setEnvironment(m.LIGHTING_PRESETS.find((p) => p.id === preset).env);
    vs.controls.enableDamping = false;
    await new Promise((r) => setTimeout(r, 300));
  }, { S, preset: opt('preset', 'bdspVs') });
  if (ball) await page.evaluate(async (bone) => { const vs = window.__viewer; vs.setHeldOnlyInClips(false); await vs.setHeldItem('/models/bdsp/ob0204_00/model.glb', { bone, side: '', clips: [] }); }, ball);
  if (ball && ballScale !== 1) await page.evaluate((k) => window.__viewer.held.obj.scale.setScalar(k), ballScale);
  const used = await page.evaluate((clip) => { const a = window.__viewer.anim; const c = [clip, 'advent_b', 'wait_b'].find((n) => a.has(n)) ?? null; if (c) a.play(c); return c; }, clip);
  const threeUrl = await page.evaluate(() => performance.getEntriesByType('resource').map((r) => r.name).find((n) => /deps\/three\.js/.test(n)));
  for (const f of frames) {
    const url = await page.evaluate(async ({ f, camera, fx, threeUrl }) => {
      const vs = window.__viewer, a = vs.anim;
      a.setPaused(true);
      const fc = Math.max(1, (a.state().frameCount || 1) - 1);
      a.scrubToFrame(Math.round(f * fc));
      vs.setCameraPreset(camera);
      if (fx) {
        const T = await import(threeUrl);
        const scene = vs.root.parent ?? vs.root;
        scene.getObjectByName('__fx')?.removeFromParent();
        const grp = new T.Group(); grp.name = '__fx'; scene.add(grp);
        vs.root.updateMatrixWorld(true);
        const c = document.createElement('canvas'); c.width = c.height = 128;
        const g = c.getContext('2d'); const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
        gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.18, 'rgba(255,255,255,0.75)'); gr.addColorStop(0.5, 'rgba(255,255,255,0.18)'); gr.addColorStop(1, 'rgba(255,255,255,0)');
        g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
        const tex = new T.CanvasTexture(c);
        const bone = (n) => { const o = vs.findBone(n); return o ? o.getWorldPosition(new T.Vector3()) : null; };
        const glow = (p, size, color, op = 1) => {
          if (!p) return;
          const m = new T.SpriteMaterial({ map: tex, color, transparent: true, opacity: op, blending: T.AdditiveBlending, depthWrite: false, depthTest: false, toneMapped: false });
          const sp = new T.Sprite(m); sp.position.copy(p); sp.scale.setScalar(size); sp.renderOrder = 999; grp.add(sp);
        };
        if (fx === 'police') {
          glow(bone('Light'), 0.36, 0xffe84a, 0.75); glow(bone('Light'), 0.12, 0xfff6b0, 0.6);
          const r = bone('RItem1'), h = bone('RHand'); if (r && h) { const d = new T.Vector3(-0.6, 0.8, 0); for (let k = 0; k < 5; k++) glow(r.clone().addScaledVector(d, 0.02 + 0.06 * k), 0.24, 0xffe84a, 0.4); glow(r, 0.3, 0xffe84a, 0.35); }
        } else if (fx === 'psychic') {
          const l = bone('LHand'), r = bone('RHand'), t = bone('RItem1');
          if (l && r && t) {
            const ctl = t.clone().multiplyScalar(2).sub(l.clone().add(r).multiplyScalar(0.5));
            const N = 36;
            for (let i = 0; i <= N; i++) {
              const u = i / N, v = 1 - u;
              const p = l.clone().multiplyScalar(v * v).add(ctl.clone().multiplyScalar(2 * u * v)).add(r.clone().multiplyScalar(u * u));
              const bead = i % 6 === 3;
              glow(p, bead ? 0.26 : 0.17, 0xffc02a, bead ? 0.85 : 0.6);
              glow(p, bead ? 0.5 : 0.36, 0xffa810, 0.16);
            }
            glow(t, 0.5, 0xffc030, 0.18);
          }
        }
      }
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      return vs.pngDataUrl();
    }, { f, camera: opt('camera', 'front'), fx, threeUrl });
    fs.writeFileSync(path.join(out, `${id.replace(/\//g, '_')}_f${f}${tag}.png`), Buffer.from(url.split(',')[1], 'base64'));
  }
  console.log(id, used);
}
await browser.close();
