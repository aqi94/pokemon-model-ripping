import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';

const OUT_DIR = path.resolve(__dirname, '../out');
const REFS_DIR = path.resolve(__dirname, '../refs');
// Reference images for the viewer's "Reference image" picker (local deployment only).
const SAMPLES_DIR = path.resolve(__dirname, process.env.REF_SAMPLES_DIR ?? '../samples');
const IMAGE_EXT = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp']);
const TYPES: Record<string, string> = {
  '.json': 'application/json',
  '.glb': 'model/gltf-binary',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.bmp': 'image/bmp',
  '.webp': 'image/webp',
};

/** Serves /<prefix>/* from a folder outside the viewer (dev + preview): /models/ -> ../out, /refs/ -> ../refs. */
function serveDir(prefix: string, dir: string): Plugin {
  const mw = (req: any, res: any, next: any) => {
    const url: string = (req.url ?? '').split('?')[0];
    if (!url.startsWith(prefix)) return next();
    const rel = decodeURIComponent(url.slice(prefix.length));
    const file = path.resolve(dir, rel);
    if (!file.startsWith(dir + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      res.statusCode = 404;
      return res.end('not found');
    }
    res.setHeader('Content-Type', TYPES[path.extname(file).toLowerCase()] ?? 'application/octet-stream');
    res.setHeader('Content-Length', fs.statSync(file).size);
    res.setHeader('Cache-Control', 'no-cache');
    fs.createReadStream(file).pipe(res);
  };
  return {
    name: 'serve-' + prefix.replace(/\//g, ''),
    configureServer: (s) => void s.middlewares.use(mw),
    configurePreviewServer: (s) => void s.middlewares.use(mw),
  };
}

/** GET /__samples.json: every image under SAMPLES_DIR (recursive, newest first). 404 when the folder is missing,
 *  which the viewer reads as "no local samples" (web build) and offers upload / drag-drop only. */
function samplesListing(): Plugin {
  const list = (dir: string, rel = ''): { path: string; size: number; mtime: number }[] => {
    const out: { path: string; size: number; mtime: number }[] = [];
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const r = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory()) out.push(...list(path.join(dir, e.name), r));
      else if (IMAGE_EXT.has(path.extname(e.name).toLowerCase())) {
        const st = fs.statSync(path.join(dir, e.name));
        out.push({ path: r, size: st.size, mtime: st.mtimeMs });
      }
    }
    return out;
  };
  const mw = (req: any, res: any, next: any) => {
    if ((req.url ?? '').split('?')[0] !== '/__samples.json') return next();
    if (!fs.existsSync(SAMPLES_DIR)) { res.statusCode = 404; return res.end('no samples folder'); }
    const body = JSON.stringify(list(SAMPLES_DIR).sort((a, b) => b.mtime - a.mtime));
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Cache-Control', 'no-cache');
    res.end(body);
  };
  return {
    name: 'samples-listing',
    configureServer: (s) => void s.middlewares.use(mw),
    configurePreviewServer: (s) => void s.middlewares.use(mw),
  };
}

/** GET/PUT /__configs/<game>/<id>: the model's saved looks in out/<game>/<id>/configs.json (local deployment only; the
 *  static build has no endpoint and the viewer falls back to localStorage). GET of a missing file = no looks; PUT of an
 *  empty list deletes the file. Only existing model folders are writable. */
function configsEndpoint(): Plugin {
  const NAME = /^[A-Za-z0-9_.-]+$/;
  const send = (res: any, code: number, body: unknown) => {
    res.statusCode = code;
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Cache-Control', 'no-cache');
    res.end(JSON.stringify(body));
  };
  const mw = (req: any, res: any, next: any) => {
    const url: string = (req.url ?? '').split('?')[0];
    if (!url.startsWith('/__configs/')) return next();
    const [game, id, ...rest] = url.slice('/__configs/'.length).split('/').map(decodeURIComponent);
    if (rest.length || !game || !id || !NAME.test(game) || !NAME.test(id) || game.startsWith('.') || id.startsWith('.')) return send(res, 400, { error: 'bad path' });
    const dir = path.resolve(OUT_DIR, game, id);
    if (!dir.startsWith(OUT_DIR + path.sep) || !fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) return send(res, 404, { error: 'no such model' });
    const file = path.join(dir, 'configs.json');
    if (req.method === 'GET') {
      if (!fs.existsSync(file)) return send(res, 200, { version: 1, looks: [] });
      try { return send(res, 200, JSON.parse(fs.readFileSync(file, 'utf8'))); } catch { return send(res, 500, { error: 'configs.json is not valid JSON' }); }
    }
    if (req.method !== 'PUT') return send(res, 405, { error: 'GET or PUT' });
    let body = '', size = 0;
    req.on('data', (c: Buffer) => { size += c.length; if (size <= 4 << 20) body += c; });
    req.on('end', () => {
      if (size > 4 << 20) return send(res, 413, { error: 'too large' });
      let data: any;
      try { data = JSON.parse(body); } catch { return send(res, 400, { error: 'invalid JSON' }); }
      const looks = data?.looks;
      if (!Array.isArray(looks) || looks.some((l: any) => !l || typeof l !== 'object' || typeof l.name !== 'string' || !l.name.trim()))
        return send(res, 400, { error: 'expected {version: 1, looks: [{name, ...}]}' });
      const names = looks.map((l: any) => l.name.trim().toLowerCase());
      if (new Set(names).size !== names.length) return send(res, 400, { error: 'duplicate look names' });
      if (!looks.length) { if (fs.existsSync(file)) fs.unlinkSync(file); return send(res, 200, { ok: true, deleted: true }); }
      const tmp = file + '.tmp';
      fs.writeFileSync(tmp, JSON.stringify({ version: 1, looks }, null, 2) + '\n');
      fs.renameSync(tmp, file);
      send(res, 200, { ok: true });
    });
  };
  return {
    name: 'configs-endpoint',
    configureServer: (s) => void s.middlewares.use(mw),
    configurePreviewServer: (s) => void s.middlewares.use(mw),
  };
}

export default defineConfig({
  base: process.env.VITE_BASE ?? '/',
  plugins: [react(), serveDir('/models/', OUT_DIR), serveDir('/refs/', REFS_DIR), serveDir('/samples/', SAMPLES_DIR), samplesListing(), configsEndpoint()],
  server: { port: 5173, fs: { allow: ['..'] } },
});
