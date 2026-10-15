import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import path from 'node:path';

const OUT_DIR = path.resolve(__dirname, '../out');
const TYPES: Record<string, string> = {
  '.json': 'application/json',
  '.glb': 'model/gltf-binary',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.webp': 'image/webp',
};

/** Serves /models/* from ../out/* (dev + preview). */
function serveModels(): Plugin {
  const mw = (req: any, res: any, next: any) => {
    const url: string = (req.url ?? '').split('?')[0];
    if (!url.startsWith('/models/')) return next();
    const rel = decodeURIComponent(url.slice('/models/'.length));
    const file = path.resolve(OUT_DIR, rel);
    if (!file.startsWith(OUT_DIR + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
      res.statusCode = 404;
      return res.end('not found');
    }
    res.setHeader('Content-Type', TYPES[path.extname(file).toLowerCase()] ?? 'application/octet-stream');
    res.setHeader('Content-Length', fs.statSync(file).size);
    res.setHeader('Cache-Control', 'no-cache');
    fs.createReadStream(file).pipe(res);
  };
  return {
    name: 'serve-models',
    configureServer: (s) => void s.middlewares.use(mw),
    configurePreviewServer: (s) => void s.middlewares.use(mw),
  };
}

export default defineConfig({
  plugins: [react(), serveModels()],
  server: { port: 5173, fs: { allow: ['..'] } },
});
