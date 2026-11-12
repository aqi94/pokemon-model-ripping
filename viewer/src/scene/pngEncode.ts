/**
 * Texture pixels in and out without a 2D canvas: canvas.toBlob('image/png') takes ~7 s for a 1024² image in Chrome, and a
 * 2D canvas premultiplies alpha, which wrecks maps that keep data in alpha (masks, packed channels). Decoding goes through
 * createImageBitmap (no premultiply, no colour conversion) and a WebGL read-back; encoding uses the browser's own deflate.
 */

let gl: WebGL2RenderingContext | null = null;

/** Raw RGBA of an encoded image, optionally resized (aspect is the caller's business). */
export async function decodePixels(bytes: Uint8Array, mime: string, width?: number, height?: number): Promise<{ data: Uint8Array; width: number; height: number }> {
  const bmp = await createImageBitmap(new Blob([bytes], { type: mime || 'image/png' }), {
    premultiplyAlpha: 'none', colorSpaceConversion: 'none',
    ...(width && height ? { resizeWidth: width, resizeHeight: height, resizeQuality: 'high' as const } : {}),
  });
  try {
    gl ??= new OffscreenCanvas(1, 1).getContext('webgl2', { premultipliedAlpha: false });
    if (!gl) throw new Error('WebGL2 unavailable');
    const tex = gl.createTexture(), fb = gl.createFramebuffer();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const data = new Uint8Array(bmp.width * bmp.height * 4);
    gl.readPixels(0, 0, bmp.width, bmp.height, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fb); gl.deleteTexture(tex);
    return { data, width: bmp.width, height: bmp.height };
  } finally {
    bmp.close();
  }
}

const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
function crc32(buf: Uint8Array, start: number, end: number) {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) c = CRC[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const v = new DataView(out.buffer);
  v.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  v.setUint32(8 + data.length, crc32(out, 4, 8 + data.length));
  return out;
}

/** PNG of RGBA pixels (RGB when every alpha is 255), Sub-filtered rows, native zlib deflate. */
export async function encodePng(rgba: Uint8Array | Uint8ClampedArray, width: number, height: number): Promise<Uint8Array> {
  let opaque = true;
  for (let i = 3; i < rgba.length; i += 4) if (rgba[i] !== 255) { opaque = false; break; }
  const bpp = opaque ? 3 : 4, stride = width * bpp;
  const raw = new Uint8Array((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    const o = y * (stride + 1);
    raw[o] = 1; // Sub
    for (let x = 0; x < width; x++) {
      const s = (y * width + x) * 4, d = o + 1 + x * bpp;
      for (let c = 0; c < bpp; c++) raw[d + c] = (rgba[s + c] - (x ? rgba[s - 4 + c] : 0)) & 0xff;
    }
  }
  const idat = new Uint8Array(await new Response(new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer());
  const ihdr = new Uint8Array(13);
  const v = new DataView(ihdr.buffer);
  v.setUint32(0, width); v.setUint32(4, height);
  ihdr[8] = 8; ihdr[9] = opaque ? 2 : 6; // 8-bit RGB / RGBA
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
