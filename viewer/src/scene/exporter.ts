export function download(blob: Blob, filename: string) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

export function exportCanvasPng(canvas: HTMLCanvasElement, filename: string, renderNow: () => void, after?: () => void) {
  renderNow(); // render immediately before reading the buffer
  canvas.toBlob((b) => b && download(b, filename), 'image/png'); // snapshots the buffer synchronously
  after?.();
}

export async function exportGlb(url: string, filename: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GLB download failed: ${res.status}`);
  download(await res.blob(), filename);
}
