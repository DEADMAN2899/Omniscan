// OmniScan vision worker: every model runs here, off the UI thread.
// Messages in:  {id, type, ...payload}   Messages out: {id, ok, result} | {id, ok:false, error} | {type:"model", ...status}
"use strict";

const ROOT = new URL("./", self.location).href;
const S = {
  manifest: null, ort: null, accel: "wasm", threads: 1,
  sessions: {}, labels: {}, ocrKeys: null, zx: null,
  loading: {},
};

const post = (m, t) => self.postMessage(m, t || []);
const status = (id, state, extra = {}) => post({ type: "model", id, state, ...extra });

/* ---------------- verified, cached, sharded downloads ---------------- */
async function sha256hex(buf) {
  const h = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(h)].map(b => b.toString(16).padStart(2, "0")).join("");
}
async function fetchVerified(entry, onProgress) {
  const out = new Uint8Array(entry.size);
  let off = 0;
  for (const part of entry.parts) {
    const url = new URL(part, ROOT).href;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Download failed (HTTP ${r.status}) for ${part}`);
    const reader = r.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (off + value.length > entry.size) throw new Error(`File larger than its manifest says: ${part}`);
      out.set(value, off); off += value.length;
      onProgress && onProgress(off / entry.size);
    }
  }
  if (off !== entry.size) throw new Error(`Incomplete download (${off} of ${entry.size} bytes)`);
  const hex = await sha256hex(out);
  if (hex !== entry.sha256) throw new Error("Checksum mismatch: refusing to load this file");
  return out;
}

/* ---------------- runtime ---------------- */
async function init({ accel, threads }) {
  S.manifest = await (await fetch(new URL("models/manifest.json", ROOT))).json();
  S.accel = accel === "webgpu" && self.navigator.gpu ? "webgpu" : "wasm";
  const webgpu = S.accel === "webgpu";
  status("runtime", "DOWNLOADING", { pct: 0 });
  const bin = await fetchVerified(S.manifest.runtime[webgpu ? "ort-jsep" : "ort-wasm"], p => status("runtime", "DOWNLOADING", { pct: p }));
  status("runtime", "LOADING");
  S.ort = await import(webgpu ? "./vendor/ort/ort.webgpu.min.mjs" : "./vendor/ort/ort.wasm.min.mjs");
  S.ort.env.wasm.wasmPaths = new URL("vendor/ort/", ROOT).href;
  S.ort.env.wasm.wasmBinary = bin.buffer;
  S.threads = self.crossOriginIsolated ? Math.max(1, Math.min(threads || 4, (self.navigator.hardwareConcurrency || 2))) : 1;
  S.ort.env.wasm.numThreads = S.threads;
  S.ort.env.wasm.simd = true;
  status("runtime", "READY", { accel: S.accel, threads: S.threads, isolated: self.crossOriginIsolated });
  return { accel: S.accel, threads: S.threads, isolated: self.crossOriginIsolated, manifest: S.manifest };
}

async function loadModel(id) {
  if (S.sessions[id]) return S.sessions[id];
  if (S.loading[id]) return S.loading[id];
  const entry = S.manifest.models[id];
  if (!entry) throw new Error(`Unknown model ${id}`);
  S.loading[id] = (async () => {
    const t0 = performance.now();
    try {
      status(id, "DOWNLOADING", { pct: 0 });
      const bytes = await fetchVerified(entry, p => status(id, "DOWNLOADING", { pct: p }));
      status(id, "VERIFYING");
      status(id, "LOADING");
      const eps = S.accel === "webgpu" && entry.webgpu !== false ? ["webgpu", "wasm"] : ["wasm"];
      let sess;
      try { sess = await S.ort.InferenceSession.create(bytes, { executionProviders: eps, graphOptimizationLevel: "all" }); }
      catch (e) {
        if (eps[0] !== "webgpu") throw e;
        sess = await S.ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      }
      if (entry.labels) S.labels[id] = await (await fetch(new URL(entry.labels, ROOT))).json();
      if (id === "ocr_rec") {
        const keys = (await (await fetch(new URL(entry.keys, ROOT))).text()).split(/\r?\n/);
        if (keys[keys.length - 1] === "") keys.pop();
        S.ocrKeys = ["", ...keys, " "];
      }
      S.sessions[id] = sess;
      status(id, "READY", { ms: Math.round(performance.now() - t0), inputNames: sess.inputNames });
      return sess;
    } catch (e) {
      status(id, "FAILED", { error: String(e && e.message || e) });
      delete S.loading[id];
      throw e;
    }
  })();
  return S.loading[id];
}

/* ---------------- pixels ---------------- */
function canvasOf(w, h) { const c = new OffscreenCanvas(w, h); return [c, c.getContext("2d", { willReadFrequently: true })]; }

/* ---------------- detection (D-FINE) ---------------- */
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return i / (a[2] * a[3] + b[2] * b[3] - i || 1);
}
async function detect({ bitmap, model, threshold }) {
  const sess = await loadModel(model);
  const entry = S.manifest.models[model], N = entry.input;
  const W = bitmap.width, H = bitmap.height;
  const [c, g] = canvasOf(N, N);
  g.drawImage(bitmap, 0, 0, N, N); bitmap.close();
  const px = g.getImageData(0, 0, N, N).data, plane = N * N;
  const x = new Float32Array(3 * plane);
  for (let i = 0, j = 0; i < plane; i++, j += 4) { x[i] = px[j] / 255; x[i + plane] = px[j + 1] / 255; x[i + 2 * plane] = px[j + 2] / 255; }
  const t0 = performance.now();
  const out = await sess.run({ [sess.inputNames[0]]: new S.ort.Tensor("float32", x, [1, 3, N, N]) });
  const ms = performance.now() - t0;
  const logits = out.logits, boxes = out.boxes;
  const [, Q, C] = logits.dims, L = logits.data, B = boxes.data, names = S.labels[model];
  const cand = [];
  for (let q = 0; q < Q; q++) {
    let best = -1e9, bc = -1;
    for (let k = 0; k < C; k++) { if (names[k] === "none") continue; const v = L[q * C + k]; if (v > best) { best = v; bc = k; } }
    const s = 1 / (1 + Math.exp(-best));
    if (s < threshold) continue;
    const cx = B[q * 4], cy = B[q * 4 + 1], bw = B[q * 4 + 2], bh = B[q * 4 + 3];
    const box = [(cx - bw / 2) * W, (cy - bh / 2) * H, bw * W, bh * H];
    if (box[2] < 4 || box[3] < 4) continue;
    cand.push({ cls: bc, label: names[bc], score: s, box });
  }
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const d of cand) { if (keep.length >= 30) break; if (!keep.some(k => k.cls === d.cls && iou(k.box, d.box) > 0.7)) keep.push(d); }
  for (const t of Object.values(out)) t.dispose && t.dispose();
  return { dets: keep, ms, model, set: entry.labelSet };
}

/* ---------------- image embeddings (TinyCLIP) ---------------- */
async function embed({ crops }) {
  const sess = await loadModel("clip");
  const m = S.manifest.models.clip, N = m.input, mean = m.mean, std = m.std, plane = N * N;
  const [c, g] = canvasOf(N, N);
  const embs = [], t0 = performance.now();
  for (const bmp of crops) {
    g.clearRect(0, 0, N, N); g.drawImage(bmp, 0, 0, N, N); bmp.close();
    const px = g.getImageData(0, 0, N, N).data, x = new Float32Array(3 * plane);
    for (let i = 0, j = 0; i < plane; i++, j += 4)
      for (let ch = 0; ch < 3; ch++) x[i + ch * plane] = (px[j + ch] / 255 - mean[ch]) / std[ch];
    const out = await sess.run({ [sess.inputNames[0]]: new S.ort.Tensor("float32", x, [1, 3, N, N]) });
    const e = Float32Array.from(Object.values(out)[0].data);
    embs.push(e);
  }
  return { embs, ms: (performance.now() - t0) / Math.max(1, crops.length) };
}

/* ---------------- OCR (PP-OCRv3 det + rec, BGR, DB post-process) ---------------- */
async function ocr({ bitmap }) {
  const det = await loadModel("ocr_det"); const rec = await loadModel("ocr_rec");
  const t0 = performance.now();
  const W = bitmap.width, H = bitmap.height;
  const [src, sg] = canvasOf(W, H); sg.drawImage(bitmap, 0, 0); bitmap.close();
  // resize: longest side <= 960, at least 640 for small crops, multiples of 32
  let s = Math.min(1, 960 / Math.max(W, H)); if (Math.max(W, H) * s < 640) s = 640 / Math.max(W, H);
  const dw = Math.max(32, Math.round(W * s / 32) * 32), dh = Math.max(32, Math.round(H * s / 32) * 32);
  const [dc, dg] = canvasOf(dw, dh); dg.drawImage(src, 0, 0, dw, dh);
  const px = dg.getImageData(0, 0, dw, dh).data, plane = dw * dh, x = new Float32Array(3 * plane);
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  for (let i = 0, j = 0; i < plane; i++, j += 4) { // BGR order, as PaddleOCR was trained
    x[i] = (px[j + 2] / 255 - mean[0]) / std[0]; x[i + plane] = (px[j + 1] / 255 - mean[1]) / std[1]; x[i + 2 * plane] = (px[j] / 255 - mean[2]) / std[2];
  }
  const dout = await det.run({ [det.inputNames[0]]: new S.ort.Tensor("float32", x, [1, 3, dh, dw]) });
  const prob = Object.values(dout)[0].data;
  // binarize + 2x2 dilation
  const bin = new Uint8Array(plane);
  for (let y = 0; y < dh; y++) for (let xx = 0; xx < dw; xx++) {
    const i = y * dw + xx;
    if (prob[i] > 0.3 || (xx > 0 && prob[i - 1] > 0.3) || (y > 0 && prob[i - dw] > 0.3) || (xx > 0 && y > 0 && prob[i - dw - 1] > 0.3)) bin[i] = 1;
  }
  // connected components -> rotated text boxes (orientation from second moments)
  const lab = new Int32Array(plane), boxes = [], stack = new Int32Array(plane);
  let n = 0;
  for (let i = 0; i < plane; i++) {
    if (!bin[i] || lab[i]) continue;
    n++; let sp = 0; stack[sp++] = i; lab[i] = n;
    let x0 = dw, y0 = dh, x1 = 0, y1 = 0, cnt = 0, sum = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    while (sp) {
      const p = stack[--sp], py = (p / dw) | 0, pxx = p - py * dw; cnt++; sum += prob[p];
      sx += pxx; sy += py; sxx += pxx * pxx; syy += py * py; sxy += pxx * py;
      if (pxx < x0) x0 = pxx; if (pxx > x1) x1 = pxx; if (py < y0) y0 = py; if (py > y1) y1 = py;
      if (pxx > 0 && bin[p - 1] && !lab[p - 1]) { lab[p - 1] = n; stack[sp++] = p - 1; }
      if (pxx < dw - 1 && bin[p + 1] && !lab[p + 1]) { lab[p + 1] = n; stack[sp++] = p + 1; }
      if (py > 0 && bin[p - dw] && !lab[p - dw]) { lab[p - dw] = n; stack[sp++] = p - dw; }
      if (py < dh - 1 && bin[p + dw] && !lab[p + dw]) { lab[p + dw] = n; stack[sp++] = p + dw; }
    }
    if (Math.min(x1 - x0 + 1, y1 - y0 + 1) < 3 || cnt < 10) continue;
    const score = sum / cnt;
    if (score < 0.5) continue;
    const mx = sx / cnt, my = sy / cnt, cxx = sxx / cnt - mx * mx, cyy = syy / cnt - my * my, cxy = sxy / cnt - mx * my;
    let th = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
    let cos = Math.cos(th), sin = Math.sin(th), u0 = 1e9, u1 = -1e9, v0 = 1e9, v1 = -1e9;
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      if (lab[y * dw + x] !== n) continue;
      const u = (x - mx) * cos + (y - my) * sin, v = -(x - mx) * sin + (y - my) * cos;
      if (u < u0) u0 = u; if (u > u1) u1 = u; if (v < v0) v0 = v; if (v > v1) v1 = v;
    }
    let rw = u1 - u0 + 1, rh = v1 - v0 + 1;
    const uc = (u0 + u1) / 2, vc = (v0 + v1) / 2;
    const ccx = mx + uc * cos - vc * sin, ccy = my + uc * sin + vc * cos;
    if (rh > rw) { th += Math.PI / 2; [rw, rh] = [rh, rw]; }
    if (th > Math.PI / 2) th -= Math.PI; if (th < -Math.PI / 2) th += Math.PI;
    const d = (rw * rh * 1.6) / (2 * (rw + rh)); // unclip
    rw += 2 * d; rh += 2 * d;
    boxes.push({ cx: (ccx + 0.5) / s, cy: (ccy + 0.5) / s, w: rw / s, h: rh / s, th, score });
  }
  for (const t of Object.values(dout)) t.dispose && t.dispose();
  boxes.sort((a, b) => b.w * b.h - a.w * a.h);
  const lines = [];
  for (const b of boxes.slice(0, 24)) {
    const ratio = b.w / b.h;
    const iw = Math.min(1600, Math.max(320, Math.ceil(48 * ratio))), tw = Math.min(iw, Math.ceil(48 * ratio));
    const [rc, rg] = canvasOf(iw, 48);
    rg.fillStyle = "#000"; rg.fillRect(0, 0, iw, 48);
    rg.save(); rg.beginPath(); rg.rect(0, 0, tw, 48); rg.clip();
    rg.translate(tw / 2, 24); rg.scale(tw / b.w, 48 / b.h); rg.rotate(-b.th); rg.translate(-b.cx, -b.cy);
    rg.drawImage(src, 0, 0); rg.restore();
    const rp = rg.getImageData(0, 0, iw, 48).data, rpl = iw * 48, rx = new Float32Array(3 * rpl);
    for (let i = 0, j = 0; i < rpl; i++, j += 4) {
      if (i % iw >= tw) continue; // padding stays 0 (normalized zero)
      rx[i] = (rp[j + 2] / 255 - 0.5) / 0.5; rx[i + rpl] = (rp[j + 1] / 255 - 0.5) / 0.5; rx[i + 2 * rpl] = (rp[j] / 255 - 0.5) / 0.5;
    }
    const rout = await rec.run({ [rec.inputNames[0]]: new S.ort.Tensor("float32", rx, [1, 3, 48, iw]) });
    const o = Object.values(rout)[0], [, T, K] = o.dims, P = o.data;
    let text = "", conf = 0, nc = 0, prev = -1;
    for (let t = 0; t < T; t++) {
      let bi = 0, bv = -1;
      for (let k = 0; k < K; k++) { const v = P[t * K + k]; if (v > bv) { bv = v; bi = k; } }
      if (bi !== 0 && bi !== prev) { text += S.ocrKeys[bi] || ""; conf += bv; nc++; }
      prev = bi;
    }
    o.dispose && o.dispose();
    text = text.trim();
    if (!text || !nc || conf / nc < 0.6) continue;
    const ex = Math.abs(b.w * Math.cos(b.th)) / 2 + Math.abs(b.h * Math.sin(b.th)) / 2, ey = Math.abs(b.w * Math.sin(b.th)) / 2 + Math.abs(b.h * Math.cos(b.th)) / 2;
    lines.push({ text, conf: conf / nc, box: [Math.max(0, b.cx - ex), Math.max(0, b.cy - ey), Math.min(W, 2 * ex), Math.min(H, 2 * ey)], angle: Math.round(b.th * 180 / Math.PI) });
  }
  lines.sort((a, b) => (Math.abs(a.box[1] - b.box[1]) < Math.min(a.box[3], b.box[3]) / 2 ? a.box[0] - b.box[0] : a.box[1] - b.box[1]));
  return { lines, ms: performance.now() - t0 };
}

/* ---------------- barcodes (zxing-cpp) ---------------- */
async function barcode({ bitmap }) {
  if (!S.zx) {
    status("barcode", "DOWNLOADING", { pct: 0 });
    const bin = await fetchVerified(S.manifest.runtime.zxing);
    const mod = await import("./vendor/zxing/zxing.mjs");
    mod.prepareZXingModule({ overrides: { wasmBinary: bin.buffer }, fireImmediately: true });
    S.zx = mod;
    status("barcode", "READY");
  }
  const W = bitmap.width, H = bitmap.height;
  const [c, g] = canvasOf(W, H); g.drawImage(bitmap, 0, 0); bitmap.close();
  const t0 = performance.now();
  const res = await S.zx.readBarcodes(g.getImageData(0, 0, W, H), { tryHarder: true, maxNumberOfSymbols: 4,
    formats: ["EAN13", "EAN8", "UPCA", "UPCE", "QRCode", "Code128", "Code39", "DataMatrix", "ITF"] });
  const codes = res.filter(r => r.isValid && r.text).map(r => {
    const p = r.position, xs = [p.topLeft.x, p.topRight.x, p.bottomLeft.x, p.bottomRight.x], ys = [p.topLeft.y, p.topRight.y, p.bottomLeft.y, p.bottomRight.y];
    const x0 = Math.min(...xs), y0 = Math.min(...ys);
    return { format: r.format, text: r.text, box: [x0, y0, Math.max(...xs) - x0, Math.max(...ys) - y0] };
  });
  return { codes, ms: performance.now() - t0 };
}

/* ---------------- dispatch ---------------- */
const handlers = { init, load: ({ model }) => loadModel(model).then(() => ({ model })), detect, embed, ocr, barcode };
self.onmessage = async ({ data }) => {
  const { id, type } = data;
  try {
    const result = await handlers[type](data);
    post({ id, ok: true, result });
  } catch (e) {
    post({ id, ok: false, error: String(e && e.message || e) });
  }
};
