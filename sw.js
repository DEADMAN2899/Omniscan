// OmniScan service worker: makes the app and its downloaded models work offline.
// App files: network first (so new deploys show up), cached copy when offline.
// Model, runtime and vocabulary files: content-addressed, so cache first, kept across deploys.
const SHELL = "omniscan-shell-20261024015357";
const ASSETS = "omniscan-assets-v1";
const SHELL_FILES = ["./", "index.html", "app.js", "worker.mjs", "styles.css", "fonts/fonts.css", "manifest.webmanifest",
  "icon-192.png", "vendor/ort/ort.wasm.min.mjs", "vendor/ort/ort-wasm-simd-threaded.mjs", "vendor/zxing/zxing.mjs", "models/manifest.json"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(SHELL).then(c => c.addAll(SHELL_FILES)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", e => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith("omniscan-shell-") && k !== SHELL) await caches.delete(k);
    // drop model parts the current manifest no longer references
    try {
      const m = await (await fetch("models/manifest.json", { cache: "no-store" })).json();
      const keep = new Set();
      for (const group of [m.runtime, m.models]) for (const v of Object.values(group)) for (const p of v.parts) keep.add(new URL(p, self.registration.scope).href);
      const c = await caches.open(ASSETS);
      for (const req of await c.keys()) if (req.url.endsWith(".bin") && !keep.has(req.url)) await c.delete(req);
    } catch {}
    await self.clients.claim();
  })());
});

// GitHub Pages can't send custom headers, so add the cross-origin isolation headers here.
// That unlocks multi-threaded inference. On hosts that already send them (Netlify), this is a no-op.
function isolate(res) {
  if (!res || res.status === 0 || res.type === "opaque") return res;
  const h = new Headers(res.headers);
  if (h.get("Cross-Origin-Embedder-Policy")) return res;
  h.set("Cross-Origin-Embedder-Policy", "require-corp");
  h.set("Cross-Origin-Opener-Policy", "same-origin");
  h.set("Cross-Origin-Resource-Policy", "same-origin");
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

self.addEventListener("fetch", e => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return; // API calls and lookups go straight to the network
  const asset = /\.bin$|\/vocab\/|\/fonts\/.+\.woff2$|\.png$/.test(url.pathname);
  if (asset) {
    e.respondWith(caches.open(ASSETS).then(async c => {
      const hit = await c.match(req);
      if (hit) return isolate(hit);
      const res = await fetch(req);
      if (res.ok) c.put(req, res.clone());
      return isolate(res);
    }));
    return;
  }
  e.respondWith((async () => {
    try {
      const res = await fetch(req);
      if (res.ok) (await caches.open(SHELL)).put(req, res.clone());
      return isolate(res);
    } catch {
      const hit = await caches.match(req, { ignoreSearch: true });
      if (hit) return isolate(hit);
      if (req.mode === "navigate") return isolate(await caches.match("index.html"));
      throw new Error("offline");
    }
  })());
});
