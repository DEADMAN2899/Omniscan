# Implementation status

Updated 2026-10-23.

## Done in this pass

- Detector replaced: COCO-SSD Lite → D-FINE N (live) and D-FINE S Objects365 (deep scan on hold).
- Multi-frame class voting and sharpest-crop selection per track.
- Second-stage identification with TinyCLIP, constrained to labels related to the detector class; 2,853-label vocabulary (LVIS, Objects365, Open Images, ImageNet, plus curated products and plants/animals lists).
- Tap-to-identify with vocabulary filters.
- Teaching and corrections, stored on the device.
- OCR (PP-OCRv3) with rotated text boxes; barcode reading (zxing-cpp); Open Food Facts lookup by number.
- Test set with stored runs.
- Model registry with SHA-256 verification, sharded downloads and lifecycle states; service worker for offline use.
- All inference moved to a module Web Worker; multi-threaded WebAssembly under cross-origin isolation; optional WebGPU.

## Tests run

Headless Chromium 1.63 build, 2 vCPU container, served with the production `_headers`:

| Check | Result |
|---|---|
| Worker harness: detection matches PyTorch export on COCO 39769 | same classes and boxes (cat 0.96, cat 0.93, remote 0.92 …) |
| CLIP encoder int8 per-channel vs fp32 | mean cosine 0.993, min 0.985 |
| OCR on synthetic label | 3 of 3 lines read; "DCD771" read as "DCDZ71" (italic font) |
| EAN-13 barcode | decoded 3017620422003 |
| Live loop, fake camera | 3.9–5 analysis fps, detection p95 ≈ 200 ms, identification ≈ 85 ms per crop |
| Hold → deep scan → identify → OCR → barcodes | completes, no console errors under strict CSP |
| Teach "Milo the cat" | both cats matched (see known issues); couch no longer matches |
| Test set, 2 photos | named 50%, top-3 or text 100% |

Not tested: real iPhone hardware, Safari's WebGPU path, cellular download behavior, Open Food Facts from a phone.

## Known issues

- Taught objects can't tell apart two very similar items of the same kind (two tabby cats). Teach the second one too; matching then picks the closer.
- Refined labels below about 30% are shown with their percentage. They're best guesses, not facts.
- First download is about 80 MB (engine, live detector, identification model, vocabulary). Deep scan, text reading and the 640 px detector add about 70 MB the first time they run. Later visits load from the cache.
- WebGPU mode falls back to CPU per model if a session fails to build, but a model that builds and computes incorrectly on WebGPU wouldn't be caught.
- The vocabulary has some odd LVIS entries (for example "cat furniture"); constrained refinement keeps them rare.

## Next

1. Benchmark on the iPhone 11 Pro Max: 416 vs 640 live detector, CPU vs WebGPU.
2. Build a real test set (50+ photos of things you care about) and tune the 0.8 relatedness cutoff and 0.25 refinement threshold against it.
3. Add a desktop eval script that runs the same models over a labeled folder on Windows.
