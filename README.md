# OmniScan (web)

Camera app that detects, identifies and reads what the phone sees. All perception runs on the device in a Web Worker; Claude is only asked when you approve sending a specific image.

## Deploy

```
python build.py          # writes ./site (about 180 MB)
```

Drag the `site` folder onto https://app.netlify.com/drop, or connect the repo (netlify.toml publishes `site`). `site/_headers` sets the cross-origin isolation headers that enable multi-threaded inference, plus a strict Content Security Policy.

Open the HTTPS link in Safari. Add to Home Screen for a full-screen app that also works offline once models have downloaded.

## Pipeline

| Stage | Model | Runs | License |
|---|---|---|---|
| Live detection | D-FINE N, COCO 80 classes, 416 or 640 px | every frame (one in flight, stale frames dropped) | Apache-2.0 |
| Deep scan | D-FINE S, Objects365 (365 classes) | when you hold a frame or open a photo | Apache-2.0 |
| Identification | TinyCLIP ViT-40M/32 image encoder (int8 per-channel) against 2,853 precomputed label embeddings | on sharp crops of stable tracks, about every 1.5 s per object | MIT |
| Text | PP-OCRv3 detector + recognizer (via RapidOCR) | on hold, or "Read text" | Apache-2.0 |
| Barcodes | zxing-cpp (zxing-wasm) | every 0.7 s live, full resolution on hold | Apache-2.0 |
| Runtime | ONNX Runtime Web 1.29 (WebAssembly, optional WebGPU) | worker | MIT |

How labels are decided:

1. **Voting.** Each track accumulates detector class scores over frames (older votes fade), so one bad frame can't flip a label.
2. **Sharpest view.** Each track keeps its sharpest crop (Laplacian variance). That crop is what identification uses and what Ask Omni sends.
3. **Constrained refinement.** The image embedding is compared only with vocabulary labels whose text embedding is close (cosine ≥ 0.8) to the detector's class, plus the class itself. This stops "person → unicycle" style errors that unconstrained CLIP makes on small crops. Embeddings are averaged over views, weighted by sharpness.
4. **Taught objects.** "Name it" stores the track's embedding with its class. A taught object matches at cosine ≥ 0.88 with a 0.02 margin over other taught objects, and only for the same kind of thing unless the views are nearly identical. "Wrong label" adds a correction the same way.
5. **Tap to identify.** Tapping a spot with no small box searches the whole vocabulary (all, objects, products, or plants & animals).
6. **Text and barcodes** are attached to the objects they sit on and included as evidence. Barcode numbers can be looked up on Open Food Facts (number only, never images).

People are detected but never identified.

## Measuring

The flask button opens the test set: add photos, type the expected label, run. Results show the share named correctly, the share in the top 3 or read from text, the whole-image open-vocabulary hit rate, and time per photo. Runs are kept so you can compare after changing settings or teaching objects.

## Rebuilding models

`tools/` holds the scripts used to make the model files (Python 3.13, torch CPU, transformers ≥ 4.53, onnxruntime, rapidocr_onnxruntime):

- `export_dfine.py <hf repo> <out.onnx> <size>`: exports and sanity-checks a D-FINE detector.
- `build_clip.py <outdir>`: builds the vocabulary, text embeddings and the TinyCLIP image encoder. It needs `vocab/lvis.py` (detectron2 `lvis_v1_categories.py`), `vocab/oi.csv` (Open Images boxable class descriptions) and `vocab/in.ipynb` (OpenAI CLIP ImageNet prompt notebook) next to `nature.txt` and `products.txt`.
- `parents.py`: text embeddings for the two detectors' class names.
- `fusion_test.py`: offline check of the constrained refinement.

After `build_clip.py`, re-quantize the encoder per-channel (MatMul only, patch embedding and projection excluded); plain dynamic int8 dropped embedding fidelity to a mean cosine of 0.94, while per-channel holds 0.993.

## This repository

The repository root is the built site, served by GitHub Pages (Settings › Pages › Deploy from a branch › `main` / root). GitHub Pages can't send the cross-origin isolation headers, so `sw.js` adds them and the app reloads once on the first visit; after that, inference runs multi-threaded. `_headers` is used only if the same folder is deployed to Netlify.
