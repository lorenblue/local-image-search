# Local Image Search

Local-first semantic image search for macOS.

Current scope:

- scan local image folders
- embed images locally with CLIP
- store metadata and vectors in SQLite
- search with natural language
- use Raycast as the desktop UI

## Privacy

Images stay on the machine. Model downloads may require network access during setup, but
indexing and search run offline after the model is cached.

## Status

- recursive scanner for JPG, JPEG, PNG, HEIC, and WEBP
- SQLite index with sqlite-vec
- incremental skip logic based on path, size, mtime, and model name
- OpenCLIP image/text embeddings
- FastAPI local search service
- Raycast extension

## Setup

```bash
cd path/to/local-image-search
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ml,heic,api]"
```

## CLI

Index a folder:

```bash
image-search init
image-search index ~/Pictures/TestPhotos
```

Search from the terminal:

```bash
image-search search "red sports car"
image-search search "person wearing glasses" --limit 20
```

Run the local search API:

```bash
image-search serve
curl "http://127.0.0.1:8765/search?q=selfie%20in%20mirror&limit=5"
```

Open the local API reference at:

```text
http://127.0.0.1:8765/scalar
```

Run the Raycast extension:

```bash
cd raycast/local-image-search
npm install
npm run dev
```

To override the OpenCLIP model:

```bash
CLIP_MODEL=ViT-B-16 CLIP_PRETRAINED=datacomp_xl_s13b_b90k image-search index ~/Pictures/TestPhotos
```

By default, the database lives at:

```text
./data/images.db
```

Override it with:

```bash
image-search --db /path/to/images.db status
```

## Layout

```text
src/local_image_search/
  cli.py            command line entrypoint
  clip.py           CLIP image/text embedder implementations
  config.py         paths and supported image formats
  db.py             SQLite schema and repositories
  scanner.py        recursive folder scanning
  server.py         FastAPI local search service
  thumbnails.py     thumbnail cache
```

## Next

1. Compare OpenCLIP models on real photos.
2. Add OCR as a separate searchable field for text inside images.
3. Consider a stronger model once the 512-dim baseline is well understood.
