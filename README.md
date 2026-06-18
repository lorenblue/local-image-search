# Local Image Search

Local Image Search is a local-first macOS image search tool. It indexes folders
with CLIP embeddings, stores vectors in SQLite, and exposes natural language and
visual similarity search through a local FastAPI service and Raycast extension.

The goal is simple: search local photos with queries like `red sports car`,
`person wearing glasses`, or `selfie in mirror` without sending image data to a
cloud service.

## Features

- Recursive indexing for JPG, JPEG, PNG, HEIC, and WEBP images
- Local OpenCLIP image and text embeddings
- SQLite metadata storage with sqlite-vec vector search
- Incremental indexing based on path, size, modified time, and model name
- Automatic pruning of deleted files under scanned folders
- FastAPI API with Scalar docs
- Raycast UI with thumbnail results, paste/copy/open actions, Quick Look, and
  visual similarity search

## Privacy Model

Images stay on the machine. Setup may download model weights and Python/Node
dependencies, but indexing and search run offline after the model is cached.

## How It Works

```text
folders -> scanner -> CLIP image embeddings -> SQLite/sqlite-vec
query   -> CLIP text embedding  -> vector search -> ranked image results
image   -> CLIP image embedding -> vector search -> visually similar images
```

CLIP maps both images and text into the same vector space. That lets the app
compare a text query such as `dog on beach` against stored image vectors, or
compare one image vector against the rest of the index for visual similarity.

## Architecture

```text
src/local_image_search/
  cli.py              CLI for init, index, search, similar, and serve
  clip.py             OpenCLIP and test embedder implementations
  db.py               SQLite schema, sqlite-vec integration, and search queries
  scanner.py          Recursive image discovery
  search_service.py   Shared search/status service used by CLI and API
  server.py           FastAPI local API
  thumbnails.py       Local thumbnail cache

raycast/local-image-search/
  src/search-images.tsx   Raycast grid UI
```

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

## Configuration

By default, the database lives at:

```text
./data/images.db
```

Override it with:

```bash
image-search --db /path/to/images.db status
```

To override the OpenCLIP model:

```bash
CLIP_MODEL=ViT-B-16 CLIP_PRETRAINED=datacomp_xl_s13b_b90k image-search index ~/Pictures/TestPhotos
```

## What This Project Demonstrates

- Designing a local-first AI workflow for private media
- Evaluating caption-based search versus direct CLIP embedding search
- Using SQLite as both metadata storage and a lightweight vector index
- Keeping CLI and API behavior shared through a service layer
- Building a desktop workflow around a local API with Raycast

## Future Work

1. Compare OpenCLIP models on real photos.
2. Add OCR as a separate searchable field for text inside images.
3. Add saved searches or folder presets.
4. Explore face clustering without identity recognition.
