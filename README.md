# Local Image Search

Local Image Search is a local-first macOS image search tool. It indexes folders
with Gemma embeddings, stores vectors in SQLite, and exposes natural language and
visual similarity search through a local FastAPI service and Raycast extension.

The goal is simple: search local photos with queries like `red sports car`,
`person wearing glasses`, or `selfie in mirror` without sending image data to a
cloud service.

## Features

- Recursive indexing for JPG, JPEG, PNG, HEIC, and WEBP images
- Local EmbeddingGemma 2 image and text embeddings
- SQLite metadata storage with sqlite-vec vector search
- Incremental indexing based on path, size, modified time, and model name
- Automatic pruning of deleted files under scanned folders
- FastAPI API with Scalar docs
- Raycast UI with thumbnail results, paste/copy/open actions, batch paste,
  Quick Look, and visual similarity search

## Privacy Model

Images stay on the machine. Setup requires local model weights and Python/Node
dependencies, but indexing and search run offline once those are installed.

## How It Works

```text
folders -> scanner -> Gemma image embeddings -> SQLite/sqlite-vec
query   -> Gemma text embedding  -> vector search -> ranked image results
image   -> Gemma image embedding -> vector search -> visually similar images
```

Gemma maps both images and text into the same vector space. That lets the app
compare a text query such as `dog on beach` against stored image vectors, or
compare one image vector against the rest of the index for visual similarity.

## Architecture

```text
src/local_image_search/
  cli.py              CLI for init, index, search, similar, and serve
  embedder.py         Shared embedding interface
  gemma_embedder.py    EmbeddingGemma 2 implementation
  stub_embedder.py     Deterministic test embedder
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
python -m pip install -e ".[heic,api,face]"
```

Create `models/` in the project root and place your downloaded
`embeddinggemma-2-text-vision-440m.litertlm` file there. Installing the Python
package does not download this model. Gemma runs through LiteRT-LM on the GPU
and produces 768-dimensional image and text embeddings.

## Raycast

Run the Raycast extension:

```bash
cd raycast/local-image-search
npm install
npm run dev
```

In Raycast preferences, set `Project Directory` to this repository and
`Indexed Folders` to the folders you want searched. Separate multiple folders
with commas or newlines, for example:

```text
~/Pictures/TestPhotos, ~/Pictures/AnotherAlbum
```

When the command opens, it starts the local API if needed and syncs those
folders in the background. Search still works while indexing is running.

Useful Raycast actions:

```text
Enter          Paste the highlighted image (outside multi-select mode)
Option+Enter   Enter or exit multi-select mode
Enter / Space  Add or remove the highlighted image in multi-select mode
Cmd+Shift+V    Paste all selected images into the previous app
Cmd+Enter      Copy the selected image
```

Batch paste writes the selected image files to the macOS pasteboard, closes
Raycast, and sends a normal paste command to the previous app. Some apps accept
multiple pasted image files better than others.

## CLI

The CLI is still useful for manual indexing and debugging:

```bash
image-search index ~/Pictures/TestPhotos
image-search search "red sports car"
image-search search "person wearing glasses" --limit 20
```

Indexing stores Gemma image embeddings and separate InsightFace face boxes and
face embeddings when the current file or model metadata is stale.

Run the local search API manually:

```bash
image-search serve
curl "http://127.0.0.1:8766/search?q=selfie%20in%20mirror&limit=5"
```

Open the local API reference at:

```text
http://127.0.0.1:8766/scalar
```

## Troubleshooting

Raycast's server log is at `data/logs/server.log` under the project directory.
For indexing progress and errors, inspect the API status:

```bash
curl http://127.0.0.1:8766/status
```

Individual embedding or face inference failures are recorded in `indexing.failures`
and do not stop the remaining files. Raycast shows a failed-file count and a
**View Indexing Failures** action on its status item. Failed work is retried on
the next folder sync; database errors still stop indexing.

If a folder appears in Raycast preferences but its images do not show up in the
index, macOS privacy permissions may be blocking the background server. Grant
Full Disk Access to Raycast, then restart the local server by quitting the old
process or rebooting Raycast.

If the server is started through Homebrew Python, macOS may also require Full
Disk Access for Python.app. For example:

```text
/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/Resources/Python.app
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

EmbeddingGemma 2 is the production GPU embedding backend. Keep
`embeddinggemma-2-text-vision-440m.litertlm` in the project's ignored `models/`
directory. The app resolves this location independently of the working directory.
Raycast uses Gemma automatically; no model selection or model path is needed.

Optionally override the model location with `--gemma-model` or `GEMMA_MODEL_PATH`.
The engine loads on first use, is reused, and closes at command exit, including
errors. The server closes it at shutdown after background indexing finishes.
WebP and HEIC inputs are converted to temporary PNGs for LiteRT decoding.
Face indexing remains configured separately.

To review stored face boxes, generate a local HTML contact sheet:

```bash
image-search faces-review ~/Pictures/TestPhotos --limit 50 --output data/faces-review.html
```

The face review command reads indexed face rows from SQLite. It does not rerun
face detection. The review page shows each stored face ID, which can be used for
similar-face search:

```bash
image-search similar-face 123 --limit 10
```

## What This Project Demonstrates

- Designing a local-first AI workflow for private media
- Evaluating caption-based search versus direct Gemma embedding search
- Using SQLite as both metadata storage and a lightweight vector index
- Keeping CLI and API behavior shared through a service layer
- Building a desktop workflow around a local API with Raycast
