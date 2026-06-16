# Local Image Search

Local-first semantic image search for macOS.

Current scope:

- scan local image folders
- caption images locally
- embed captions locally
- store metadata and embeddings in SQLite
- search with natural language

The eventual UI target is a Raycast extension backed by a local service. The first version is a CLI so the model/search path can be tested before building UI.

## Privacy

Images should stay on the machine. Model downloads may require network access during setup, but indexing and search should run offline after that.

## Status

- recursive scanner for JPG, JPEG, PNG, HEIC, and WEBP
- SQLite index
- incremental skip logic based on path, size, mtime, and model names
- stub captioner/embedder for smoke tests
- Moondream captioner hook
- sentence-transformers embedder hook
- cosine similarity search

## Setup

```bash
cd /Users/lorenzo/Projects/local-image-search
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional local ML dependencies:

```bash
python -m pip install -e ".[ml,heic]"
```

## CLI

Stub mode does not require model downloads:

```bash
image-search init
image-search index ~/Pictures/test-images --captioner stub --embedder stub
image-search search "person wearing glasses" --embedder stub
image-search status
```

Local model mode uses Moondream 2 by default:

```bash
image-search index ~/Pictures/TestPhotos --captioner moondream --embedder sentence-transformers
image-search search "selfie in mirror" --embedder sentence-transformers
```

To override the Moondream model:

```bash
MOONDREAM_MODEL=moondream3-preview image-search index ~/Pictures/TestPhotos --captioner moondream --embedder sentence-transformers
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
  config.py         paths and supported image formats
  db.py             SQLite schema and repositories
  scanner.py        recursive folder scanning
  captioning.py     captioner interface and implementations
  embeddings.py     embedder interface and implementations
  search.py         cosine similarity ranking
```

## Next

1. Test Moondream caption quality on real photos.
2. Compare Florence-2 if Moondream captions are too generic.
3. Add a local API.
4. Add a Raycast extension.
