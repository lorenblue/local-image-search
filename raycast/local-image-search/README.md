# Local Image Search Raycast Extension

Raycast UI for the local image search API.

## Run

Start the backend first:

```bash
cd path/to/local-image-search
source .venv/bin/activate
image-search serve --embedder sentence-transformers --clip-embedder open-clip
```

Then run the extension:

```bash
cd path/to/local-image-search/raycast/local-image-search
npm install
npm run dev
```

The extension has a Search Mode preference. Use Caption for the existing
caption-based search or CLIP after running `image-search index-clip`.
