# Local Image Search Raycast Extension

Raycast UI for the local image search API.

## Run

Run the extension:

```bash
cd path/to/local-image-search/raycast/local-image-search
npm install
npm run dev
```

Set the extension preferences:

```text
API Base URL: http://127.0.0.1:8766
Project Directory: ~/Projects/local-image-search
Indexed Folders: ~/Pictures/TestPhotos
```

The extension starts the local API if needed and syncs indexed folders in the
background when it opens.

Multi-file paste uses the macOS file pasteboard and a simulated Command-V after
Raycast closes. On macOS 27, allow Raycast in System Settings > Privacy &
Security > Device Control and Data Access if multi-file paste does not run.
