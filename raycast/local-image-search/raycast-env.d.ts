/// <reference types="@raycast/api">

/* 🚧 🚧 🚧
 * This file is auto-generated from the extension's manifest.
 * Do not modify manually. Instead, update the `package.json` file.
 * 🚧 🚧 🚧 */

/* eslint-disable @typescript-eslint/ban-types */

type ExtensionPreferences = {
  /** API Base URL - Local Image Search API base URL */
  "apiBaseUrl": string,
  /** Search Mode - Search captions or direct CLIP image embeddings */
  "searchMode": "caption" | "clip"
}

/** Preferences accessible in all the extension's commands */
declare type Preferences = ExtensionPreferences

declare namespace Preferences {
  /** Preferences accessible in the `search-images` command */
  export type SearchImages = ExtensionPreferences & {}
}

declare namespace Arguments {
  /** Arguments passed to the `search-images` command */
  export type SearchImages = {}
}

