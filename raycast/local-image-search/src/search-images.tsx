import {
  Action,
  ActionPanel,
  Color,
  Detail,
  Grid,
  Icon,
  Toast,
  closeMainWindow,
  getPreferenceValues,
  open,
  showToast,
} from "@raycast/api";
import { execFile } from "child_process";
import { useEffect, useMemo, useState } from "react";
import { promisify } from "util";

import { ensureServerRunning } from "./server";

type Preferences = {
  apiBaseUrl: string;
  projectDirectory: string;
  indexedFolders?: string;
  clipModelPreset: string;
};

type SearchResult = {
  id: number;
  faceId?: number;
  path: string;
  fileName: string;
  score: number;
  embeddingModel: string;
  thumbnailPath: string | null;
};

type SearchResponse = {
  query: string;
  limit: number;
  elapsedMs: number;
  results: SearchResult[];
};

type SimilarResponse = {
  path: string;
  limit: number;
  elapsedMs: number;
  results: SearchResult[];
};

type FaceBox = {
  x: number;
  y: number;
  width: number;
  height: number;
  detectionScore: number | null;
};

type FaceResult = {
  faceId: number;
  imageId: number;
  path: string;
  fileName: string;
  score: number;
  faceEmbeddingModel: string;
  thumbnailPath: string | null;
  box: FaceBox;
};

type SimilarFaceResponse = {
  faceId: number;
  limit: number;
  elapsedMs: number;
  results: FaceResult[];
};

type PrimaryFaceResponse = {
  faceId: number;
  imageId: number;
  path: string;
  fileName: string;
  faceEmbeddingModel: string;
  thumbnailPath: string | null;
  box: FaceBox;
};

type StatusResponse = {
  database: string;
  clipEmbedder: string;
  clipModelPreset: string | null;
  clipEmbeddingDimensions: number;
  clipModelLoaded: boolean;
  indexEmbeddingDimensions: number | null;
  indexedImages: number;
  searchableImages: number;
  indexing: IndexingStatus;
  memory: {
    currentMb: number;
    peakMb: number;
  };
  uptimeSeconds: number;
};

type IndexingStatus = {
  roots: string[];
  running: boolean;
  total: number;
  processed: number;
  indexed: number;
  skipped: number;
  deleted: number;
  phase: string;
  lastFile: string | null;
  startedAt: number | null;
  finishedAt: number | null;
  error: string | null;
};

const DEFAULT_LIMIT = 30;
const STATUS_POLL_MS = 2000;
const execFileAsync = promisify(execFile);

export default function Command() {
  const preferences = getPreferenceValues<Preferences>();
  const apiBaseUrl = normalizeBaseUrl(preferences.apiBaseUrl);
  const clipModelPreset = normalizeClipModelPreset(preferences.clipModelPreset);
  const [query, setQuery] = useState("");
  const [similarSource, setSimilarSource] = useState<SearchResult | null>(null);
  const [similarFaceSource, setSimilarFaceSource] =
    useState<SearchResult | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedItemId, setSelectedItemId] = useState<string | undefined>();
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [multiSelectMode, setMultiSelectMode] = useState(false);
  const [pasteSelection, setPasteSelection] = useState<string[]>([]);
  const pasteSelectionSet = useMemo(
    () => new Set(pasteSelection),
    [pasteSelection],
  );

  useEffect(() => {
    let cancelled = false;

    async function loadStatus() {
      setIsLoading(true);
      setError(null);
      try {
        await ensureServerRunning(
          apiBaseUrl,
          preferences.projectDirectory,
          clipModelPreset,
        );
        const indexedFolders = parseIndexedFolders(preferences.indexedFolders);
        let response = await fetchJson<StatusResponse>(
          `${apiBaseUrl}/status`,
        );
        if (indexedFolders.length > 0 && !response.indexing.running) {
          await postJson(`${apiBaseUrl}/sync`, { roots: indexedFolders });
          response = await fetchJson<StatusResponse>(`${apiBaseUrl}/status`);
        }
        if (!cancelled) {
          setStatus(response);
        }
      } catch (unknownError) {
        if (!cancelled) {
          setError(errorMessage(unknownError));
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    loadStatus();
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl, preferences.projectDirectory, preferences.indexedFolders, clipModelPreset]);

  useEffect(() => {
    if (!status?.indexing.running) {
      return;
    }

    let cancelled = false;
    const interval = setInterval(async () => {
      try {
        const response = await fetchJson<StatusResponse>(
          `${apiBaseUrl}/status`,
        );
        if (!cancelled) {
          setStatus(response);
        }
      } catch (unknownError) {
        if (!cancelled) {
          setError(errorMessage(unknownError));
        }
      }
    }, STATUS_POLL_MS);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [apiBaseUrl, status?.indexing.running]);

  useEffect(() => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery && !similarSource && !similarFaceSource) {
      setResults([]);
      setSelectedItemId(undefined);
      return;
    }

    const controller = new AbortController();
    const timeout = setTimeout(async () => {
      setIsLoading(true);
      setError(null);
      try {
        const url = new URL(searchUrl(apiBaseUrl, trimmedQuery, similarFaceSource));
        if (trimmedQuery) {
          url.searchParams.set("q", trimmedQuery);
        } else if (similarFaceSource?.faceId) {
          url.searchParams.set("faceId", String(similarFaceSource.faceId));
        } else if (similarSource) {
          url.searchParams.set("path", similarSource.path);
        }
        url.searchParams.set("limit", String(DEFAULT_LIMIT));
        const response = await fetchJson<
          SearchResponse | SimilarResponse | SimilarFaceResponse
        >(
          url.toString(),
          controller.signal,
        );
        const nextResults = similarFaceSource
          ? (response as SimilarFaceResponse).results.map(faceResultToSearchResult)
          : (response as SearchResponse | SimilarResponse).results;
        setResults(nextResults);
        setSelectedItemId(resultItemId(nextResults[0]));
      } catch (unknownError) {
        if (!controller.signal.aborted) {
          setError(errorMessage(unknownError));
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      }
    }, 180);

    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, [apiBaseUrl, query, similarSource, similarFaceSource]);

  const searchBarPlaceholder = useMemo(() => {
    if (similarFaceSource) {
      return `Similar faces to ${similarFaceSource.fileName}`;
    }
    if (similarSource) {
      return `Similar to ${similarSource.fileName}`;
    }
    if (status) {
      return `Search ${status.searchableImages} indexed images`;
    }
    return "Search indexed images";
  }, [similarSource, status]);
  const baseNavigationTitle = navigationTitle(
    query,
    similarSource,
    similarFaceSource,
  );
  const gridNavigationTitle = [
    baseNavigationTitle,
    multiSelectMode ? "Multi-select mode" : null,
    pasteSelection.length > 0
      ? `${pasteSelection.length} selected`
      : null,
  ]
    .filter(Boolean)
    .join(" · ");

  if (error) {
    return <ServerError apiBaseUrl={apiBaseUrl} message={error} />;
  }

  function handleSearchTextChange(text: string) {
    setQuery(text);
    if (text.trim()) {
      setSimilarSource(null);
      setSimilarFaceSource(null);
    }
  }

  async function handleFindSimilarFace(result: SearchResult) {
    setIsLoading(true);
    setError(null);
    try {
      const url = new URL(`${apiBaseUrl}/primary-face`);
      url.searchParams.set("imageId", String(result.id));
      const primaryFace = await fetchJson<PrimaryFaceResponse>(url.toString());
      setSimilarFaceSource({ ...result, faceId: primaryFace.faceId });
      setSimilarSource(null);
      setQuery("");
      setSelectedItemId(undefined);
    } catch (unknownError) {
      await showToast({
        style: Toast.Style.Failure,
        title: "No Similar Face Search",
        message: errorMessage(unknownError),
      });
    } finally {
      setIsLoading(false);
    }
  }

  function handleResultsTrashed(paths: string[]) {
    const trashedPaths = new Set(paths);
    const selectedIndex = results.findIndex(
      (result) => resultItemId(result) === selectedItemId,
    );
    const selectedResultWasTrashed =
      selectedIndex >= 0 && trashedPaths.has(results[selectedIndex].path);
    const nextResults = results.filter((result) => !trashedPaths.has(result.path));
    const nextSelection = selectedResultWasTrashed
      ? results
          .slice(selectedIndex + 1)
          .find((result) => !trashedPaths.has(result.path)) ??
        results
          .slice(0, selectedIndex)
          .reverse()
          .find((result) => !trashedPaths.has(result.path))
      : nextResults.find((result) => resultItemId(result) === selectedItemId);

    setPasteSelection((currentSelection) =>
      currentSelection.filter((selectedPath) => !trashedPaths.has(selectedPath)),
    );
    setResults(nextResults);
    setSelectedItemId(resultItemId(nextSelection));
  }

  function handleTogglePasteSelection(path: string) {
    setPasteSelection((currentSelection) => {
      if (currentSelection.includes(path)) {
        return currentSelection.filter((selectedPath) => selectedPath !== path);
      }
      return [...currentSelection, path];
    });
  }

  async function handlePasteSelection(paths = pasteSelection) {
    if (paths.length === 0) {
      await showToast({
        style: Toast.Style.Failure,
        title: "No Images Selected",
        message: "Add images to the paste selection first.",
      });
      return;
    }

    try {
      await pasteFiles(paths);
      setPasteSelection([]);
    } catch (unknownError) {
      await showToast({
        style: Toast.Style.Failure,
        title: "Paste Failed",
        message: errorMessage(unknownError),
      });
    }
  }

  return (
    <Grid
      columns={5}
      fit={Grid.Fit.Fill}
      inset={Grid.Inset.Small}
      isLoading={isLoading}
      navigationTitle={gridNavigationTitle}
      onSearchTextChange={handleSearchTextChange}
      onSelectionChange={(id) => setSelectedItemId(id ?? undefined)}
      searchBarPlaceholder={searchBarPlaceholder}
      searchText={query}
      selectedItemId={selectedItemId}
      throttle
    >
      {!query.trim() && !similarSource && !similarFaceSource && status ? (
        <StatusItem status={status} apiBaseUrl={apiBaseUrl} />
      ) : null}
      {results.map((result) => (
        <ResultItem
          key={resultItemId(result)}
          result={result}
          onFindSimilar={() => {
            setSimilarSource(result);
            setSimilarFaceSource(null);
            setQuery("");
          }}
          onFindSimilarFace={() => handleFindSimilarFace(result)}
          onPasteSelection={() => handlePasteSelection()}
          onTogglePasteSelection={() => handleTogglePasteSelection(result.path)}
          onToggleMultiSelectMode={() =>
            setMultiSelectMode((currentMode) => !currentMode)
          }
          onTrash={() => handleResultsTrashed([result.path])}
          onTrashSelection={() => handleResultsTrashed(pasteSelection)}
          multiSelectMode={multiSelectMode}
          pasteSelectionCount={pasteSelection.length}
          selectedPaths={pasteSelection}
          isSelectedForPaste={pasteSelectionSet.has(result.path)}
        />
      ))}
    </Grid>
  );
}

function StatusItem({
  status,
  apiBaseUrl,
}: {
  status: StatusResponse;
  apiBaseUrl: string;
}) {
  const indexingLabel = status.indexing.running
    ? ` · ${indexingPhaseLabel(status.indexing)}`
    : "";
  const errorLabel = status.indexing.error ? " · indexing error" : "";
  const modelLabel = status.clipModelPreset
    ? ` · ${status.clipModelPreset}`
    : "";
  const modelLoadLabel = status.clipModelLoaded ? "" : " · loading model";

  return (
    <Grid.Item
      id="status"
      title="Local Image Search"
      subtitle={`${status.searchableImages} searchable images${indexingLabel}${errorLabel}${modelLabel}${modelLoadLabel} · ${status.memory.currentMb.toFixed(0)} MB`}
      content={{ source: Icon.MagnifyingGlass }}
      actions={
        <ActionPanel>
          <Action.OpenInBrowser
            title="Open API Reference"
            url={`${apiBaseUrl}/scalar`}
          />
        </ActionPanel>
      }
    />
  );
}

function ResultItem({
  result,
  onFindSimilar,
  onFindSimilarFace,
  onPasteSelection,
  onTogglePasteSelection,
  onToggleMultiSelectMode,
  onTrash,
  onTrashSelection,
  multiSelectMode,
  pasteSelectionCount,
  selectedPaths,
  isSelectedForPaste,
}: {
  result: SearchResult;
  onFindSimilar: () => void;
  onFindSimilarFace: () => void;
  onPasteSelection: () => void;
  onTogglePasteSelection: () => void;
  onToggleMultiSelectMode: () => void;
  onTrash: () => void;
  onTrashSelection: () => void;
  multiSelectMode: boolean;
  pasteSelectionCount: number;
  selectedPaths: string[];
  isSelectedForPaste: boolean;
}) {
  const content = result.thumbnailPath
    ? { source: result.thumbnailPath }
    : { source: Icon.Image, tintColor: Color.SecondaryText };
  const pasteSelectionTitle = isSelectedForPaste
    ? "Remove from Selection"
    : "Add to Selection";
  const multiSelectModeTitle = multiSelectMode
    ? "Exit Multi-Select Mode"
    : "Enter Multi-Select Mode";

  return (
    <Grid.Item
      id={resultItemId(result)}
      title={result.fileName}
      subtitle={scoreLabel(result.score)}
      content={content}
      accessory={
        isSelectedForPaste
          ? { icon: Icon.CheckCircle, tooltip: "Selected for batch paste" }
          : undefined
      }
      quickLook={{ name: result.fileName, path: result.path }}
      actions={
        <ActionPanel>
          <ActionPanel.Section>
            {multiSelectMode ? (
              <>
                <Action
                  title={pasteSelectionTitle}
                  icon={isSelectedForPaste ? Icon.MinusCircle : Icon.PlusCircle}
                  onAction={onTogglePasteSelection}
                />
                <Action
                  title={`${pasteSelectionTitle} (Space)`}
                  icon={isSelectedForPaste ? Icon.MinusCircle : Icon.PlusCircle}
                  shortcut={{ modifiers: [], key: "space" }}
                  onAction={onTogglePasteSelection}
                />
              </>
            ) : null}
            <Action.Paste title="Paste Image" content={{ file: result.path }} />
            <Action.CopyToClipboard
              title="Copy Image"
              content={{ file: result.path }}
              shortcut={{ modifiers: ["cmd"], key: "enter" }}
            />
            <Action
              title={multiSelectModeTitle}
              icon={multiSelectMode ? Icon.CheckCircle : Icon.PlusCircle}
              shortcut={{ modifiers: ["opt"], key: "enter" }}
              onAction={onToggleMultiSelectMode}
            />
            {pasteSelectionCount > 0 ? (
              <Action
                title={`Paste ${pasteSelectionCount} Selected ${pluralizeImage(pasteSelectionCount)}`}
                icon={Icon.Clipboard}
                shortcut={{ modifiers: ["cmd", "shift"], key: "v" }}
                onAction={onPasteSelection}
              />
            ) : null}
            <Action
              title="Open Image"
              icon={Icon.Image}
              onAction={() => open(result.path)}
            />
            <Action.ToggleQuickLook />
            <Action.ShowInFinder path={result.path} />
            {selectedPaths.length > 1 ? (
              <Action.Trash
                title={`Trash ${selectedPaths.length} Selected ${pluralizeImage(selectedPaths.length)}`}
                paths={selectedPaths}
                onTrash={onTrashSelection}
                shortcut={{ modifiers: ["cmd", "shift"], key: "d" }}
              />
            ) : null}
            <Action.Trash
              paths={result.path}
              onTrash={onTrash}
              shortcut={{ modifiers: ["cmd"], key: "d" }}
            />
          </ActionPanel.Section>
          <ActionPanel.Section>
            <Action
              title="Find Similar Images"
              icon={Icon.BullsEye}
              onAction={onFindSimilar}
            />
            <Action
              title="Find Similar Faces"
              icon={Icon.Person}
              onAction={onFindSimilarFace}
            />
            <Action.CopyToClipboard title="Copy Path" content={result.path} />
          </ActionPanel.Section>
        </ActionPanel>
      }
    />
  );
}

function ServerError({
  apiBaseUrl,
  message,
}: {
  apiBaseUrl: string;
  message: string;
}) {
  const markdown = [
    "# Server Not Available",
    "",
    `Could not reach \`${apiBaseUrl}\`.`,
    "",
    "Start the local API server manually, or check the Raycast Project Directory preference:",
    "",
    "```bash",
    "cd path/to/local-image-search",
    "source .venv/bin/activate",
    "image-search serve --port 8766",
    "```",
    "",
    `Error: \`${message}\``,
  ].join("\n");

  return <Detail markdown={markdown} />;
}

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${body}`);
  }
  return (await response.json()) as T;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const responseBody = await response.text();
    throw new Error(
      `${response.status} ${response.statusText}: ${responseBody}`,
    );
  }
  return (await response.json()) as T;
}

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, "");
}

function normalizeClipModelPreset(value: string): string {
  return value === "fast" ? "fast" : "better";
}

function scoreLabel(score: number): string {
  return score.toFixed(3);
}

function indexingPhaseLabel(indexing: IndexingStatus): string {
  if (indexing.phase === "starting") {
    return "starting index";
  }
  if (indexing.phase === "preparingDatabase") {
    return "preparing database";
  }
  if (indexing.phase === "scanning") {
    return "scanning folders";
  }
  if (indexing.total > 0) {
    return `indexing ${indexing.processed}/${indexing.total}`;
  }
  return "indexing";
}

function navigationTitle(
  query: string,
  similarSource: SearchResult | null,
  similarFaceSource: SearchResult | null,
): string {
  if (query.trim() || (!similarSource && !similarFaceSource)) {
    return "Search Images";
  }
  return similarFaceSource ? "Similar Faces" : "Similar Images";
}

async function pasteFiles(paths: string[]) {
  await setPasteboardFiles(paths);
  await closeMainWindow({ clearRootSearch: false });
  await delay(150);
  await execFileAsync("/usr/bin/osascript", [
    "-e",
    'tell application "System Events" to keystroke "v" using command down',
  ]);
}

async function setPasteboardFiles(paths: string[]) {
  const script = `
import AppKit

let urls = CommandLine.arguments.dropFirst().map { path in
  NSURL(fileURLWithPath: path)
}

let pasteboard = NSPasteboard.general
pasteboard.clearContents()

if !pasteboard.writeObjects(urls) {
  fputs("Could not write files to the pasteboard.\\n", stderr)
  exit(1)
}
`;

  await execFileAsync("/usr/bin/swift", [
    "-module-cache-path",
    "/tmp/local-image-search-swift-cache",
    "-e",
    script,
    ...paths,
  ]);
}

function delay(milliseconds: number) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function pluralizeImage(count: number): string {
  return count === 1 ? "Image" : "Images";
}

function resultItemId(result: SearchResult | undefined): string | undefined {
  if (!result) {
    return undefined;
  }
  return result.faceId ? `face-${result.faceId}` : `image-${result.id}`;
}

function searchUrl(
  apiBaseUrl: string,
  trimmedQuery: string,
  similarFaceSource: SearchResult | null,
): string {
  if (trimmedQuery) {
    return `${apiBaseUrl}/search`;
  }
  if (similarFaceSource) {
    return `${apiBaseUrl}/similar-face`;
  }
  return `${apiBaseUrl}/similar`;
}

function faceResultToSearchResult(result: FaceResult): SearchResult {
  return {
    id: result.imageId,
    faceId: result.faceId,
    path: result.path,
    fileName: result.fileName,
    score: result.score,
    embeddingModel: result.faceEmbeddingModel,
    thumbnailPath: result.thumbnailPath,
  };
}

function parseIndexedFolders(value: string | undefined): string[] {
  if (!value) {
    return [];
  }
  return value
    .split(/[,\n]/)
    .map((folder) => folder.trim())
    .filter(Boolean);
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}
