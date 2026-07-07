import {
  Action,
  ActionPanel,
  Color,
  Detail,
  Grid,
  Icon,
  Toast,
  getPreferenceValues,
  open,
  showToast,
} from "@raycast/api";
import { useEffect, useMemo, useState } from "react";

import { ensureServerRunning } from "./server";

type Preferences = {
  apiBaseUrl: string;
  projectDirectory: string;
  indexedFolders?: string;
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
  lastFile: string | null;
  startedAt: number | null;
  finishedAt: number | null;
  error: string | null;
};

const DEFAULT_LIMIT = 30;
const STATUS_POLL_MS = 2000;

export default function Command() {
  const preferences = getPreferenceValues<Preferences>();
  const apiBaseUrl = normalizeBaseUrl(preferences.apiBaseUrl);
  const [query, setQuery] = useState("");
  const [similarSource, setSimilarSource] = useState<SearchResult | null>(null);
  const [similarFaceSource, setSimilarFaceSource] =
    useState<SearchResult | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedItemId, setSelectedItemId] = useState<string | undefined>();
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadStatus() {
      setIsLoading(true);
      setError(null);
      try {
        await ensureServerRunning(apiBaseUrl, preferences.projectDirectory);
        const indexedFolders = parseIndexedFolders(preferences.indexedFolders);
        if (indexedFolders.length > 0) {
          await postJson(`${apiBaseUrl}/sync`, { roots: indexedFolders });
        }
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
  }, [apiBaseUrl, preferences.projectDirectory, preferences.indexedFolders]);

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

  function handleResultTrashed(path: string) {
    setResults((currentResults) => {
      const nextResults = currentResults.filter((result) => result.path !== path);
      setSelectedItemId(resultItemId(nextResults[0]));
      return nextResults;
    });
  }

  return (
    <Grid
      columns={5}
      fit={Grid.Fit.Fill}
      inset={Grid.Inset.Small}
      isLoading={isLoading}
      navigationTitle={
        query.trim() || (!similarSource && !similarFaceSource)
          ? "Search Images"
          : similarFaceSource
            ? "Similar Faces"
            : "Similar Images"
      }
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
          onTrash={() => handleResultTrashed(result.path)}
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
    ? ` · indexing ${status.indexing.processed}/${status.indexing.total}`
    : "";
  const errorLabel = status.indexing.error ? " · indexing error" : "";

  return (
    <Grid.Item
      id="status"
      title="Local Image Search"
      subtitle={`${status.searchableImages} searchable images${indexingLabel}${errorLabel} · ${status.memory.currentMb.toFixed(0)} MB`}
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
  onTrash,
}: {
  result: SearchResult;
  onFindSimilar: () => void;
  onFindSimilarFace: () => void;
  onTrash: () => void;
}) {
  const content = result.thumbnailPath
    ? { source: result.thumbnailPath }
    : { source: Icon.Image, tintColor: Color.SecondaryText };

  return (
    <Grid.Item
      id={resultItemId(result)}
      title={result.fileName}
      subtitle={scoreLabel(result.score)}
      content={content}
      quickLook={{ name: result.fileName, path: result.path }}
      actions={
        <ActionPanel>
          <ActionPanel.Section>
            <Action.Paste title="Paste Image" content={{ file: result.path }} />
            <Action.CopyToClipboard
              title="Copy Image"
              content={{ file: result.path }}
              shortcut={{ modifiers: ["cmd"], key: "enter" }}
            />
            <Action
              title="Open Image"
              icon={Icon.Image}
              onAction={() => open(result.path)}
            />
            <Action.ToggleQuickLook />
            <Action.ShowInFinder path={result.path} />
            <Action.Trash paths={result.path} onTrash={onTrash} />
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

function scoreLabel(score: number): string {
  return score.toFixed(3);
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
