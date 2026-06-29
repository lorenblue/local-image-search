import {
  Action,
  ActionPanel,
  Color,
  Detail,
  Grid,
  Icon,
  getPreferenceValues,
  open,
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
    if (!trimmedQuery && !similarSource) {
      setResults([]);
      setSelectedItemId(undefined);
      return;
    }

    const controller = new AbortController();
    const timeout = setTimeout(async () => {
      setIsLoading(true);
      setError(null);
      try {
        const url = new URL(
          trimmedQuery ? `${apiBaseUrl}/search` : `${apiBaseUrl}/similar`,
        );
        if (trimmedQuery) {
          url.searchParams.set("q", trimmedQuery);
        } else if (similarSource) {
          url.searchParams.set("path", similarSource.path);
        }
        url.searchParams.set("limit", String(DEFAULT_LIMIT));
        const response = await fetchJson<SearchResponse | SimilarResponse>(
          url.toString(),
          controller.signal,
        );
        setResults(response.results);
        setSelectedItemId(resultItemId(response.results[0]));
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
  }, [apiBaseUrl, query, similarSource]);

  const searchBarPlaceholder = useMemo(() => {
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
        query.trim() || !similarSource ? "Search Images" : "Similar Images"
      }
      onSearchTextChange={handleSearchTextChange}
      onSelectionChange={(id) => setSelectedItemId(id ?? undefined)}
      searchBarPlaceholder={searchBarPlaceholder}
      searchText={query}
      selectedItemId={selectedItemId}
      throttle
    >
      {!query.trim() && !similarSource && status ? (
        <StatusItem status={status} apiBaseUrl={apiBaseUrl} />
      ) : null}
      {results.map((result) => (
        <ResultItem
          key={resultItemId(result)}
          result={result}
          onFindSimilar={() => {
            setSimilarSource(result);
            setQuery("");
          }}
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
  onTrash,
}: {
  result: SearchResult;
  onFindSimilar: () => void;
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
  return result ? String(result.id) : undefined;
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
