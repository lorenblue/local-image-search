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

type Preferences = {
  apiBaseUrl: string;
  searchMode: "caption" | "clip";
};

type SearchResult = {
  id: number;
  path: string;
  fileName: string;
  score: number;
  caption: string;
  captionModel: string;
  embeddingModel: string;
  thumbnailPath: string | null;
};

type SearchResponse = {
  query: string;
  limit: number;
  mode: "caption" | "clip";
  elapsedMs: number;
  results: SearchResult[];
};

type StatusResponse = {
  database: string;
  embedder: string;
  clipEmbedder: string | null;
  indexedImages: number;
  searchableImages: number;
  clipSearchableImages: number;
  memory: {
    currentMb: number;
    peakMb: number;
  };
  uptimeSeconds: number;
};

const DEFAULT_LIMIT = 30;

export default function Command() {
  const preferences = getPreferenceValues<Preferences>();
  const apiBaseUrl = normalizeBaseUrl(preferences.apiBaseUrl);
  const searchMode = preferences.searchMode;
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadStatus() {
      setIsLoading(true);
      setError(null);
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
  }, [apiBaseUrl]);

  useEffect(() => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      setResults([]);
      return;
    }

    const controller = new AbortController();
    const timeout = setTimeout(async () => {
      setIsLoading(true);
      setError(null);
      try {
        const url = new URL(`${apiBaseUrl}/search`);
        url.searchParams.set("q", trimmedQuery);
        url.searchParams.set("limit", String(DEFAULT_LIMIT));
        url.searchParams.set("mode", searchMode);
        const response = await fetchJson<SearchResponse>(
          url.toString(),
          controller.signal,
        );
        setResults(response.results);
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
  }, [apiBaseUrl, query, searchMode]);

  const searchBarPlaceholder = useMemo(() => {
    if (status) {
      const count =
        searchMode === "clip"
          ? status.clipSearchableImages
          : status.searchableImages;
      return `Search ${count} indexed images`;
    }
    return "Search indexed images";
  }, [searchMode, status]);

  if (error) {
    return <ServerError apiBaseUrl={apiBaseUrl} message={error} />;
  }

  return (
    <Grid
      columns={5}
      fit={Grid.Fit.Fill}
      inset={Grid.Inset.Small}
      isLoading={isLoading}
      navigationTitle="Search Images"
      onSearchTextChange={setQuery}
      searchBarPlaceholder={searchBarPlaceholder}
      throttle
    >
      {!query.trim() && status ? (
        <StatusItem
          status={status}
          apiBaseUrl={apiBaseUrl}
          searchMode={searchMode}
        />
      ) : null}
      {results.map((result) => (
        <ResultItem key={result.id} result={result} />
      ))}
    </Grid>
  );
}

function StatusItem({
  status,
  apiBaseUrl,
  searchMode,
}: {
  status: StatusResponse;
  apiBaseUrl: string;
  searchMode: "caption" | "clip";
}) {
  const searchableImages =
    searchMode === "clip"
      ? status.clipSearchableImages
      : status.searchableImages;

  return (
    <Grid.Item
      title="Local Image Search"
      subtitle={`${searchableImages} searchable images · ${status.memory.currentMb.toFixed(0)} MB`}
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

function ResultItem({ result }: { result: SearchResult }) {
  const content = result.thumbnailPath
    ? { source: result.thumbnailPath }
    : { source: Icon.Image, tintColor: Color.SecondaryText };

  return (
    <Grid.Item
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
          </ActionPanel.Section>
          <ActionPanel.Section>
            <Action.Push
              title="Show Details"
              icon={Icon.Text}
              target={<ResultDetail result={result} />}
            />
            <Action.CopyToClipboard title="Copy Path" content={result.path} />
            <Action.CopyToClipboard
              title="Copy Caption"
              content={result.caption}
            />
          </ActionPanel.Section>
        </ActionPanel>
      }
    />
  );
}

function ResultDetail({ result }: { result: SearchResult }) {
  const markdown = [
    result.thumbnailPath
      ? `![${escapeMarkdown(result.fileName)}](${result.thumbnailPath})`
      : "",
    `# ${escapeMarkdown(result.fileName)}`,
    "",
    `**Score:** ${scoreLabel(result.score)}`,
    "",
    `**Path:** \`${result.path}\``,
    "",
    "## Caption",
    "",
    result.caption,
  ]
    .filter(Boolean)
    .join("\n");

  return (
    <Detail
      markdown={markdown}
      metadata={
        <Detail.Metadata>
          <Detail.Metadata.Label title="File" text={result.fileName} />
          <Detail.Metadata.Label
            title="Score"
            text={scoreLabel(result.score)}
          />
          <Detail.Metadata.Label title="Path" text={result.path} />
        </Detail.Metadata>
      }
      actions={
        <ActionPanel>
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
          <Action.CopyToClipboard title="Copy Path" content={result.path} />
          <Action.CopyToClipboard
            title="Copy Caption"
            content={result.caption}
          />
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
    "Start the local API server:",
    "",
    "```bash",
    "cd path/to/local-image-search",
    "source .venv/bin/activate",
    "image-search serve --embedder sentence-transformers",
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

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, "");
}

function scoreLabel(score: number): string {
  return score.toFixed(3);
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function escapeMarkdown(value: string): string {
  return value.replace(/([\\`*_{}[\]()#+\-.!])/g, "\\$1");
}
