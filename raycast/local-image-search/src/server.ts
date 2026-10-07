import { execFile, spawn } from "child_process";
import { homedir } from "os";
import { dirname, join } from "path";
import { promisify } from "util";

const SERVER_START_TIMEOUT_MS = 20_000;
const SERVER_STOP_TIMEOUT_MS = 5_000;
const REQUIRED_API_VERSION = 4;
const SERVER_HEALTH_RETRY_MS = 500;
const execFileAsync = promisify(execFile);

let serverStartPromise: Promise<void> | null = null;

type ServerStatus = {
  apiVersion?: number;
  database?: string;
  embedder?: string;
};

export async function ensureServerRunning(
  apiBaseUrl: string,
  projectDirectory: string,
): Promise<void> {
  const normalizedBaseUrl = normalizeBaseUrl(apiBaseUrl);
  const statusUrl = `${normalizedBaseUrl}/status`;
  const { port } = apiAddress(normalizedBaseUrl);
  const status = await fetchStatus(statusUrl);

  if (status && !isLocalImageSearchStatus(status)) {
    throw new Error(`${statusUrl} is responding, but it does not look like Local Image Search`);
  }

  if (status && isCompatibleServer(status)) {
    return;
  }

  if (status) {
    await stopServer(port, statusUrl);
  }

  if (!serverStartPromise) {
    const command = buildServerCommand(projectDirectory, apiBaseUrl);
    serverStartPromise = startServer(command, statusUrl).finally(() => {
      serverStartPromise = null;
    });
  }

  await serverStartPromise;
}

function buildServerCommand(
  projectDirectory: string,
  apiBaseUrl: string,
): string {
  const projectDir = expandHome(projectDirectory);
  const executablePath = join(projectDir, ".venv", "bin", "image-search");
  const dbPath = join(projectDir, "data", "images.db");
  const logPath = join(projectDir, "data", "logs", "server.log");
  const { host, port } = apiAddress(apiBaseUrl);

  return [
    "mkdir",
    "-p",
    shellQuote(dirname(logPath)),
    "&&",
    shellQuote(executablePath),
    "--db",
    shellQuote(dbPath),
    "serve",
    "--host",
    shellQuote(host),
    "--port",
    shellQuote(port),
    ">>",
    shellQuote(logPath),
    "2>&1",
  ].join(" ");
}

async function startServer(command: string, statusUrl: string): Promise<void> {
  const child = spawn("/bin/zsh", ["-lc", command], {
    detached: true,
    stdio: "ignore",
  });
  child.unref();

  const startedAt = Date.now();
  while (Date.now() - startedAt < SERVER_START_TIMEOUT_MS) {
    const status = await fetchStatus(statusUrl);
    if (status && isLocalImageSearchStatus(status) && isCompatibleServer(status)) {
      return;
    }
    await sleep(SERVER_HEALTH_RETRY_MS);
  }

  throw new Error(
    `Started server command but ${statusUrl} did not become available within ${
      SERVER_START_TIMEOUT_MS / 1000
    } seconds`,
  );
}

async function fetchStatus(url: string): Promise<ServerStatus | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as ServerStatus;
  } catch {
    return null;
  }
}

async function stopServer(port: string, statusUrl: string): Promise<void> {
  const pids = await listeningPids(port);
  if (pids.length === 0) {
    return;
  }

  await execFileAsync("/bin/kill", pids);
  const startedAt = Date.now();
  while (Date.now() - startedAt < SERVER_STOP_TIMEOUT_MS) {
    if (!(await fetchStatus(statusUrl))) {
      return;
    }
    await sleep(SERVER_HEALTH_RETRY_MS);
  }
  throw new Error("The previous server is still shutting down. Wait for indexing to finish and retry.");
}

async function listeningPids(port: string): Promise<string[]> {
  try {
    const { stdout } = await execFileAsync("/usr/sbin/lsof", [
      `-tiTCP:${port}`,
      "-sTCP:LISTEN",
    ]);
    return stdout
      .split("\n")
      .map((pid) => pid.trim())
      .filter(Boolean);
  } catch {
    return [];
  }
}

function isLocalImageSearchStatus(status: ServerStatus): boolean {
  return typeof status.database === "string" && typeof status.apiVersion === "number";
}

function isCompatibleServer(status: ServerStatus): boolean {
  return status.apiVersion === REQUIRED_API_VERSION &&
    status.embedder === "embeddinggemma-2/text-vision-440m";
}

function expandHome(path: string): string {
  if (path === "~") {
    return homedir();
  }
  if (path.startsWith("~/")) {
    return join(homedir(), path.slice(2));
  }
  return path;
}

function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, milliseconds);
  });
}

function normalizeBaseUrl(value: string): string {
  return value.replace(/\/+$/, "");
}

function apiAddress(apiBaseUrl: string): { host: string; port: string } {
  const url = new URL(normalizeBaseUrl(apiBaseUrl));
  const port = url.port || (url.protocol === "https:" ? "443" : "80");
  return { host: url.hostname, port };
}
