import { spawn } from "child_process";
import { homedir } from "os";
import { dirname, join } from "path";

const SERVER_START_TIMEOUT_MS = 20_000;
const SERVER_HEALTH_RETRY_MS = 500;

let serverStartPromise: Promise<void> | null = null;

export async function ensureServerRunning(
  apiBaseUrl: string,
  projectDirectory: string,
): Promise<void> {
  const statusUrl = `${normalizeBaseUrl(apiBaseUrl)}/status`;
  if (await isAvailable(statusUrl)) {
    return;
  }

  if (!serverStartPromise) {
    const command = buildServerCommand(projectDirectory, apiBaseUrl);
    serverStartPromise = startServer(command, statusUrl).finally(() => {
      serverStartPromise = null;
    });
  }

  await serverStartPromise;
}

function buildServerCommand(projectDirectory: string, apiBaseUrl: string): string {
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
    if (await isAvailable(statusUrl)) {
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

async function isAvailable(url: string): Promise<boolean> {
  try {
    const response = await fetch(url);
    return response.ok;
  } catch {
    return false;
  }
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
  const port =
    url.port || (url.protocol === "https:" ? "443" : "80");
  return { host: url.hostname, port };
}
