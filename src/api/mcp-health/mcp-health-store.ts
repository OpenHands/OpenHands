import type { McpServerHealth } from "#/types/mcp-health";

type Listener = () => void;

export type McpHealthMap = Record<string, McpServerHealth>;

/**
 * In-memory MCP connection-health store, keyed by
 * `getMcpServerHealthKey(server)`. Mirrors the module-store shape of
 * `#/api/backend-registry/health-store` but is deliberately NOT persisted:
 * a health verdict is only as fresh as its probe, so a page reload resets
 * every server to "unchecked" instead of resurrecting a possibly-stale
 * "healthy".
 */
let healthMap: McpHealthMap = {};
const listeners = new Set<Listener>();
let nextCheckId = 1;

function commit(next: McpHealthMap): void {
  healthMap = next;
  listeners.forEach((listener) => listener());
}

export function getMcpHealthSnapshot(): McpHealthMap {
  return healthMap;
}

export function subscribeMcpHealth(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Mark a probe as in flight and return its id for `resolveMcpHealthCheck`. */
export function beginMcpHealthCheck(key: string): number {
  const checkId = nextCheckId;
  nextCheckId += 1;
  commit({ ...healthMap, [key]: { status: "checking", checkId } });
  return checkId;
}

/**
 * Commit a probe result, but only while the entry is still the `checking`
 * state that probe created. A slow older probe can never overwrite a newer
 * probe's result, and a probe whose server was edited, deleted, or reseeded
 * mid-flight (entry cleared or replaced) is silently dropped — a stale
 * check can never land a false verdict.
 */
export function resolveMcpHealthCheck(
  key: string,
  checkId: number,
  health: McpServerHealth,
): void {
  const current = healthMap[key];
  if (current?.status !== "checking" || current.checkId !== checkId) {
    return;
  }
  commit({ ...healthMap, [key]: health });
}

/** Unconditional write — used to seed health from a pre-save test result. */
export function setMcpServerHealth(key: string, health: McpServerHealth): void {
  commit({ ...healthMap, [key]: health });
}

/** Drop the entry (back to "unchecked") — used when a server changes or is deleted. */
export function clearMcpServerHealth(key: string): void {
  if (!(key in healthMap)) return;
  const { [key]: _removed, ...rest } = healthMap;
  commit(rest);
}

/** Test-only: reset state and listeners. */
export function __resetMcpHealthStoreForTests(): void {
  healthMap = {};
  listeners.clear();
  nextCheckId = 1;
}
