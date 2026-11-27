import { beforeEach, describe, expect, it } from "vitest";
import {
  __resetMcpHealthStoreForTests,
  beginMcpHealthCheck,
  clearMcpServerHealth,
  getMcpHealthSnapshot,
  resolveMcpHealthCheck,
} from "#/api/mcp-health/mcp-health-store";
import type { McpServerHealth } from "#/types/mcp-health";

const HEALTHY: McpServerHealth = {
  status: "healthy",
  verification: "verified",
  toolCount: 3,
  checkedAt: 1,
};

const FAILED: McpServerHealth = {
  status: "failed",
  kind: "connection",
  error: "refused",
  checkedAt: 2,
};

describe("mcp-health-store", () => {
  beforeEach(() => {
    __resetMcpHealthStoreForTests();
  });

  it("marks a check as in flight and commits its result", () => {
    const checkId = beginMcpHealthCheck("key");
    expect(getMcpHealthSnapshot().key).toEqual({ status: "checking", checkId });

    resolveMcpHealthCheck("key", checkId, HEALTHY);

    expect(getMcpHealthSnapshot().key).toEqual(HEALTHY);
  });

  it("drops a stale result that was superseded by a newer check", () => {
    const staleId = beginMcpHealthCheck("key");
    const freshId = beginMcpHealthCheck("key");

    // The slow older probe must not overwrite the newer probe's lifecycle.
    resolveMcpHealthCheck("key", staleId, HEALTHY);
    expect(getMcpHealthSnapshot().key).toEqual({
      status: "checking",
      checkId: freshId,
    });

    resolveMcpHealthCheck("key", freshId, FAILED);
    expect(getMcpHealthSnapshot().key).toEqual(FAILED);
  });

  it("drops a result whose entry was cleared mid-flight", () => {
    // e.g. the server was edited or deleted while its probe was running —
    // the late result must not resurrect a verdict for a gone config.
    const checkId = beginMcpHealthCheck("key");
    clearMcpServerHealth("key");

    resolveMcpHealthCheck("key", checkId, HEALTHY);

    expect(getMcpHealthSnapshot().key).toBeUndefined();
  });
});
