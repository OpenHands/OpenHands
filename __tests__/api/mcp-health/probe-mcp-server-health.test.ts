import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetMcpHealthStoreForTests,
  getMcpHealthSnapshot,
  setMcpServerHealth,
} from "#/api/mcp-health/mcp-health-store";
import {
  interpretMcpTestResponse,
  probeMcpServerHealth,
  seedMcpServerHealth,
} from "#/api/mcp-health/probe-mcp-server-health";
import McpService from "#/api/mcp-service/mcp-service.api";
import type { ExtendedMCPTestResponse, MCPServerConfig } from "#/types/mcp-server";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";

/** Matches the catalog `github` entry, so the `get_me` probe spec applies. */
const GITHUB: MCPServerConfig = {
  id: "shttp-0",
  type: "shttp",
  name: "github",
  url: "https://api.githubcopilot.com/mcp/",
  auth: { strategy: "api_key", value: "github_pat_x" },
};

/** Matches no catalog entry, so no probe spec exists. */
const CUSTOM: MCPServerConfig = {
  id: "shttp-1",
  type: "shttp",
  name: "custom",
  url: "https://mcp.example.com/mcp",
};

describe("interpretMcpTestResponse", () => {
  it("passes a non-auth connection failure through with its kind", () => {
    const health = interpretMcpTestResponse(CUSTOM, {
      ok: false,
      error: "connection refused",
      error_kind: "connection",
    });

    expect(health).toMatchObject({
      status: "failed",
      kind: "connection",
      error: "connection refused",
    });
  });

  it("reclassifies a connection failure with auth-rejection text as a credentials failure", () => {
    // Hosted servers reject bad tokens at the HTTP handshake, which the
    // backend can only report as a connection failure.
    const health = interpretMcpTestResponse(GITHUB, {
      ok: false,
      error: "Client error '401 Unauthorized' for url …",
      error_kind: "connection",
    });

    expect(health).toMatchObject({ status: "failed", kind: "credentials" });
  });

  it("reports verified health when the read-only probe tool ran cleanly", () => {
    const health = interpretMcpTestResponse(GITHUB, {
      ok: true,
      tools: ["get_me", "search_code"],
      tool_result: { is_error: false, text: '{"login":"octocat"}' },
    });

    expect(health).toMatchObject({
      status: "healthy",
      verification: "verified",
      toolCount: 2,
    });
  });

  it("downgrades to connectivity-only when the probe tool is not advertised", () => {
    const health = interpretMcpTestResponse(GITHUB, {
      ok: true,
      tools: ["search_code"],
      tool_result: {
        is_error: true,
        text: "Tool 'get_me' not advertised by server",
      },
    });

    expect(health).toMatchObject({
      status: "healthy",
      verification: "connectivity-only",
    });
  });

  it("reports connectivity-only for servers without a probe spec", () => {
    const health = interpretMcpTestResponse(CUSTOM, {
      ok: true,
      tools: ["a"],
    });

    expect(health).toMatchObject({
      status: "healthy",
      verification: "connectivity-only",
      toolCount: 1,
    });
  });
});

describe("probeMcpServerHealth", () => {
  beforeEach(() => {
    __resetMcpHealthStoreForTests();
    vi.restoreAllMocks();
  });

  it("publishes checking while the probe runs, then the interpreted result", async () => {
    let resolveProbe!: (value: ExtendedMCPTestResponse) => void;
    vi.spyOn(McpService, "testServer").mockReturnValue(
      new Promise((resolve) => {
        resolveProbe = resolve;
      }),
    );
    const key = getMcpServerHealthKey(CUSTOM);

    const probe = probeMcpServerHealth(CUSTOM);
    expect(getMcpHealthSnapshot()[key]).toMatchObject({ status: "checking" });

    resolveProbe({ ok: true, tools: ["a", "b"] });
    await probe;

    expect(getMcpHealthSnapshot()[key]).toMatchObject({
      status: "healthy",
      verification: "connectivity-only",
      toolCount: 2,
    });
  });

  it("converts a thrown transport error into a failed verdict", async () => {
    vi.spyOn(McpService, "testServer").mockRejectedValue(
      new Error("network down"),
    );

    await probeMcpServerHealth(CUSTOM);

    expect(getMcpHealthSnapshot()[getMcpServerHealthKey(CUSTOM)]).toMatchObject(
      { status: "failed", kind: "unknown", error: "network down" },
    );
  });
});

describe("seedMcpServerHealth", () => {
  beforeEach(() => {
    __resetMcpHealthStoreForTests();
  });

  it("publishes the saved server's health from its pre-save test result", () => {
    seedMcpServerHealth(CUSTOM, { ok: true, tools: ["a"] }, []);

    expect(getMcpHealthSnapshot()[getMcpServerHealthKey(CUSTOM)]).toMatchObject(
      { status: "healthy" },
    );
  });

  it("skips seeding when another installed server shares the health key", () => {
    // A second install of the same catalog entry collides on the key until
    // the save suffixes its name; it must not overwrite the first card.
    const existingVerdict = {
      status: "failed",
      kind: "credentials",
      error: "bad token",
      checkedAt: 1,
    } as const;
    setMcpServerHealth(getMcpServerHealthKey(GITHUB), existingVerdict);

    seedMcpServerHealth(
      { ...GITHUB, id: "shttp-9" },
      { ok: true, tools: ["get_me"] },
      [GITHUB],
    );

    expect(getMcpHealthSnapshot()[getMcpServerHealthKey(GITHUB)]).toEqual(
      existingVerdict,
    );
  });
});
