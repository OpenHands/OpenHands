import { describe, expect, it } from "vitest";
import type { MCPConfig } from "@openhands/typescript-client";
import type { MCPServerConfig } from "#/types/mcp-server";
import {
  buildMcpServerPatch,
  buildRenameMcpConfigPatch,
  parseMcpConfig,
  REDACTED_MCP_SECRET_VALUE,
  toCanonicalMcpServer,
} from "#/utils/mcp-config";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";

describe("canonical MCP settings", () => {
  // @spec MCP-003 — Settings map keys are stable MCP identities
  it("keeps settings map keys as stable identities across transport grouping", () => {
    const first = parseMcpConfig({
      github: {
        transport: "http",
        url: "https://github.example/mcp",
      },
      filesystem: {
        transport: "stdio",
        command: "npx",
        args: ["-y", "@modelcontextprotocol/server-filesystem"],
      },
    });
    const reordered = parseMcpConfig({
      filesystem: {
        transport: "stdio",
        command: "npx",
        args: ["-y", "@modelcontextprotocol/server-filesystem"],
      },
      github: {
        transport: "http",
        url: "https://github.example/mcp",
      },
    });

    expect(
      flattenMcpConfig(first)
        .map(({ id }) => id)
        .sort(),
    ).toEqual(["filesystem", "github"]);
    expect(
      flattenMcpConfig(reordered)
        .map(({ id }) => id)
        .sort(),
    ).toEqual(["filesystem", "github"]);
  });

  it("normalizes the cloud wrapper while preserving tagged auth and OAuth state", () => {
    expect(
      parseMcpConfig({
        mcpServers: {
          github: {
            url: "https://github.example/mcp",
            transport: "streamable-http",
            auth: {
              strategy: "oauth2",
              authentication: {
                type: "oauth",
                client_auth_method: "client_secret_post",
              },
              state: {
                tokens: { access_token: REDACTED_MCP_SECRET_VALUE },
              },
            },
          },
        },
      }),
    ).toEqual({
      github: {
        transport: "streamable-http",
        url: "https://github.example/mcp",
        auth: {
          strategy: "oauth2",
          authentication: {
            type: "oauth",
            client_auth_method: "client_secret_post",
          },
          state: {
            tokens: { access_token: REDACTED_MCP_SECRET_VALUE },
          },
        },
      },
    });
  });
});

describe("MCP sparse patches", () => {
  const storedRemote: MCPConfig["github"] = {
    transport: "http",
    url: "https://github.example/mcp",
    auth: {
      strategy: "bearer",
      value: REDACTED_MCP_SECRET_VALUE,
    },
  };

  // @spec MCP-002 — Secret patches preserve user intent
  it("omits unchanged redacted auth while updating a non-secret field", () => {
    const edited: MCPServerConfig = {
      id: "github",
      type: "shttp",
      name: "github",
      url: "https://github.example/v2/mcp",
      auth: storedRemote.auth ?? undefined,
    };

    expect(buildMcpServerPatch(storedRemote, edited)).toEqual({
      transport: "http",
      url: "https://github.example/v2/mcp",
    });
  });

  it("replaces auth when the user enters a new credential", () => {
    const edited: MCPServerConfig = {
      id: "github",
      type: "shttp",
      name: "github",
      url: storedRemote.url,
      auth: { strategy: "bearer", value: "github_pat_replacement" },
    };

    expect(buildMcpServerPatch(storedRemote, edited)).toMatchObject({
      auth: { strategy: "bearer", value: "github_pat_replacement" },
    });
  });

  it("clears auth explicitly when the user selects no authentication", () => {
    const edited: MCPServerConfig = {
      id: "github",
      type: "shttp",
      name: "github",
      url: storedRemote.url,
    };

    expect(buildMcpServerPatch(storedRemote, edited)).toMatchObject({
      auth: null,
    });
  });

  it("omits unchanged redacted env leaves and deletes removed env entries", () => {
    const stored = {
      transport: "stdio" as const,
      command: "npx",
      env: {
        API_KEY: REDACTED_MCP_SECRET_VALUE,
        REGION: "us-east-1",
      },
    };
    const edited: MCPServerConfig = {
      id: "worker",
      type: "stdio",
      name: "worker",
      command: "npx",
      env: {
        API_KEY: REDACTED_MCP_SECRET_VALUE,
        REGION: "eu-west-1",
      },
    };

    expect(buildMcpServerPatch(stored, edited)).toMatchObject({
      env: { REGION: "eu-west-1" },
    });

    delete edited.env!.REGION;
    expect(buildMcpServerPatch(stored, edited)).toMatchObject({
      env: { REGION: null },
    });
  });

  // @spec MCP-003 — Settings map keys are stable MCP identities
  it("builds a rename as one map patch and rejects hidden secrets", () => {
    const uncredentialed = {
      transport: "http" as const,
      url: "https://docs.example/mcp",
    };
    const renamed: MCPServerConfig = {
      id: "docs",
      type: "shttp",
      name: "reference",
      url: uncredentialed.url,
    };

    expect(
      buildRenameMcpConfigPatch("docs", "reference", uncredentialed, renamed),
    ).toEqual({
      docs: null,
      reference: toCanonicalMcpServer(renamed),
    });
    expect(() =>
      buildRenameMcpConfigPatch("github", "github-renamed", storedRemote, {
        id: "github",
        type: "shttp",
        name: "github-renamed",
        url: storedRemote.url,
        auth: storedRemote.auth ?? undefined,
      }),
    ).toThrow(/credential/i);
  });
});
