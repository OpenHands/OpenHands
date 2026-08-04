import { describe, expect, it } from "vitest";
import type { MCPConfig } from "@openhands/typescript-client";
import type { MCPServerConfig } from "#/types/mcp-server";
import {
  buildMcpServerPatch,
  buildRenameMcpConfigPatch,
  MCP_HEADER_REMOVAL_ERROR,
  MCP_RENAME_CREDENTIAL_ERROR,
  parseMcpConfig,
  REDACTED_MCP_SECRET_VALUE,
  toCanonicalMcpServer,
} from "#/utils/mcp-config";
import { flattenMcpConfig } from "#/utils/mcp-installed-servers";

describe("canonical MCP configuration", () => {
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

  it("preserves a disabled server while treating omitted enabled as true", () => {
    const config = parseMcpConfig({
      disabled: { command: "npx", enabled: false },
      enabled: { command: "npx" },
    });

    expect(flattenMcpConfig(config)).toEqual([
      expect.objectContaining({ id: "disabled", enabled: false }),
      expect.objectContaining({ id: "enabled", enabled: undefined }),
    ]);
    expect(toCanonicalMcpServer({
      id: "disabled",
      type: "stdio",
      command: "npx",
      enabled: false,
    })).toMatchObject({ enabled: false });
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

  it("deletes stale auth fields when replacing one strategy with another", () => {
    const storedOAuth = {
      transport: "http" as const,
      url: "https://mail.example/mcp",
      auth: {
        strategy: "oauth2" as const,
        authentication: { type: "oauth" as const, scopes: "mail.read" },
        state: { tokens: { access_token: REDACTED_MCP_SECRET_VALUE } },
      },
    };
    const bearerEdit: MCPServerConfig = {
      id: "mail",
      type: "shttp",
      name: "mail",
      url: storedOAuth.url,
      auth: { strategy: "bearer", value: "replacement-token" },
    };

    expect(buildMcpServerPatch(storedOAuth, bearerEdit).auth).toEqual({
      strategy: "bearer",
      value: "replacement-token",
      authentication: null,
      state: null,
    });

    const storedHeader = {
      transport: "http" as const,
      url: "https://mail.example/mcp",
      auth: {
        strategy: "header" as const,
        headers: { "X-API-Key": REDACTED_MCP_SECRET_VALUE },
      },
    };
    const oauthEdit: MCPServerConfig = {
      id: "mail",
      type: "shttp",
      name: "mail",
      url: storedHeader.url,
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", scopes: "mail.read" },
      },
    };

    expect(buildMcpServerPatch(storedHeader, oauthEdit).auth).toEqual({
      strategy: "oauth2",
      authentication: { type: "oauth", scopes: "mail.read" },
      headers: null,
    });
  });

  // @spec MCP-002 — Secret patches preserve user intent
  it("applies added and changed header-auth headers while preserving redacted leaves", () => {
    const stored = {
      transport: "http" as const,
      url: "https://mail.example/mcp",
      auth: {
        strategy: "header" as const,
        headers: {
          "X-API-Key": REDACTED_MCP_SECRET_VALUE,
          "X-Region": "us-east-1",
        },
      },
    };
    const edited: MCPServerConfig = {
      id: "mail",
      type: "shttp",
      name: "mail",
      url: stored.url,
      auth: {
        strategy: "header",
        headers: {
          "X-API-Key": REDACTED_MCP_SECRET_VALUE,
          "X-Region": "eu-west-1",
          "X-Trace": "on",
        },
      },
    };

    expect(buildMcpServerPatch(stored, edited).auth).toEqual({
      strategy: "header",
      headers: { "X-Region": "eu-west-1", "X-Trace": "on" },
    });
  });

  // @spec MCP-002 — Secret patches preserve user intent
  it("rejects removing an individual header from header auth", () => {
    const stored = {
      transport: "http" as const,
      url: "https://mail.example/mcp",
      auth: {
        strategy: "header" as const,
        headers: {
          "X-API-Key": REDACTED_MCP_SECRET_VALUE,
          "X-Region": "us-east-1",
        },
      },
    };
    const edited: MCPServerConfig = {
      id: "mail",
      type: "shttp",
      name: "mail",
      url: stored.url,
      auth: {
        strategy: "header",
        headers: { "X-Region": "eu-west-1" },
      },
    };

    expect(() => buildMcpServerPatch(stored, edited)).toThrow(
      MCP_HEADER_REMOVAL_ERROR,
    );
  });

  it("patches OAuth metadata and a replacement secret without sending redacted state", () => {
    const stored = {
      transport: "http" as const,
      url: "https://mail.example/mcp",
      auth: {
        strategy: "oauth2" as const,
        authentication: {
          type: "oauth" as const,
          client_auth_method: "client_secret_post" as const,
          scopes: "mail.read",
          client_name: "OpenHands Canvas",
          client_metadata_url: "https://mail.example/oauth/client.json",
          client_id: "old-client",
          client_secret: REDACTED_MCP_SECRET_VALUE,
        },
        state: {
          tokens: {
            access_token: REDACTED_MCP_SECRET_VALUE,
            refresh_token: REDACTED_MCP_SECRET_VALUE,
          },
          token_expires_at: 123,
        },
      },
    };
    const edited: MCPServerConfig = {
      id: "mail",
      type: "shttp",
      name: "mail",
      url: stored.url,
      auth: {
        strategy: "oauth2",
        authentication: {
          type: "oauth",
          client_auth_method: "client_secret_basic",
          scopes: "mail.read mail.send",
          client_id: "new-client",
          client_secret: "replacement-secret",
        },
        state: stored.auth.state,
      },
    };

    const patch = buildMcpServerPatch(stored, edited);

    expect(patch.auth).toEqual({
      strategy: "oauth2",
      authentication: {
        type: "oauth",
        client_auth_method: "client_secret_basic",
        scopes: "mail.read mail.send",
        client_id: "new-client",
        client_secret: "replacement-secret",
      },
    });
    expect(JSON.stringify(patch)).not.toContain(REDACTED_MCP_SECRET_VALUE);
  });

  it("sends nested nulls for explicitly cleared OAuth authentication fields", () => {
    const stored = {
      transport: "http" as const,
      url: "https://mail.example/mcp",
      auth: {
        strategy: "oauth2" as const,
        authentication: {
          type: "oauth" as const,
          client_auth_method: "client_secret_post" as const,
          scopes: "mail.read",
          client_id: "old-client",
          client_secret: REDACTED_MCP_SECRET_VALUE,
        },
        state: {
          tokens: { access_token: REDACTED_MCP_SECRET_VALUE },
        },
      },
    };
    const edited: MCPServerConfig = {
      id: "mail",
      type: "shttp",
      name: "mail",
      url: stored.url,
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth" },
        state: stored.auth.state,
      },
    };

    expect(buildMcpServerPatch(stored, edited).auth).toEqual({
      strategy: "oauth2",
      authentication: {
        type: "oauth",
        client_auth_method: null,
        scopes: null,
        client_id: null,
        client_secret: null,
      },
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
    ).toThrow(MCP_RENAME_CREDENTIAL_ERROR);
  });

  // @spec MCP-003 — Settings map keys are stable MCP identities
  it("preserves stored remote metadata when renaming a server", () => {
    const stored = {
      transport: "http" as const,
      url: "https://catalog.example/mcp",
      description: "Catalog-managed server",
      icon: "catalog-icon",
      headers: { "X-Catalog-Mode": "managed" },
      sse_read_timeout: 5_000,
      keep_alive: true,
    };
    const renamed: MCPServerConfig = {
      id: "catalog",
      type: "shttp",
      name: "renamed-catalog",
      url: "https://catalog.example/v2/mcp",
    };

    expect(
      buildRenameMcpConfigPatch("catalog", "renamed-catalog", stored, renamed),
    ).toEqual({
      catalog: null,
      "renamed-catalog": {
        ...stored,
        url: renamed.url,
      },
    });
  });
});
