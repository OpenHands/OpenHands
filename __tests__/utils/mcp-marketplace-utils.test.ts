import { describe, expect, it } from "vitest";
import {
  findCatalogEntryForServer,
  findInstalledMatch,
  getDefaultMcpTransport,
  getInstallableMcpConnectionOption,
  getMcpMarketplaceCatalog,
  installedServerMatchesQuery,
  marketplaceEntryMatchesQuery,
} from "#/utils/mcp-marketplace-utils";
import { INTEGRATION_CATALOG as MCP_MARKETPLACE } from "@openhands/extensions/integrations";

const mcpMarketplace = getMcpMarketplaceCatalog(MCP_MARKETPLACE);
const slackEntry = mcpMarketplace.find((e) => e.id === "slack")!;
const tavilyEntry = mcpMarketplace.find((e) => e.id === "tavily")!;
const linearEntry = mcpMarketplace.find((e) => e.id === "linear")!;
const filesystemEntry = mcpMarketplace.find((e) => e.id === "filesystem")!;

function optionTransport(entry: typeof slackEntry, optionId = "api") {
  const transport = entry.connectionOptions.find(
    (option) => option.id === optionId,
  )?.transport;
  if (!transport) throw new Error(`Missing ${optionId} transport`);
  return transport;
}

describe("findInstalledMatch", () => {
  it("matches stdio servers by name", () => {
    const result = findInstalledMatch(optionTransport(slackEntry), [
      {
        id: "stdio-0",
        type: "stdio",
        name: "slack",
        command: "npx",
        args: ["-y", "@zencoderai/slack-mcp-server"],
      },
    ]);
    expect(result).toEqual(expect.objectContaining({ id: "stdio-0" }));
  });

  it("does not match a different stdio name", () => {
    const result = findInstalledMatch(optionTransport(slackEntry), [
      {
        id: "stdio-0",
        type: "stdio",
        name: "github",
        command: "npx",
        args: [],
      },
    ]);
    expect(result).toBeNull();
  });

  it("matches Tavily as a stdio server by name", () => {
    // Tavily lives in the catalog as a stdio MCP entry (the previous
    // tavily-builtin / search_api_key flow never persisted anywhere
    // and silently dropped the key); confirm the now-uniform match.
    const result = findInstalledMatch(getDefaultMcpTransport(tavilyEntry)!, [
      {
        id: "stdio-0",
        type: "stdio",
        name: "tavily",
        command: "npx",
        args: ["-y", "tavily-mcp"],
        env: { TAVILY_API_KEY: "tvly-secret" },
      },
    ]);
    expect(result).toEqual(expect.objectContaining({ id: "stdio-0" }));
  });

  it("matches HTTP servers loosely on URL", () => {
    const result = findInstalledMatch(getDefaultMcpTransport(linearEntry)!, [
      {
        id: "shttp-0",
        type: "shttp",
        url: "https://mcp.linear.app/mcp/",
      },
    ]);
    expect(result).toEqual(expect.objectContaining({ id: "shttp-0" }));
  });

  it("returns null when servers carry malformed urls (defensive)", () => {
    const result = findInstalledMatch(getDefaultMcpTransport(linearEntry)!, [
      // Cast to any to simulate runtime data slipping past the type.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      { id: "shttp-0", type: "shttp", url: undefined as any },
    ]);
    expect(result).toBeNull();
  });
});

describe("getInstallableMcpConnectionOption", () => {
  it("prefers Slack's API fallback over the default OAuth option", () => {
    const option = getInstallableMcpConnectionOption(slackEntry);
    expect(option?.id).toBe("api");
    expect(option?.auth.strategy).toBe("api_key");
    expect(option?.transport.kind).toBe("stdio");
  });

  it("returns undefined for provider OAuth entries without a local MCP auth contract", () => {
    const oauthOnlyEntry: Parameters<
      typeof getInstallableMcpConnectionOption
    >[0] = {
      ...slackEntry,
      id: "oauth-only",
      connectionOptions: [
        {
          id: "oauth",
          provider: "mcp",
          auth: { strategy: "oauth2" },
          transport: { kind: "shttp", url: "https://example.com/mcp" },
        } as Parameters<
          typeof getInstallableMcpConnectionOption
        >[0]["connectionOptions"][number],
      ],
    };
    const option = getInstallableMcpConnectionOption(oauthOnlyEntry);
    expect(option).toBeUndefined();
  });

  it("returns MCP-server-managed OAuth options", () => {
    const oauthOnlyEntry: Parameters<
      typeof getInstallableMcpConnectionOption
    >[0] = {
      ...slackEntry,
      id: "oauth-only",
      connectionOptions: [
        {
          id: "oauth",
          provider: "mcp",
          auth: {
            strategy: "oauth2",
            oauth: { clientAuthentication: "none" },
          },
          transport: { kind: "shttp", url: "https://example.com/mcp" },
        } as Parameters<
          typeof getInstallableMcpConnectionOption
        >[0]["connectionOptions"][number],
      ],
    };
    const option = getInstallableMcpConnectionOption(oauthOnlyEntry);
    expect(option).toBeDefined();
    expect(option?.auth.strategy).toBe("oauth2");
    expect(option?.transport.kind).toBe("shttp");
  });

  it("returns undefined when the entry has no MCP connection options", () => {
    const noOptionsEntry: Parameters<
      typeof getInstallableMcpConnectionOption
    >[0] = {
      ...slackEntry,
      id: "no-mcp",
      connectionOptions: [],
    };
    const option = getInstallableMcpConnectionOption(noOptionsEntry);
    expect(option).toBeUndefined();
  });
});

describe("marketplaceEntryMatchesQuery", () => {
  it("matches by name (case-insensitive)", () => {
    expect(marketplaceEntryMatchesQuery(slackEntry, "slack")).toBe(true);
    expect(marketplaceEntryMatchesQuery(slackEntry, "SLACK")).toBe(true);
  });

  it("matches by keyword", () => {
    expect(marketplaceEntryMatchesQuery(slackEntry, "messaging")).toBe(true);
  });

  it("matches by substring of description", () => {
    expect(marketplaceEntryMatchesQuery(tavilyEntry, "web search")).toBe(true);
  });

  it("returns true for empty/whitespace queries", () => {
    expect(marketplaceEntryMatchesQuery(slackEntry, "")).toBe(true);
    expect(marketplaceEntryMatchesQuery(slackEntry, "   ")).toBe(true);
  });

  it("returns false for non-matches", () => {
    expect(marketplaceEntryMatchesQuery(slackEntry, "zzzz-no-match")).toBe(
      false,
    );
  });
});

describe("installedServerMatchesQuery", () => {
  const slackServer = {
    id: "stdio-0",
    type: "stdio" as const,
    name: "slack",
    command: "npx",
    args: ["-y", "@zencoderai/slack-mcp-server"],
  };

  it("matches by stdio server name", () => {
    expect(installedServerMatchesQuery(slackServer, undefined, "slack")).toBe(
      true,
    );
  });

  it("matches via the catalog entry's name even if server.name differs", () => {
    const renamed = { ...slackServer, name: "my-slack-instance" };
    expect(installedServerMatchesQuery(renamed, slackEntry, "slack")).toBe(
      true,
    );
  });

  it("matches by url for shttp/sse servers", () => {
    const sseServer = {
      id: "sse-0",
      type: "sse" as const,
      url: "https://mcp.linear.app/sse",
    };
    expect(installedServerMatchesQuery(sseServer, undefined, "linear")).toBe(
      true,
    );
  });

  it("empty query always matches", () => {
    expect(installedServerMatchesQuery(slackServer, undefined, "")).toBe(true);
  });
});

describe("findCatalogEntryForServer", () => {
  it("finds the Slack catalog entry for an installed Slack stdio server", () => {
    const match = findCatalogEntryForServer(
      {
        id: "stdio-0",
        type: "stdio",
        name: "slack",
        command: "npx",
        args: [],
      },
      mcpMarketplace,
    );
    expect(match?.id).toBe("slack");
  });

  it("returns undefined for unknown servers", () => {
    expect(
      findCatalogEntryForServer(
        {
          id: "stdio-0",
          type: "stdio",
          name: "unknown",
          command: "npx",
          args: [],
        },
        mcpMarketplace,
      ),
    ).toBeUndefined();
  });

  it("matches an HTTP server whose URL differs only by trailing slash", () => {
    // Regression coverage for the strict-=== URL match that previously
    // diverged from findInstalledMatch and caused installed cards to
    // render the generic icon while the marketplace tile said
    // "Installed".
    const linear = mcpMarketplace.find((e) => e.id === "linear")!;
    const linearTransport = getDefaultMcpTransport(linear);
    if (linearTransport?.kind !== "shttp") {
      throw new Error("Linear template should be shttp");
    }
    const normalizedUrl = linearTransport.url.replace(/\/$/, "");
    const match = findCatalogEntryForServer(
      { id: "shttp-0", type: "shttp", url: `${normalizedUrl}/` },
      mcpMarketplace,
    );
    expect(match?.id).toBe("linear");
  });
});

describe("GitHub hosted MCP entry", () => {
  function getGitHubTransport(
    catalog: ReturnType<typeof getMcpMarketplaceCatalog>,
  ) {
    const github = catalog.find((e) => e.id === "github");
    expect(github).toBeDefined();
    const transport = getDefaultMcpTransport(github!);
    expect(transport?.kind).toBe("shttp");
    if (transport?.kind !== "shttp") throw new Error("expected shttp");
    return transport;
  }

  it("uses GitHub's hosted streamable HTTP endpoint", () => {
    const transport = getGitHubTransport(
      getMcpMarketplaceCatalog(MCP_MARKETPLACE),
    );
    expect(transport.url).toBe("https://api.githubcopilot.com/mcp/");
  });

  it("matches installed hosted GitHub servers by URL", () => {
    const github = getMcpMarketplaceCatalog(MCP_MARKETPLACE).find(
      (e) => e.id === "github",
    )!;
    const match = findCatalogEntryForServer(
      {
        id: "shttp-0",
        type: "shttp",
        url: "https://api.githubcopilot.com/mcp/",
      },
      [github],
    );
    expect(match?.id).toBe("github");
  });
});

describe("local Appwrite MCP catalog entry", () => {
  it("exposes the uvx stdio Appwrite server in the marketplace", () => {
    // @spec — Heimdall local catalog overlay for Appwrite MCP
    const catalog = getMcpMarketplaceCatalog(MCP_MARKETPLACE);
    const appwrite = catalog.find((entry) => entry.id === "appwrite");
    expect(appwrite).toBeDefined();
    expect(appwrite?.name).toBe("Appwrite");
    expect(appwrite?.docsUrl).toBe(
      "https://appwrite.io/docs/tooling/ai/mcp-servers",
    );

    const transport = getDefaultMcpTransport(appwrite!);
    expect(transport?.kind).toBe("stdio");
    if (transport?.kind !== "stdio") {
      throw new Error("Appwrite transport should be stdio");
    }
    expect(transport.serverName).toBe("appwrite");
    expect(transport.command).toBe("uvx");
    expect(transport.args).toEqual(["mcp-server-appwrite"]);
    expect(transport.envFields?.map((field) => field.key)).toEqual([
      "APPWRITE_API_KEY",
      "APPWRITE_PROJECT_ID",
      "APPWRITE_ENDPOINT",
    ]);
  });

  it("matches installed Appwrite stdio servers by name", () => {
    const appwrite = getMcpMarketplaceCatalog(MCP_MARKETPLACE).find(
      (entry) => entry.id === "appwrite",
    )!;
    const match = findInstalledMatch(getDefaultMcpTransport(appwrite)!, [
      {
        id: "stdio-0",
        type: "stdio",
        name: "appwrite",
        command: "uvx",
        args: ["mcp-server-appwrite"],
        env: {
          APPWRITE_API_KEY: "secret",
          APPWRITE_PROJECT_ID: "proj",
          APPWRITE_ENDPOINT: "https://cloud.appwrite.io/v1",
        },
      },
    ]);
    expect(match).toEqual(expect.objectContaining({ id: "stdio-0" }));
  });
});

describe("local Plane MCP catalog entry", () => {
  it("exposes the uvx stdio Plane server in the marketplace", () => {
    // @spec — Heimdall local catalog overlay for Plane MCP
    const catalog = getMcpMarketplaceCatalog(MCP_MARKETPLACE);
    const plane = catalog.find((entry) => entry.id === "plane");
    expect(plane).toBeDefined();
    expect(plane?.name).toBe("Plane");
    expect(plane?.docsUrl).toBe(
      "https://developers.plane.so/api-reference/introduction",
    );

    const transport = getDefaultMcpTransport(plane!);
    expect(transport?.kind).toBe("stdio");
    if (transport?.kind !== "stdio") {
      throw new Error("Plane transport should be stdio");
    }
    expect(transport.serverName).toBe("plane");
    expect(transport.command).toBe("uvx");
    expect(transport.args).toEqual(["plane-mcp-server", "stdio"]);
    expect(transport.envFields?.map((field) => field.key)).toEqual([
      "PLANE_API_KEY",
      "PLANE_WORKSPACE_SLUG",
      "PLANE_BASE_URL",
    ]);
  });

  it("matches installed Plane stdio servers by name", () => {
    const plane = getMcpMarketplaceCatalog(MCP_MARKETPLACE).find(
      (entry) => entry.id === "plane",
    )!;
    const match = findInstalledMatch(getDefaultMcpTransport(plane)!, [
      {
        id: "stdio-plane",
        type: "stdio",
        name: "plane",
        command: "uvx",
        args: ["plane-mcp-server", "stdio"],
        env: {
          PLANE_API_KEY: "secret",
          PLANE_WORKSPACE_SLUG: "heimdall",
          PLANE_BASE_URL: "https://plane.heimdallsec.com.br",
        },
      },
    ]);
    expect(match).toEqual(expect.objectContaining({ id: "stdio-plane" }));
  });
});
