import { describe, expect, it, vi } from "vitest";
import {
  findCatalogEntryForServer,
  findInstalledEntryMatch,
  findInstalledMatch,
  getDefaultMcpConnectionOption,
  getDefaultMcpTransport,
  getInstallableMcpConnectionOption,
  getMarketplaceEntriesByPopularity,
  getMarketplaceEntryById,
  getMcpConnectionOptions,
  getMcpMarketplaceCatalog,
  getMcpOAuthAuthenticationConfig,
  installedServerMatchesQuery,
  marketplaceEntryMatchesQuery,
  urlsMatch,
} from "#/utils/mcp-marketplace-utils";
import { INTEGRATION_CATALOG as MCP_MARKETPLACE } from "@openhands/extensions/integrations";

const mcpMarketplace = getMcpMarketplaceCatalog(MCP_MARKETPLACE);
const slackEntry = mcpMarketplace.find((e) => e.id === "slack")!;
const tavilyEntry = mcpMarketplace.find((e) => e.id === "tavily")!;
const linearEntry = mcpMarketplace.find((e) => e.id === "linear")!;

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

  it("keeps searchable fields separated", () => {
    const entry = {
      ...slackEntry,
      name: "alpha",
      description: "beta",
      id: "gamma",
      keywords: [],
    };

    expect(marketplaceEntryMatchesQuery(entry, "alpha beta")).toBe(true);
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

  it("returns false when no installed-server field matches", () => {
    expect(
      installedServerMatchesQuery(slackServer, undefined, "zzzz-no-match"),
    ).toBe(false);
    expect(
      installedServerMatchesQuery(slackServer, undefined, "Stryker was here"),
    ).toBe(false);
  });

  it("searches HTTP server fields without gaps from missing fields", () => {
    const server = {
      id: "shttp-0",
      type: "shttp" as const,
      url: "https://example.com/mcp",
    };

    expect(
      installedServerMatchesQuery(
        server,
        undefined,
        "shttp https://example.com/mcp",
      ),
    ).toBe(true);
  });

  it.each([
    ["name", "slack"],
    ["command", "npx"],
    ["args", "-y @zencoderai/slack-mcp-server"],
    ["name and command", "slack npx"],
  ])("searches isolated %s values", (_field, query) => {
    expect(installedServerMatchesQuery(slackServer, undefined, query)).toBe(
      true,
    );
  });

  it("handles a runtime stdio server whose args are missing", () => {
    const server = { ...slackServer, args: undefined };

    expect(
      installedServerMatchesQuery(server as never, undefined, "no-match"),
    ).toBe(false);
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

describe("marketplace option normalization", () => {
  it("filters non-MCP and transport-less connection options", () => {
    const entry = {
      ...slackEntry,
      connectionOptions: [
        ...slackEntry.connectionOptions,
        { id: "other", provider: "github", auth: { strategy: "none" } },
        {
          id: "missing-transport",
          provider: "mcp",
          auth: { strategy: "none" },
        },
      ],
    } as typeof slackEntry;
    const options = getMcpConnectionOptions(entry);
    expect(options.length).toBe(slackEntry.connectionOptions.length);
    expect(getDefaultMcpConnectionOption(entry)).toBe(options[0]);
  });

  it("applies option filtering when the module is initialized", async () => {
    vi.resetModules();
    const fresh = await import("#/utils/mcp-marketplace-utils");
    const invalidEntry = {
      ...slackEntry,
      id: "invalid",
      connectionOptions: [
        {
          id: "other",
          provider: "github",
          auth: { strategy: "none" },
          transport: { kind: "shttp", url: "https://example.com/mcp" },
        },
        {
          id: "missing-transport",
          provider: "mcp",
          auth: { strategy: "none" },
        },
      ],
    } as typeof slackEntry;

    expect(fresh.getMcpConnectionOptions(invalidEntry)).toEqual([]);
    expect(fresh.getMcpMarketplaceCatalog([invalidEntry, slackEntry])).toEqual([
      slackEntry,
    ]);
  });

  it("returns no default transport when an entry has no MCP option", () => {
    const entry = { ...slackEntry, connectionOptions: [] };

    expect(getDefaultMcpTransport(entry)).toBeUndefined();
  });

  it.each([
    ["none", "none"],
    ["body", "client_secret_post"],
    ["basic", "client_secret_basic"],
  ] as const)("maps %s OAuth client authentication", (input, expected) => {
    const option = {
      id: "oauth",
      provider: "mcp",
      auth: {
        strategy: "oauth2",
        oauth: { clientAuthentication: input, scopes: ["read", "write"] },
      },
      transport: { kind: "shttp", url: "https://example.com/mcp" },
    } as Parameters<typeof getMcpOAuthAuthenticationConfig>[0];
    expect(getMcpOAuthAuthenticationConfig(option)).toEqual({
      type: "oauth",
      client_auth_method: expected,
      scopes: ["read", "write"],
    });
  });

  it("omits empty or unsupported OAuth metadata and ignores non-OAuth options", () => {
    const base = {
      id: "oauth",
      provider: "mcp",
      transport: { kind: "shttp", url: "https://example.com/mcp" },
    } as const;
    expect(
      getMcpOAuthAuthenticationConfig({
        ...base,
        auth: {
          strategy: "none",
          oauth: { clientAuthentication: "none" },
        },
      } as never),
    ).toBeUndefined();
    expect(
      getMcpOAuthAuthenticationConfig({
        ...base,
        auth: { strategy: "oauth2" },
      } as never),
    ).toBeUndefined();
    expect(
      getMcpOAuthAuthenticationConfig({
        ...base,
        auth: {
          strategy: "oauth2",
          oauth: { clientAuthentication: "unsupported", scopes: [] },
        },
      } as never),
    ).toBeUndefined();
  });

  it("supports client authentication when OAuth scopes are omitted", () => {
    const option = {
      id: "oauth",
      provider: "mcp",
      auth: {
        strategy: "oauth2",
        oauth: { clientAuthentication: "body" },
      },
      transport: { kind: "shttp", url: "https://example.com/mcp" },
    } as Parameters<typeof getMcpOAuthAuthenticationConfig>[0];

    expect(getMcpOAuthAuthenticationConfig(option)).toEqual({
      type: "oauth",
      client_auth_method: "client_secret_post",
    });
  });

  it.each([
    "authorizationUrl",
    "tokenUrl",
    "registrationUrl",
    "additionalAuthorizationParams",
    "additionalTokenParams",
  ] as const)("rejects hosted OAuth metadata: %s", (field) => {
    const option = {
      id: field,
      provider: "mcp",
      auth: {
        strategy: "oauth2",
        oauth: { clientAuthentication: "none", [field]: "configured" },
      },
      transport: { kind: "shttp", url: "https://example.com/mcp" },
    } as never;
    const entry = {
      ...slackEntry,
      connectionOptions: [option],
    } as typeof slackEntry;
    expect(getInstallableMcpConnectionOption(entry)).toBeUndefined();
  });
});

describe("URL and entry matching", () => {
  it("normalizes default ports, queries, and trailing slashes", () => {
    expect(
      urlsMatch(
        "https://example.com:443/mcp/?token=one",
        "https://example.com/mcp?token=two",
      ),
    ).toBe(true);
  });

  it.each([
    [undefined, "https://example.com"],
    ["https://example.com", undefined],
    ["http://example.com/mcp", "https://example.com/mcp"],
    ["https://one.example/mcp", "https://two.example/mcp"],
    ["https://example.com/one", "https://example.com/two"],
    ["not a url/", "not a url"],
    ["not a url", "different"],
  ])("compares defensive URL inputs", (left, right) => {
    const expected = left === "not a url/" && right === "not a url";
    expect(urlsMatch(left, right)).toBe(expected);
  });

  it.each([
    [1, "not-a-url"],
    ["not-a-url", 1],
    [null, "Stryker was here!"],
    ["Stryker was here!", null],
    [undefined, undefined],
    [undefined, "/"],
    ["/", undefined],
    ["not-a-url", "https://example.com"],
    ["https://example.com", "not-a-url"],
  ])("rejects mixed defensive URL values", (left, right) => {
    expect(urlsMatch(left, right)).toBe(false);
  });

  it.each([
    ["alpha/beta", "alphabeta", false],
    ["alphabeta", "alpha/beta", false],
    ["not-a-url///", "not-a-url", true],
    ["not-a-url", "not-a-url///", true],
    ["https://example.com/path///", "https://example.com/path", true],
    ["https://example.com/path", "https://example.com/path///", true],
  ])("normalizes only trailing URL separators", (left, right, expected) => {
    expect(urlsMatch(left, right)).toBe(expected);
  });

  it("matches SSE transports and rejects wrong or missing server URLs", () => {
    const entry = {
      ...linearEntry,
      connectionOptions: [
        {
          id: "sse",
          provider: "mcp",
          auth: { strategy: "none" },
          transport: { kind: "sse", url: "https://example.com/events" },
        },
      ],
    } as typeof linearEntry;
    expect(
      findInstalledEntryMatch(entry, [
        { id: "shttp", type: "shttp", url: "https://example.com/events" },
        { id: "empty", type: "sse", url: "" },
        { id: "sse", type: "sse", url: "https://example.com/events/" },
      ]),
    ).toMatchObject({ id: "sse" });
    expect(findInstalledEntryMatch(entry, [])).toBeNull();
  });

  it("checks later connection options after an earlier option misses", () => {
    const entry = {
      ...linearEntry,
      connectionOptions: [
        {
          id: "missing",
          provider: "mcp",
          auth: { strategy: "none" },
          transport: { kind: "shttp", url: "https://missing.example/mcp" },
        },
        {
          id: "matching",
          provider: "mcp",
          auth: { strategy: "none" },
          transport: { kind: "shttp", url: "https://example.com/mcp" },
        },
      ],
    } as typeof linearEntry;

    expect(
      findInstalledEntryMatch(entry, [
        { id: "match", type: "shttp", url: "https://example.com/mcp" },
      ]),
    ).toMatchObject({ id: "match" });
  });

  it("requires the installed server transport type to match", () => {
    const shttp = {
      kind: "shttp" as const,
      url: "https://example.com/mcp",
    };
    expect(
      findInstalledMatch(shttp, [
        { id: "sse", type: "sse", url: "https://example.com/mcp" },
      ]),
    ).toBeNull();

    const stdio = {
      kind: "stdio" as const,
      serverName: "matching-name",
      command: "npx",
      args: [],
    };
    expect(
      findInstalledMatch(stdio, [
        {
          id: "wrong-type",
          type: "shttp",
          url: "https://example.com/mcp",
          name: "matching-name",
        } as never,
      ]),
    ).toBeNull();
  });
});

describe("marketplace catalog ordering and lookup", () => {
  const entries = [
    { ...slackEntry, id: "first", popularityRank: undefined },
    { ...slackEntry, id: "popular", popularityRank: 10 },
    { ...slackEntry, id: "stable-a", popularityRank: 5 },
    { ...slackEntry, id: "stable-b", popularityRank: 5 },
  ];

  it("orders by popularity while keeping equal ranks stable", () => {
    expect(
      getMarketplaceEntriesByPopularity(entries).map((entry) => entry.id),
    ).toEqual(["popular", "stable-a", "stable-b", "first"]);
    expect(getMarketplaceEntriesByPopularity([entries[1], entries[0]])).toEqual(
      [entries[1], entries[0]],
    );
  });

  it("finds entries by id and returns undefined for misses", () => {
    expect(getMarketplaceEntryById("popular", entries)?.id).toBe("popular");
    expect(getMarketplaceEntryById("missing", entries)).toBeUndefined();
  });

  it("matches queries when keywords are absent", () => {
    const entry = { ...slackEntry, keywords: undefined };
    expect(marketplaceEntryMatchesQuery(entry, entry.id)).toBe(true);
    expect(marketplaceEntryMatchesQuery(entry, "Stryker was here")).toBe(false);
  });
});
