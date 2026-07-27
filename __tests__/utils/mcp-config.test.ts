import { describe, expect, it } from "vitest";

import { parseMcpConfig, toSdkMcpConfig } from "#/utils/mcp-config";
import type { MCPConfig } from "#/types/settings";

describe("toSdkMcpConfig", () => {
  it("uses bare base names when there are no collisions across server types", () => {
    // The bug we're guarding against: a shared monotonic counter would
    // emit "sse", "shttp_1", "myname_2" — bumping the stdio suffix every
    // time another server type's count changes. With per-base collision
    // suffixing, unrelated entries keep their bare names.
    const config: MCPConfig = {
      sse_servers: [{ url: "https://sse.example" }],
      shttp_servers: [{ url: "https://shttp.example" }],
      stdio_servers: [{ name: "myname", command: "/bin/run" }],
    };

    const out = toSdkMcpConfig(config);

    expect(out).not.toBeNull();
    expect(Object.keys(out!)).toEqual(["sse", "shttp", "myname"]);
  });

  it("only suffixes when the same base actually collides", () => {
    const config: MCPConfig = {
      sse_servers: [
        { url: "https://a.example" },
        { url: "https://b.example" },
        { url: "https://c.example" },
      ],
      shttp_servers: [
        { url: "https://d.example" },
        { url: "https://e.example" },
      ],
      stdio_servers: [],
    };

    const out = toSdkMcpConfig(config);

    expect(Object.keys(out!)).toEqual([
      "sse",
      "sse_1",
      "sse_2",
      "shttp",
      "shttp_1",
    ]);
  });

  it("preserves stdio names verbatim when distinct, even with sse/shttp present", () => {
    // Adding sse/shttp servers must not rename existing stdio entries.
    // This is the exact scenario the user reported: numbers appearing
    // on their stdio MCP server names when they edit unrelated entries.
    const config: MCPConfig = {
      sse_servers: [{ url: "https://x" }, { url: "https://y" }],
      shttp_servers: [{ url: "https://z" }],
      stdio_servers: [
        { name: "github", command: "/bin/gh" },
        { name: "filesystem", command: "/bin/fs" },
      ],
    };

    const out = toSdkMcpConfig(config);

    expect(out!).toMatchObject({
      sse: { url: "https://x" },
      sse_1: { url: "https://y" },
      shttp: { url: "https://z" },
      github: { command: "/bin/gh" },
      filesystem: { command: "/bin/fs" },
    });
  });

  it("suffixes only colliding stdio names", () => {
    const config: MCPConfig = {
      sse_servers: [],
      shttp_servers: [],
      stdio_servers: [
        { name: "tool", command: "/bin/a" },
        { name: "tool", command: "/bin/b" },
        { name: "other", command: "/bin/c" },
      ],
    };

    const out = toSdkMcpConfig(config);

    expect(Object.keys(out!)).toEqual(["tool", "tool_1", "other"]);
  });

  it("falls back to a 'stdio' base when a stdio entry has no name", () => {
    const config: MCPConfig = {
      sse_servers: [],
      shttp_servers: [],
      stdio_servers: [
        { name: "", command: "/bin/a" },
        { name: "", command: "/bin/b" },
      ],
    };

    const out = toSdkMcpConfig(config);

    expect(Object.keys(out!)).toEqual(["stdio", "stdio_1"]);
  });

  it("uses a user-given name as the sse/shttp dict key", () => {
    const config: MCPConfig = {
      sse_servers: [{ name: "my_search", url: "https://sse.example" }],
      shttp_servers: [{ name: "my_docs", url: "https://shttp.example" }],
      stdio_servers: [],
    };

    const out = toSdkMcpConfig(config);

    expect(Object.keys(out!)).toEqual(["my_search", "my_docs"]);
    expect(out!.my_search).toMatchObject({
      url: "https://sse.example",
      transport: "sse",
    });
  });

  it("preserves valid hyphenated MCP server names as SDK keys", () => {
    const config: MCPConfig = {
      sse_servers: [
        { name: "integrations-hub", url: "https://hub.example/mcp" },
      ],
      shttp_servers: [],
      stdio_servers: [{ name: "docs-server", command: "npx" }],
    };

    const out = toSdkMcpConfig(config);

    expect(Object.keys(out!)).toEqual([
      "integrations-hub",
      "docs-server",
    ]);
  });

  it("falls back to the base name for unnamed sse/shttp entries", () => {
    const config: MCPConfig = {
      sse_servers: [{ name: "named", url: "https://a" }, { url: "https://b" }],
      shttp_servers: [{ url: "https://c" }],
      stdio_servers: [],
    };

    const out = toSdkMcpConfig(config);

    expect(Object.keys(out!)).toEqual(["named", "sse", "shttp"]);
  });

  it("de-dups colliding user-given sse/shttp names with a suffix", () => {
    const config: MCPConfig = {
      sse_servers: [
        { name: "search", url: "https://a" },
        { name: "search", url: "https://b" },
      ],
      shttp_servers: [],
      stdio_servers: [],
    };

    const out = toSdkMcpConfig(config);

    expect(Object.keys(out!)).toEqual(["search", "search_1"]);
  });

  it("round-trips a user-given sse/shttp name through parse → write", () => {
    const persisted = {
      my_search: { url: "https://x", transport: "sse" },
      my_docs: { url: "https://y" },
    };

    const parsed = parseMcpConfig(persisted);

    expect(parsed.sse_servers).toEqual([
      { name: "my_search", url: "https://x" },
    ]);
    expect(parsed.shttp_servers).toEqual([
      { name: "my_docs", url: "https://y" },
    ]);

    const written = toSdkMcpConfig(parsed);
    expect(Object.keys(written!).sort()).toEqual(["my_docs", "my_search"]);
  });

  it("parses cloud SDK MCPConfig wrapper shape", () => {
    const persisted = {
      mcpServers: {
        "cloud-weather": {
          url: "https://weather.example/mcp",
          transport: "http",
        },
        "cloud-files": {
          command: "uvx",
          args: ["mcp-server-files"],
        },
      },
    };

    const parsed = parseMcpConfig(persisted);

    expect(parsed.shttp_servers).toEqual([
      { name: "cloud-weather", url: "https://weather.example/mcp" },
    ]);
    expect(parsed.stdio_servers).toEqual([
      {
        name: "cloud-files",
        command: "uvx",
        args: ["mcp-server-files"],
      },
    ]);
  });

  it("does not unwrap a valid server named mcpServers", () => {
    const parsed = parseMcpConfig({
      mcpServers: {
        url: "https://meta.example/mcp",
      },
    });

    expect(parsed.shttp_servers).toEqual([
      { name: "mcpServers", url: "https://meta.example/mcp" },
    ]);
  });

  it("does not surface auto-generated sse/shttp keys as user names", () => {
    // Keys matching the fallback pattern carry no user intent, so parsing
    // must leave `name` unset — otherwise the auto key would become a
    // sticky, user-facing name on the next edit.
    const persisted = {
      sse: { url: "https://a", transport: "sse" },
      sse_1: { url: "https://b", transport: "sse" },
      shttp: { url: "https://c" },
      shttp_2: { url: "https://d" },
    };

    const parsed = parseMcpConfig(persisted);

    expect(parsed.sse_servers).toEqual([
      { url: "https://a" },
      { url: "https://b" },
    ]);
    expect(parsed.shttp_servers).toEqual([
      { url: "https://c" },
      { url: "https://d" },
    ]);
  });

  it("returns null when there are no servers", () => {
    expect(
      toSdkMcpConfig({ sse_servers: [], shttp_servers: [], stdio_servers: [] }),
    ).toBeNull();
  });

  it("serializes remote API keys through the FastMCP auth field", () => {
    const config: MCPConfig = {
      sse_servers: [
        {
          url: "https://sse.example",
          auth: { strategy: "bearer", value: "sse-secret" },
        },
      ],
      shttp_servers: [
        {
          url: "https://shttp.example",
          auth: { strategy: "bearer", value: "shttp-secret" },
        },
      ],
      stdio_servers: [],
    };

    const out = toSdkMcpConfig(config);

    expect(out).toEqual({
      sse: {
        url: "https://sse.example",
        transport: "sse",
        auth: { strategy: "bearer", value: "sse-secret" },
      },
      shttp: {
        url: "https://shttp.example",
        transport: "http",
        auth: { strategy: "bearer", value: "shttp-secret" },
      },
    });
  });

  it("keeps names stable across a parse → write round trip", () => {
    // Simulates loading the user's persisted settings, parsing them,
    // and re-serializing on save (which is what happens on every edit).
    // The keys must not drift between trips.
    const persisted = {
      sse: { url: "https://x", transport: "sse" },
      sse_1: { url: "https://y", transport: "sse" },
      shttp: { url: "https://z" },
      github: { command: "/bin/gh" },
    };

    const parsed = parseMcpConfig(persisted);
    const written = toSdkMcpConfig(parsed);

    expect(written).not.toBeNull();
    expect(Object.keys(written!).sort()).toEqual(Object.keys(persisted).sort());
  });

  it("does not bump the suffix on a stdio name when an sse server is added", () => {
    // Concretely demonstrates the user's report: editing/adding an sse
    // server must leave the stdio name untouched. The previous shared
    // counter implementation would rename "myname" → "myname_2" here.
    const before: MCPConfig = {
      sse_servers: [{ url: "https://a" }],
      shttp_servers: [],
      stdio_servers: [{ name: "myname", command: "/bin/run" }],
    };
    const after: MCPConfig = {
      sse_servers: [{ url: "https://a" }, { url: "https://b" }],
      shttp_servers: [],
      stdio_servers: [{ name: "myname", command: "/bin/run" }],
    };

    const out1 = toSdkMcpConfig(before)!;
    const out2 = toSdkMcpConfig(after)!;

    expect("myname" in out1).toBe(true);
    expect("myname" in out2).toBe(true);
  });
});

describe("parseMcpConfig / toSdkMcpConfig — auth: oauth round-trip", () => {
  it("round-trips auth metadata and state for remote OAuth servers", () => {
    const persisted = {
      superhuman_mail: {
        url: "https://mcp.mail.superhuman.com/mcp",
        transport: "http",
        auth: {
          strategy: "oauth2",
          authentication: {
            type: "oauth",
            client_auth_method: "none",
          },
          state: {
            tokens: { access_token: "gAAAAencrypted-access-token" },
            token_expires_at: 12345,
          },
        },
      },
    };

    const roundTripped = toSdkMcpConfig(parseMcpConfig(persisted));

    expect(roundTripped).toEqual({
      superhuman_mail: {
        url: "https://mcp.mail.superhuman.com/mcp",
        transport: "http",
        auth: {
          strategy: "oauth2",
          authentication: {
            type: "oauth",
            client_auth_method: "none",
          },
          state: {
            tokens: { access_token: "gAAAAencrypted-access-token" },
            token_expires_at: 12345,
          },
        },
      },
    });
  });

  it("keeps private_key_jwt OAuth client authentication metadata", () => {
    const persisted = {
      oauth: {
        url: "https://mcp.example.com/mcp",
        transport: "http",
        auth: {
          strategy: "oauth2",
          authentication: {
            type: "oauth",
            client_auth_method: "private_key_jwt",
          },
        },
      },
    };

    expect(toSdkMcpConfig(parseMcpConfig(persisted))).toEqual({
      oauth: {
        url: "https://mcp.example.com/mcp",
        transport: "http",
        auth: {
          strategy: "oauth2",
          authentication: {
            type: "oauth",
            client_auth_method: "private_key_jwt",
          },
        },
      },
    });
  });
});
