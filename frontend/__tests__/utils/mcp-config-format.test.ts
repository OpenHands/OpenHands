import { describe, it, expect } from "vitest";
import { parseMcpConfig, toSdkMcpConfig } from "#/utils/mcp-config";
import { MCPConfig } from "#/types/settings";

// SDK 1.31.x dropped the ``mcpServers`` wrapper from ``model_dump``, so
// ``GET /api/v1/settings`` now returns a flat server map directly. These tests
// cover both the new wire format (flat) and the legacy wrapped shape, on both
// the parse and serialize paths.

describe("parseMcpConfig — SDK 1.31.x flat wire format", () => {
  it("parses an shttp server from the flat shape", () => {
    const input = {
      shttp: { url: "https://mcp.deepwiki.com/mcp", timeout: 60 },
    };

    const result = parseMcpConfig(input);

    expect(result).toEqual({
      sse_servers: [],
      stdio_servers: [],
      shttp_servers: [
        { name: "shttp", url: "https://mcp.deepwiki.com/mcp", timeout: 60 },
      ],
    });
  });

  it("parses mixed server types from the flat shape", () => {
    const input = {
      "sse-server": { url: "https://sse.example", transport: "sse" },
      "http-server": { url: "https://http.example", transport: "http" },
      "stdio-server": { command: "/usr/bin/cmd" },
    };

    const result = parseMcpConfig(input);

    expect(result.sse_servers).toHaveLength(1);
    expect(result.shttp_servers).toHaveLength(1);
    expect(result.stdio_servers).toHaveLength(1);
  });

  it("treats mcpServers: null as the flat shape (empty server map)", () => {
    // Defensive: a null ``mcpServers`` key on the new flat input shouldn't be
    // confused with the legacy wrapper — it's just a stray key on the flat map.
    const input = { mcpServers: null, real: { url: "https://x", transport: "sse" } };

    const result = parseMcpConfig(input);

    expect(result.sse_servers).toHaveLength(1);
    expect(result.sse_servers[0]).toMatchObject({ name: "real" });
  });
});

describe("parseMcpConfig — legacy wrapped shape still works", () => {
  it("parses an shttp server from the wrapped shape", () => {
    const input = {
      mcpServers: {
        shttp: { url: "https://mcp.deepwiki.com/mcp", timeout: 60 },
      },
    };

    const result = parseMcpConfig(input);

    expect(result.shttp_servers).toHaveLength(1);
    expect(result.shttp_servers[0]).toEqual({
      name: "shttp",
      url: "https://mcp.deepwiki.com/mcp",
      timeout: 60,
    });
  });
});

describe("toSdkMcpConfig — emits flat server map", () => {
  it("returns a flat server map without the mcpServers wrapper", () => {
    const config: MCPConfig = {
      sse_servers: [{ name: "my-custom-name", url: "https://example.com" }],
      stdio_servers: [],
      shttp_servers: [],
    };

    const result = toSdkMcpConfig(config);

    expect(result).not.toHaveProperty("mcpServers");
    expect(result).toEqual({
      "my-custom-name": {
        url: "https://example.com",
        transport: "sse",
      },
    });
  });

  it("round-trips through parseMcpConfig unchanged", () => {
    const config: MCPConfig = {
      sse_servers: [{ name: "sse", url: "https://a.example" }],
      stdio_servers: [{ name: "stdio", command: "/bin/x" }],
      shttp_servers: [
        { name: "shttp", url: "https://b.example", timeout: 90 },
      ],
    };

    const serialized = toSdkMcpConfig(config);
    const reparsed = parseMcpConfig(serialized);

    expect(reparsed).toEqual(config);
  });
});
