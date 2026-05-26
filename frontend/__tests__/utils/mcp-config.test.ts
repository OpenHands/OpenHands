import { describe, it, expect } from "vitest";
import { parseMcpConfig, toSdkMcpConfig } from "#/utils/mcp-config";
import { MCPConfig } from "#/types/settings";

describe("parseMcpConfig", () => {
  it("should return empty config for null/undefined input", () => {
    expect(parseMcpConfig(null)).toEqual({
      sse_servers: [],
      stdio_servers: [],
      shttp_servers: [],
    });
    expect(parseMcpConfig(undefined)).toEqual({
      sse_servers: [],
      stdio_servers: [],
      shttp_servers: [],
    });
  });

  it("should return empty config for invalid input", () => {
    expect(parseMcpConfig("string")).toEqual({
      sse_servers: [],
      stdio_servers: [],
      shttp_servers: [],
    });
    expect(parseMcpConfig({})).toEqual({
      sse_servers: [],
      stdio_servers: [],
      shttp_servers: [],
    });
    expect(parseMcpConfig({ mcpServers: null })).toEqual({
      sse_servers: [],
      stdio_servers: [],
      shttp_servers: [],
    });
  });

  it("should preserve server names for SSE servers", () => {
    const input = {
      mcpServers: {
        "my-custom-name": {
          url: "https://example.com",
          transport: "sse",
        },
      },
    };

    const result = parseMcpConfig(input);

    expect(result.sse_servers).toHaveLength(1);
    expect(result.sse_servers[0]).toEqual({
      name: "my-custom-name",
      url: "https://example.com",
    });
  });

  it("should preserve server names for shttp servers", () => {
    const input = {
      mcpServers: {
        "http-server": {
          url: "https://example.com",
          transport: "http",
        },
      },
    };

    const result = parseMcpConfig(input);

    expect(result.shttp_servers).toHaveLength(1);
    expect(result.shttp_servers[0]).toEqual({
      name: "http-server",
      url: "https://example.com",
    });
  });

  it("should preserve server names for stdio servers", () => {
    const input = {
      mcpServers: {
        "my-stdio-server": {
          command: "/usr/bin/my-server",
          args: ["--port", "8080"],
        },
      },
    };

    const result = parseMcpConfig(input);

    expect(result.stdio_servers).toHaveLength(1);
    expect(result.stdio_servers[0]).toEqual({
      name: "my-stdio-server",
      command: "/usr/bin/my-server",
      args: ["--port", "8080"],
    });
  });

  it("should parse api_key from auth field", () => {
    const input = {
      mcpServers: {
        "auth-server": {
          url: "https://example.com",
          transport: "sse",
          auth: "my-secret-key",
        },
      },
    };

    const result = parseMcpConfig(input);

    expect(result.sse_servers[0]).toEqual({
      name: "auth-server",
      url: "https://example.com",
      api_key: "my-secret-key",
    });
  });

  it("should not include api_key when auth is 'oauth'", () => {
    const input = {
      mcpServers: {
        "oauth-server": {
          url: "https://example.com",
          transport: "sse",
          auth: "oauth",
        },
      },
    };

    const result = parseMcpConfig(input);

    expect(result.sse_servers[0]).toEqual({
      name: "oauth-server",
      url: "https://example.com",
    });
    expect((result.sse_servers[0] as { api_key?: string }).api_key).toBeUndefined();
  });

  it("should parse timeout for shttp servers", () => {
    const input = {
      mcpServers: {
        "timeout-server": {
          url: "https://example.com",
          transport: "http",
          timeout: 30,
        },
      },
    };

    const result = parseMcpConfig(input);

    expect(result.shttp_servers[0]).toEqual({
      name: "timeout-server",
      url: "https://example.com",
      timeout: 30,
    });
  });
});

describe("toSdkMcpConfig", () => {
  it("should return null for empty config", () => {
    const config: MCPConfig = {
      sse_servers: [],
      stdio_servers: [],
      shttp_servers: [],
    };

    expect(toSdkMcpConfig(config)).toBeNull();
  });

  it("should use preserved names when available", () => {
    const config: MCPConfig = {
      sse_servers: [{ name: "my-custom-name", url: "https://example.com" }],
      stdio_servers: [],
      shttp_servers: [],
    };

    const result = toSdkMcpConfig(config);

    expect(result).toEqual({
      mcpServers: {
        "my-custom-name": {
          url: "https://example.com",
          transport: "sse",
        },
      },
    });
  });

  it("should generate default names for servers without names", () => {
    const config: MCPConfig = {
      sse_servers: ["https://example.com"],
      stdio_servers: [],
      shttp_servers: [],
    };

    const result = toSdkMcpConfig(config);

    expect(result).toEqual({
      mcpServers: {
        sse: {
          url: "https://example.com",
          transport: "sse",
        },
      },
    });
  });

  it("should handle name collisions with suffix", () => {
    const config: MCPConfig = {
      sse_servers: [
        { name: "sse", url: "https://example1.com" },
        { name: "sse", url: "https://example2.com" },
        { name: "sse", url: "https://example3.com" },
      ],
      stdio_servers: [],
      shttp_servers: [],
    };

    const result = toSdkMcpConfig(config);

    expect(result?.mcpServers).toHaveProperty("sse");
    expect(result?.mcpServers).toHaveProperty("sse_1");
    expect(result?.mcpServers).toHaveProperty("sse_2");
    expect(result?.mcpServers["sse"].url).toBe("https://example1.com");
    expect(result?.mcpServers["sse_1"].url).toBe("https://example2.com");
    expect(result?.mcpServers["sse_2"].url).toBe("https://example3.com");
  });

  it("should include api_key as auth field", () => {
    const config: MCPConfig = {
      sse_servers: [
        { name: "secure", url: "https://example.com", api_key: "my-secret" },
      ],
      stdio_servers: [],
      shttp_servers: [],
    };

    const result = toSdkMcpConfig(config);

    expect(result?.mcpServers["secure"]).toEqual({
      url: "https://example.com",
      transport: "sse",
      auth: "my-secret",
    });
  });

  it("should include timeout for shttp servers", () => {
    const config: MCPConfig = {
      sse_servers: [],
      stdio_servers: [],
      shttp_servers: [
        { name: "timeout", url: "https://example.com", timeout: 60 },
      ],
    };

    const result = toSdkMcpConfig(config);

    expect(result?.mcpServers["timeout"]).toEqual({
      url: "https://example.com",
      timeout: 60,
    });
  });

  it("should convert stdio servers correctly", () => {
    const config: MCPConfig = {
      sse_servers: [],
      stdio_servers: [
        {
          name: "my-stdio",
          command: "/usr/bin/server",
          args: ["--flag"],
          env: { KEY: "value" },
        },
      ],
      shttp_servers: [],
    };

    const result = toSdkMcpConfig(config);

    expect(result?.mcpServers["my-stdio"]).toEqual({
      command: "/usr/bin/server",
      args: ["--flag"],
      env: { KEY: "value" },
    });
  });
});

describe("round-trip preservation", () => {
  it("should preserve server names through parse -> serialize round-trip", () => {
    const original = {
      mcpServers: {
        "custom-sse-name": {
          url: "https://sse.example.com",
          transport: "sse",
        },
        "custom-http-name": {
          url: "https://http.example.com",
          transport: "http",
        },
        "custom-stdio-name": {
          command: "/usr/bin/server",
        },
      },
    };

    const parsed = parseMcpConfig(original);
    const serialized = toSdkMcpConfig(parsed);

    expect(serialized?.mcpServers).toHaveProperty("custom-sse-name");
    expect(serialized?.mcpServers).toHaveProperty("custom-http-name");
    expect(serialized?.mcpServers).toHaveProperty("custom-stdio-name");
  });

  it("should not generate new names if original names are preserved", () => {
    const original = {
      mcpServers: {
        server1: { url: "https://server1.com", transport: "sse" },
        server2: { url: "https://server2.com", transport: "sse" },
      },
    };

    const parsed = parseMcpConfig(original);
    const serialized = toSdkMcpConfig(parsed);

    // Should use original names, not "sse" or "sse_1"
    expect(serialized?.mcpServers).toHaveProperty("server1");
    expect(serialized?.mcpServers).toHaveProperty("server2");
    expect(serialized?.mcpServers).not.toHaveProperty("sse");
    expect(serialized?.mcpServers).not.toHaveProperty("sse_1");
  });
});

describe("edge cases", () => {
  it("should handle empty mcpServers object", () => {
    const input = { mcpServers: {} };
    const result = parseMcpConfig(input);

    expect(result).toEqual({
      sse_servers: [],
      stdio_servers: [],
      shttp_servers: [],
    });
  });

  it("should treat server with missing url as stdio server", () => {
    const input = {
      mcpServers: {
        "no-url-server": {
          transport: "sse",
          command: "/usr/bin/server",
        },
      },
    };

    const result = parseMcpConfig(input);

    // Servers without url are treated as stdio servers
    expect(result.stdio_servers).toHaveLength(1);
    expect(result.stdio_servers[0]).toEqual({
      name: "no-url-server",
      command: "/usr/bin/server",
    });
  });

  it("should treat unknown transport type as shttp server", () => {
    const input = {
      mcpServers: {
        "unknown-transport": {
          url: "https://example.com",
          transport: "websocket", // not "sse", so falls through to shttp
        },
      },
    };

    const result = parseMcpConfig(input);

    // Non-SSE transports with url go to shttp_servers
    expect(result.sse_servers).toHaveLength(0);
    expect(result.shttp_servers).toHaveLength(1);
    expect(result.shttp_servers[0]).toEqual({
      name: "unknown-transport",
      url: "https://example.com",
    });
  });

  it("should handle mixed server types correctly", () => {
    const input = {
      mcpServers: {
        "sse-server": { url: "https://sse.com", transport: "sse" },
        "http-server": { url: "https://http.com", transport: "http" },
        "stdio-server": { command: "/usr/bin/cmd" },
      },
    };

    const result = parseMcpConfig(input);

    expect(result.sse_servers).toHaveLength(1);
    expect(result.shttp_servers).toHaveLength(1);
    expect(result.stdio_servers).toHaveLength(1);
  });

  it("should handle deleting all servers (empty result)", () => {
    const config: MCPConfig = {
      sse_servers: [],
      stdio_servers: [],
      shttp_servers: [],
    };

    const result = toSdkMcpConfig(config);

    expect(result).toBeNull();
  });

  it("should handle servers with special characters in names", () => {
    const input = {
      mcpServers: {
        "server/with/slashes": { url: "https://s1.com", transport: "sse" },
        "server:with:colons": { url: "https://s2.com", transport: "sse" },
        "server with spaces": { url: "https://s3.com", transport: "sse" },
      },
    };

    const parsed = parseMcpConfig(input);
    const serialized = toSdkMcpConfig(parsed);

    expect(serialized?.mcpServers).toHaveProperty("server/with/slashes");
    expect(serialized?.mcpServers).toHaveProperty("server:with:colons");
    expect(serialized?.mcpServers).toHaveProperty("server with spaces");
  });

  it("should handle cross-type name collisions", () => {
    const config: MCPConfig = {
      sse_servers: [{ name: "myserver", url: "https://sse.com" }],
      stdio_servers: [{ name: "myserver", command: "/usr/bin/cmd" }],
      shttp_servers: [],
    };

    const result = toSdkMcpConfig(config);

    const names = Object.keys(result?.mcpServers || {});
    expect(names).toContain("myserver");
    expect(names).toContain("myserver_1");
    expect(names).toHaveLength(2);
  });
});
