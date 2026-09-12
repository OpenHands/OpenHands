import { describe, expect, it } from "vitest";
import type {
  MCPConfig,
  MCPServer,
} from "@openhands/typescript-client";

import {
  allocateMcpSettingsKey,
  buildMcpServerPatch,
  buildRenameMcpConfigPatch,
  getMcpServerEnabled,
  getSdkMcpServerMap,
  hasRedactedMcpSecretLeaf,
  MCP_HEADER_REMOVAL_ERROR,
  MCP_RENAME_CREDENTIAL_ERROR,
  parseMcpConfig,
  REDACTED_MCP_SECRET_VALUE,
  stringRecord,
  toCanonicalMcpServer,
} from "#/utils/mcp-config";
import type { MCPServerConfig } from "#/types/mcp-server";

describe("REDACTED_MCP_SECRET_VALUE", () => {
  it("is the ten-asterisk sentinel", () => {
    expect(REDACTED_MCP_SECRET_VALUE).toBe("**********");
  });
});

describe("getSdkMcpServerMap", () => {
  it("returns null for non-record values", () => {
    expect(getSdkMcpServerMap(null)).toBeNull();
    expect(getSdkMcpServerMap(undefined)).toBeNull();
    expect(getSdkMcpServerMap("string")).toBeNull();
    expect(getSdkMcpServerMap(42)).toBeNull();
    expect(getSdkMcpServerMap([])).toBeNull();
  });

  it("unwraps the mcpServers wrapper when it is a plain server map", () => {
    const wrapped = { mcpServers: { docs: { url: "https://docs.example" } } };
    expect(getSdkMcpServerMap(wrapped)).toEqual({
      docs: { url: "https://docs.example" },
    });
  });

  it("does not unwrap when mcpServers is itself a single server", () => {
    const urlServer = { mcpServers: { url: "https://meta.example" } };
    expect(getSdkMcpServerMap(urlServer)).toEqual(urlServer);

    const commandServer = { mcpServers: { command: "npx" } };
    expect(getSdkMcpServerMap(commandServer)).toEqual(commandServer);
  });

  it("does not unwrap when mcpServers is not a record", () => {
    const value = { mcpServers: "invalid" };
    expect(getSdkMcpServerMap(value)).toEqual(value);
  });

  it("returns the value unchanged when there is no mcpServers key", () => {
    const flat = { docs: { url: "https://docs.example" } };
    expect(getSdkMcpServerMap(flat)).toBe(flat);
  });
});

describe("stringRecord", () => {
  it("returns undefined for non-records", () => {
    expect(stringRecord(null)).toBeUndefined();
    expect(stringRecord(["a"])).toBeUndefined();
    expect(stringRecord("x")).toBeUndefined();
  });

  it("keeps only string-valued entries", () => {
    expect(stringRecord({ valid: "yes", count: 1, nested: {} })).toEqual({
      valid: "yes",
    });
  });

  it("returns undefined when no entry is a string", () => {
    expect(stringRecord({ count: 1, flag: true })).toBeUndefined();
  });
});

describe("hasRedactedMcpSecretLeaf", () => {
  it("detects the redacted sentinel directly", () => {
    expect(hasRedactedMcpSecretLeaf(REDACTED_MCP_SECRET_VALUE)).toBe(true);
  });

  it("returns false for plain scalars", () => {
    expect(hasRedactedMcpSecretLeaf("plain")).toBe(false);
    expect(hasRedactedMcpSecretLeaf(123)).toBe(false);
    expect(hasRedactedMcpSecretLeaf(null)).toBe(false);
  });

  it("recurses into arrays", () => {
    expect(
      hasRedactedMcpSecretLeaf(["plain", { nested: REDACTED_MCP_SECRET_VALUE }]),
    ).toBe(true);
    expect(hasRedactedMcpSecretLeaf(["plain", "other"])).toBe(false);
  });

  it("recurses into records", () => {
    expect(
      hasRedactedMcpSecretLeaf({ a: { b: REDACTED_MCP_SECRET_VALUE } }),
    ).toBe(true);
    expect(hasRedactedMcpSecretLeaf({ nested: ["plain"] })).toBe(false);
  });
});

describe("getMcpServerEnabled", () => {
  it("returns false only for an explicit disabled flag", () => {
    expect(getMcpServerEnabled({ enabled: false } as unknown as MCPServer)).toBe(
      false,
    );
  });

  it("returns undefined when enabled is true or absent", () => {
    expect(
      getMcpServerEnabled({ enabled: true } as unknown as MCPServer),
    ).toBeUndefined();
    expect(getMcpServerEnabled({} as unknown as MCPServer)).toBeUndefined();
  });
});

describe("parseMcpConfig", () => {
  it("returns an empty config for non-record input", () => {
    expect(parseMcpConfig("invalid")).toEqual({});
    expect(parseMcpConfig(null)).toEqual({});
  });

  it("unwraps the cloud mcpServers wrapper", () => {
    expect(
      parseMcpConfig({
        mcpServers: {
          cloud: { url: "https://cloud.example", transport: "http" },
        },
      }),
    ).toEqual({
      cloud: { transport: "http", url: "https://cloud.example" },
    });
  });

  it("normalizes a fully-populated stdio server", () => {
    expect(
      parseMcpConfig({
        local: {
          command: "npx",
          args: ["a", "b"],
          env: { TOKEN: "value", ignored: 2 },
          cwd: "/tmp",
          description: "desc",
          icon: "icon",
          timeout: 5,
          enabled: false,
        },
      }),
    ).toEqual({
      local: {
        transport: "stdio",
        command: "npx",
        args: ["a", "b"],
        env: { TOKEN: "value" },
        cwd: "/tmp",
        description: "desc",
        icon: "icon",
        timeout: 5,
        enabled: false,
      },
    });
  });

  it("omits stdio args when not every entry is a string", () => {
    expect(
      parseMcpConfig({ local: { command: "x", args: ["ok", 2] } }),
    ).toEqual({ local: { transport: "stdio", command: "x" } });
  });

  it("normalizes remote transports and their optional fields", () => {
    expect(
      parseMcpConfig({
        http_default: {
          url: "https://http.example",
          headers: { "X-Test": "yes", ignored: 1 },
          auth: { strategy: "bearer", value: "token" },
          description: "d",
          icon: "i",
          timeout: 0,
          sse_read_timeout: 10,
          keep_alive: true,
          enabled: false,
        },
        sse_server: { url: "https://sse.example", transport: "sse" },
        streamable: { url: "https://x.example", transport: "streamable-http" },
        shttp_alias: { url: "https://y.example", transport: "shttp" },
      }),
    ).toEqual({
      http_default: {
        transport: "http",
        url: "https://http.example",
        headers: { "X-Test": "yes" },
        auth: { strategy: "bearer", value: "token" },
        description: "d",
        icon: "i",
        timeout: 0,
        sse_read_timeout: 10,
        keep_alive: true,
        enabled: false,
      },
      sse_server: { transport: "sse", url: "https://sse.example" },
      streamable: { transport: "streamable-http", url: "https://x.example" },
      shttp_alias: { transport: "http", url: "https://y.example" },
    });
  });

  it("drops invalid auth credentials while keeping the server", () => {
    expect(
      parseMcpConfig({
        remote: {
          url: "https://remote.example",
          auth: { strategy: "not-a-real-strategy" },
        },
      }),
    ).toEqual({
      remote: { transport: "http", url: "https://remote.example" },
    });
  });

  it("skips entries that are not usable server records", () => {
    expect(
      parseMcpConfig({
        scalar: "nope",
        null_entry: null,
        empty_object: { foo: "bar" },
        bad_transport: { url: "https://z.example", transport: "weird" },
        stdio_transport_url: {
          url: "https://s.example",
          transport: "stdio",
        },
        keep: { url: "https://keep.example" },
      }),
    ).toEqual({
      keep: { transport: "http", url: "https://keep.example" },
    });
  });

  it("ignores optional fields whose types are wrong", () => {
    expect(
      parseMcpConfig({
        stdio_bad: {
          command: "c",
          args: "not-array",
          env: "not-record",
          cwd: 1,
          description: 2,
          icon: 3,
          timeout: "x",
          enabled: "no",
        },
        remote_bad: {
          url: "https://bad.example",
          headers: "not-record",
          auth: 123,
          description: 1,
          icon: 2,
          timeout: "x",
          sse_read_timeout: "y",
          keep_alive: "z",
          enabled: "no",
        },
      }),
    ).toEqual({
      stdio_bad: { transport: "stdio", command: "c" },
      remote_bad: { transport: "http", url: "https://bad.example" },
    });
  });
});

describe("toCanonicalMcpServer", () => {
  it("serializes a full stdio server", () => {
    const server: MCPServerConfig = {
      id: "local",
      type: "stdio",
      command: "npx",
      args: ["server"],
      env: { TOKEN: "value" },
      enabled: false,
    };
    expect(toCanonicalMcpServer(server)).toEqual({
      transport: "stdio",
      command: "npx",
      args: ["server"],
      env: { TOKEN: "value" },
      enabled: false,
    });
  });

  it("omits empty stdio args and env", () => {
    const server: MCPServerConfig = {
      id: "local",
      type: "stdio",
      command: "uvx",
      args: [],
      env: {},
    };
    expect(toCanonicalMcpServer(server)).toEqual({
      transport: "stdio",
      command: "uvx",
    });
  });

  it("serializes an sse server and ignores timeout", () => {
    const server: MCPServerConfig = {
      id: "events",
      type: "sse",
      url: "https://events.example",
      headers: { "X-Test": "yes" },
      auth: { strategy: "bearer", value: "secret" },
      timeout: 5,
    };
    expect(toCanonicalMcpServer(server)).toEqual({
      transport: "sse",
      url: "https://events.example",
      headers: { "X-Test": "yes" },
      auth: { strategy: "bearer", value: "secret" },
    });
  });

  it("serializes an shttp server and keeps its timeout", () => {
    const server: MCPServerConfig = {
      id: "http",
      type: "shttp",
      url: "https://http.example",
      timeout: 15,
    };
    expect(toCanonicalMcpServer(server)).toEqual({
      transport: "http",
      url: "https://http.example",
      timeout: 15,
    });
  });

  it("omits empty headers and undefined timeout for remote servers", () => {
    const server: MCPServerConfig = {
      id: "http-min",
      type: "shttp",
      url: "https://http.example",
      headers: {},
    };
    expect(toCanonicalMcpServer(server)).toStrictEqual({
      transport: "http",
      url: "https://http.example",
    });
  });

  it("omits stdio args and env when they are absent", () => {
    const server: MCPServerConfig = { id: "m", type: "stdio", command: "uvx" };
    expect(toCanonicalMcpServer(server)).toEqual({
      transport: "stdio",
      command: "uvx",
    });
  });

  it("keeps enabled:false on a remote server", () => {
    const server: MCPServerConfig = {
      id: "r",
      type: "shttp",
      url: "https://r.example",
      enabled: false,
    };
    expect(toCanonicalMcpServer(server)).toEqual({
      transport: "http",
      url: "https://r.example",
      enabled: false,
    });
  });
});

describe("buildMcpServerPatch", () => {
  const stdioPrevious = { transport: "stdio", command: "old" } as MCPServer;

  it("adds args and env for a stdio edit", () => {
    const edited: MCPServerConfig = {
      id: "s",
      type: "stdio",
      command: "npx",
      args: ["a"],
      env: { TOKEN: "value" },
    };
    expect(buildMcpServerPatch(stdioPrevious, edited)).toEqual({
      transport: "stdio",
      command: "npx",
      args: ["a"],
      env: { TOKEN: "value" },
    });
  });

  it("nulls previously-set args that are removed on edit", () => {
    const previous = {
      transport: "stdio",
      command: "c",
      args: ["x"],
    } as MCPServer;
    const edited: MCPServerConfig = { id: "s", type: "stdio", command: "c" };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "stdio",
      command: "c",
      args: null,
    });
  });

  it("skips redacted env values and nulls removed env keys", () => {
    const previous = {
      transport: "stdio",
      command: "c",
      env: { OLD: "1", KEEP: "2" },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "stdio",
      command: "c",
      env: { KEEP: REDACTED_MCP_SECRET_VALUE, NEW: "3" },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "stdio",
      command: "c",
      env: { NEW: "3", OLD: null },
    });
  });

  it("emits enabled:false when the edit disables the server", () => {
    const edited: MCPServerConfig = {
      id: "s",
      type: "sse",
      url: "https://e.example",
      enabled: false,
    };
    expect(buildMcpServerPatch(stdioPrevious, edited)).toEqual({
      transport: "sse",
      url: "https://e.example",
      enabled: false,
    });
  });

  it("re-enables a previously disabled server", () => {
    const previous = {
      transport: "http",
      url: "https://e.example",
      enabled: false,
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://e.example",
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://e.example",
      enabled: true,
    });
  });

  it("sets and clears shttp timeout", () => {
    const previousWithTimeout = {
      transport: "http",
      url: "https://u.example",
      timeout: 10,
    } as MCPServer;
    const clearEdit: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
    };
    expect(buildMcpServerPatch(previousWithTimeout, clearEdit)).toEqual({
      transport: "http",
      url: "https://u.example",
      timeout: null,
    });

    const setEdit: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      timeout: 5,
    };
    expect(buildMcpServerPatch(stdioPrevious, setEdit)).toEqual({
      transport: "http",
      url: "https://u.example",
      timeout: 5,
    });
  });

  it("clears remote auth when the edit removes it", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "bearer", value: "tok" },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: null,
    });
  });

  it("writes a new non-oauth credential", () => {
    const edited: MCPServerConfig = {
      id: "s",
      type: "sse",
      url: "https://u.example",
      auth: { strategy: "bearer", value: "tok" },
    };
    expect(buildMcpServerPatch(stdioPrevious, edited)).toEqual({
      transport: "sse",
      url: "https://u.example",
      auth: { strategy: "bearer", value: "tok" },
    });
  });

  it("skips a redacted credential rather than persisting the mask", () => {
    const edited: MCPServerConfig = {
      id: "s",
      type: "sse",
      url: "https://u.example",
      auth: { strategy: "bearer", value: REDACTED_MCP_SECRET_VALUE },
    };
    expect(buildMcpServerPatch(stdioPrevious, edited)).toEqual({
      transport: "sse",
      url: "https://u.example",
    });
  });

  it("nulls fields that differ across a credential strategy replacement", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "bearer", value: "tok" },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: { strategy: "header", headers: { "X-One": "one" } },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "header", headers: { "X-One": "one" }, value: null },
    });
  });

  it("patches header auth by merging headers", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "header", headers: { A: "1" } },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: { strategy: "header", headers: { A: "1", B: "2" } },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "header", headers: { A: "1", B: "2" } },
    });
  });

  it("throws when a header removal is attempted", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "header", headers: { A: "1", B: "2" } },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: { strategy: "header", headers: { A: "1" } },
    };
    expect(() => buildMcpServerPatch(previous, edited)).toThrow(
      MCP_HEADER_REMOVAL_ERROR,
    );
  });

  it("creates a new oauth credential and drops redacted state leaves", () => {
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: {
          type: "oauth",
          client_id: "cid",
          scopes: ["read", "write"],
        },
        state: {
          token_expires_at: 5,
          tokens: { access_token: REDACTED_MCP_SECRET_VALUE },
        },
      },
    };
    expect(buildMcpServerPatch(stdioPrevious, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: {
          type: "oauth",
          client_id: "cid",
          scopes: ["read", "write"],
        },
        state: { token_expires_at: 5 },
      },
    });
  });

  it("nulls oauth authentication fields removed on edit", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_id: "cid", scopes: ["a"] },
      },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", scopes: ["a"] },
      },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_id: null },
      },
    });
  });

  it("omits the auth key when an oauth credential is unchanged", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_id: "cid" },
        state: { token_expires_at: 5 },
      },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_id: "cid" },
        state: { token_expires_at: 5 },
      },
    };
    expect(buildMcpServerPatch(previous, edited)).toStrictEqual({
      transport: "http",
      url: "https://u.example",
    });
  });

  it("omits the env patch when the only change is a redacted secret", () => {
    const previous = {
      transport: "stdio",
      command: "c",
      env: { SECRET: "real" },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "stdio",
      command: "c",
      env: { SECRET: REDACTED_MCP_SECRET_VALUE },
    };
    expect(buildMcpServerPatch(previous, edited)).toStrictEqual({
      transport: "stdio",
      command: "c",
    });
  });

  it("ignores timeout for sse edits", () => {
    const edited: MCPServerConfig = {
      id: "s",
      type: "sse",
      url: "https://u.example",
      timeout: 5,
    };
    expect(buildMcpServerPatch(stdioPrevious, edited)).toEqual({
      transport: "sse",
      url: "https://u.example",
    });
  });

  it("does not clear timeout when the previous server was stdio", () => {
    const previous = {
      transport: "stdio",
      command: "c",
      timeout: 10,
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
    });
  });

  it("uses stdio env/args baselines only when the previous server was stdio", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      env: { X: "1" },
      args: ["x"],
    } as unknown as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "stdio",
      command: "c",
      env: { Y: "2" },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "stdio",
      command: "c",
      env: { Y: "2" },
    });
  });

  it("nulls a removed field when replacing a same-strategy credential", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "bearer", value: "old" },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: { strategy: "bearer" },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "bearer" },
    });
  });

  it("replaces a header credential with a different strategy", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "header", headers: { A: "1" } },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: { strategy: "bearer", value: "x" },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "bearer", value: "x", headers: null },
    });
  });

  it("adds a header credential when the previous server was stdio", () => {
    const previous = { transport: "stdio", command: "c" } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: { strategy: "header", headers: { A: "1" } },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "header", headers: { A: "1" } },
    });
  });

  it("adds a header credential when the previous remote had no auth", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: { strategy: "header", headers: { A: "1" } },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "header", headers: { A: "1" } },
    });
  });

  it("makes no header change when the edit only re-sends a redacted header", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "header", headers: { A: "1" } },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: { strategy: "header", headers: { A: REDACTED_MCP_SECRET_VALUE } },
    };
    expect(buildMcpServerPatch(previous, edited)).toStrictEqual({
      transport: "http",
      url: "https://u.example",
    });
  });

  it("treats a non-oauth previous credential as having no oauth baseline", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "bearer", value: "old" },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: { strategy: "oauth2" },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "oauth2", value: null },
    });
  });

  it("patches only oauth state when the authentication block is omitted", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_id: "cid" },
        state: { token_expires_at: 5 },
      },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: { strategy: "oauth2", state: { token_expires_at: 9 } },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "oauth2", state: { token_expires_at: 9 } },
    });
  });

  it("clears oauth authentication when it is set to null", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_id: "cid" },
      },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: { strategy: "oauth2", authentication: null },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "oauth2", authentication: null },
    });
  });

  it("drops oauth scopes that contain a redacted secret", () => {
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: {
          type: "oauth",
          client_id: "cid",
          scopes: ["read", REDACTED_MCP_SECRET_VALUE],
        },
      },
    };
    expect(buildMcpServerPatch(stdioPrevious, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_id: "cid" },
      },
    });
  });

  it("does not resurrect previous oauth state when the edit omits it", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_id: "cid" },
        state: { token_expires_at: 5 },
      },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_id: "NEW" },
      },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_id: "NEW" },
      },
    });
  });

  it("keeps an explicit null oauth state", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_id: "cid" },
        state: null,
      },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
      auth: {
        strategy: "oauth2",
        authentication: { type: "oauth", client_id: "cid" },
        state: null,
      },
    };
    expect(buildMcpServerPatch(previous, edited)).toEqual({
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "oauth2", state: null },
    });
  });
});

describe("buildRenameMcpConfigPatch", () => {
  it("throws when a stdio credential is still redacted", () => {
    const previous = {
      transport: "stdio",
      command: "c",
      env: { TOKEN: REDACTED_MCP_SECRET_VALUE },
    } as MCPServer;
    const edited: MCPServerConfig = { id: "s", type: "stdio", command: "c" };
    expect(() =>
      buildRenameMcpConfigPatch("old", "new", previous, edited),
    ).toThrow(MCP_RENAME_CREDENTIAL_ERROR);
  });

  it("throws when a remote credential is still redacted", () => {
    const previous = {
      transport: "http",
      url: "https://u.example",
      auth: { strategy: "bearer", value: REDACTED_MCP_SECRET_VALUE },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://u.example",
    };
    expect(() =>
      buildRenameMcpConfigPatch("old", "new", previous, edited),
    ).toThrow(MCP_RENAME_CREDENTIAL_ERROR);
  });

  it("removes the old key and writes the merged server under the new key", () => {
    const previous = {
      transport: "http",
      url: "https://old.example",
      auth: { strategy: "bearer", value: "tok" },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://new.example",
    };
    expect(buildRenameMcpConfigPatch("old", "new", previous, edited)).toEqual({
      old: null,
      new: { transport: "http", url: "https://new.example" },
    });
  });

  it("merges nested record fields when renaming", () => {
    const previous = {
      transport: "stdio",
      command: "c",
      env: { A: "1" },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "stdio",
      command: "c",
      env: { A: "1", B: "2" },
    };
    expect(buildRenameMcpConfigPatch("old", "new", previous, edited)).toEqual({
      old: null,
      new: { transport: "stdio", command: "c", env: { A: "1", B: "2" } },
    });
  });

  it("preserves fields the edit does not touch", () => {
    const previous = {
      transport: "http",
      url: "https://old.example",
      description: "keep",
      icon: "ico",
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "shttp",
      url: "https://new.example",
    };
    expect(buildRenameMcpConfigPatch("old", "new", previous, edited)).toEqual({
      old: null,
      new: {
        transport: "http",
        url: "https://new.example",
        description: "keep",
        icon: "ico",
      },
    });
  });

  it("deep-merges nested records so untouched secrets survive the rename", () => {
    const previous = {
      transport: "stdio",
      command: "c",
      env: { A: "1", SECRET: "real" },
    } as MCPServer;
    const edited: MCPServerConfig = {
      id: "s",
      type: "stdio",
      command: "c",
      env: { A: "1", SECRET: REDACTED_MCP_SECRET_VALUE },
    };
    expect(buildRenameMcpConfigPatch("old", "new", previous, edited)).toEqual({
      old: null,
      new: {
        transport: "stdio",
        command: "c",
        env: { A: "1", SECRET: "real" },
      },
    });
  });
});

describe("allocateMcpSettingsKey", () => {
  it("normalizes the preferred name when it does not collide", () => {
    expect(allocateMcpSettingsKey({}, "My Server", "sse")).toBe("My_Server");
  });

  it("falls back to the provided base when no name is given", () => {
    expect(allocateMcpSettingsKey({}, undefined, "shttp")).toBe("shttp");
  });

  it("appends the first free numeric suffix on collision", () => {
    const config = {
      tool: {} as MCPServer,
      tool_1: {} as MCPServer,
    } satisfies MCPConfig;
    expect(allocateMcpSettingsKey(config, "tool", "stdio")).toBe("tool_2");
  });

  it("uses a single suffix when only the base collides", () => {
    const config = { tool: {} as MCPServer } satisfies MCPConfig;
    expect(allocateMcpSettingsKey(config, "tool", "stdio")).toBe("tool_1");
  });
});
