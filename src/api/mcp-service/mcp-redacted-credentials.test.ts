import { afterEach, describe, expect, it, vi } from "vitest";
import SettingsService, {
  type SettingsApiResponse,
} from "#/api/settings-service/settings-service.api";
import type { MCPServerConfig } from "#/types/mcp-server";
import { substituteRedactedMcpCredentials } from "./mcp-redacted-credentials";
import { REDACTED_MCP_SECRET_VALUE } from "#/utils/mcp-config";

const getStdioServer = (
  overrides: Partial<MCPServerConfig> = {},
): MCPServerConfig => ({
  id: "stdio-0",
  type: "stdio",
  name: "my-server",
  command: "npx",
  env: { API_KEY: REDACTED_MCP_SECRET_VALUE },
  ...overrides,
});

const getRemoteServer = (
  overrides: Partial<MCPServerConfig> = {},
): MCPServerConfig => ({
  id: "shttp-0",
  type: "shttp",
  url: "https://example.com/mcp",
  auth: { strategy: "bearer", value: REDACTED_MCP_SECRET_VALUE },
  ...overrides,
});

const mockEncryptedMcpConfig = (mcpConfig: unknown) =>
  vi.spyOn(SettingsService, "fetchSettingsFromApi").mockResolvedValue({
    agent_settings: { mcp_config: mcpConfig },
  } as unknown as SettingsApiResponse);

describe("substituteRedactedMcpCredentials", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("substitutes a redacted stdio env value with the encrypted stored secret", async () => {
    // The stored server is resolved by its stable id, so renaming the display
    // name must not lose the encrypted secret behind the redaction placeholder.
    mockEncryptedMcpConfig({
      "stdio-0": {
        command: "npx",
        env: { API_KEY: "gAAAAA-encrypted-api-key" },
      },
    });

    const result = await substituteRedactedMcpCredentials(
      getStdioServer({ name: "new-name" }),
    );

    expect(result.env).toEqual({ API_KEY: "gAAAAA-encrypted-api-key" });
    expect(result.name).toBe("new-name");
  });

  it("resolves the stored stdio entry by id and ignores other entries", async () => {
    // The lookup keys on the server id, so a decoy entry stored under a
    // different key must never leak into the resolved secret.
    mockEncryptedMcpConfig({
      "stdio-0": { command: "npx", env: { TOKEN: "gAAAAA-alpha-token" } },
      other: { command: "npx", env: { TOKEN: "gAAAAA-beta-token" } },
    });

    const result = await substituteRedactedMcpCredentials(
      getStdioServer({
        name: "beta",
        env: { TOKEN: REDACTED_MCP_SECRET_VALUE },
      }),
    );

    expect(result.env).toEqual({ TOKEN: "gAAAAA-alpha-token" });
  });

  it("leaves typed (non-redacted) env values untouched", async () => {
    mockEncryptedMcpConfig({
      "stdio-0": {
        command: "npx",
        env: { API_KEY: "gAAAAA-encrypted", REGION: "us-east-1" },
      },
    });

    const result = await substituteRedactedMcpCredentials(
      getStdioServer({
        env: {
          API_KEY: REDACTED_MCP_SECRET_VALUE,
          REGION: "eu-west-1",
        },
      }),
    );

    expect(result.env).toEqual({
      API_KEY: "gAAAAA-encrypted",
      REGION: "eu-west-1",
    });
  });

  it("returns the server unchanged when no env value is redacted", async () => {
    const fetchSpy = vi
      .spyOn(SettingsService, "fetchSettingsFromApi")
      .mockRejectedValue(new Error("unexpected fetch"));
    const server = getStdioServer({ env: { API_KEY: "plaintext" } });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toBe(server);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("keeps the placeholder when the stored stdio entry is missing", async () => {
    mockEncryptedMcpConfig({});

    const result = await substituteRedactedMcpCredentials(getStdioServer());

    expect(result.env).toEqual({ API_KEY: REDACTED_MCP_SECRET_VALUE });
  });

  it("replaces redacted OAuth state with the encrypted stored subtree", async () => {
    mockEncryptedMcpConfig({
      "shttp-0": {
        url: "https://mcp.mail.superhuman.com/mcp",
        auth: {
          strategy: "oauth2",
          state: {
            tokens: {
              access_token: "gAAAAA-encrypted-access-token",
              refresh_token: "gAAAAA-encrypted-refresh-token",
            },
            client_info: {
              client_id: "superhuman-client",
              client_secret: "gAAAAA-encrypted-client-secret",
            },
          },
        },
      },
    });

    const result = await substituteRedactedMcpCredentials(
      getRemoteServer({
        name: "superhuman-mail",
        url: "https://mcp.mail.superhuman.com/mcp",
        auth: {
          strategy: "oauth2",
          state: {
            tokens: {
              access_token: REDACTED_MCP_SECRET_VALUE,
              refresh_token: REDACTED_MCP_SECRET_VALUE,
            },
            client_info: {
              client_id: "superhuman-client",
              client_secret: REDACTED_MCP_SECRET_VALUE,
            },
          },
        },
      }),
    );

    expect(result.auth).toEqual({
      strategy: "oauth2",
      state: {
        tokens: {
          access_token: "gAAAAA-encrypted-access-token",
          refresh_token: "gAAAAA-encrypted-refresh-token",
        },
        client_info: {
          client_id: "superhuman-client",
          client_secret: "gAAAAA-encrypted-client-secret",
        },
      },
    });
  });

  it("substitutes a stdio env by id while leaving unrelated entries alone", async () => {
    mockEncryptedMcpConfig({
      remote: {
        url: "https://example.com/mcp",
        transport: "shttp",
        env: { API_KEY: "gAAAAA-remote" },
      },
      "stdio-1": { command: "beta", env: { API_KEY: "gAAAAA-beta" } },
    });
    const server = getStdioServer({
      id: "stdio-1",
      name: "renamed-beta",
      args: ["beta"],
      timeout: 4_000,
      env: {
        API_KEY: REDACTED_MCP_SECRET_VALUE,
        REGION: "eu-west-1",
      },
    });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toEqual({
      ...server,
      env: { API_KEY: "gAAAAA-beta", REGION: "eu-west-1" },
    });
  });

  it.each([
    { label: "an empty id", id: "" },
    { label: "a non-positional id", id: "custom-id" },
    { label: "an id with a leading prefix", id: "prefix-stdio-0" },
    { label: "an id with a trailing suffix", id: "stdio-0-suffix" },
  ])(
    "resolves the stored stdio entry by the exact id for $label",
    async ({ id }) => {
      const fetchSpy = mockEncryptedMcpConfig({
        positional: {
          command: "npx",
          env: { API_KEY: "gAAAAA-wrong-position" },
        },
        [id]: {
          command: "npx",
          env: { API_KEY: "gAAAAA-id-match" },
        },
      });
      const server = getStdioServer({ id });

      const result = await substituteRedactedMcpCredentials(server);

      expect(result.env).toEqual({ API_KEY: "gAAAAA-id-match" });
      expect(fetchSpy).toHaveBeenCalledOnce();
      expect(fetchSpy).toHaveBeenCalledWith("encrypted");
    },
  );

  it("resolves a multi-digit stdio id exactly", async () => {
    mockEncryptedMcpConfig(
      Object.fromEntries(
        Array.from({ length: 11 }, (_, index) => [
          `stdio-${index}`,
          {
            command: "npx",
            env: { API_KEY: `gAAAAA-position-${index}` },
          },
        ]),
      ),
    );
    const server = getStdioServer({
      id: "stdio-10",
      name: undefined,
    });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toEqual({
      ...server,
      env: { API_KEY: "gAAAAA-position-10" },
    });
  });

  it("does not fall back to the stored name (lookup is id-only)", async () => {
    mockEncryptedMcpConfig({
      "my-server": {
        command: "npx",
        env: { API_KEY: "gAAAAA-name-fallback" },
      },
    });
    const server = getStdioServer({ id: "stdio-9" });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toBe(server);
  });

  it("keeps an exact stdio round trip when the id is not stored", async () => {
    mockEncryptedMcpConfig({
      alpha: { command: "alpha", env: { API_KEY: "gAAAAA-alpha" } },
    });
    const server = getStdioServer({ id: "stdio-9", name: undefined });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toBe(server);
  });

  it.each([
    { label: "missing agent settings", response: {} },
    {
      label: "a null MCP config",
      response: { agent_settings: { mcp_config: null } },
    },
    {
      label: "a primitive MCP config",
      response: { agent_settings: { mcp_config: "invalid" } },
    },
    {
      label: "an array MCP config",
      response: { agent_settings: { mcp_config: [] } },
    },
  ])(
    "returns the exact server when encrypted settings contain $label",
    async ({ response }) => {
      const fetchSpy = vi
        .spyOn(SettingsService, "fetchSettingsFromApi")
        .mockResolvedValue(response as unknown as SettingsApiResponse);
      const server = getStdioServer();

      const result = await substituteRedactedMcpCredentials(server);

      expect(result).toBe(server);
      expect(fetchSpy).toHaveBeenCalledOnce();
      expect(fetchSpy).toHaveBeenCalledWith("encrypted");
    },
  );

  it.each([
    { label: "null", storedEnv: null },
    { label: "an array", storedEnv: ["gAAAAA-array-value"] },
    { label: "a primitive", storedEnv: "gAAAAA-primitive-value" },
    { label: "a non-string record", storedEnv: { API_KEY: 42 } },
  ])(
    "does not substitute a redacted env from $label stored env",
    async ({ storedEnv }) => {
      mockEncryptedMcpConfig({
        "custom-id": { command: "npx", env: storedEnv },
      });
      const server = getStdioServer({ id: "custom-id" });

      const result = await substituteRedactedMcpCredentials(server);

      expect(result).toEqual(server);
      expect(result).not.toBe(server);
    },
  );

  it("only restores string values from a partially malformed stored env", async () => {
    mockEncryptedMcpConfig({
      "custom-id": {
        command: "npx",
        env: { API_KEY: "gAAAAA-encrypted", NUMERIC_SECRET: 42 },
      },
    });
    const server = getStdioServer({
      id: "custom-id",
      env: {
        API_KEY: REDACTED_MCP_SECRET_VALUE,
        NUMERIC_SECRET: REDACTED_MCP_SECRET_VALUE,
        REGION: "eu-west-1",
      },
    });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toEqual({
      ...server,
      env: {
        API_KEY: "gAAAAA-encrypted",
        NUMERIC_SECRET: REDACTED_MCP_SECRET_VALUE,
        REGION: "eu-west-1",
      },
    });
  });

  it.each(["sse", "shttp"] as const)(
    "substitutes redacted bearer auth for %s servers",
    async (type) => {
      const encryptedAuth = {
        strategy: "bearer" as const,
        value: `gAAAAA-${type}`,
      };
      mockEncryptedMcpConfig({
        [`${type}-0`]: {
          url: "https://example.com/mcp",
          auth: encryptedAuth,
        },
      });
      const server = getRemoteServer({ id: `${type}-0`, type });

      const result = await substituteRedactedMcpCredentials(server);

      expect(result).toEqual({ ...server, auth: encryptedAuth });
    },
  );

  it("restores a redacted api_key credential while preserving other fields", async () => {
    const encryptedAuth = {
      strategy: "api_key" as const,
      value: "gAAAAA-url-match",
      header_name: "X-API-Key",
    };
    mockEncryptedMcpConfig({
      "shttp-0": {
        url: "https://example.com/mcp",
        auth: encryptedAuth,
      },
    });
    const server = getRemoteServer({
      name: "renamed-remote",
      auth: {
        strategy: "api_key",
        value: REDACTED_MCP_SECRET_VALUE,
        header_name: "X-API-Key",
      },
    });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toEqual({ ...server, auth: encryptedAuth });
  });

  it("resolves remote auth by id and ignores decoy entries", async () => {
    const namedAuth = {
      strategy: "bearer" as const,
      value: "gAAAAA-name-match",
    };
    mockEncryptedMcpConfig({
      "sse-0": {
        url: "https://old.example.com/mcp",
        auth: namedAuth,
      },
      urlMatch: {
        url: "https://example.com/mcp",
        auth: { strategy: "bearer", value: "gAAAAA-url-match" },
      },
    });
    const server = getRemoteServer({
      id: "sse-0",
      type: "sse",
      name: "preferred",
    });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toEqual({ ...server, auth: namedAuth });
  });

  it("substitutes redacted remote headers with the stored encrypted values", async () => {
    mockEncryptedMcpConfig({
      "shttp-0": {
        url: "https://example.com/mcp",
        headers: { "X-Token": "gAAAAA-header", "X-Region": "us-east-1" },
      },
    });
    const server = getRemoteServer({
      auth: undefined,
      headers: {
        "X-Token": REDACTED_MCP_SECRET_VALUE,
        "X-Region": "eu-west-1",
      },
    });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toEqual({
      ...server,
      headers: { "X-Token": "gAAAAA-header", "X-Region": "eu-west-1" },
    });
  });

  it("substitutes both redacted remote auth and headers together", async () => {
    const encryptedAuth = {
      strategy: "bearer" as const,
      value: "gAAAAA-both-auth",
    };
    mockEncryptedMcpConfig({
      "shttp-0": {
        url: "https://example.com/mcp",
        auth: encryptedAuth,
        headers: { "X-Token": "gAAAAA-both-header" },
      },
    });
    const server = getRemoteServer({
      headers: { "X-Token": REDACTED_MCP_SECRET_VALUE },
    });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toEqual({
      ...server,
      auth: encryptedAuth,
      headers: { "X-Token": "gAAAAA-both-header" },
    });
  });

  it("keeps a redacted remote auth leaf when the stored value is not a string", async () => {
    mockEncryptedMcpConfig({
      "shttp-0": {
        url: "https://example.com/mcp",
        auth: { strategy: "bearer", value: 42 },
      },
    });
    const server = getRemoteServer();

    const result = await substituteRedactedMcpCredentials(server);

    expect(result.auth).toEqual({
      strategy: "bearer",
      value: REDACTED_MCP_SECRET_VALUE,
    });
  });

  it("substitutes redacted sse headers and keeps leaves absent from storage", async () => {
    mockEncryptedMcpConfig({
      "sse-0": {
        url: "https://example.com/mcp",
        headers: { "X-Token": "gAAAAA-header" },
      },
    });
    const server = getRemoteServer({
      id: "sse-0",
      type: "sse",
      auth: undefined,
      headers: {
        "X-Token": REDACTED_MCP_SECRET_VALUE,
        "X-Missing": REDACTED_MCP_SECRET_VALUE,
        "X-Region": "eu-west-1",
      },
    });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toEqual({
      ...server,
      headers: {
        "X-Token": "gAAAAA-header",
        "X-Missing": REDACTED_MCP_SECRET_VALUE,
        "X-Region": "eu-west-1",
      },
    });
  });

  it("ignores redacted headers on a stdio server", async () => {
    const fetchSpy = vi
      .spyOn(SettingsService, "fetchSettingsFromApi")
      .mockRejectedValue(new Error("unexpected fetch"));
    const server = getStdioServer({
      env: { API_KEY: "plaintext" },
      headers: { "X-Token": REDACTED_MCP_SECRET_VALUE },
    });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toBe(server);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("keeps redacted remote auth when the stored auth is malformed", async () => {
    mockEncryptedMcpConfig({
      "shttp-0": {
        url: "https://example.com/mcp",
        transport: "shttp",
        auth: { strategy: "unsupported", value: "gAAAAA-invalid" },
      },
    });
    const server = getRemoteServer({ name: "malformed" });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toEqual(server);
    expect(result).not.toBe(server);
  });

  it("keeps the exact server when encrypted settings cannot be fetched", async () => {
    const fetchSpy = vi
      .spyOn(SettingsService, "fetchSettingsFromApi")
      .mockRejectedValue(new Error("settings unavailable"));
    const server = getRemoteServer({ name: "unavailable" });

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toBe(server);
    expect(fetchSpy).toHaveBeenCalledOnce();
    expect(fetchSpy).toHaveBeenCalledWith("encrypted");
  });

  it("does not fetch encrypted settings for absent or non-redacted secrets", async () => {
    const fetchSpy = vi
      .spyOn(SettingsService, "fetchSettingsFromApi")
      .mockRejectedValue(new Error("unexpected fetch"));
    const servers = [
      getStdioServer({ env: undefined }),
      getRemoteServer({
        id: "sse-0",
        type: "sse",
        auth: { strategy: "bearer", value: "user-entered-token" },
      }),
      getRemoteServer({
        auth: { strategy: "bearer", value: "user-entered-token" },
        env: { API_KEY: REDACTED_MCP_SECRET_VALUE },
      }),
      getStdioServer({
        env: undefined,
        auth: { strategy: "bearer", value: REDACTED_MCP_SECRET_VALUE },
      }),
    ];

    for (const server of servers) {
      expect(await substituteRedactedMcpCredentials(server)).toBe(server);
    }
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
