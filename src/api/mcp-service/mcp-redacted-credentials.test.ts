import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService, {
  type SettingsApiResponse,
} from "#/api/settings-service/settings-service.api";
import { substituteRedactedMcpCredentials } from "./mcp-redacted-credentials";
import { REDACTED_MCP_SECRET_VALUE } from "#/utils/mcp-config";

describe("substituteRedactedMcpCredentials", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("preserves encrypted stdio env when the server is renamed", async () => {
    // Regression: renaming a stdio server left a redacted env value unchanged,
    // so the lookup by the new display name missed the stored entry and the
    // literal redaction placeholder overwrote the stored encrypted secret.
    vi.spyOn(SettingsService, "fetchSettingsFromApi").mockResolvedValue({
      agent_settings: {
        mcp_config: {
          "old-name": {
            command: "npx",
            env: { API_KEY: "gAAAAA-encrypted-api-key" },
          },
        },
      },
    } as unknown as SettingsApiResponse);

    const result = await substituteRedactedMcpCredentials({
      id: "stdio-0",
      type: "stdio",
      name: "new-name",
      command: "npx",
      env: { API_KEY: REDACTED_MCP_SECRET_VALUE },
    });

    expect(result.env).toEqual({ API_KEY: "gAAAAA-encrypted-api-key" });
    expect(result.name).toBe("new-name");
  });

  it("does not restore another server's secret when renamed onto an existing name", async () => {
    // Renaming "alpha" onto "beta"'s name must still resolve alpha's stored
    // entry by position, not beta's entry (which the name match would return).
    vi.spyOn(SettingsService, "fetchSettingsFromApi").mockResolvedValue({
      agent_settings: {
        mcp_config: {
          alpha: { command: "npx", env: { TOKEN: "gAAAAA-alpha-token" } },
          beta: { command: "npx", env: { TOKEN: "gAAAAA-beta-token" } },
        },
      },
    } as unknown as SettingsApiResponse);

    const result = await substituteRedactedMcpCredentials({
      id: "stdio-0",
      type: "stdio",
      name: "beta",
      command: "npx",
      env: { TOKEN: REDACTED_MCP_SECRET_VALUE },
    });

    expect(result.env).toEqual({ TOKEN: "gAAAAA-alpha-token" });
  });

  it("leaves typed (non-redacted) env values untouched", async () => {
    vi.spyOn(SettingsService, "fetchSettingsFromApi").mockResolvedValue({
      agent_settings: {
        mcp_config: {
          "my-server": {
            command: "npx",
            env: { API_KEY: "gAAAAA-encrypted", REGION: "us-east-1" },
          },
        },
      },
    } as unknown as SettingsApiResponse);

    const result = await substituteRedactedMcpCredentials({
      id: "stdio-0",
      type: "stdio",
      name: "my-server",
      command: "npx",
      env: {
        API_KEY: REDACTED_MCP_SECRET_VALUE,
        REGION: "eu-west-1",
      },
    });

    expect(result.env).toEqual({
      API_KEY: "gAAAAA-encrypted",
      REGION: "eu-west-1",
    });
  });

  it("returns the server unchanged when no env value is redacted", async () => {
    const server = {
      id: "stdio-0",
      type: "stdio" as const,
      name: "my-server",
      command: "npx",
      env: { API_KEY: "plaintext" },
    };

    const result = await substituteRedactedMcpCredentials(server);

    expect(result).toBe(server);
    expect(SettingsService.fetchSettingsFromApi).not.toHaveBeenCalled();
  });

  it("keeps the placeholder when the stored stdio entry is missing", async () => {
    vi.spyOn(SettingsService, "fetchSettingsFromApi").mockResolvedValue({
      agent_settings: { mcp_config: {} },
    } as unknown as SettingsApiResponse);

    const result = await substituteRedactedMcpCredentials({
      id: "stdio-0",
      type: "stdio",
      name: "new-name",
      command: "npx",
      env: { API_KEY: REDACTED_MCP_SECRET_VALUE },
    });

    expect(result.env).toEqual({ API_KEY: REDACTED_MCP_SECRET_VALUE });
  });

  it("replaces redacted OAuth state with the encrypted stored subtree", async () => {
    vi.spyOn(SettingsService, "fetchSettingsFromApi").mockResolvedValue({
      agent_settings: {
        mcp_config: {
          "superhuman-mail": {
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
        },
      },
    } as unknown as SettingsApiResponse);

    const result = await substituteRedactedMcpCredentials({
      id: "shttp-0",
      type: "shttp",
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
    });

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
});
