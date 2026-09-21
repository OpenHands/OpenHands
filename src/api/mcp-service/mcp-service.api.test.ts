import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MCPClient } from "@openhands/typescript-client/clients";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "../backend-registry/active-store";
import {
  getCloudMcpOAuthStatus,
  startCloudMcpOAuth,
} from "../cloud/mcp-service.api";
import SettingsService from "../settings-service/settings-service.api";
import McpService from "./mcp-service.api";
import type { MCPServerConfig } from "#/types/mcp-server";
import { REDACTED_MCP_SECRET_VALUE } from "#/utils/mcp-config";

vi.mock("@openhands/typescript-client/clients", () => ({
  MCPClient: vi.fn(),
}));

vi.mock("../cloud/mcp-service.api", () => ({
  testCloudMcpServer: vi.fn(),
  startCloudMcpOAuth: vi.fn(),
  getCloudMcpOAuthStatus: vi.fn(),
}));

const testServer = vi.fn();
const startOAuth = vi.fn();
const getOAuthStatus = vi.fn();
const submitOAuthCallback = vi.fn();
const close = vi.fn();

const encryptedAuth = "gAAAAAencrypted-auth";

describe("McpService.testServer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setRegisteredBackends([
      {
        id: "local",
        name: "Local",
        host: "http://127.0.0.1:8001",
        apiKey: "session-key",
        kind: "local",
      },
    ]);
    setActiveSelection({ backendId: "local", orgId: null });
    vi.mocked(MCPClient).mockImplementation(function MockMCPClient() {
      return {
        testServer,
        startOAuth,
        getOAuthStatus,
        submitOAuthCallback,
        close,
      } as unknown as MCPClient;
    } as unknown as typeof MCPClient);
    testServer.mockResolvedValue({ ok: true, tools: [] });
    startOAuth.mockResolvedValue({
      ok: true,
      job_id: "job-1",
      authorization_url: "https://auth.example/authorize",
    });
    getOAuthStatus.mockResolvedValue({
      ok: true,
      status: "succeeded",
      job_id: "job-1",
      tools: ["search_mail"],
    });
    submitOAuthCallback.mockResolvedValue({
      ok: true,
      status: "succeeded",
      job_id: "job-1",
      tools: ["search_mail"],
    });
  });

  it("tests stored remote MCP credentials as encrypted auth, not redacted text", async () => {
    vi.spyOn(SettingsService, "fetchSettingsFromApi").mockResolvedValue({
      llm_api_key_is_set: false,
      conversation_settings: {},
      agent_settings: {
        mcp_config: {
          linear: {
            url: "https://mcp.linear.app/mcp",
            transport: "http",
            auth: { strategy: "bearer", value: encryptedAuth },
          },
        },
      },
    });

    await McpService.testServer({
      id: "linear",
      type: "shttp",
      name: "linear",
      url: "https://mcp.linear.app/mcp",
      auth: { strategy: "bearer", value: REDACTED_MCP_SECRET_VALUE },
    });

    expect(SettingsService.fetchSettingsFromApi).toHaveBeenCalledWith(
      "encrypted",
    );
    expect(testServer).toHaveBeenCalledTimes(1);
    expect(testServer.mock.calls[0][0]).toMatchObject({
      name: "linear",
      server: {
        type: "http",
        url: "https://mcp.linear.app/mcp",
        auth: { strategy: "bearer", value: encryptedAuth },
      },
    });
    expect(testServer.mock.calls[0][0].server).not.toHaveProperty("api_key");
    expect(testServer.mock.calls[0][0].server).not.toHaveProperty("headers");
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("forwards explicit OAuth authentication metadata to the MCP test endpoint", async () => {
    await McpService.testServer({
      id: "shttp-0",
      type: "shttp",
      name: "superhuman-mail",
      url: "https://mcp.mail.superhuman.com/mcp",
      auth: {
        strategy: "oauth2",
        authentication: {
          type: "oauth",
          client_auth_method: "none",
        },
        state: {
          tokens: {
            access_token: "gAAAAexisting-access-token",
          },
        },
      },
    });

    expect(testServer).toHaveBeenCalledTimes(1);
    expect(testServer.mock.calls[0][0]).toMatchObject({
      name: "superhuman-mail",
      server: {
        type: "http",
        url: "https://mcp.mail.superhuman.com/mcp",
        auth: {
          strategy: "oauth2",
          authentication: {
            type: "oauth",
            client_auth_method: "none",
          },
          state: {
            tokens: {
              access_token: "gAAAAexisting-access-token",
            },
          },
        },
      },
      timeout: 120,
    });
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("returns OAuth state captured by the MCP test endpoint", async () => {
    testServer.mockResolvedValueOnce({
      ok: true,
      tools: ["search_mail"],
      oauth_state: {
        tokens: {
          access_token: "gAAAAencrypted-access-token",
        },
        token_expires_at: 12345,
      },
    });

    const result = await McpService.testServer({
      id: "shttp-0",
      type: "shttp",
      name: "superhuman-mail",
      url: "https://mcp.mail.superhuman.com/mcp",
      auth: {
        strategy: "oauth2",
        authentication: {
          type: "oauth",
          client_auth_method: "none",
        },
      },
    });

    expect(result.ok).toBe(true);
    if (!result.ok) throw new Error("expected successful MCP test");
    expect(result.oauth_state).toMatchObject({
      tokens: {
        access_token: "gAAAAencrypted-access-token",
      },
      token_expires_at: 12345,
    });
  });

  it("starts OAuth through the TypeScript MCP client", async () => {
    const result = await McpService.startOAuth({
      id: "shttp-0",
      type: "shttp",
      name: "superhuman-mail",
      url: "https://mcp.mail.superhuman.com/mcp",
      auth: {
        strategy: "oauth2",
        authentication: {
          type: "oauth",
          client_auth_method: "none",
        },
      },
    });

    expect(result.job_id).toBe("job-1");
    expect(startOAuth).toHaveBeenCalledTimes(1);
    expect(startOAuth.mock.calls[0][0]).toMatchObject({
      name: "superhuman-mail",
      server: {
        type: "http",
        url: "https://mcp.mail.superhuman.com/mcp",
        auth: {
          strategy: "oauth2",
          authentication: {
            type: "oauth",
            client_auth_method: "none",
          },
        },
      },
      timeout: 120,
    });
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("submits OAuth callback through the TypeScript MCP client", async () => {
    await McpService.submitOAuthCallback(
      "job/1",
      "http://localhost:1234/callback?code=abc",
    );

    expect(submitOAuthCallback).toHaveBeenCalledWith("job/1", {
      callback_url: "http://localhost:1234/callback?code=abc",
    });
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("gets OAuth status through the TypeScript MCP client", async () => {
    await McpService.getOAuthStatus("job/1");

    expect(getOAuthStatus).toHaveBeenCalledWith("job/1");
    expect(close).toHaveBeenCalledTimes(1);
  });
});

describe("McpService.authorizeOAuth on cloud backends", () => {
  const ROVO: MCPServerConfig = {
    id: "atlassian_rovo",
    type: "shttp",
    name: "Atlassian Rovo",
    url: "https://mcp.atlassian.com/v1/mcp/authv2",
    auth: {
      strategy: "oauth2",
      authentication: { type: "oauth", client_auth_method: "none" },
    },
  };
  const AUTHORIZATION_URL = "https://auth.atlassian.com/authorize?state=s";
  let popup: { close: ReturnType<typeof vi.fn>; location: { href: string } };

  beforeEach(() => {
    vi.clearAllMocks();
    setRegisteredBackends([
      {
        id: "cloud",
        name: "Cloud",
        host: "https://app.all-hands.dev",
        apiKey: "bearer-token",
        kind: "cloud",
      },
    ]);
    setActiveSelection({ backendId: "cloud", orgId: null });
    popup = { close: vi.fn(), location: { href: "" } };
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window);
    vi.mocked(startCloudMcpOAuth).mockReset();
    vi.mocked(getCloudMcpOAuthStatus).mockReset();
    vi.mocked(startCloudMcpOAuth).mockResolvedValue({
      ok: true,
      job_id: "job-1",
      authorization_url: AUTHORIZATION_URL,
    });
    vi.mocked(getCloudMcpOAuthStatus)
      .mockResolvedValueOnce({
        ok: true,
        status: "authorizing",
        job_id: "job-1",
        callback_ready: true,
      })
      .mockResolvedValue({
        ok: true,
        status: "succeeded",
        job_id: "job-1",
        callback_ready: true,
        tools: ["search"],
        oauth_state: { tokens: { access_token: "plain-access-token" } },
      });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("runs the OAuth flow through the app server instead of a local agent-server", async () => {
    vi.useFakeTimers();
    try {
      // Act
      const pending = McpService.authorizeOAuth(ROVO);
      await vi.advanceTimersByTimeAsync(1000);
      const result = await pending;

      // Assert
      expect(MCPClient).not.toHaveBeenCalled();
      expect(startCloudMcpOAuth).toHaveBeenCalledTimes(1);
      expect(vi.mocked(startCloudMcpOAuth).mock.calls[0][0]).toMatchObject({
        name: "atlassian_rovo",
        server: { type: "http", url: ROVO.url, auth: ROVO.auth },
        timeout: 120,
      });
      expect(popup.location.href).toBe(AUTHORIZATION_URL);
      expect(getCloudMcpOAuthStatus).toHaveBeenCalledWith("job-1");
      expect(popup.close).toHaveBeenCalledTimes(1);
      expect(result).toMatchObject({
        ok: true,
        tools: ["search"],
        oauth_state: { tokens: { access_token: "plain-access-token" } },
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it("closes the popup when the app server call fails", async () => {
    vi.mocked(startCloudMcpOAuth).mockRejectedValue(new Error("boom"));

    await expect(McpService.authorizeOAuth(ROVO)).rejects.toThrow("boom");

    expect(popup.close).toHaveBeenCalledTimes(1);
  });
});
