import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MCPClient } from "@openhands/typescript-client/clients";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "../backend-registry/active-store";
import SettingsService from "../settings-service/settings-service.api";
import McpService from "./mcp-service.api";
import type { MCPServerConfig } from "#/types/mcp-server";
import { REDACTED_MCP_SECRET_VALUE } from "#/utils/mcp-config";

vi.mock("@openhands/typescript-client/clients", () => ({
  MCPClient: vi.fn(),
}));

const testServer = vi.fn();
const startOAuth = vi.fn();
const getOAuthStatus = vi.fn();
const submitOAuthCallback = vi.fn();
const close = vi.fn();

const encryptedAuth = "gAAAAAencrypted-auth";

const oauthServer = (): MCPServerConfig => ({
  id: "shttp-oauth",
  type: "shttp",
  name: "oauth-server",
  url: "https://mcp.example.com/mcp",
  auth: {
    strategy: "oauth2",
    authentication: {
      type: "oauth",
      client_auth_method: "none",
    },
  },
});

const popupWindow = () => ({
  close: vi.fn(),
  location: { href: "about:blank" },
});

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

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("tests stored remote MCP credentials as encrypted auth, not redacted text", async () => {
    vi.spyOn(SettingsService, "fetchSettingsFromApi").mockResolvedValue({
      llm_api_key_is_set: false,
      conversation_settings: {},
      agent_settings: {
        mcp_config: {
          "shttp-0": {
            url: "https://mcp.linear.app/mcp",
            transport: "http",
            auth: { strategy: "bearer", value: encryptedAuth },
          },
        },
      },
    });

    await McpService.testServer({
      id: "shttp-0",
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

  it("passes through a tool result when the server has no credential validator", async () => {
    const response = {
      ok: true as const,
      tools: ["ping"],
      tool_result: { is_error: true, text: "opaque server result" },
    };
    testServer.mockResolvedValueOnce(response);

    const result = await McpService.testServer({
      id: "shttp-generic",
      type: "shttp",
      url: "https://mcp.example.com/mcp",
    });

    expect(result).toEqual(response);
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

  it("omits empty stdio options and optional local credentials", async () => {
    setRegisteredBackends([
      {
        id: "local",
        name: "Local",
        host: "http://127.0.0.1:8001",
        apiKey: "",
        kind: "local",
      },
    ]);
    setActiveSelection({ backendId: "local", orgId: null });

    await McpService.testServer({
      id: "stdio-empty-options",
      type: "stdio",
      command: "node",
      args: [],
      env: {},
      timeout: 17,
    });

    expect(MCPClient).toHaveBeenCalledWith({
      host: "http://127.0.0.1:8001",
    });
    expect(testServer).toHaveBeenCalledWith({
      server: { type: "stdio", command: "node" },
      timeout: 17,
    });
  });

  it("accepts a stdio server with omitted args and omits an unset timeout", async () => {
    await McpService.testServer({
      id: "stdio-minimal",
      type: "stdio",
      command: "node",
    });

    const request = testServer.mock.calls[0][0];
    expect(request).toEqual({
      server: { type: "stdio", command: "node" },
    });
    expect(request).not.toHaveProperty("timeout");
  });

  it("maps SSE headers to the connectivity request", async () => {
    await McpService.testServer({
      id: "sse-with-headers",
      type: "sse",
      url: "https://mcp.example.com/events",
      headers: { Authorization: "Bearer secret" },
    });

    expect(testServer).toHaveBeenCalledWith({
      server: {
        type: "sse",
        url: "https://mcp.example.com/events",
        headers: { Authorization: "Bearer secret" },
      },
    });
  });

  it("omits empty remote headers from the connectivity request", async () => {
    await McpService.testServer({
      id: "shttp-empty-headers",
      type: "shttp",
      url: "https://mcp.example.com/mcp",
      headers: {},
    });

    expect(testServer).toHaveBeenCalledWith({
      server: {
        type: "http",
        url: "https://mcp.example.com/mcp",
      },
    });
  });

  it("uses the active local backend without an optional API key for OAuth probes", async () => {
    setRegisteredBackends([
      {
        id: "local",
        name: "Local",
        host: "http://127.0.0.1:8001",
        apiKey: "",
        kind: "local",
      },
    ]);
    setActiveSelection({ backendId: "local", orgId: null });

    await McpService.getOAuthStatus("job-without-key");

    expect(MCPClient).toHaveBeenCalledWith({
      host: "http://127.0.0.1:8001",
      timeout: 125_000,
    });
  });

  it("prefers the active local backend over another registered local backend", async () => {
    setRegisteredBackends([
      {
        id: "other-local",
        name: "Other local",
        host: "http://127.0.0.1:9000",
        apiKey: "other-key",
        kind: "local",
      },
      {
        id: "active-local",
        name: "Active local",
        host: "http://127.0.0.1:8002///",
        apiKey: "active-key",
        kind: "local",
      },
    ]);
    setActiveSelection({ backendId: "active-local", orgId: null });

    await McpService.getOAuthStatus("job-on-active-local");

    expect(MCPClient).toHaveBeenCalledWith({
      host: "http://127.0.0.1:8002",
      apiKey: "active-key",
      timeout: 125_000,
    });
  });

  it("passes the active local API key to connectivity tests", async () => {
    await McpService.testServer({
      id: "shttp-keyed",
      type: "shttp",
      url: "https://mcp.example.com/mcp",
    });

    expect(MCPClient).toHaveBeenCalledWith({
      host: "http://127.0.0.1:8001",
      apiKey: "session-key",
    });
  });

  it("uses a registered local backend when OAuth starts from a cloud session", async () => {
    setRegisteredBackends([
      {
        id: "cloud",
        name: "Cloud",
        host: "https://app.all-hands.dev",
        apiKey: "cloud-key",
        kind: "cloud",
      },
      {
        id: "local",
        name: "Local",
        host: "http://127.0.0.1:8001///",
        apiKey: "local-key",
        kind: "local",
      },
    ]);
    setActiveSelection({ backendId: "cloud", orgId: "org-1" });

    await McpService.startOAuth(oauthServer());

    expect(MCPClient).toHaveBeenCalledWith({
      host: "http://127.0.0.1:8001",
      apiKey: "local-key",
      timeout: 125_000,
    });
  });

  it("omits an empty fallback API key when probing OAuth from the cloud", async () => {
    setRegisteredBackends([
      {
        id: "cloud",
        name: "Cloud",
        host: "https://app.all-hands.dev",
        apiKey: "cloud-key",
        kind: "cloud",
      },
      {
        id: "local",
        name: "Local",
        host: "http://127.0.0.1:8001",
        apiKey: "",
        kind: "local",
      },
    ]);
    setActiveSelection({ backendId: "cloud", orgId: "org-1" });

    await McpService.getOAuthStatus("job-with-fallback");

    expect(MCPClient).toHaveBeenCalledWith({
      host: "http://127.0.0.1:8001",
      timeout: 125_000,
    });
  });

  it("rejects OAuth when no registered local backend is reachable", async () => {
    setRegisteredBackends([
      {
        id: "cloud",
        name: "Cloud",
        host: "https://app.all-hands.dev",
        apiKey: "cloud-key",
        kind: "cloud",
      },
      {
        id: "unreachable-local",
        name: "Unreachable local",
        host: "",
        apiKey: "",
        kind: "local",
      },
    ]);
    setActiveSelection({ backendId: "cloud", orgId: "org-1" });

    await expect(McpService.startOAuth(oauthServer())).rejects.toThrow(
      "OAuth authorization requires a reachable local backend.",
    );
    expect(MCPClient).not.toHaveBeenCalled();
  });

  it("returns a reported OAuth start failure and closes the popup", async () => {
    const popup = popupWindow();
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window);
    startOAuth.mockResolvedValueOnce({
      ok: false,
      error: "OAuth client registration failed",
      error_kind: "connection",
    });

    const result = await McpService.authorizeOAuth(oauthServer());

    expect(result).toEqual({
      ok: false,
      error: "OAuth client registration failed",
      error_kind: "connection",
    });
    expect(popup.close).toHaveBeenCalledOnce();
    expect(close).toHaveBeenCalledOnce();
  });

  it("uses default details when OAuth starts without a job", async () => {
    vi.spyOn(window, "open").mockReturnValue(null);
    startOAuth.mockResolvedValueOnce({
      ok: true,
      authorization_url: "https://auth.example/authorize",
    });

    const result = await McpService.authorizeOAuth(oauthServer());

    expect(result).toEqual({
      ok: false,
      error: "Could not start OAuth authorization",
      error_kind: "unknown",
    });
    expect(close).toHaveBeenCalledOnce();
  });

  it("rejects an OAuth start response without an authorization URL", async () => {
    const popup = popupWindow();
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window);
    startOAuth.mockResolvedValueOnce({ ok: true, job_id: "job-1" });

    const result = await McpService.authorizeOAuth(oauthServer());

    expect(result).toEqual({
      ok: false,
      error: "Could not start OAuth authorization",
      error_kind: "unknown",
    });
    expect(popup.close).toHaveBeenCalledOnce();
  });

  it("returns an immediately completed OAuth result with optional state", async () => {
    const popup = popupWindow();
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window);
    getOAuthStatus.mockResolvedValueOnce({
      ok: true,
      status: "succeeded",
      job_id: "job-1",
      tools: null,
      tool_result: null,
      oauth_state: null,
    });

    const result = await McpService.authorizeOAuth(oauthServer());

    expect(result).toEqual({
      ok: true,
      tools: [],
      tool_result: null,
      oauth_state: null,
    });
    expect(popup.close).toHaveBeenCalledOnce();
  });

  it("omits absent optional fields from an immediately completed OAuth result", async () => {
    const popup = popupWindow();
    const open = vi
      .spyOn(window, "open")
      .mockReturnValue(popup as unknown as Window);
    getOAuthStatus.mockResolvedValueOnce({
      ok: true,
      status: "succeeded",
      job_id: "job-1",
      tools: ["search"],
    });

    const result = await McpService.authorizeOAuth(oauthServer());

    expect(open).toHaveBeenCalledWith("about:blank", "_blank");
    expect(result).toEqual({ ok: true, tools: ["search"] });
    expect(result).not.toHaveProperty("tool_result");
    expect(result).not.toHaveProperty("oauth_state");
  });

  it("returns an immediately failed OAuth result with default details", async () => {
    vi.spyOn(window, "open").mockReturnValue(null);
    getOAuthStatus.mockResolvedValueOnce({
      ok: false,
      status: "failed",
      job_id: "job-1",
    });

    const result = await McpService.authorizeOAuth(oauthServer());

    expect(result).toEqual({
      ok: false,
      error: "OAuth authorization did not complete",
      error_kind: "unknown",
    });
  });

  it("opens the authorization URL and returns the completed tool result", async () => {
    vi.useFakeTimers();
    const popup = popupWindow();
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window);
    getOAuthStatus
      .mockResolvedValueOnce({
        ok: true,
        status: "authorizing",
        job_id: "job-1",
        callback_ready: true,
      })
      .mockResolvedValueOnce({
        ok: true,
        status: "succeeded",
        job_id: "job-1",
        tools: ["search"],
        tool_result: { is_error: false, text: '{"ok":true}' },
        oauth_state: { token_expires_at: 12_345 },
      });

    const authorization = McpService.authorizeOAuth(oauthServer());
    await vi.runAllTimersAsync();
    const result = await authorization;

    expect(popup.location.href).toBe("https://auth.example/authorize");
    expect(result).toEqual({
      ok: true,
      tools: ["search"],
      tool_result: { is_error: false, text: '{"ok":true}' },
      oauth_state: { token_expires_at: 12_345 },
    });
    expect(popup.close).toHaveBeenCalledOnce();
  });

  it("detects a failure while waiting for callback readiness", async () => {
    vi.useFakeTimers();
    const popup = popupWindow();
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window);
    getOAuthStatus
      .mockResolvedValueOnce({
        ok: true,
        status: "pending",
        job_id: "job-1",
        callback_ready: false,
      })
      .mockResolvedValueOnce({
        ok: false,
        status: "failed",
        job_id: "job-1",
        error: "Authorization was denied",
        error_kind: "credentials",
      });

    const authorization = McpService.authorizeOAuth(oauthServer());
    await vi.runAllTimersAsync();
    const result = await authorization;

    expect(result).toEqual({
      ok: false,
      error: "Authorization was denied",
      error_kind: "credentials",
    });
    expect(popup.close).toHaveBeenCalledOnce();
  });

  it("returns a failure discovered after the authorization popup opens", async () => {
    vi.useFakeTimers();
    const popup = popupWindow();
    vi.spyOn(window, "open").mockReturnValue(popup as unknown as Window);
    getOAuthStatus
      .mockResolvedValueOnce({
        ok: true,
        status: "authorizing",
        job_id: "job-1",
        callback_ready: true,
      })
      .mockResolvedValueOnce({
        ok: false,
        status: "failed",
        job_id: "job-1",
        error: "OAuth callback was rejected",
        error_kind: "credentials",
      });

    const authorization = McpService.authorizeOAuth(oauthServer());
    await vi.runAllTimersAsync();
    const result = await authorization;

    expect(result).toEqual({
      ok: false,
      error: "OAuth callback was rejected",
      error_kind: "credentials",
    });
    expect(popup.close).toHaveBeenCalledOnce();
  });

  it("completes authorization after popup blocking without trying to close a window", async () => {
    vi.useFakeTimers();
    vi.spyOn(window, "open").mockReturnValue(null);
    getOAuthStatus
      .mockResolvedValueOnce({
        ok: true,
        status: "authorizing",
        job_id: "job-1",
        callback_ready: true,
      })
      .mockResolvedValueOnce({
        ok: true,
        status: "succeeded",
        job_id: "job-1",
        tools: ["search"],
      });

    const authorization = McpService.authorizeOAuth(oauthServer());
    await vi.runAllTimersAsync();

    await expect(authorization).resolves.toEqual({
      ok: true,
      tools: ["search"],
    });
  });

  it("waits between callback-readiness status checks", async () => {
    vi.useFakeTimers();
    vi.spyOn(window, "open").mockReturnValue(null);
    getOAuthStatus
      .mockResolvedValueOnce({
        ok: true,
        status: "pending",
        job_id: "job-1",
        callback_ready: false,
      })
      .mockResolvedValueOnce({
        ok: false,
        status: "failed",
        job_id: "job-1",
        error: "Authorization was denied",
        error_kind: "credentials",
      });

    const authorization = McpService.authorizeOAuth(oauthServer());
    await vi.advanceTimersByTimeAsync(0);
    expect(getOAuthStatus).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(1);

    await vi.advanceTimersByTimeAsync(249);
    expect(getOAuthStatus).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(1);

    await vi.advanceTimersByTimeAsync(1);
    await expect(authorization).resolves.toEqual({
      ok: false,
      error: "Authorization was denied",
      error_kind: "credentials",
    });
    expect(getOAuthStatus).toHaveBeenCalledTimes(2);
  });

  it("times out when OAuth never becomes ready or completes", async () => {
    vi.useFakeTimers();
    vi.spyOn(window, "open").mockReturnValue(null);
    getOAuthStatus.mockResolvedValue({
      ok: true,
      status: "pending",
      job_id: "job-1",
      callback_ready: false,
    });

    const authorization = McpService.authorizeOAuth(oauthServer());
    await vi.runAllTimersAsync();
    const result = await authorization;

    expect(result).toEqual({
      ok: false,
      error: "OAuth authorization timed out",
      error_kind: "timeout",
    });
    expect(getOAuthStatus).toHaveBeenCalledTimes(141);
    expect(close).toHaveBeenCalledOnce();
  });
});
