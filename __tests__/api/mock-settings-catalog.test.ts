import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  OPENAI_SUBSCRIPTION_DEVICE_POLL_PATH,
  OPENAI_SUBSCRIPTION_DEVICE_START_PATH,
  OPENAI_SUBSCRIPTION_LOGOUT_PATH,
  OPENAI_SUBSCRIPTION_MODELS_PATH,
  OPENAI_SUBSCRIPTION_STATUS_PATH,
} from "#/constants/llm-subscription";
import {
  createMockWebClientConfig,
  resetTestHandlersMockSettings,
} from "#/mocks/settings-handlers";

const BASE_URL = "http://localhost:3000";

const fetchJson = async <T>(path: string, init?: RequestInit) => {
  const response = await fetch(`${BASE_URL}${path}`, init);
  expect(response.ok).toBe(true);
  return (await response.json()) as T;
};

beforeEach(() => {
  resetTestHandlersMockSettings();
});

describe("mock agent-server discovery", () => {
  it("builds a complete web-client config with optional overrides", () => {
    const defaults = createMockWebClientConfig();
    expect(defaults).toEqual({
      feature_flags: {
        hide_llm_settings: false,
        hide_users_page: false,
      },
      providers_configured: [],
      maintenance_start_time: null,
      recaptcha_site_key: null,
      faulty_models: [],
      error_message: null,
      updated_at: expect.any(String),
    });

    expect(
      createMockWebClientConfig({
        feature_flags: {
          hide_llm_settings: true,
          hide_users_page: false,
        },
        updated_at: "2030-01-01T00:00:00.000Z",
      }),
    ).toEqual({
      ...defaults,
      feature_flags: {
        hide_llm_settings: true,
        hide_users_page: false,
      },
      updated_at: "2030-01-01T00:00:00.000Z",
    });
  });

  it("publishes server metadata and the legacy model catalog", async () => {
    const serverInfo = await fetchJson<{
      version: string;
      usable_tools: string[];
      agents: string[];
      default_agent: string;
      models: string[];
      security_analyzers: string[];
    }>("/server_info");
    expect(serverInfo).toMatchObject({
      version: "1.36.1",
      usable_tools: [
        "terminal",
        "file_editor",
        "task_tracker",
        "browser_tool_set",
      ],
      agents: ["CodeActAgent"],
      default_agent: "CodeActAgent",
      security_analyzers: ["llm", "none"],
    });
    expect(serverInfo.models).toContain("openhands/glm-5.2");

    const options = await fetchJson<{
      models: string[];
      verified_models: string[];
      verified_providers: string[];
      default_model: string;
    }>("/api/options/models");
    expect(options.default_model).toBe("openai/gpt-5.6-sol");
    expect(options.models).toEqual(serverInfo.models);
    expect(options.verified_models).toEqual([
      "claude-opus-4-5-20251101",
      "claude-sonnet-4-5-20250929",
    ]);
    expect(options.verified_providers).toContain("openai");

    await expect(
      fetchJson<string[]>("/api/options/security-analyzers"),
    ).resolves.toEqual(["llm", "none"]);
  });

  it("publishes normalized model and provider endpoints", async () => {
    const modelCatalog = await fetchJson<{ models: string[] }>(
      "/api/llm/models",
    );
    expect(modelCatalog.models).toContain("openai/gpt-4o");

    const verified = await fetchJson<{
      models: Record<string, string[]>;
    }>("/api/llm/models/verified");
    expect(verified.models.openai).toEqual([
      "gpt-5.5",
      "gpt-5.6-sol",
      "gpt-6-astra",
    ]);
    expect(verified.models.openhands).toEqual([
      "claude-sonnet-4-5-20250929",
      "claude-opus-4-5-20251101",
      "deepseek-v4-flash",
      "glm-5.2",
    ]);

    const providers = await fetchJson<{ providers: string[] }>(
      "/api/llm/providers",
    );
    expect(providers.providers).toEqual([
      "anthropic",
      "openai",
      "openhands",
      "sambanova",
    ]);
  });

  it("searches providers case-insensitively and filters verification", async () => {
    const all = await fetchJson<{
      items: { name: string; verified: boolean }[];
      next_page_id: null;
    }>("/api/v1/config/providers/search");
    expect(all.next_page_id).toBeNull();
    expect(all.items).toEqual([
      { name: "anthropic", verified: true },
      { name: "openai", verified: true },
      { name: "openhands", verified: true },
      { name: "sambanova", verified: false },
    ]);

    const matching = await fetchJson<{ items: { name: string }[] }>(
      "/api/v1/config/providers/search?query=OPEN&verified__eq=true",
    );
    expect(matching.items.map(({ name }) => name)).toEqual([
      "openai",
      "openhands",
    ]);

    const unverified = await fetchJson<{
      items: { name: string; verified: boolean }[];
    }>("/api/v1/config/providers/search?verified__eq=false");
    expect(unverified.items).toEqual([{ name: "sambanova", verified: false }]);
  });

  it("searches models by provider, name, and verification", async () => {
    const all = await fetchJson<{
      items: {
        provider: string | null;
        name: string;
        verified: boolean;
        free: boolean;
        default: boolean;
      }[];
    }>("/api/v1/config/models/search");
    expect(all.items).toContainEqual({
      provider: "openai",
      name: "gpt-4o",
      verified: false,
      free: false,
      default: false,
    });
    expect(all.items).toContainEqual({
      provider: "anthropic",
      name: "claude-opus-4-8",
      verified: true,
      free: false,
      default: false,
    });

    const matching = await fetchJson<{ items: unknown[] }>(
      "/api/v1/config/models/search?provider__eq=anthropic&query=OPUS&verified__eq=true",
    );
    expect(matching.items).toEqual([
      {
        provider: "anthropic",
        name: "claude-opus-4-5-20251101",
        verified: true,
        free: false,
        default: false,
      },
      {
        provider: "anthropic",
        name: "claude-opus-4-8",
        verified: true,
        free: false,
        default: false,
      },
    ]);

    const unverified = await fetchJson<{ items: unknown[] }>(
      "/api/v1/config/models/search?provider__eq=openai&verified__eq=false",
    );
    expect(unverified.items).toEqual([
      {
        provider: "openai",
        name: "gpt-3.5-turbo",
        verified: false,
        free: false,
        default: false,
      },
      {
        provider: "openai",
        name: "gpt-4o",
        verified: false,
        free: false,
        default: false,
      },
      {
        provider: "openai",
        name: "gpt-4o-mini",
        verified: false,
        free: false,
        default: false,
      },
    ]);
  });

  it("publishes the web-client configuration endpoint", async () => {
    const config = await fetchJson<{
      feature_flags: Record<string, boolean>;
      updated_at: string;
    }>("/api/v1/web-client/config");
    expect(config).toMatchObject({
      feature_flags: {
        hide_llm_settings: false,
        hide_users_page: false,
      },
      providers_configured: [],
      faulty_models: [],
      updated_at: expect.any(String),
    });
  });

  it("falls back to deterministic settings when shared defaults omit optional sections", async () => {
    vi.resetModules();
    vi.doMock("#/services/settings", () => ({ DEFAULT_SETTINGS: {} }));

    try {
      const { MOCK_DEFAULT_USER_SETTINGS: settings } =
        await import("#/mocks/settings-handlers");

      expect(settings).toMatchObject({
        agent_settings: {
          llm: { model: "openhands/claude-opus-4-5-20251101" },
        },
        conversation_settings: {},
      });
    } finally {
      vi.doUnmock("#/services/settings");
      vi.resetModules();
    }
  });
});

describe("mock OpenAI subscription", () => {
  it("offers models and a deterministic device-login challenge", async () => {
    await expect(fetchJson(OPENAI_SUBSCRIPTION_MODELS_PATH)).resolves.toEqual({
      vendor: "openai",
      models: ["gpt-5.2", "gpt-5.3-codex"],
    });

    await expect(
      fetchJson(OPENAI_SUBSCRIPTION_DEVICE_START_PATH, { method: "POST" }),
    ).resolves.toEqual({
      device_code: "mock-device-code",
      user_code: "MOCK-CODE",
      verification_uri: "https://auth.openai.com/activate",
      verification_uri_complete:
        "https://auth.openai.com/activate?user_code=MOCK-CODE",
      interval: 1,
      expires_in: 900,
    });
  });

  it("moves through disconnected, connected, and logged-out states", async () => {
    await expect(fetchJson(OPENAI_SUBSCRIPTION_STATUS_PATH)).resolves.toEqual({
      connected: false,
      account_email: null,
      expires_at: null,
    });

    await expect(
      fetchJson(OPENAI_SUBSCRIPTION_DEVICE_POLL_PATH, { method: "POST" }),
    ).resolves.toEqual({
      connected: true,
      account_email: "mock-chatgpt@example.com",
      expires_at: null,
    });
    await expect(fetchJson(OPENAI_SUBSCRIPTION_STATUS_PATH)).resolves.toEqual({
      connected: true,
      account_email: "mock-chatgpt@example.com",
      expires_at: null,
    });

    await expect(
      fetchJson(OPENAI_SUBSCRIPTION_LOGOUT_PATH, { method: "POST" }),
    ).resolves.toEqual({ connected: false });
    await expect(fetchJson(OPENAI_SUBSCRIPTION_STATUS_PATH)).resolves.toEqual({
      connected: false,
      account_email: null,
      expires_at: null,
    });
  });
});
