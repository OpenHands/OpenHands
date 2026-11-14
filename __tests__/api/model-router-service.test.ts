import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelRouterService from "#/api/conversation-service/model-router-service.api";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { assertSubscriptionAuthReady } from "#/api/agent-server-adapter";
import { ConversationClient } from "@openhands/typescript-client/clients";

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  patch: vi.fn(),
}));

vi.mock("uuid", () => ({ v4: () => "uuid-1" }));

vi.mock("@openhands/typescript-client/clients", () => ({
  ConversationClient: vi.fn(function ConversationClientMock(
    this: Record<string, unknown>,
    options: { host: string; apiKey?: string },
  ) {
    Object.assign(this, {
      host: options.host,
      apiKey: options.apiKey,
      client: transport,
    });
  }),
}));

vi.mock("#/api/agent-server-client-options", () => ({
  getAgentServerClientOptions: vi.fn((options = {}) => ({
    host: options.conversationUrl ?? "http://local-agent-server",
    apiKey: options.sessionApiKey ?? "local-session-key",
    timeout: 60000,
  })),
}));

vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  default: { getProfile: vi.fn() },
}));

vi.mock("#/api/agent-server-adapter", () => ({
  assertSubscriptionAuthReady: vi.fn(),
}));

describe("ModelRouterService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    transport.get.mockReset();
    transport.post.mockReset();
    transport.patch.mockReset();
  });

  it("reads model-router status through the conversation runtime client", async () => {
    transport.get.mockResolvedValue({
      data: {
        enabled: false,
        selection: null,
        last_route_key: null,
        routes: [],
      },
    });

    const status = await ModelRouterService.getModelRouter(
      "conv-1",
      "https://runtime.example",
      "session-key",
    );

    expect(getAgentServerClientOptions).toHaveBeenCalledWith({
      conversationUrl: "https://runtime.example",
      sessionApiKey: "session-key",
    });
    expect(ConversationClient).toHaveBeenCalledWith({
      host: "https://runtime.example",
      apiKey: "session-key",
      timeout: 60000,
    });
    expect(transport.get).toHaveBeenCalledWith(
      "/api/conversations/conv-1/model_router",
    );
    expect(status.enabled).toBe(false);
  });

  it("adds an encrypted LLM profile as a model-router route", async () => {
    vi.mocked(ProfilesService.getProfile).mockResolvedValue({
      config: {
        model: "openrouter/example-model",
        api_key: "encrypted-key",
        custom_llm_provider: "openrouter",
      },
    } as never);
    transport.post.mockResolvedValue({
      data: {
        enabled: true,
        selection: { mode: "auto" },
        last_route_key: null,
        routes: [
          {
            key: "reviewer",
            model: "openrouter/example-model",
            label: "reviewer",
          },
        ],
      },
    });

    const status = await ModelRouterService.addProfileRoute(
      "conv-1",
      null,
      null,
      "Reviewer",
      "reviewer",
    );

    expect(ProfilesService.getProfile).toHaveBeenCalledWith(
      "Reviewer",
      "encrypted",
    );
    expect(assertSubscriptionAuthReady).toHaveBeenCalledWith({
      llm: {
        model: "openrouter/example-model",
        api_key: "encrypted-key",
        custom_llm_provider: "openrouter",
      },
    });
    expect(transport.post).toHaveBeenCalledWith(
      "/api/conversations/conv-1/model_router/routes",
      {
        route_key: "reviewer",
        base_route_key: "default",
        llm: {
          model: "openrouter/example-model",
          api_key: "encrypted-key",
          custom_llm_provider: "openrouter",
          stream: true,
          usage_id: "profile:Reviewer:uuid-1",
        },
      },
    );
    expect(status.routes).toHaveLength(1);
  });

  it("switches model-router selection through PATCH", async () => {
    transport.patch.mockResolvedValue({
      data: {
        enabled: true,
        selection: { mode: "manual", manual_route_key: "reviewer" },
        last_route_key: null,
        routes: [],
      },
    });

    const status = await ModelRouterService.switchRoute(
      "conv-1",
      null,
      null,
      "reviewer",
    );

    expect(transport.patch).toHaveBeenCalledWith(
      "/api/conversations/conv-1/model_router",
      { selection: { mode: "manual", manual_route_key: "reviewer" } },
    );
    expect(status.selection?.manual_route_key).toBe("reviewer");
  });
});
