import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelRouterService from "#/api/conversation-service/model-router-service.api";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { ConversationClient } from "@openhands/typescript-client/clients";

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  patch: vi.fn(),
}));

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
        last_route_name: null,
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

  it("adds a saved LLM profile as a model-router route", async () => {
    transport.post.mockResolvedValue({
      data: {
        enabled: true,
        selection: { mode: "auto" },
        last_route_name: null,
        routes: [
          {
            name: "reviewer",
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

    expect(transport.post).toHaveBeenCalledWith(
      "/api/conversations/conv-1/model_router/routes",
      {
        profile_name: "Reviewer",
        route_name: "reviewer",
        base_route_name: "default",
      },
    );
    expect(status.routes).toHaveLength(1);
  });

  it("switches model-router selection through PATCH", async () => {
    transport.patch.mockResolvedValue({
      data: {
        enabled: true,
        selection: { mode: "manual", manual_route_name: "reviewer" },
        last_route_name: null,
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
      { selection: { mode: "manual", manual_route_name: "reviewer" } },
    );
    expect(status.selection?.manual_route_name).toBe("reviewer");
  });
});
