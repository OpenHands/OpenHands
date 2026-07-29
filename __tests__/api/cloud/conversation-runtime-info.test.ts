import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import {
  getFetchCall,
  getJsonBody,
  mockJsonResponse,
} from "./fetch-test-utils";

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const localBackend: Backend = {
  id: "self-hosted",
  name: "Self-hosted",
  host: "http://192.168.1.99:9999",
  apiKey: "local-key",
  kind: "local",
};

const runtimeResponse = {
  id: "conv-abc",
  title: "Test conversation",
  created_at: "2026-04-16T00:00:00Z",
  updated_at: "2026-04-16T00:00:00Z",
  execution_status: "idle",
  metrics: null,
  stats: {
    usage_to_metrics: {
      agent: {
        model_name: "test-model",
        accumulated_cost: 1.23,
        max_budget_per_task: null,
        accumulated_token_usage: null,
        costs: [],
        response_latencies: [],
        token_usages: [],
      },
    },
  },
};

const fetchMock = vi.fn();

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  fetchMock.mockReset();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("AgentServerConversationService.getRuntimeConversation", () => {
  describe("cloud mode", () => {
    beforeEach(() => {
      __resetActiveStoreForTests();
      setRegisteredBackends([cloudBackend]);
      setActiveSelection({ backendId: cloudBackend.id });
      fetchMock.mockReset();
      vi.stubGlobal("fetch", fetchMock);
    });

    it("routes through /api/cloud-proxy targeting the conversation runtime host", async () => {
      // Arrange
      fetchMock.mockResolvedValue(mockJsonResponse(runtimeResponse));
      const conversationUrl =
        "http://abc123.runtime.all-hands.dev/api/conversations/conv-abc";

      // Act
      const result =
        await AgentServerConversationService.getRuntimeConversation(
          "conv-abc",
          conversationUrl,
          "session-xyz",
        );

      // Assert
      expect(fetchMock).toHaveBeenCalledOnce();
      const [url, init] = getFetchCall(fetchMock);
      const body = getJsonBody(init);
      expect(url).toMatch(/\/api\/cloud-proxy$/);
      expect(body).toMatchObject({
        host: "http://abc123.runtime.all-hands.dev",
        method: "GET",
        path: "/api/conversations/conv-abc",
        headers: { "X-Session-API-Key": "session-xyz" },
      });
      expect(result.stats.usage_to_metrics.agent?.accumulated_cost).toBe(1.23);
    });
  });

  describe("local mode", () => {
    beforeEach(() => {
      __resetActiveStoreForTests();
      setRegisteredBackends([localBackend]);
      setActiveSelection({ backendId: localBackend.id });
      fetchMock.mockReset();
      vi.stubGlobal("fetch", fetchMock);
    });

    it("targets the conversation_url host (not the active backend host) and forwards X-Session-API-Key", async () => {
      // Arrange
      fetchMock.mockResolvedValue(mockJsonResponse(runtimeResponse));
      const conversationUrl =
        "http://192.168.1.42:8888/api/conversations/conv-abc";

      // Act
      const result =
        await AgentServerConversationService.getRuntimeConversation(
          "conv-abc",
          conversationUrl,
          "session-xyz",
        );

      // Assert
      expect(fetchMock).toHaveBeenCalledOnce();
      const [url, init] = getFetchCall(fetchMock);
      expect(url).toContain("192.168.1.42:8888");
      expect(url).not.toContain(localBackend.host);
      expect(init.headers).toMatchObject({
        "X-Session-API-Key": "session-xyz",
      });
      expect(result.id).toBe("conv-abc");
    });
  });
});
