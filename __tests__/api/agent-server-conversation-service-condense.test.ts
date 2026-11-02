import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ConversationClient } from "@openhands/typescript-client/clients";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { callCloudProxy } from "#/api/cloud/proxy";
import { buildHttpBaseUrl } from "#/utils/websocket-url";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";

const { mockCondenseConversation } = vi.hoisted(() => ({
  mockCondenseConversation: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", async () => {
  const actual = await vi.importActual<
    typeof import("@openhands/typescript-client/clients")
  >("@openhands/typescript-client/clients");
  return {
    ...actual,
    ConversationClient: vi.fn(function ConversationClientMock() {
      return { condenseConversation: mockCondenseConversation };
    }),
  };
});

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: vi.fn(),
}));

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

const localBackend: Backend = {
  id: "local",
  name: "Local",
  host: "http://127.0.0.1:8001",
  apiKey: "",
  kind: "local",
};

const RUNTIME_URL = "http://localhost:54928/api/conversations/conv-1";

describe("AgentServerConversationService.condenseConversation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCondenseConversation.mockReset().mockResolvedValue(undefined);
    vi.mocked(callCloudProxy).mockReset().mockResolvedValue(undefined);
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  afterEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  it("routes cloud conversations straight to the runtime via ConversationClient", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    await AgentServerConversationService.condenseConversation(
      "conv-1",
      RUNTIME_URL,
      "sess-key",
    );

    // Cloud now calls the runtime host directly (CORS allowlisted), not
    // the /api/cloud-proxy envelope.
    expect(callCloudProxy).not.toHaveBeenCalled();
    expect(ConversationClient).toHaveBeenCalledWith(
      expect.objectContaining({
        host: buildHttpBaseUrl(RUNTIME_URL),
        apiKey: "sess-key",
      }),
    );
    expect(mockCondenseConversation).toHaveBeenCalledWith("conv-1");
  });

  it("throws when a cloud conversation has no runtime URL to call", async () => {
    // Symmetric with every other cloud runtime call: a missing conversation
    // URL on a cloud backend is a caller bug, surfaced with a specific
    // message rather than the generic NoBackendAvailableError.
    setRegisteredBackends([cloudBackend, localBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    await expect(
      AgentServerConversationService.condenseConversation(
        "conv-1",
        null,
        "sess-key",
      ),
    ).rejects.toThrow(/requires a conversation URL on cloud backends/);
    expect(callCloudProxy).not.toHaveBeenCalled();
    expect(mockCondenseConversation).not.toHaveBeenCalled();
  });

  it("uses the ConversationClient directly on a local backend", async () => {
    await AgentServerConversationService.condenseConversation(
      "conv-1",
      RUNTIME_URL,
      "sess-key",
    );

    expect(mockCondenseConversation).toHaveBeenCalledWith("conv-1");
    expect(ConversationClient).toHaveBeenCalledWith(
      expect.objectContaining({
        host: buildHttpBaseUrl(RUNTIME_URL),
        apiKey: "sess-key",
      }),
    );
    expect(callCloudProxy).not.toHaveBeenCalled();
  });
});
