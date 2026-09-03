import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";

const { mockCondenseConversation, mockConversationClient } = vi.hoisted(() => ({
  mockCondenseConversation: vi.fn(),
  mockConversationClient: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", async () => {
  const actual = await vi.importActual<
    typeof import("@openhands/typescript-client/clients")
  >("@openhands/typescript-client/clients");
  return {
    ...actual,
    ConversationClient: mockConversationClient.mockImplementation(
      function ConversationClientMock() {
      return { condenseConversation: mockCondenseConversation };
      },
    ),
  };
});

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
    mockConversationClient.mockClear();
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  afterEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  it("routes cloud conversations through the runtime condense endpoint", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    await AgentServerConversationService.condenseConversation(
      "conv-1",
      RUNTIME_URL,
      "sess-key",
    );

    expect(mockConversationClient).toHaveBeenCalledWith(
      expect.objectContaining({
        host: "http://localhost:54928",
        apiKey: "sess-key",
      }),
    );
    expect(mockCondenseConversation).toHaveBeenCalledWith("conv-1");
  });

  it("requires a runtime URL for cloud conversations", async () => {
    setRegisteredBackends([cloudBackend, localBackend]);
    setActiveSelection({ backendId: cloudBackend.id });

    await expect(
      AgentServerConversationService.condenseConversation(
        "conv-1",
        null,
        "sess-key",
      ),
    ).rejects.toThrow("No backend is configured");

    expect(mockCondenseConversation).not.toHaveBeenCalled();
  });

  it("uses the ConversationClient directly on a local backend", async () => {
    await AgentServerConversationService.condenseConversation(
      "conv-1",
      RUNTIME_URL,
      "sess-key",
    );

    expect(mockCondenseConversation).toHaveBeenCalledWith("conv-1");
  });
});
