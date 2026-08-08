import { beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationClient } from "@openhands/typescript-client/clients";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { condenseConversation } from "./conversation-mutation-utils";

const { mockCondenseConversation } = vi.hoisted(() => ({
  mockCondenseConversation: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", () => ({
  ConversationClient: vi.fn(function ConversationClientMock() {
    return { condenseConversation: mockCondenseConversation };
  }),
}));

describe("condenseConversation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(
      AgentServerConversationService,
      "batchGetAppConversations",
    ).mockResolvedValue([
      {
        id: "conv-1",
        conversation_url: "https://runtime.example.test",
        session_api_key: "session-key",
        sandbox_id: "sandbox-1",
      } as never,
    ]);
    mockCondenseConversation.mockResolvedValue({ success: true });
  });

  // @spec SC-005 — Conversation condensation
  it("uses the active conversation runtime credentials and typed client", async () => {
    await condenseConversation("conv-1");

    expect(ConversationClient).toHaveBeenCalledWith(
      expect.objectContaining({
        host: "https://runtime.example.test",
        apiKey: "session-key",
      }),
    );
    expect(mockCondenseConversation).toHaveBeenCalledWith("conv-1");
  });
});
