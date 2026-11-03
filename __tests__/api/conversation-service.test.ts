import { describe, expect, it, vi, beforeEach } from "vitest";

import ConversationService from "#/api/conversation-service/conversation-service.api";

const { getVSCodeUrlMock, vscodeClientMock } = vi.hoisted(() => {
  const getVSCodeUrlMock = vi.fn();
  return {
    getVSCodeUrlMock,
    vscodeClientMock: vi.fn(function VSCodeClientMock() {
      return { getUrl: getVSCodeUrlMock };
    }),
  };
});

vi.mock("@openhands/typescript-client/clients", () => ({
  VSCodeClient: vscodeClientMock,
}));

describe("ConversationService", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    ConversationService.setCurrentConversation(null);
  });

  describe("getVSCodeUrl", () => {
    it("targets the current conversation runtime URL and session key", async () => {
      ConversationService.setCurrentConversation({
        id: "conv-1",
        conversation_url:
          "https://runtime.example.com/api/conversations/conv-1",
        session_api_key: "runtime-session-key",
        workspace: { working_dir: "/workspace/project/conv-1" },
      } as never);
      getVSCodeUrlMock.mockResolvedValue("https://runtime.example.com/vscode");

      await expect(ConversationService.getVSCodeUrl("conv-1")).resolves.toEqual(
        { vscode_url: "https://runtime.example.com/vscode" },
      );

      expect(vscodeClientMock).toHaveBeenCalledWith(
        expect.objectContaining({
          host: "https://runtime.example.com",
          apiKey: "runtime-session-key",
        }),
      );
    });
  });
});
