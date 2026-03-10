import { describe, expect, it, vi } from "vitest";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

const { mockGet } = vi.hoisted(() => ({ mockGet: vi.fn() }));
vi.mock("#/api/open-hands-axios", () => ({
  openHands: { get: mockGet },
}));

describe("V1ConversationService", () => {
  describe("readConversationFile", () => {
    it("uses default plan path when filePath is not provided", async () => {
      // Arrange
      const conversationId = "conv-123";
      mockGet.mockResolvedValue({ data: "# PLAN content" });

      // Act
      await V1ConversationService.readConversationFile(conversationId);

      // Assert
      expect(mockGet).toHaveBeenCalledTimes(1);
      const callUrl = mockGet.mock.calls[0][0] as string;
      expect(callUrl).toContain(
        "file_path=%2Fworkspace%2Fproject%2F.agents_tmp%2FPLAN.md",
      );
    });
  });
});
