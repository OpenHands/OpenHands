import { describe, expect, it, vi, beforeEach, afterEach, Mock } from "vitest";
import axios from "axios";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";

const { mockGet } = vi.hoisted(() => ({ mockGet: vi.fn() }));
vi.mock("#/api/open-hands-axios", () => ({
  openHands: { get: mockGet },
}));

vi.mock("axios");

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

  describe("uploadFile", () => {
    beforeEach(() => {
      vi.clearAllMocks();
      (axios.post as Mock).mockResolvedValue({ data: {} });
    });

    afterEach(() => {
      vi.resetAllMocks();
    });

    it("uses query params for file upload path", async () => {
      // Arrange
      const conversationUrl = "http://localhost:54928/api/conversations/conv-123";
      const sessionApiKey = "test-api-key";
      const file = new File(["test content"], "test.txt", { type: "text/plain" });
      const uploadPath = "/workspace/custom/path.txt";

      // Act
      await V1ConversationService.uploadFile(
        conversationUrl,
        sessionApiKey,
        file,
        uploadPath,
      );

      // Assert
      expect(axios.post).toHaveBeenCalledTimes(1);
      const callUrl = (axios.post as Mock).mock.calls[0][0] as string;

      // Verify URL uses query params format
      expect(callUrl).toContain("/api/file/upload?");
      expect(callUrl).toContain("path=%2Fworkspace%2Fcustom%2Fpath.txt");

      // Verify it's NOT using path params format
      expect(callUrl).not.toContain("/api/file/upload/%2F");
    });

    it("uses default workspace path when no path provided", async () => {
      // Arrange
      const conversationUrl = "http://localhost:54928/api/conversations/conv-123";
      const sessionApiKey = "test-api-key";
      const file = new File(["test content"], "myfile.txt", { type: "text/plain" });

      // Act
      await V1ConversationService.uploadFile(
        conversationUrl,
        sessionApiKey,
        file,
      );

      // Assert
      expect(axios.post).toHaveBeenCalledTimes(1);
      const callUrl = (axios.post as Mock).mock.calls[0][0] as string;

      // Default path should be /workspace/{filename}
      expect(callUrl).toContain("path=%2Fworkspace%2Fmyfile.txt");
    });

    it("sends file as FormData with correct headers", async () => {
      // Arrange
      const conversationUrl = "http://localhost:54928/api/conversations/conv-123";
      const sessionApiKey = "test-api-key";
      const file = new File(["test content"], "test.txt", { type: "text/plain" });

      // Act
      await V1ConversationService.uploadFile(
        conversationUrl,
        sessionApiKey,
        file,
      );

      // Assert
      expect(axios.post).toHaveBeenCalledTimes(1);
      const callArgs = (axios.post as Mock).mock.calls[0];

      // Verify FormData is sent
      const formData = callArgs[1];
      expect(formData).toBeInstanceOf(FormData);
      expect(formData.get("file")).toBe(file);

      // Verify headers include session API key and content type
      const headers = callArgs[2].headers;
      expect(headers).toHaveProperty("X-Session-API-Key", sessionApiKey);
      expect(headers).toHaveProperty("Content-Type", "multipart/form-data");
    });
  });
});
