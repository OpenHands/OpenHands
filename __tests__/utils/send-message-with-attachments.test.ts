import { beforeEach, describe, expect, it, vi } from "vitest";
import { sendMessageWithAttachments } from "#/utils/send-message-with-attachments";

const mocks = vi.hoisted(() => ({
  partition: vi.fn(),
  validate: vi.fn(),
  convert: vi.fn(),
  resolveRuntime: vi.fn(),
  upload: vi.fn(),
  sendMessage: vi.fn(),
  toast: vi.fn(),
}));

vi.mock("#/components/features/chat/utils/chat-input.utils", () => ({
  partitionImagesForUpload: mocks.partition,
}));

vi.mock("#/utils/file-validation", () => ({
  validateFiles: mocks.validate,
}));

vi.mock("#/utils/convert-image-to-base-64", () => ({
  convertImageToBase64: mocks.convert,
}));

vi.mock("#/api/conversation-file-upload.api", () => ({
  resolveConversationRuntime: mocks.resolveRuntime,
  uploadFilesToConversation: mocks.upload,
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: { sendMessage: mocks.sendMessage },
  }),
);

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: mocks.toast,
}));

const translate = vi.fn((key: string) => `translated:${key}`);
const image = new File(["image"], "image.png", { type: "image/png" });
const imageFile = new File(["large image"], "large.png", {
  type: "image/png",
});
const documentFile = new File(["notes"], "notes.txt", {
  type: "text/plain",
});

function options() {
  return {
    conversationId: "conversation-1",
    content: "Please inspect these files",
    images: [image, imageFile],
    files: [documentFile],
    imagesMarkedUploadAsFile: ["large.png"],
    t: translate as never,
  };
}

describe("sendMessageWithAttachments", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.partition.mockReturnValue({
      imagesToEmbed: [image],
      imagesAsFiles: [imageFile],
    });
    mocks.validate.mockReturnValue({ isValid: true });
    mocks.convert.mockResolvedValue("data:image/png;base64,aW1hZ2U=");
    mocks.resolveRuntime.mockResolvedValue({ kind: "remote" });
    mocks.upload.mockResolvedValue({
      skipped_files: [{ reason: "large.png was skipped" }],
      uploaded_files: [
        "/workspace/notes.txt",
        "/workspace/large.png",
      ],
    });
    mocks.sendMessage.mockResolvedValue(undefined);
  });

  it("uploads files, embeds images, reports skips, and sends the augmented prompt", async () => {
    const result = await sendMessageWithAttachments(options());

    expect(mocks.partition).toHaveBeenCalledWith(
      [image, imageFile],
      ["large.png"],
    );
    expect(mocks.validate).toHaveBeenCalledWith([
      image,
      documentFile,
      imageFile,
    ]);
    expect(mocks.upload).toHaveBeenCalledWith("conversation-1", [
      documentFile,
      imageFile,
    ]);
    expect(mocks.toast).toHaveBeenCalledWith("large.png was skipped");
    expect(mocks.sendMessage).toHaveBeenCalledWith(
      "conversation-1",
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Please inspect these files\n\ntranslated:CHAT_INTERFACE$AUGMENTED_PROMPT_FILES_TITLE: /workspace/notes.txt\n\n/workspace/large.png",
          },
          {
            type: "image",
            image_urls: ["data:image/png;base64,aW1hZ2U="],
          },
        ],
      },
      { kind: "remote" },
    );
    expect(result).toMatchObject({
      text: "Please inspect these files",
      content:
        "Please inspect these files\n\ntranslated:CHAT_INTERFACE$AUGMENTED_PROMPT_FILES_TITLE: /workspace/notes.txt\n\n/workspace/large.png",
      imageUrls: ["data:image/png;base64,aW1hZ2U="],
      fileUrls: ["/workspace/notes.txt", "/workspace/large.png"],
    });
    expect(Number.isNaN(Date.parse(result.timestamp))).toBe(false);
  });

  it("sends plain text without calling the upload or image converters", async () => {
    mocks.partition.mockReturnValue({ imagesToEmbed: [], imagesAsFiles: [] });
    const plain = options();
    plain.images = [];
    plain.files = [];
    plain.imagesMarkedUploadAsFile = [];

    const result = await sendMessageWithAttachments(plain);

    expect(mocks.upload).not.toHaveBeenCalled();
    expect(mocks.convert).not.toHaveBeenCalled();
    expect(mocks.toast).not.toHaveBeenCalled();
    expect(mocks.sendMessage.mock.calls[0][1]).toEqual({
      role: "user",
      content: [{ type: "text", text: plain.content }],
    });
    expect(result.content).toBe(plain.content);
  });

  it("throws the validator's attachment error before resolving a runtime", async () => {
    mocks.validate.mockReturnValue({
      isValid: false,
      errorMessage: "Attachment is too large",
    });
    await expect(sendMessageWithAttachments(options())).rejects.toThrow(
      "Attachment is too large",
    );
    expect(mocks.resolveRuntime).not.toHaveBeenCalled();
  });

  it("uses a fallback error for invalid attachments without a message", async () => {
    mocks.validate.mockReturnValue({ isValid: false });
    await expect(sendMessageWithAttachments(options())).rejects.toThrow(
      "Invalid attachments",
    );
  });
});
