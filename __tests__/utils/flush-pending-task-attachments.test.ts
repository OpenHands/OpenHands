import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  setPendingTaskAttachments,
  usePendingTaskAttachmentsStore,
} from "#/stores/pending-task-attachments-store";
import { flushPendingTaskAttachments } from "#/utils/flush-pending-task-attachments";

const sendMessageWithAttachments = vi.fn();

vi.mock("#/utils/send-message-with-attachments", () => ({
  sendMessageWithAttachments: (...args: unknown[]) =>
    sendMessageWithAttachments(...args),
}));

vi.mock("#/i18n", () => ({
  default: { getFixedT: () => (key: string) => key },
  OPENHANDS_I18N_NAMESPACE: "openhands",
  waitForI18n: vi.fn().mockResolvedValue(undefined),
}));

describe("flushPendingTaskAttachments", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    usePendingTaskAttachmentsStore.setState({ byTaskId: {} });

    sendMessageWithAttachments.mockResolvedValue({
      text: "hello",
      content: "hello",
      imageUrls: ["data:image/png;base64,abc"],
      fileUrls: [],
      timestamp: "2020-01-01T00:00:00.000Z",
    });
  });

  it("sends queued attachments to the real conversation id", async () => {
    const image = new File(["x"], "shot.png", { type: "image/png" });
    setPendingTaskAttachments("start-task-id", {
      content: "hello",
      images: [image],
      files: [],
      imagesMarkedUploadAsFile: [],
    });

    await flushPendingTaskAttachments(
      "start-task-id",
      "550e8400-e29b-41d4-a716-446655440000",
    );

    expect(sendMessageWithAttachments).toHaveBeenCalledWith(
      expect.objectContaining({
        conversationId: "550e8400-e29b-41d4-a716-446655440000",
        content: "hello",
      }),
    );
  });
  it("retains files and the prompt when delivery fails", async () => {
    const file = new File(["notes"], "notes.txt");
    setPendingTaskAttachments("failed", {
      content: "Read this",
      images: [],
      files: [file],
      imagesMarkedUploadAsFile: [],
    });
    sendMessageWithAttachments.mockRejectedValueOnce(
      new Error("Upload failed"),
    );
    await expect(
      flushPendingTaskAttachments("failed", "conversation-1"),
    ).rejects.toThrow("Upload failed");
    expect(usePendingTaskAttachmentsStore.getState().byTaskId.failed).toEqual(
      expect.objectContaining({ content: "Read this", files: [file] }),
    );
  });

  it("sends only once when background delivery and the open page see READY together", async () => {
    setPendingTaskAttachments("shared", {
      content: "hello",
      images: [],
      files: [],
      imagesMarkedUploadAsFile: [],
    });
    await Promise.all([
      flushPendingTaskAttachments("shared", "conversation-1"),
      flushPendingTaskAttachments("shared", "conversation-1"),
    ]);
    expect(sendMessageWithAttachments).toHaveBeenCalledTimes(1);
    expect(
      usePendingTaskAttachmentsStore.getState().byTaskId.shared,
    ).toBeUndefined();
  });
});
