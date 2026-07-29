import { beforeEach, describe, expect, it } from "vitest";
import {
  consumePendingTaskAttachments,
  setPendingTaskAttachments,
  usePendingTaskAttachmentsStore,
} from "#/stores/pending-task-attachments-store";

describe("pending-task-attachments-store", () => {
  beforeEach(() => {
    usePendingTaskAttachmentsStore.setState({ byTaskId: {} });
  });

  it("stores and consumes attachments by task id", () => {
    const image = new File(["x"], "shot.png", { type: "image/png" });

    setPendingTaskAttachments("task-uuid", {
      content: "see this",
      images: [image],
      files: [],
      imagesMarkedUploadAsFile: [],
    });

    const consumed = consumePendingTaskAttachments("task-uuid");
    expect(consumed?.content).toBe("see this");
    expect(consumed?.images).toHaveLength(1);
    expect(consumePendingTaskAttachments("task-uuid")).toBeNull();
  });
});
