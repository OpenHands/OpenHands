import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useChatAttachmentUpload } from "#/hooks/chat/use-chat-attachment-upload";

const store = vi.hoisted(() => ({
  addImages: vi.fn(),
  addFiles: vi.fn(),
  addFileLoading: vi.fn(),
  removeFileLoading: vi.fn(),
  addImageLoading: vi.fn(),
  removeImageLoading: vi.fn(),
  markImagesAsPasted: vi.fn(),
}));

vi.mock("#/stores/conversation-store", () => ({
  useConversationStore: () => ({
    images: [],
    files: [],
    ...store,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

const MB = 1024 * 1024;

function makeFile(name: string, size: number, type: string) {
  const file = new File([""], name, { type });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

describe("useChatAttachmentUpload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("routes images over the inline limit through file upload", async () => {
    const smallImage = makeFile("small.png", 2 * MB, "image/png");
    const largeImage = makeFile("large.png", 4 * MB, "image/png");
    const document = makeFile("design.pdf", 10 * MB, "application/pdf");
    const { result } = renderHook(() => useChatAttachmentUpload());

    await act(() =>
      result.current.handleUpload([smallImage, largeImage, document]),
    );

    expect(store.addImages).toHaveBeenCalledWith([smallImage]);
    expect(store.addFiles).toHaveBeenCalledWith([largeImage, document]);
    expect(store.markImagesAsPasted).toHaveBeenCalledWith(["small.png"]);
  });
});
