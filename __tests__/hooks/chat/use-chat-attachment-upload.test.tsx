import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useChatAttachmentUpload } from "#/hooks/chat/use-chat-attachment-upload";

const {
  validateFilesMock,
  processFilesMock,
  processImagesMock,
  displayErrorToastMock,
  store,
} = vi.hoisted(() => ({
  validateFilesMock: vi.fn(),
  processFilesMock: vi.fn(),
  processImagesMock: vi.fn(),
  displayErrorToastMock: vi.fn(),
  store: {
    images: [] as File[],
    files: [] as File[],
    addImages: vi.fn(),
    addFiles: vi.fn(),
    addFileLoading: vi.fn(),
    removeFileLoading: vi.fn(),
    addImageLoading: vi.fn(),
    removeImageLoading: vi.fn(),
    markImagesAsPasted: vi.fn(),
  },
}));

vi.mock("#/utils/file-validation", () => ({
  validateFiles: (...args: unknown[]) => validateFilesMock(...args),
}));

vi.mock("#/utils/file-processing", () => ({
  processFiles: (...args: unknown[]) => processFilesMock(...args),
  processImages: (...args: unknown[]) => processImagesMock(...args),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: (...args: unknown[]) => displayErrorToastMock(...args),
}));

vi.mock("#/stores/conversation-store", () => ({
  useConversationStore: () => store,
}));

const createFile = (name: string, type: string) =>
  new File([`${name} content`], name, { type });

const setupMocks = () => {
  vi.clearAllMocks();
  store.images = [];
  store.files = [];
  validateFilesMock.mockReturnValue({ isValid: true });
  processFilesMock.mockResolvedValue({ successful: [], failed: [] });
  processImagesMock.mockResolvedValue({ successful: [], failed: [] });
};

describe("chat attachment upload", () => {
  it("rejects invalid selections against all existing attachments", async () => {
    setupMocks();
    const existingImage = createFile("existing.png", "image/png");
    const existingFile = createFile("existing.txt", "text/plain");
    const selected = createFile("too-large.txt", "text/plain");
    store.images = [existingImage];
    store.files = [existingFile];
    validateFilesMock.mockReturnValue({
      isValid: false,
      errorMessage: "Total size exceeds 3MB",
    });
    const { result } = renderHook(() => useChatAttachmentUpload());

    await act(async () => result.current.handleUpload([selected]));

    expect(validateFilesMock).toHaveBeenCalledWith(
      [selected],
      [existingImage, existingFile],
    );
    expect(displayErrorToastMock).toHaveBeenCalledWith(
      "Error: Total size exceeds 3MB",
    );
    expect(processFilesMock).not.toHaveBeenCalled();
    expect(processImagesMock).not.toHaveBeenCalled();
    expect(store.addFileLoading).not.toHaveBeenCalled();
  });

  it("processes mixed files and images and publishes successful attachments", async () => {
    setupMocks();
    const textFile = createFile("notes.txt", "text/plain");
    const image = createFile("diagram.png", "image/png");
    processFilesMock.mockResolvedValue({
      successful: [textFile],
      failed: [],
    });
    processImagesMock.mockResolvedValue({
      successful: [image],
      failed: [],
    });
    const { result } = renderHook(() => useChatAttachmentUpload());

    await act(async () =>
      result.current.handleUpload([textFile, image], { fromPaste: true }),
    );

    expect(store.markImagesAsPasted).toHaveBeenCalledWith(["diagram.png"]);
    expect(store.addFileLoading).toHaveBeenCalledWith("notes.txt");
    expect(store.addImageLoading).toHaveBeenCalledWith("diagram.png");
    expect(processFilesMock).toHaveBeenCalledWith([textFile]);
    expect(processImagesMock).toHaveBeenCalledWith([image]);
    expect(store.addFiles).toHaveBeenCalledWith([textFile]);
    expect(store.addImages).toHaveBeenCalledWith([image]);
    expect(store.removeFileLoading).toHaveBeenCalledWith("notes.txt");
    expect(store.removeImageLoading).toHaveBeenCalledWith("diagram.png");
    expect(displayErrorToastMock).not.toHaveBeenCalled();
  });

  it("reports individual file and image processing failures", async () => {
    setupMocks();
    const textFile = createFile("broken.txt", "text/plain");
    const image = createFile("broken.png", "image/png");
    processFilesMock.mockResolvedValue({
      successful: [],
      failed: [{ file: textFile, error: new Error("cannot read text") }],
    });
    processImagesMock.mockResolvedValue({
      successful: [],
      failed: [{ file: image, error: new Error("cannot decode image") }],
    });
    const { result } = renderHook(() => useChatAttachmentUpload());

    await act(async () => result.current.handleUpload([textFile, image]));

    expect(store.addFiles).not.toHaveBeenCalled();
    expect(store.addImages).not.toHaveBeenCalled();
    expect(store.removeFileLoading).toHaveBeenCalledWith("broken.txt");
    expect(store.removeImageLoading).toHaveBeenCalledWith("broken.png");
    expect(displayErrorToastMock).toHaveBeenCalledWith(
      "Failed to process file broken.txt: cannot read text",
    );
    expect(displayErrorToastMock).toHaveBeenCalledWith(
      "Failed to process image broken.png: cannot decode image",
    );
  });

  it("cleans every loading marker when processing rejects unexpectedly", async () => {
    setupMocks();
    const textFile = createFile("notes.txt", "text/plain");
    const image = createFile("diagram.png", "image/png");
    processFilesMock.mockRejectedValue(new Error("worker crashed"));
    const { result } = renderHook(() => useChatAttachmentUpload());

    await act(async () => result.current.handleUpload([textFile, image]));

    expect(store.removeFileLoading).toHaveBeenCalledWith("notes.txt");
    expect(store.removeImageLoading).toHaveBeenCalledWith("diagram.png");
    expect(displayErrorToastMock).toHaveBeenCalledWith(
      "An unexpected error occurred while processing files",
    );
  });

  it("handles a regular-file-only selection without marking pasted images", async () => {
    setupMocks();
    const textFile = createFile("notes.txt", "text/plain");
    processFilesMock.mockResolvedValue({ successful: [textFile], failed: [] });
    const { result } = renderHook(() => useChatAttachmentUpload());

    await act(async () => result.current.handleUpload([textFile]));

    expect(store.markImagesAsPasted).not.toHaveBeenCalled();
    expect(processFilesMock).toHaveBeenCalledWith([textFile]);
    expect(processImagesMock).toHaveBeenCalledWith([]);
    expect(store.addFiles).toHaveBeenCalledWith([textFile]);
  });

  it("validates against the latest attachments after rerender", async () => {
    setupMocks();
    const selected = createFile("selected.txt", "text/plain");
    const latestExisting = createFile("latest.txt", "text/plain");
    const { result, rerender } = renderHook(() => useChatAttachmentUpload());

    store.files = [latestExisting];
    rerender();
    await act(async () => result.current.handleUpload([selected]));

    expect(validateFilesMock).toHaveBeenCalledWith(
      [selected],
      [latestExisting],
    );
  });
});
