import { beforeEach, describe, expect, it, vi } from "vitest";
import { useConversationStore } from "#/stores/conversation-store";

const defaultConversationState = {
  selectedTab: "files" as const,
  unpinnedTabs: [] as string[],
  conversationMode: "code" as const,
};

const mockGetConversationState = vi.fn(
  (_id: string) => defaultConversationState,
);
const mockSetConversationState = vi.fn();

vi.mock("#/utils/conversation-local-storage", () => ({
  getConversationState: (id: string) => mockGetConversationState(id),
  setConversationState: (id: string, updates: object) =>
    mockSetConversationState(id, updates),
}));

const CONV_ID = "conv-test-1";

describe("conversation store", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetConversationState.mockReturnValue(defaultConversationState);
    Object.defineProperty(window, "location", {
      value: { pathname: `/conversations/${CONV_ID}` },
      writable: true,
    });
    useConversationStore.setState({
      conversationMode: "code",
      planContent: null,
      subConversationTaskId: null,
      shouldHideSuggestions: false,
      imagesMarkedUploadAsFile: [],
      pastedImageNames: [],
    });
  });

  describe("setConversationMode", () => {
    it("updates store state and persists via setConversationState when conversation ID is in location", () => {
      useConversationStore.getState().setConversationMode("plan");

      expect(useConversationStore.getState().conversationMode).toBe("plan");
      expect(mockSetConversationState).toHaveBeenCalledWith(CONV_ID, {
        conversationMode: "plan",
      });
    });
  });

  describe("imagesMarkedUploadAsFile", () => {
    it("toggles per-image upload-as-file marks by file name", () => {
      expect(useConversationStore.getState().imagesMarkedUploadAsFile).toEqual(
        [],
      );

      useConversationStore.getState().toggleImageUploadAsFile("paste.png");
      expect(useConversationStore.getState().imagesMarkedUploadAsFile).toEqual(
        ["paste.png"],
      );

      useConversationStore.getState().toggleImageUploadAsFile("paste.png");
      expect(useConversationStore.getState().imagesMarkedUploadAsFile).toEqual(
        [],
      );
    });

    it("clears marks when an image is removed", () => {
      const image = new File(["x"], "paste.png", { type: "image/png" });
      useConversationStore.getState().addImages([image]);
      useConversationStore.getState().toggleImageUploadAsFile("paste.png");
      useConversationStore.getState().removeImage(0);

      expect(useConversationStore.getState().imagesMarkedUploadAsFile).toEqual(
        [],
      );
    });

    it("is reset by clearAllFiles", () => {
      useConversationStore.getState().toggleImageUploadAsFile("paste.png");
      useConversationStore.getState().clearAllFiles();
      expect(useConversationStore.getState().imagesMarkedUploadAsFile).toEqual(
        [],
      );
    });
  });

  describe("pastedImageNames", () => {
    it("tracks attached image names for the upload-as-file control", () => {
      useConversationStore.getState().markImagesAsPasted(["shot.png"]);
      expect(useConversationStore.getState().pastedImageNames).toEqual([
        "shot.png",
      ]);
    });

    it("clears pasted names when the image is removed", () => {
      const image = new File(["x"], "shot.png", { type: "image/png" });
      useConversationStore.getState().addImages([image]);
      useConversationStore.getState().markImagesAsPasted(["shot.png"]);
      useConversationStore.getState().removeImage(0);
      expect(useConversationStore.getState().pastedImageNames).toEqual([]);
    });
  });

  describe("resetConversationState", () => {
    it("sets conversationMode from getConversationState", () => {
      useConversationStore.setState({ conversationMode: "plan" });
      mockGetConversationState.mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "code",
      });

      useConversationStore.getState().resetConversationState();

      expect(useConversationStore.getState().conversationMode).toBe("code");
      expect(mockGetConversationState).toHaveBeenCalledWith(CONV_ID);
    });
  });
});
