import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getComposerBucket,
  useConversationStore,
} from "#/stores/conversation-store";

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
const OTHER_ID = "conv-test-2";

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
      byConversation: {},
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
      expect(
        getComposerBucket(useConversationStore.getState(), CONV_ID)
          .imagesMarkedUploadAsFile,
      ).toEqual([]);

      useConversationStore
        .getState()
        .toggleImageUploadAsFile(CONV_ID, "paste.png");
      expect(
        getComposerBucket(useConversationStore.getState(), CONV_ID)
          .imagesMarkedUploadAsFile,
      ).toEqual(["paste.png"]);

      useConversationStore
        .getState()
        .toggleImageUploadAsFile(CONV_ID, "paste.png");
      expect(
        getComposerBucket(useConversationStore.getState(), CONV_ID)
          .imagesMarkedUploadAsFile,
      ).toEqual([]);
    });

    it("clears marks when an image is removed", () => {
      const image = new File(["x"], "paste.png", { type: "image/png" });
      useConversationStore.getState().addImages(CONV_ID, [image]);
      useConversationStore
        .getState()
        .toggleImageUploadAsFile(CONV_ID, "paste.png");
      useConversationStore.getState().removeImage(CONV_ID, 0);

      expect(
        getComposerBucket(useConversationStore.getState(), CONV_ID)
          .imagesMarkedUploadAsFile,
      ).toEqual([]);
    });

    it("is reset by clearAllFiles", () => {
      useConversationStore
        .getState()
        .toggleImageUploadAsFile(CONV_ID, "paste.png");
      useConversationStore.getState().clearAllFiles(CONV_ID);
      expect(
        getComposerBucket(useConversationStore.getState(), CONV_ID)
          .imagesMarkedUploadAsFile,
      ).toEqual([]);
    });
  });

  describe("pastedImageNames", () => {
    it("tracks attached image names for the upload-as-file control", () => {
      useConversationStore.getState().markImagesAsPasted(CONV_ID, ["shot.png"]);
      expect(
        getComposerBucket(useConversationStore.getState(), CONV_ID)
          .pastedImageNames,
      ).toEqual(["shot.png"]);
    });

    it("clears pasted names when the image is removed", () => {
      const image = new File(["x"], "shot.png", { type: "image/png" });
      useConversationStore.getState().addImages(CONV_ID, [image]);
      useConversationStore.getState().markImagesAsPasted(CONV_ID, ["shot.png"]);
      useConversationStore.getState().removeImage(CONV_ID, 0);
      expect(
        getComposerBucket(useConversationStore.getState(), CONV_ID)
          .pastedImageNames,
      ).toEqual([]);
    });
  });

  describe("composer isolation", () => {
    it("keeps attachments and programmatic messages per conversation", () => {
      const primaryImage = new File(["a"], "primary.png", {
        type: "image/png",
      });
      const popoutImage = new File(["b"], "popout.png", { type: "image/png" });

      useConversationStore.getState().addImages(CONV_ID, [primaryImage]);
      useConversationStore.getState().addImages(OTHER_ID, [popoutImage]);
      useConversationStore.getState().setMessageToSend(CONV_ID, "primary draft");
      useConversationStore.getState().setMessageToSend(OTHER_ID, "popout draft");

      useConversationStore.getState().clearAllFiles(OTHER_ID);
      useConversationStore.getState().clearMessageToSend(OTHER_ID);

      const primary = getComposerBucket(
        useConversationStore.getState(),
        CONV_ID,
      );
      const other = getComposerBucket(
        useConversationStore.getState(),
        OTHER_ID,
      );

      expect(primary.images.map((file) => file.name)).toEqual(["primary.png"]);
      expect(primary.messageToSend?.text).toBe("primary draft");
      expect(other.images).toEqual([]);
      expect(other.messageToSend).toBeNull();
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
