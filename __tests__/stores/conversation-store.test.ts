import { beforeEach, describe, expect, it, vi } from "vitest";

let useConversationStore: (typeof import("#/stores/conversation-store"))["useConversationStore"];

const defaultConversationState: {
  selectedTab: "files";
  unpinnedTabs: string[];
  conversationMode: "code" | "plan" | "deep-plan";
  deepPlan?: {
    activePhase: string | null;
    confirmed: string[];
    documents: Record<string, string>;
  };
} = {
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
  beforeEach(async () => {
    vi.clearAllMocks();
    mockGetConversationState.mockReturnValue(defaultConversationState);
    Object.defineProperty(window, "location", {
      value: { pathname: `/conversations/${CONV_ID}` },
      writable: true,
    });
    vi.resetModules();
    ({ useConversationStore } = await import("#/stores/conversation-store"));
  });

  it("starts with the complete transient state reset", () => {
    expect(useConversationStore.getState()).toMatchObject({
      isRightPanelShown: false,
      selectedTab: "files",
      images: [],
      files: [],
      imagesMarkedUploadAsFile: [],
      pastedImageNames: [],
      loadingFiles: [],
      loadingImages: [],
      messageToSend: null,
      messageRestoreIfEmpty: null,
      shouldShownAgentLoading: false,
      submittedMessage: null,
      shouldHideSuggestions: false,
      hasRightPanelToggled: false,
      planContent: null,
      conversationMode: "code",
      subConversationTaskId: null,
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
      expect(useConversationStore.getState().imagesMarkedUploadAsFile).toEqual([
        "paste.png",
      ]);

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

  describe("deep plan", () => {
    it("starts on the first phase, switches mode, and persists the machine", () => {
      useConversationStore.getState().startDeepPlan();

      expect(useConversationStore.getState().conversationMode).toBe(
        "deep-plan",
      );
      expect(useConversationStore.getState().deepPlan.activePhase).toBe(
        "analysis",
      );
      expect(mockSetConversationState).toHaveBeenCalledWith(CONV_ID, {
        deepPlan: expect.objectContaining({ activePhase: "analysis" }),
        conversationMode: "deep-plan",
      });
    });

    it("persists the mode alongside the machine so a refresh stays in deep plan", () => {
      // `startDeepPlan` flips `conversationMode` in memory. The route's
      // mount-time reset re-derives the mode from storage, so unless the mode
      // is persisted too, a refresh lands the user back in `code` mode with a
      // restored machine they can no longer see or use.
      useConversationStore.getState().startDeepPlan();

      expect(mockSetConversationState).toHaveBeenCalledWith(
        CONV_ID,
        expect.objectContaining({ conversationMode: "deep-plan" }),
      );

      mockGetConversationState.mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "deep-plan",
        deepPlan: {
          activePhase: "analysis",
          confirmed: [],
          documents: {},
        },
      });
      useConversationStore.getState().resetConversationState();

      expect(useConversationStore.getState().conversationMode).toBe(
        "deep-plan",
      );
    });

    it("clears the mode when the machine is reset, without re-persisting it", () => {
      // `resetDeepPlan` shares the persistence helper. It must not inherit the
      // "persist deep-plan" behavior, or cancelling would immediately re-arm
      // the mode it just cleared.
      useConversationStore.getState().startDeepPlan();
      mockSetConversationState.mockClear();
      mockGetConversationState.mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "code",
      });

      useConversationStore.getState().resetDeepPlan();

      expect(useConversationStore.getState().conversationMode).toBe("code");
      expect(useConversationStore.getState().deepPlan.activePhase).toBeNull();
      expect(mockSetConversationState).toHaveBeenCalledWith(CONV_ID, {
        deepPlan: expect.objectContaining({ activePhase: null }),
      });
    });

    it("keeps an in-progress chain when Deep Plan is entered again", () => {
      // Shift+Tab, the context menu and `/deep-plan` all call `startDeepPlan`.
      // Re-entering must not wipe confirmations and documents the user already
      // built, or an accidental second entry silently destroys the chain.
      const store = useConversationStore.getState();
      store.startDeepPlan();
      store.setDeepPlanDocument("requirements", "## 3.1 Authentication\n");
      store.confirmDeepPlanPhase("analysis");
      store.confirmDeepPlanPhase("requirements");
      expect(useConversationStore.getState().deepPlan.activePhase).toBe(
        "database",
      );

      useConversationStore.getState().startDeepPlan();

      expect(useConversationStore.getState().deepPlan.activePhase).toBe(
        "database",
      );
      expect(useConversationStore.getState().deepPlan.confirmed).toEqual([
        "analysis",
        "requirements",
      ]);
      expect(
        useConversationStore.getState().deepPlan.documents.requirements,
      ).toBe("## 3.1 Authentication\n");
      expect(useConversationStore.getState().conversationMode).toBe(
        "deep-plan",
      );
    });

    it("starts a fresh chain when there is no chain yet", () => {
      expect(useConversationStore.getState().deepPlan.activePhase).toBeNull();

      useConversationStore.getState().startDeepPlan();

      expect(useConversationStore.getState().deepPlan.activePhase).toBe(
        "analysis",
      );
    });

    it("blocks entering a later phase until its predecessors are confirmed", () => {
      useConversationStore.getState().startDeepPlan();
      useConversationStore.getState().setDeepPlanPhase("database");

      expect(useConversationStore.getState().deepPlan.activePhase).toBe(
        "analysis",
      );
    });

    it("advances the phase when the reference chain is valid", () => {
      const store = useConversationStore.getState();
      store.startDeepPlan();
      store.setDeepPlanDocument("requirements", "## 3.1 Authentication\n");
      store.confirmDeepPlanPhase("analysis");

      const result = useConversationStore
        .getState()
        .confirmDeepPlanPhase("requirements");

      expect(result).toEqual({ ok: true });
      expect(useConversationStore.getState().deepPlan.activePhase).toBe(
        "database",
      );
    });

    it("refuses a checkpoint whose chain cites a section that does not exist", () => {
      const store = useConversationStore.getState();
      store.startDeepPlan();
      store.setDeepPlanDocument("requirements", "## 3.1 Authentication\n");
      store.confirmDeepPlanPhase("analysis");
      store.confirmDeepPlanPhase("requirements");
      // Requirements are confirmed; the database document now cites a section
      // that no requirement defines.
      store.setDeepPlanDocument("database", "## 2.1 Users [Req 9.9]\n");

      const result = useConversationStore
        .getState()
        .confirmDeepPlanPhase("database");

      expect(result.ok).toBe(false);
      expect(result.error).toContain("[Req 9.9]");
      expect(useConversationStore.getState().deepPlan.activePhase).toBe(
        "database",
      );
    });

    it("drops confirmations built on a document that is rewritten", () => {
      // A checkpoint vouches for the chain it saw. Rewriting an upstream
      // document must re-open the checkpoints that depend on it, or the chain
      // would keep reporting "confirmed" for documents it never re-validated.
      const store = useConversationStore.getState();
      store.startDeepPlan();
      store.setDeepPlanDocument("requirements", "## 3.1 Authentication\n");
      store.confirmDeepPlanPhase("analysis");
      store.confirmDeepPlanPhase("requirements");
      store.setDeepPlanDocument("database", "## 2.1 Users [Req 3.1]\n");
      store.confirmDeepPlanPhase("database");
      expect(useConversationStore.getState().deepPlan.activePhase).toBe(
        "backend",
      );

      // The agent rewrites requirements, changing a section the database cites.
      useConversationStore
        .getState()
        .setDeepPlanDocument("requirements", "## 3.2 Sessions\n");

      expect(useConversationStore.getState().deepPlan.confirmed).toEqual([
        "analysis",
      ]);
      expect(useConversationStore.getState().deepPlan.activePhase).toBe(
        "requirements",
      );
    });

    it("keeps confirmations when replay re-hydrates an unchanged document", () => {
      // Reload replays the same writes from history; identical bytes are not an
      // edit, so the confirmations restored from disk must survive.
      const store = useConversationStore.getState();
      store.startDeepPlan();
      store.setDeepPlanDocument("requirements", "## 3.1 Authentication\n");
      store.confirmDeepPlanPhase("analysis");
      store.confirmDeepPlanPhase("requirements");

      useConversationStore
        .getState()
        .setDeepPlanDocument("requirements", "## 3.1 Authentication\n");

      expect(useConversationStore.getState().deepPlan.confirmed).toEqual([
        "analysis",
        "requirements",
      ]);
    });

    it("restores the phase machine from persisted state on load", async () => {
      mockGetConversationState.mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "deep-plan",
        deepPlan: {
          activePhase: "backend",
          confirmed: ["analysis", "requirements", "database"],
          documents: {},
        },
      });
      vi.resetModules();
      const { useConversationStore: freshStore } =
        await import("#/stores/conversation-store");

      expect(freshStore.getState().deepPlan.activePhase).toBe("backend");
      expect(freshStore.getState().conversationMode).toBe("deep-plan");
    });

    it("keeps the persisted phase across the mount-time reset", () => {
      // The conversation route calls `resetConversationState()` on every
      // mount. If that reset writes the empty machine, a refresh drops the
      // user back to phase 1 even though the confirmations are on disk.
      useConversationStore.getState().startDeepPlan();
      mockGetConversationState.mockReturnValue({
        selectedTab: "files",
        unpinnedTabs: [],
        conversationMode: "deep-plan",
        deepPlan: {
          activePhase: "backend",
          confirmed: ["analysis", "requirements", "database"],
          documents: { requirements: "## 1.1 Login\n" },
        },
      });

      useConversationStore.getState().resetConversationState();

      expect(useConversationStore.getState().deepPlan.activePhase).toBe(
        "backend",
      );
      expect(useConversationStore.getState().deepPlan.confirmed).toEqual([
        "analysis",
        "requirements",
        "database",
      ]);
    });
  });

  it("updates the panel, tab, loading, and suggestion flags", () => {
    const store = useConversationStore.getState();
    store.setIsRightPanelShown(true);
    store.setSelectedTab("terminal");
    store.setShouldShownAgentLoading(true);
    store.setShouldHideSuggestions(true);
    store.setHasRightPanelToggled(true);

    expect(useConversationStore.getState()).toMatchObject({
      isRightPanelShown: true,
      selectedTab: "terminal",
      shouldShownAgentLoading: true,
      shouldHideSuggestions: true,
      hasRightPanelToggled: true,
    });
  });

  it("appends and removes files and images while preserving unrelated entries", () => {
    const firstImage = new File(["a"], "first.png", { type: "image/png" });
    const secondImage = new File(["b"], "second.png", { type: "image/png" });
    const firstFile = new File(["a"], "first.txt");
    const secondFile = new File(["b"], "second.txt");
    const store = useConversationStore.getState();

    store.addImages([firstImage]);
    store.addImages([secondImage]);
    store.addFiles([firstFile]);
    store.addFiles([secondFile]);
    store.markImagesAsPasted(["first.png", "first.png", "second.png"]);
    store.toggleImageUploadAsFile("first.png");
    store.toggleImageUploadAsFile("second.png");
    store.removeImage(99);
    expect(useConversationStore.getState()).toMatchObject({
      images: [firstImage, secondImage],
      imagesMarkedUploadAsFile: ["first.png", "second.png"],
      pastedImageNames: ["first.png", "second.png"],
    });

    store.removeImage(0);
    store.removeFile(0);
    expect(useConversationStore.getState()).toMatchObject({
      images: [secondImage],
      files: [secondFile],
      imagesMarkedUploadAsFile: ["second.png"],
      pastedImageNames: ["second.png"],
    });

    store.removeFile(99);
    expect(useConversationStore.getState().files).toEqual([secondFile]);
    store.clearImages();
    store.clearFiles();
    expect(useConversationStore.getState()).toMatchObject({
      images: [],
      files: [],
    });
  });

  it("deduplicates and removes file and image loading indicators", () => {
    const store = useConversationStore.getState();
    store.addFileLoading("one.txt");
    store.addFileLoading("one.txt");
    store.addFileLoading("two.txt");
    store.addImageLoading("one.png");
    store.addImageLoading("one.png");
    store.addImageLoading("two.png");

    expect(useConversationStore.getState()).toMatchObject({
      loadingFiles: ["one.txt", "two.txt"],
      loadingImages: ["one.png", "two.png"],
    });

    store.removeFileLoading("one.txt");
    store.removeImageLoading("one.png");
    expect(useConversationStore.getState()).toMatchObject({
      loadingFiles: ["two.txt"],
      loadingImages: ["two.png"],
    });

    store.clearAllLoading();
    expect(useConversationStore.getState()).toMatchObject({
      loadingFiles: [],
      loadingImages: [],
    });
  });

  it("clears every attachment and loading collection together", () => {
    const image = new File(["a"], "one.png", { type: "image/png" });
    const file = new File(["a"], "one.txt");
    const store = useConversationStore.getState();
    store.addImages([image]);
    store.addFiles([file]);
    store.toggleImageUploadAsFile("one.png");
    store.markImagesAsPasted(["one.png"]);
    store.addFileLoading("one.txt");
    store.addImageLoading("one.png");

    store.clearAllFiles();

    expect(useConversationStore.getState()).toMatchObject({
      images: [],
      files: [],
      imagesMarkedUploadAsFile: [],
      pastedImageNames: [],
      loadingFiles: [],
      loadingImages: [],
    });
  });

  it("creates and consumes timestamped send and restore requests", () => {
    vi.spyOn(Date, "now").mockReturnValueOnce(100).mockReturnValueOnce(200);
    const store = useConversationStore.getState();
    store.setMessageToSend("send me");
    store.restoreMessageToInputIfEmpty("restore me");
    expect(useConversationStore.getState()).toMatchObject({
      messageToSend: { text: "send me", timestamp: 100 },
      messageRestoreIfEmpty: { text: "restore me", timestamp: 200 },
    });

    store.clearMessageToSend();
    store.clearMessageRestoreIfEmpty();
    expect(useConversationStore.getState()).toMatchObject({
      messageToSend: null,
      messageRestoreIfEmpty: null,
    });
  });

  it("updates submitted, task, plan, and conversation-mode state without a route id", () => {
    Object.defineProperty(window, "location", {
      value: { pathname: "/" },
      writable: true,
    });
    const store = useConversationStore.getState();
    store.setSubmittedMessage("submitted");
    store.setSubConversationTaskId("task-1");
    store.setPlanContent("# Plan");
    store.setConversationMode("plan");

    expect(useConversationStore.getState()).toMatchObject({
      submittedMessage: "submitted",
      subConversationTaskId: "task-1",
      planContent: "# Plan",
      conversationMode: "plan",
    });
    expect(mockSetConversationState).not.toHaveBeenCalled();

    store.setSubmittedMessage(null);
    store.setSubConversationTaskId(null);
    store.setPlanContent(null);
    expect(useConversationStore.getState()).toMatchObject({
      submittedMessage: null,
      subConversationTaskId: null,
      planContent: null,
    });
  });

  it("resets transient conversation planning state from persisted mode", () => {
    mockGetConversationState.mockReturnValue({
      selectedTab: "files",
      unpinnedTabs: [],
      conversationMode: "plan",
    });
    useConversationStore.setState({
      shouldHideSuggestions: true,
      subConversationTaskId: "task",
      planContent: "plan",
    });

    useConversationStore.getState().resetConversationState();

    expect(useConversationStore.getState()).toMatchObject({
      shouldHideSuggestions: false,
      conversationMode: "plan",
      subConversationTaskId: null,
      planContent: null,
    });
  });

  it.each([
    [
      "setIsRightPanelShown",
      () => useConversationStore.getState().setIsRightPanelShown(true),
    ],
    ["clearImages", () => useConversationStore.getState().clearImages()],
    ["clearAllFiles", () => useConversationStore.getState().clearAllFiles()],
    [
      "clearAllLoading",
      () => useConversationStore.getState().clearAllLoading(),
    ],
    [
      "setMessageToSend",
      () => useConversationStore.getState().setMessageToSend("message"),
    ],
    [
      "clearMessageToSend",
      () => useConversationStore.getState().clearMessageToSend(),
    ],
    [
      "setSubmittedMessage",
      () => useConversationStore.getState().setSubmittedMessage("message"),
    ],
    [
      "resetConversationState",
      () => useConversationStore.getState().resetConversationState(),
    ],
  ])("%s preserves unrelated store state and actions", (_name, update) => {
    useConversationStore.setState({ selectedTab: "planner" });

    update();

    expect(useConversationStore.getState().selectedTab).toBe("planner");
    expect(useConversationStore.getState().setSelectedTab).toEqual(
      expect.any(Function),
    );
  });

  it("initializes safely without window and keeps mode changes in memory", async () => {
    vi.resetModules();
    vi.stubGlobal("window", undefined);

    const serverModule = await import("#/stores/conversation-store");
    expect(serverModule.useConversationStore.getState().conversationMode).toBe(
      "code",
    );
    serverModule.useConversationStore.getState().setConversationMode("plan");
    expect(serverModule.useConversationStore.getState().conversationMode).toBe(
      "plan",
    );
    expect(mockSetConversationState).not.toHaveBeenCalled();

    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it("initializes code mode without reading persistence when the route has no id", async () => {
    Object.defineProperty(window, "location", {
      value: { pathname: "/" },
      writable: true,
    });
    mockGetConversationState.mockClear();
    vi.resetModules();

    const freshModule = await import("#/stores/conversation-store");

    expect(freshModule.useConversationStore.getState().conversationMode).toBe(
      "code",
    );
    expect(mockGetConversationState).not.toHaveBeenCalled();
  });

  it("initializes a fresh module from persisted conversation mode", async () => {
    mockGetConversationState.mockReturnValue({
      selectedTab: "files",
      unpinnedTabs: [],
      conversationMode: "plan",
    });
    Object.defineProperty(window, "location", {
      value: { pathname: "/conversations/loaded-id/details" },
      writable: true,
    });
    vi.resetModules();

    const loadedModule = await import("#/stores/conversation-store");

    expect(loadedModule.useConversationStore.getState().conversationMode).toBe(
      "plan",
    );
    expect(mockGetConversationState).toHaveBeenCalledWith("loaded-id");
    vi.resetModules();
  });
});
