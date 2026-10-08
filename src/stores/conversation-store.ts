import { create } from "zustand";
import { devtools } from "zustand/middleware";
import {
  getConversationState,
  setConversationState,
} from "#/utils/conversation-local-storage";
import {
  EMPTY_DEEP_PLAN_STATE,
  canEnterPhase,
  confirmPhase,
  hashDeepPlanDocument,
  hydrateDeepPlanState,
  invalidateFrom,
  startDeepPlan as createDeepPlanState,
  toPersistedDeepPlan,
  type ConfirmFailure,
  type DeepPlanState,
} from "#/utils/deep-plan-machine";
import type { DeepPlanPhaseId } from "#/utils/deep-plan";

export type ConversationTab =
  | "files"
  | "commits"
  | "browser"
  | "terminal"
  | "planner"
  | "tasklist"
  | "usage";

export type ConversationMode = "code" | "plan" | "deep-plan";

export type CommitsPaneSection = "uncommitted";

export interface IMessageToSend {
  text: string;
  timestamp: number;
}

interface ConversationState {
  isRightPanelShown: boolean;
  isOverviewPanelShown: boolean;
  isOverviewPanelPeeked: boolean;
  selectedTab: ConversationTab | null;
  commitsAutoExpandSection: CommitsPaneSection | null;
  images: File[];
  files: File[];
  /** Image file names (e.g. pasted screenshots) to send via file upload instead of vision embed. */
  imagesMarkedUploadAsFile: string[];
  /** Image file names attached in chat (controls per-image upload-as-file UI). */
  pastedImageNames: string[];
  loadingFiles: string[]; // File names currently being processed
  loadingImages: string[]; // Image names currently being processed
  messageToSend: IMessageToSend | null;
  /** One-shot restore request consumed by the chat input when empty. */
  messageRestoreIfEmpty: IMessageToSend | null;
  shouldShownAgentLoading: boolean;
  submittedMessage: string | null;
  shouldHideSuggestions: boolean; // New state to hide suggestions when input expands
  hasRightPanelToggled: boolean;
  planContent: string | null;
  conversationMode: ConversationMode;
  /** Deep Planning phase machine state; the store is its only writer. */
  deepPlan: DeepPlanState;
  subConversationTaskId: string | null; // Task ID for cloud sub-conversation creation
  localPlanningConversationId: string | null;
  /**
   * Deep Planning phase the current local planner was created for. The planner
   * is pinned to one document at creation (its `plan_path`), so switching phase
   * requires a new planner; this records which phase the live one owns so the
   * switch can be detected. `null` for a plain `plan`-mode planner.
   */
  deepPlanPlannerPhase: DeepPlanPhaseId | null;
}

interface ConversationActions {
  setIsRightPanelShown: (isRightPanelShown: boolean) => void;
  setIsOverviewPanelShown: (isOverviewPanelShown: boolean) => void;
  setIsOverviewPanelPeeked: (isOverviewPanelPeeked: boolean) => void;
  setSelectedTab: (selectedTab: ConversationTab | null) => void;
  setCommitsAutoExpandSection: (
    commitsAutoExpandSection: CommitsPaneSection | null,
  ) => void;
  setShouldShownAgentLoading: (shouldShownAgentLoading: boolean) => void;
  setShouldHideSuggestions: (shouldHideSuggestions: boolean) => void;
  addImages: (images: File[]) => void;
  addFiles: (files: File[]) => void;
  toggleImageUploadAsFile: (fileName: string) => void;
  markImagesAsPasted: (fileNames: string[]) => void;
  removeImage: (index: number) => void;
  removeFile: (index: number) => void;
  clearImages: () => void;
  clearFiles: () => void;
  clearAllFiles: () => void;
  addFileLoading: (fileName: string) => void;
  removeFileLoading: (fileName: string) => void;
  addImageLoading: (imageName: string) => void;
  removeImageLoading: (imageName: string) => void;
  clearAllLoading: () => void;
  setMessageToSend: (text: string) => void;
  clearMessageToSend: () => void;
  restoreMessageToInputIfEmpty: (text: string) => void;
  clearMessageRestoreIfEmpty: () => void;
  setSubmittedMessage: (message: string | null) => void;
  resetConversationState: () => void;
  setHasRightPanelToggled: (hasRightPanelToggled: boolean) => void;
  setConversationMode: (conversationMode: ConversationMode) => void;
  setSubConversationTaskId: (taskId: string | null) => void;
  setLocalPlanningConversationId: (
    conversationId: string | null,
    deepPlanPlannerPhase?: DeepPlanPhaseId | null,
  ) => void;
  setPlanContent: (planContent: string | null) => void;
  startDeepPlan: () => void;
  setDeepPlanPhase: (phase: DeepPlanPhaseId) => void;
  setDeepPlanDocument: (phase: DeepPlanPhaseId, content: string) => void;
  /** Record that a phase document's reload re-read was rejected (file gone). */
  failDeepPlanDocumentRestore: (phase: DeepPlanPhaseId) => void;
  /**
   * Clear every recorded restore failure so the provider re-attempts the disk
   * re-read on the next render. The user triggers this from the panel's Retry
   * button; there is no automatic retry, since a genuinely missing file would
   * then loop forever.
   */
  retryDeepPlanDocumentRestore: () => void;
  /** Runs the reference validator; returns the structured reason on failure. */
  confirmDeepPlanPhase: (phase: DeepPlanPhaseId) => {
    ok: boolean;
    failure?: ConfirmFailure;
  };
  resetDeepPlan: () => void;
}

type ConversationStore = ConversationState & ConversationActions;

const getConversationIdFromLocation = (): string | null => {
  if (typeof window === "undefined") {
    return null;
  }

  const match = window.location.pathname.match(/\/conversations\/([^/]+)/);
  return match ? match[1] : null;
};

const getInitialConversationMode = (): ConversationMode => {
  if (typeof window === "undefined") {
    return "code";
  }

  const conversationId = getConversationIdFromLocation();
  if (!conversationId) {
    return "code";
  }

  const state = getConversationState(conversationId);
  return state.conversationMode;
};

/**
 * Restores the phase machine across a refresh. Without this the user would
 * come back to phase 1 with their confirmations gone, even though the
 * documents are still on disk.
 */
const getInitialDeepPlanState = (): DeepPlanState => {
  if (typeof window === "undefined") {
    return EMPTY_DEEP_PLAN_STATE;
  }

  const conversationId = getConversationIdFromLocation();
  if (!conversationId) {
    return EMPTY_DEEP_PLAN_STATE;
  }

  // Only phase/confirmations/hashes are persisted; the bodies are re-read from
  // disk via history replay, so the machine starts with empty documents.
  return hydrateDeepPlanState(getConversationState(conversationId).deepPlan);
};

/**
 * Persists the phase machine for the conversation in the current URL.
 *
 * `conversationMode` is only passed by callers that also change the mode.
 * `startDeepPlan` flips the mode in memory, and the mount-time reset
 * re-derives it from storage — so without persisting it a refresh restores
 * the machine but drops the user back to `code`, unable to see it.
 */
const persistDeepPlan = (
  deepPlan: DeepPlanState,
  conversationMode?: ConversationMode,
): void => {
  const conversationId = getConversationIdFromLocation();
  if (conversationId) {
    // Persist the slim form: document bodies live on disk and are re-read on
    // reload, so storing them here only risks the localStorage quota.
    const persisted = toPersistedDeepPlan(deepPlan);
    setConversationState(
      conversationId,
      conversationMode
        ? { deepPlan: persisted, conversationMode }
        : { deepPlan: persisted },
    );
  }
};

export const useConversationStore = create<ConversationStore>()(
  devtools(
    (set) => ({
      // Initial state.
      //
      // The right-side drawer (`isRightPanelShown` / `hasRightPanelToggled`)
      // is intentionally *session-only* state: it always starts closed on
      // app load (or on opening a fresh/existing conversation after a
      // restart), but it survives in-app navigation because the Zustand
      // store stays alive across React Router transitions. Persisting the
      // open/closed state in localStorage made the panel feel sticky in
      // a way users didn't expect — they want a clean, focused chat view
      // when they come back to the app and only want the panel back when
      // they themselves opened it during the current session.
      isRightPanelShown: false,
      isOverviewPanelShown: false,
      isOverviewPanelPeeked: false,
      selectedTab: "files" as ConversationTab,
      commitsAutoExpandSection: null,
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
      conversationMode: getInitialConversationMode(),
      deepPlan: getInitialDeepPlanState(),
      subConversationTaskId: null,
      localPlanningConversationId: null,
      deepPlanPlannerPhase: null,

      // Actions
      setIsRightPanelShown: (isRightPanelShown) =>
        set({ isRightPanelShown }, false, "setIsRightPanelShown"),

      setIsOverviewPanelShown: (isOverviewPanelShown) =>
        set(
          { isOverviewPanelShown, isOverviewPanelPeeked: false },
          false,
          "setIsOverviewPanelShown",
        ),

      setIsOverviewPanelPeeked: (isOverviewPanelPeeked) =>
        set({ isOverviewPanelPeeked }, false, "setIsOverviewPanelPeeked"),

      setSelectedTab: (selectedTab) =>
        set({ selectedTab }, false, "setSelectedTab"),

      setCommitsAutoExpandSection: (commitsAutoExpandSection) =>
        set({ commitsAutoExpandSection }, false, "setCommitsAutoExpandSection"),

      setShouldShownAgentLoading: (shouldShownAgentLoading) =>
        set({ shouldShownAgentLoading }, false, "setShouldShownAgentLoading"),

      setShouldHideSuggestions: (shouldHideSuggestions) =>
        set({ shouldHideSuggestions }, false, "setShouldHideSuggestions"),

      addImages: (images) =>
        set(
          (state) => ({ images: [...state.images, ...images] }),
          false,
          "addImages",
        ),

      addFiles: (files) =>
        set(
          (state) => ({ files: [...state.files, ...files] }),
          false,
          "addFiles",
        ),

      toggleImageUploadAsFile: (fileName) =>
        set(
          (state) => {
            const marked = new Set(state.imagesMarkedUploadAsFile);
            if (marked.has(fileName)) {
              marked.delete(fileName);
            } else {
              marked.add(fileName);
            }
            return { imagesMarkedUploadAsFile: [...marked] };
          },
          false,
          "toggleImageUploadAsFile",
        ),

      markImagesAsPasted: (fileNames) =>
        set(
          (state) => {
            const merged = new Set([...state.pastedImageNames, ...fileNames]);
            return { pastedImageNames: [...merged] };
          },
          false,
          "markImagesAsPasted",
        ),

      removeImage: (index) =>
        set(
          (state) => {
            const removed = state.images[index];
            const newImages = [...state.images];
            newImages.splice(index, 1);
            return {
              images: newImages,
              imagesMarkedUploadAsFile: removed
                ? state.imagesMarkedUploadAsFile.filter(
                    (name) => name !== removed.name,
                  )
                : state.imagesMarkedUploadAsFile,
              pastedImageNames: removed
                ? state.pastedImageNames.filter((name) => name !== removed.name)
                : state.pastedImageNames,
            };
          },
          false,
          "removeImage",
        ),

      removeFile: (index) =>
        set(
          (state) => {
            const newFiles = [...state.files];
            newFiles.splice(index, 1);
            return { files: newFiles };
          },
          false,
          "removeFile",
        ),

      clearImages: () => set({ images: [] }, false, "clearImages"),

      clearFiles: () => set({ files: [] }, false, "clearFiles"),

      clearAllFiles: () =>
        set(
          {
            images: [],
            files: [],
            imagesMarkedUploadAsFile: [],
            pastedImageNames: [],
            loadingFiles: [],
            loadingImages: [],
          },
          false,
          "clearAllFiles",
        ),

      addFileLoading: (fileName) =>
        set(
          (state) => {
            if (!state.loadingFiles.includes(fileName)) {
              return { loadingFiles: [...state.loadingFiles, fileName] };
            }
            return state;
          },
          false,
          "addFileLoading",
        ),

      removeFileLoading: (fileName) =>
        set(
          (state) => ({
            loadingFiles: state.loadingFiles.filter(
              (name) => name !== fileName,
            ),
          }),
          false,
          "removeFileLoading",
        ),

      addImageLoading: (imageName) =>
        set(
          (state) => {
            if (!state.loadingImages.includes(imageName)) {
              return { loadingImages: [...state.loadingImages, imageName] };
            }
            return state;
          },
          false,
          "addImageLoading",
        ),

      removeImageLoading: (imageName) =>
        set(
          (state) => ({
            loadingImages: state.loadingImages.filter(
              (name) => name !== imageName,
            ),
          }),
          false,
          "removeImageLoading",
        ),

      clearAllLoading: () =>
        set({ loadingFiles: [], loadingImages: [] }, false, "clearAllLoading"),

      setMessageToSend: (text) =>
        set(
          {
            messageToSend: {
              text,
              timestamp: Date.now(),
            },
          },
          false,
          "setMessageToSend",
        ),

      // One-shot consume: clear after the composer applies it, so a never-sent
      // value can't replay into another conversation's composer on remount.
      clearMessageToSend: () =>
        set({ messageToSend: null }, false, "clearMessageToSend"),

      restoreMessageToInputIfEmpty: (text) =>
        set(
          {
            messageRestoreIfEmpty: {
              text,
              timestamp: Date.now(),
            },
          },
          false,
          "restoreMessageToInputIfEmpty",
        ),

      clearMessageRestoreIfEmpty: () =>
        set(
          { messageRestoreIfEmpty: null },
          false,
          "clearMessageRestoreIfEmpty",
        ),

      setSubmittedMessage: (submittedMessage) =>
        set({ submittedMessage }, false, "setSubmittedMessage"),

      resetConversationState: () =>
        set(
          {
            shouldHideSuggestions: false,
            conversationMode: getInitialConversationMode(),
            // Re-derive from storage, exactly like `conversationMode`: this
            // runs on every conversation mount, so resetting to the empty
            // machine would discard the persisted phase and drop the user
            // back to phase 1 on refresh.
            deepPlan: getInitialDeepPlanState(),
            subConversationTaskId: null,
            localPlanningConversationId: null,
            // Clear the phase with the id. They describe the same planner, so a
            // stale phase alongside a nulled id would make the next
            // conversation inherit this one's planner decision and skip
            // provisioning its own.
            deepPlanPlannerPhase: null,
            planContent: null,
          },
          false,
          "resetConversationState",
        ),

      setHasRightPanelToggled: (hasRightPanelToggled) =>
        set({ hasRightPanelToggled }, false, "setHasRightPanelToggled"),

      setConversationMode: (conversationMode) => {
        const conversationId = getConversationIdFromLocation();
        if (conversationId) {
          setConversationState(conversationId, { conversationMode });
        }
        set({ conversationMode }, false, "setConversationMode");
      },

      setSubConversationTaskId: (subConversationTaskId) =>
        set({ subConversationTaskId }, false, "setSubConversationTaskId"),

      setLocalPlanningConversationId: (
        localPlanningConversationId,
        deepPlanPlannerPhase = null,
      ) =>
        set(
          { localPlanningConversationId, deepPlanPlannerPhase },
          false,
          "setLocalPlanningConversationId",
        ),

      setPlanContent: (planContent) =>
        set({ planContent }, false, "setPlanContent"),

      startDeepPlan: () => {
        // Re-entering the mode (Shift+Tab, the context menu, `/deep-plan`)
        // must not discard a chain the user already advanced; only open a
        // fresh one when there is nothing to keep.
        const existing = useConversationStore.getState().deepPlan;
        const deepPlan =
          existing.activePhase === null ? createDeepPlanState() : existing;
        persistDeepPlan(deepPlan, "deep-plan");
        set(
          { deepPlan, conversationMode: "deep-plan" as ConversationMode },
          false,
          "startDeepPlan",
        );
      },

      setDeepPlanPhase: (phase) =>
        set(
          (state) => {
            if (!canEnterPhase(state.deepPlan, phase)) return {};
            const deepPlan = { ...state.deepPlan, activePhase: phase };
            persistDeepPlan(deepPlan);
            return { deepPlan };
          },
          false,
          "setDeepPlanPhase",
        ),

      setDeepPlanDocument: (phase, content) => {
        const current = useConversationStore.getState().deepPlan;
        // History replay re-reads the persisted documents after a refresh;
        // re-hydrating identical bytes is not an edit, so it must not drop the
        // confirmations the user already earned. Only a real change invalidates
        // the phase and everything built on it.
        if (current.documents[phase] === content) {
          return;
        }
        // After a reload the bodies start empty but the persisted per-document
        // hashes remain, so a replay of the bytes the persisted chain vouches
        // for is recognized as a rehydrate (not an edit) even though the body
        // is not yet in memory. The body is still stored so the panel can
        // render it and the validator can read it.
        const isRehydrate =
          current.documentHashes?.[phase] === hashDeepPlanDocument(content);
        set(
          (state) => {
            const documentHashes = {
              ...state.deepPlan.documentHashes,
              [phase]: hashDeepPlanDocument(content),
            };
            const withDocument = {
              ...state.deepPlan,
              documents: { ...state.deepPlan.documents, [phase]: content },
              documentHashes,
              // A successful re-read clears any earlier restore failure for the
              // phase, so the checkpoint stops reporting it as unavailable.
              restoreFailed: (state.deepPlan.restoreFailed ?? []).filter(
                (failed) => failed !== phase,
              ),
            };
            const deepPlan = isRehydrate
              ? withDocument
              : invalidateFrom(withDocument, phase);
            persistDeepPlan(deepPlan);
            return { deepPlan };
          },
          false,
          "setDeepPlanDocument",
        );
      },

      failDeepPlanDocumentRestore: (phase) =>
        set(
          (state) => {
            if (state.deepPlan.documents[phase] !== undefined) return {};
            const restoreFailed = state.deepPlan.restoreFailed ?? [];
            if (restoreFailed.includes(phase)) return {};
            return {
              deepPlan: {
                ...state.deepPlan,
                restoreFailed: [...restoreFailed, phase],
              },
            };
          },
          false,
          "failDeepPlanDocumentRestore",
        ),

      retryDeepPlanDocumentRestore: () =>
        set(
          (state) => {
            if ((state.deepPlan.restoreFailed ?? []).length === 0) return {};
            // `restoreFailed` is deliberately not persisted: it describes an
            // in-flight read failure, so clearing it in memory is enough to let
            // the restore effect re-issue the read.
            return {
              deepPlan: { ...state.deepPlan, restoreFailed: [] },
            };
          },
          false,
          "retryDeepPlanDocumentRestore",
        ),

      confirmDeepPlanPhase: (phase) => {
        const result = confirmPhase(
          useConversationStore.getState().deepPlan,
          phase,
        );
        if (!result.ok) {
          return { ok: false, failure: result.failure };
        }
        persistDeepPlan(result.state);
        set({ deepPlan: result.state }, false, "confirmDeepPlanPhase");
        return { ok: true };
      },

      resetDeepPlan: () => {
        // Persist `"code"` explicitly: `startDeepPlan` wrote `"deep-plan"` to
        // storage, so re-deriving the mode from storage (the mount-time reset
        // path) would leave the user stuck in Deep Plan after a reset.
        persistDeepPlan(EMPTY_DEEP_PLAN_STATE, "code");
        set(
          {
            deepPlan: EMPTY_DEEP_PLAN_STATE,
            conversationMode: "code",
          },
          false,
          "resetDeepPlan",
        );
      },
    }),
    {
      name: "conversation-store",
    },
  ),
);
