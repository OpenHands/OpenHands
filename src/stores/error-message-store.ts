import { create } from "zustand";
import type { ErrorClassification } from "@openhands/typescript-client";
import type { PluginErrorDetail } from "#/types/plugin-error";

/**
 * "connection" errors auto-clear once connectivity recovers; "conversation"
 * errors (e.g. a wrong API key) are sticky and clear only on an explicit user
 * action (dismiss, retry, new message).
 */
export type ErrorMessageType = "connection" | "conversation";

interface ErrorMessageState {
  errorMessage: string | null;
  errorType: ErrorMessageType | null;
  /**
   * Structured error code from a ConversationErrorEvent/ServerErrorEvent (e.g.
   * "ACPAuthRequired"), used to render a code-specific header + recovery action.
   * null for errors that carry no code (connection errors, plain strings).
   */
  errorCode: string | null;
  errorClassification: ErrorClassification | null;
  /**
   * Structured plugin error payload when the error is a PluginManifestError.
   * When non-null, the UI renders PluginErrorBanner instead of ErrorMessageBanner.
   */
  pluginError: PluginErrorDetail | null;
}

interface ErrorMessageActions {
  setErrorMessage: (
    message: string,
    type?: ErrorMessageType,
    code?: string | null,
    classification?: ErrorClassification | null,
    pluginError?: PluginErrorDetail | null,
  ) => void;
  removeErrorMessage: () => void;
  /** Clears the error only when it is a transient connection error. */
  clearConnectionError: () => void;
}

type ErrorMessageStore = ErrorMessageState & ErrorMessageActions;

const initialState: ErrorMessageState = {
  errorMessage: null,
  errorType: null,
  errorCode: null,
  errorClassification: null,
  pluginError: null,
};

export const useErrorMessageStore = create<ErrorMessageStore>((set) => ({
  ...initialState,

  setErrorMessage: (
    message: string,
    type: ErrorMessageType = "conversation",
    code: string | null = null,
    classification: ErrorClassification | null = null,
    pluginError: PluginErrorDetail | null = null,
  ) =>
    set(() => ({
      errorMessage: message,
      errorType: type,
      errorCode: code,
      errorClassification: classification,
      pluginError,
    })),

  removeErrorMessage: () => set(() => ({ ...initialState })),

  clearConnectionError: () =>
    set((state) =>
      state.errorType === "connection" ? { ...initialState } : state,
    ),
}));
