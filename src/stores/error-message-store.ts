import { create } from "zustand";
import type { ErrorClassification } from "@openhands/typescript-client";

/**
 * "connection" errors auto-clear once connectivity recovers; "conversation"
 * errors (e.g. a wrong API key) are sticky and clear only on an explicit user
 * action (dismiss, retry, new message).
 */
export type ErrorMessageType = "connection" | "conversation";

/** Which agent of the open conversation reported an error. */
export type ErrorRole = "main" | "planner";

export interface ErrorEntry {
  message: string;
  type: ErrorMessageType;
  /**
   * Structured error code from a ConversationErrorEvent/ServerErrorEvent (e.g.
   * "ACPAuthRequired"), used to render a code-specific header + recovery action.
   * null for errors that carry no code (connection errors, plain strings).
   */
  code: string | null;
  classification: ErrorClassification | null;
}

interface ErrorMessageState {
  /** The main conversation's error. */
  errorMessage: string | null;
  errorType: ErrorMessageType | null;
  errorCode: string | null;
  errorClassification: ErrorClassification | null;
  /**
   * The planning sub-conversation's error, kept apart so a planner failure
   * never replaces or clears the main conversation's error (or vice versa).
   */
  plannerError: ErrorEntry | null;
}

type SetError = (
  message: string,
  type?: ErrorMessageType,
  code?: string | null,
  classification?: ErrorClassification | null,
) => void;

interface ErrorMessageActions {
  setErrorMessage: SetError;
  removeErrorMessage: () => void;
  /** Clears the error only when it is a transient connection error. */
  clearConnectionError: () => void;
  setPlannerError: SetError;
  removePlannerError: () => void;
  clearPlannerConnectionError: () => void;
  /** Clears both roles' errors, e.g. when the open conversation changes. */
  resetErrorMessages: () => void;
}

type ErrorMessageStore = ErrorMessageState & ErrorMessageActions;

const initialMainState = {
  errorMessage: null,
  errorType: null,
  errorCode: null,
  errorClassification: null,
} satisfies Omit<ErrorMessageState, "plannerError">;

export const useErrorMessageStore = create<ErrorMessageStore>((set) => ({
  ...initialMainState,
  plannerError: null,

  setErrorMessage: (
    message,
    type = "conversation",
    code = null,
    classification = null,
  ) =>
    set(() => ({
      errorMessage: message,
      errorType: type,
      errorCode: code,
      errorClassification: classification,
    })),

  removeErrorMessage: () => set(() => ({ ...initialMainState })),

  clearConnectionError: () =>
    set((state) =>
      state.errorType === "connection" ? { ...initialMainState } : state,
    ),

  setPlannerError: (
    message,
    type = "conversation",
    code = null,
    classification = null,
  ) => set(() => ({ plannerError: { message, type, code, classification } })),

  removePlannerError: () => set(() => ({ plannerError: null })),

  clearPlannerConnectionError: () =>
    set((state) =>
      state.plannerError?.type === "connection"
        ? { plannerError: null }
        : state,
    ),

  resetErrorMessages: () =>
    set(() => ({ ...initialMainState, plannerError: null })),
}));
