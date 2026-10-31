import React from "react";
import { PromptEnhancementUnavailableError } from "@openhands/typescript-client/clients";
import PromptEnhancementService from "#/api/prompt-enhancement-service";
import {
  focusContentEditableAtEnd,
  getTextContent,
  replaceContentEditableText,
} from "#/components/features/chat/utils/chat-input.utils";
import type { PromptEnhancementAvailabilityState } from "#/hooks/query/use-prompt-enhancement-availability";
import { isSdkHttpError } from "#/utils/sdk-http-error";

/** Why an enhancement produced no suggestion. The draft never changes. */
export type PromptEnhancementErrorReason =
  /** The user edited the draft while the request ran. */
  | "draft_changed"
  | "input_too_large"
  | "timeout"
  | "invalid_output"
  | "profile_unavailable"
  | "unsupported"
  | "failed";

export type PromptEnhancementState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "preview"; enhancedText: string }
  | { status: "error"; reason: PromptEnhancementErrorReason };

const IDLE: PromptEnhancementState = { status: "idle" };

const readErrorCode = (error: unknown): string | null => {
  if (
    !(error instanceof Error) ||
    !isSdkHttpError(error) ||
    !("response" in error)
  ) {
    return null;
  }
  const { response } = error;
  return typeof response === "object" &&
    response !== null &&
    "code" in response &&
    typeof response.code === "string"
    ? response.code
    : null;
};

const toErrorReason = (error: unknown): PromptEnhancementErrorReason => {
  if (error instanceof PromptEnhancementUnavailableError) return "unsupported";
  switch (readErrorCode(error)) {
    case "input_too_large":
      return "input_too_large";
    case "enhancement_timeout":
      return "timeout";
    case "invalid_model_output":
    case "output_too_large":
      return "invalid_output";
    case "profile_not_found":
    case "profile_unavailable":
    case "profile_store_timeout":
      return "profile_unavailable";
    case "unsupported_configuration":
      return "unsupported";
    default:
      return "failed";
  }
};

/**
 * The composer's "Enhance prompt" flow: request a suggestion, let the user
 * edit it, and replace the draft only on explicit acceptance. Accepting never
 * sends the message. A suggestion is dropped when the draft changed after the
 * request started, so a late response never overwrites newer input.
 */
export function usePromptEnhancement(
  chatInputRef: React.RefObject<HTMLDivElement | null>,
  availability: PromptEnhancementAvailabilityState,
) {
  const [state, setState] = React.useState<PromptEnhancementState>(IDLE);
  const abortRef = React.useRef<AbortController | null>(null);
  const requestedDraftRef = React.useRef("");

  // Abort an in-flight request when the composer unmounts.
  React.useEffect(() => () => abortRef.current?.abort(), []);

  const enhance = React.useCallback(async () => {
    const draft = getTextContent(chatInputRef.current);
    if (!availability.isAvailable || !draft.trim() || abortRef.current) return;

    const controller = new AbortController();
    abortRef.current = controller;
    requestedDraftRef.current = draft;
    setState({ status: "loading" });
    try {
      const enhancedText = await PromptEnhancementService.enhancePrompt(
        availability.profileName,
        draft,
        controller.signal,
      );
      if (controller.signal.aborted) return;
      setState(
        getTextContent(chatInputRef.current) === draft
          ? { status: "preview", enhancedText }
          : { status: "error", reason: "draft_changed" },
      );
    } catch (error) {
      if (controller.signal.aborted) return;
      setState({ status: "error", reason: toErrorReason(error) });
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
    }
  }, [availability, chatInputRef]);

  /** Close the request or the suggestion without changing the draft. */
  const dismiss = React.useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setState(IDLE);
    focusContentEditableAtEnd(chatInputRef.current);
  }, [chatInputRef]);

  const setEnhancedText = React.useCallback((enhancedText: string) => {
    setState((current) =>
      current.status === "preview" ? { ...current, enhancedText } : current,
    );
  }, []);

  /** Replace the draft with the (possibly edited) suggestion. Does not send. */
  const accept = React.useCallback(() => {
    if (state.status !== "preview" || !state.enhancedText.trim()) return;
    const input = chatInputRef.current;
    if (getTextContent(input) !== requestedDraftRef.current) {
      setState({ status: "error", reason: "draft_changed" });
      return;
    }
    replaceContentEditableText(input, state.enhancedText);
    setState(IDLE);
  }, [chatInputRef, state]);

  return { state, enhance, dismiss, setEnhancedText, accept };
}

export type PromptEnhancementController = ReturnType<
  typeof usePromptEnhancement
>;
