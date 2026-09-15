import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSystemCommandInterceptor } from "#/hooks/chat/use-system-command-interceptor";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import { useEventStore } from "#/stores/use-event-store";

const {
  mockCondenseConversation,
  mockDisplayErrorToast,
  mockDisplaySuccessToast,
} = vi.hoisted(() => ({
  mockCondenseConversation: vi.fn(),
  mockDisplayErrorToast: vi.fn(),
  mockDisplaySuccessToast: vi.fn(),
}));

vi.mock("#/hooks/mutation/conversation-mutation-utils", () => ({
  condenseConversation: mockCondenseConversation,
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: mockDisplayErrorToast,
  displaySuccessToast: mockDisplaySuccessToast,
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const CONVERSATION_ID = "conv-1";

const makeUserMessage = (id: string) => ({
  id,
  timestamp: "2026-08-02T00:00:00.000Z",
  source: "user" as const,
  llm_message: {
    role: "user" as const,
    content: [{ type: "text" as const, text: "hello" }],
  },
  activated_microagents: [],
  extended_content: [],
});

describe("useSystemCommandInterceptor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCondenseConversation.mockResolvedValue(undefined);
    useEventStore.getState().clearEvents();
    useSlashCommandOutputStore.getState().clearAll();
  });

  // @spec SC-002 — Inline help
  it("renders only built-in commands in /help", () => {
    useEventStore.getState().addEvent(makeUserMessage("message-1"));
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current(" /help "));

    const entry =
      useSlashCommandOutputStore.getState().entriesByConversation[
        CONVERSATION_ID
      ]?.[0];
    expect(entry).toMatchObject({
      kind: "help",
      anchorEventId: "message-1",
    });
    if (entry?.kind === "help") {
      expect(entry.commands.map((command) => command.command)).toEqual([
        "/new",
        "/btw",
        "/model",
        "/goal",
        "/help",
        "/plan",
        "/code",
        "/condense",
      ]);
    }
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("shows an error when /help is used without a conversation", () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(null, onSubmit),
    );

    act(() => result.current("/help"));

    expect(mockDisplayErrorToast).toHaveBeenCalledWith(
      "SLASH_COMMAND$ACTIVE_CONVERSATION_REQUIRED",
    );
    expect(onSubmit).not.toHaveBeenCalled();
  });

  // @spec SC-005 — Conversation condensation
  it("condenses the active conversation and reports success", async () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current("/condense"));

    await waitFor(() => {
      expect(mockCondenseConversation).toHaveBeenCalledWith(CONVERSATION_ID);
      expect(mockDisplaySuccessToast).toHaveBeenCalledWith(
        "SLASH_COMMAND$CONDENSE_SUCCESS",
      );
    });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  // @spec SC-005 — Conversation condensation
  it.each([404, 405, 501])(
    "reports unsupported condensation for an HTTP %s response",
    async (status) => {
      mockCondenseConversation.mockRejectedValueOnce({ response: { status } });
      const onSubmit = vi.fn();
      const { result } = renderHook(() =>
        useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
      );

      act(() => result.current("/condense"));

      await waitFor(() => {
        expect(mockDisplayErrorToast).toHaveBeenCalledWith(
          "SLASH_COMMAND$CONDENSE_UNSUPPORTED",
        );
      });
    },
  );

  // @spec SC-005 — Conversation condensation
  it("explains when the conversation has too little history to condense", async () => {
    mockCondenseConversation.mockRejectedValueOnce({
      status: 500,
      response: {
        detail:
          "NoCondensationAvailableException: Cannot condense 0 events.",
      },
    });
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current("/condense"));

    await waitFor(() => {
      expect(mockDisplayErrorToast).toHaveBeenCalledWith(
        "SLASH_COMMAND$CONDENSE_NOT_ENOUGH_HISTORY",
      );
    });
  });

  // @spec SC-005 — Conversation condensation
  it("uses the localized generic message for other condensation failures", async () => {
    mockCondenseConversation.mockRejectedValueOnce(new Error("private detail"));
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current("/condense"));

    await waitFor(() => {
      expect(mockDisplayErrorToast).toHaveBeenCalledWith(
        "SLASH_COMMAND$CONDENSE_FAILED",
      );
    });
  });

  it("passes ordinary messages through unchanged", () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useSystemCommandInterceptor(CONVERSATION_ID, onSubmit),
    );

    act(() => result.current("please run /help later"));

    expect(onSubmit).toHaveBeenCalledWith("please run /help later");
  });
});
