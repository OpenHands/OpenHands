import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { useModelInterceptor } from "#/hooks/chat/use-model-interceptor";
import { useEventStore } from "#/stores/use-event-store";
import { useModelStore } from "#/stores/model-store";

const renderInterceptor = (
  conversationId: string | null,
  onSubmit: (m: string) => void,
) => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
  return {
    queryClient,
    ...renderHook(() => useModelInterceptor(conversationId, onSubmit), {
      wrapper,
    }),
  };
};

const mockListProfiles = vi.hoisted(() =>
  vi.fn<
    () => Promise<{
      profiles: LlmProfileSummary[];
      active_profile: string | null;
    }>
  >(),
);
const mockSwitchProfile = vi.hoisted(() =>
  vi.fn<(id: string, name: string) => Promise<void>>(),
);
const mockDisplayErrorToast = vi.hoisted(() => vi.fn<(msg: string) => void>());
const mockDisplaySuccessToast = vi.hoisted(() =>
  vi.fn<(msg: string) => void>(),
);

vi.mock("#/api/settings-service/profiles-service.api", () => ({
  default: { listProfiles: () => mockListProfiles() },
}));

vi.mock("#/api/conversation-service/v1-conversation-service.api", () => ({
  default: {
    switchProfile: (id: string, name: string) => mockSwitchProfile(id, name),
  },
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: (msg: string) => mockDisplayErrorToast(msg),
  displaySuccessToast: (msg: string) => mockDisplaySuccessToast(msg),
}));

const CONV = "conv-1";
const entries = () =>
  useModelStore.getState().entriesByConversation[CONV] ?? [];

describe("useModelInterceptor", () => {
  beforeEach(() => {
    useModelStore.setState({ entriesByConversation: {} });
    useEventStore.setState({ events: [], uiEvents: [] });
    mockListProfiles.mockReset();
    mockSwitchProfile.mockReset();
    mockDisplayErrorToast.mockReset();
    mockDisplaySuccessToast.mockReset();
  });

  it("falls through to onSubmit for non-/model messages", () => {
    const onSubmit = vi.fn();
    const { result } = renderInterceptor(CONV, onSubmit);
    act(() => result.current("hello world"));
    expect(onSubmit).toHaveBeenCalledWith("hello world");
    expect(mockListProfiles).not.toHaveBeenCalled();
    expect(mockSwitchProfile).not.toHaveBeenCalled();
  });

  it("intercepts bare /model and pushes the listed profiles to the store", async () => {
    mockListProfiles.mockResolvedValueOnce({
      profiles: [
        {
          name: "default",
          model: "anthropic/claude-sonnet-4-6",
          base_url: null,
          api_key_set: true,
        },
      ],
      active_profile: "default",
    });
    const onSubmit = vi.fn();
    const { result } = renderInterceptor(CONV, onSubmit);

    act(() => result.current("/model"));

    expect(onSubmit).not.toHaveBeenCalled();
    await waitFor(() => expect(entries()).toHaveLength(1));
    expect(entries()[0]).toMatchObject({
      anchorEventId: null,
      profiles: [{ name: "default" }],
    });
  });

  it("anchors to the latest *rendered* event, skipping events filtered out by shouldRenderEvent", async () => {
    const renderedMessage = {
      id: "evt-77",
      timestamp: "2026-05-01T10:00:00Z",
      source: "user",
      llm_message: { role: "user", content: "hi" },
    };
    // ConversationStateUpdateEvent is excluded by shouldRenderEvent — if the
    // interceptor anchored to it the toggle would have no slot to mount in.
    const trailingStateUpdate = {
      id: "evt-78",
      timestamp: "2026-05-01T10:00:01Z",
      source: "agent",
      kind: "ConversationStateUpdateEvent",
      key: "execution_status",
    };
    useEventStore.setState({
      events: [],
      uiEvents: [renderedMessage as never, trailingStateUpdate as never],
    });
    mockListProfiles.mockResolvedValueOnce({
      profiles: [
        {
          name: "default",
          model: "anthropic/claude-sonnet-4-6",
          base_url: null,
          api_key_set: true,
        },
      ],
      active_profile: "default",
    });
    const { result } = renderInterceptor(CONV, vi.fn());

    act(() => result.current("/model"));

    await waitFor(() => expect(entries()).toHaveLength(1));
    expect(entries()[0].anchorEventId).toBe("evt-77");
  });

  it("intercepts /model <name> and switches via V1ConversationService", async () => {
    mockSwitchProfile.mockResolvedValueOnce(undefined);
    const { result } = renderInterceptor(CONV, vi.fn());

    act(() => result.current("/model gpt-5"));

    await waitFor(() =>
      expect(mockSwitchProfile).toHaveBeenCalledWith(CONV, "gpt-5"),
    );
    // No success toast — the chat info block (recorded in the store) is the
    // sole user-visible confirmation; toast would be redundant.
    expect(mockDisplaySuccessToast).not.toHaveBeenCalled();
  });

  it("records a switch entry in the store after a successful /model <name>", async () => {
    mockSwitchProfile.mockResolvedValueOnce(undefined);
    const { result } = renderInterceptor(CONV, vi.fn());

    act(() => result.current("/model gpt-5"));

    await waitFor(() => expect(entries()).toHaveLength(1));
    expect(entries()[0]).toMatchObject({
      anchorEventId: null,
      profiles: [],
      switchedTo: "gpt-5",
    });
  });

  it("anchors the switch entry to the latest rendered event", async () => {
    const renderedMessage = {
      id: "evt-91",
      timestamp: "2026-05-01T10:00:00Z",
      source: "user",
      llm_message: { role: "user", content: "hi" },
    };
    useEventStore.setState({
      events: [],
      uiEvents: [renderedMessage as never],
    });
    mockSwitchProfile.mockResolvedValueOnce(undefined);
    const { result } = renderInterceptor(CONV, vi.fn());

    act(() => result.current("/model gpt-5"));

    await waitFor(() => expect(entries()).toHaveLength(1));
    expect(entries()[0].anchorEventId).toBe("evt-91");
    expect(entries()[0].switchedTo).toBe("gpt-5");
  });

  it("does not record a switch entry when switchProfile rejects", async () => {
    mockSwitchProfile.mockRejectedValueOnce({
      response: { data: { detail: "Profile 'ghost' not found" } },
    });
    const { result } = renderInterceptor(CONV, vi.fn());

    act(() => result.current("/model ghost"));

    await waitFor(() => expect(mockDisplayErrorToast).toHaveBeenCalled());
    expect(entries()).toEqual([]);
  });

  it("invalidates the active conversation query after a successful switch", async () => {
    mockSwitchProfile.mockResolvedValueOnce(undefined);
    const { result, queryClient } = renderInterceptor(CONV, vi.fn());
    const invalidate = vi.spyOn(queryClient, "invalidateQueries");

    act(() => result.current("/model gpt-5"));

    await waitFor(() =>
      expect(invalidate).toHaveBeenCalledWith({
        queryKey: ["user", "conversation", CONV],
      }),
    );
  });

  it("falls back to the i18n switch-failed key when no error detail is present", async () => {
    mockSwitchProfile.mockRejectedValueOnce({});
    const { result } = renderInterceptor(CONV, vi.fn());

    act(() => result.current("/model gpt-5"));

    await waitFor(() =>
      expect(mockDisplayErrorToast).toHaveBeenCalledWith("MODEL$SWITCH_FAILED"),
    );
  });

  it("falls back to the i18n list-failed key when listProfiles rejects without a message", async () => {
    mockListProfiles.mockRejectedValueOnce({});
    const { result } = renderInterceptor(CONV, vi.fn());

    act(() => result.current("/model"));

    await waitFor(() =>
      expect(mockDisplayErrorToast).toHaveBeenCalledWith("MODEL$LIST_FAILED"),
    );
  });

  it("surfaces backend detail when switchProfile rejects", async () => {
    mockSwitchProfile.mockRejectedValueOnce({
      response: { data: { detail: "Profile 'ghost' not found" } },
    });
    const { result } = renderInterceptor(CONV, vi.fn());

    act(() => result.current("/model ghost"));

    await waitFor(() =>
      expect(mockDisplayErrorToast).toHaveBeenCalledWith(
        "Profile 'ghost' not found",
      ),
    );
  });

  it("falls through when conversationId is null (feature off for V0)", () => {
    const onSubmit = vi.fn();
    const { result } = renderInterceptor(null, onSubmit);
    act(() => result.current("/model"));
    expect(onSubmit).toHaveBeenCalledWith("/model");
    expect(mockListProfiles).not.toHaveBeenCalled();
    expect(mockSwitchProfile).not.toHaveBeenCalled();
  });
});
