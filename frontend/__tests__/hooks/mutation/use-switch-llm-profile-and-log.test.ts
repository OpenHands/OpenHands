import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSwitchLlmProfileAndLog } from "#/hooks/mutation/use-switch-llm-profile-and-log";
import { useEventStore } from "#/stores/use-event-store";
import { useModelStore } from "#/stores/model-store";

const mockSwitchProfile = vi.hoisted(() =>
  vi.fn<(id: string, name: string) => Promise<void>>(),
);
const mockDisplayErrorToast = vi.hoisted(() => vi.fn<(msg: string) => void>());

vi.mock("#/api/conversation-service/v1-conversation-service.api", () => ({
  default: {
    switchProfile: (id: string, name: string) => mockSwitchProfile(id, name),
  },
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: (msg: string) => mockDisplayErrorToast(msg),
  displaySuccessToast: vi.fn(),
}));

const renderTestHook = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
  return renderHook(() => useSwitchLlmProfileAndLog(), { wrapper });
};

const CONV = "conv-1";
const entries = () =>
  useModelStore.getState().entriesByConversation[CONV] ?? [];

describe("useSwitchLlmProfileAndLog", () => {
  beforeEach(() => {
    useModelStore.setState({ entriesByConversation: {} });
    useEventStore.setState({ events: [], uiEvents: [] });
    mockSwitchProfile.mockReset();
    mockDisplayErrorToast.mockReset();
  });

  it("calls V1ConversationService.switchProfile with the conversation id and name", async () => {
    mockSwitchProfile.mockResolvedValueOnce(undefined);
    const { result } = renderTestHook();
    act(() => result.current.switchAndLog(CONV, "gpt-5"));
    await waitFor(() =>
      expect(mockSwitchProfile).toHaveBeenCalledWith(CONV, "gpt-5"),
    );
  });

  it("records a switch entry on success, anchored to the latest rendered v1 event", async () => {
    useEventStore.setState({
      events: [],
      uiEvents: [
        {
          id: "evt-7",
          timestamp: "2026-05-01T10:00:00Z",
          source: "user",
          llm_message: { role: "user", content: "hi" },
        } as never,
      ],
    });
    mockSwitchProfile.mockResolvedValueOnce(undefined);
    const { result } = renderTestHook();
    act(() => result.current.switchAndLog(CONV, "gpt-5"));
    await waitFor(() => expect(entries()).toHaveLength(1));
    expect(entries()[0]).toMatchObject({
      anchorEventId: "evt-7",
      profiles: [],
      switchedTo: "gpt-5",
    });
  });

  it("anchors to null when there are no rendered events yet", async () => {
    mockSwitchProfile.mockResolvedValueOnce(undefined);
    const { result } = renderTestHook();
    act(() => result.current.switchAndLog(CONV, "gpt-5"));
    await waitFor(() => expect(entries()).toHaveLength(1));
    expect(entries()[0].anchorEventId).toBeNull();
  });

  it("does not record an entry when switchProfile rejects, and surfaces backend detail", async () => {
    mockSwitchProfile.mockRejectedValueOnce({
      response: { data: { detail: "Profile 'ghost' not found" } },
    });
    const { result } = renderTestHook();
    act(() => result.current.switchAndLog(CONV, "ghost"));
    await waitFor(() =>
      expect(mockDisplayErrorToast).toHaveBeenCalledWith(
        "Profile 'ghost' not found",
      ),
    );
    expect(entries()).toEqual([]);
  });

  it("falls back to the i18n switch-failed key when no error detail is present", async () => {
    mockSwitchProfile.mockRejectedValueOnce({});
    const { result } = renderTestHook();
    act(() => result.current.switchAndLog(CONV, "gpt-5"));
    await waitFor(() =>
      expect(mockDisplayErrorToast).toHaveBeenCalledWith("MODEL$SWITCH_FAILED"),
    );
  });
});
