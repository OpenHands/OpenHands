import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import React from "react";
import { renderHook, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AxiosError, AxiosHeaders } from "axios";
import { useReadConversationFile } from "#/hooks/mutation/use-read-conversation-file";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: { readConversationFile: vi.fn() },
  }),
);
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

function makeAxiosError(status: number): AxiosError {
  return new AxiosError(
    `Request failed with status code ${status}`,
    String(status),
    undefined,
    undefined,
    {
      status,
      statusText: String(status),
      headers: new AxiosHeaders(),
      config: { headers: new AxiosHeaders() },
      data: undefined,
    },
  );
}

describe("useReadConversationFile", () => {
  let queryClient: QueryClient;
  let wrapper: ({
    children,
  }: {
    children: React.ReactNode;
  }) => React.ReactElement;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(
        QueryClientProvider,
        { client: queryClient },
        children,
      );
  });

  afterEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
  });

  it("resolves with the file content on success", async () => {
    vi.mocked(
      AgentServerConversationService.readConversationFile,
    ).mockResolvedValue("# Plan\n\n- step 1");

    const { result } = renderHook(() => useReadConversationFile(), {
      wrapper,
    });

    let content: string | undefined;
    await act(async () => {
      content = await result.current.mutateAsync({
        conversationId: "conv-1",
      });
    });

    expect(content).toBe("# Plan\n\n- step 1");
    expect(displayErrorToast).not.toHaveBeenCalled();
  });

  it("stays silent on a 404 — the expected no-plan-yet state", async () => {
    vi.mocked(
      AgentServerConversationService.readConversationFile,
    ).mockRejectedValue(makeAxiosError(404));

    const { result } = renderHook(() => useReadConversationFile(), {
      wrapper,
    });

    act(() => {
      result.current.mutate({ conversationId: "conv-1" });
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(displayErrorToast).not.toHaveBeenCalled();
  });

  it("shows an error toast for a non-404 failure, e.g. a transient 500", async () => {
    // Regression: meta.disableToast used to suppress the toast
    // unconditionally, silently stranding the Planner tab on its empty
    // state for real failures too, not just the expected 404.
    vi.mocked(
      AgentServerConversationService.readConversationFile,
    ).mockRejectedValue(makeAxiosError(500));

    const { result } = renderHook(() => useReadConversationFile(), {
      wrapper,
    });

    act(() => {
      result.current.mutate({ conversationId: "conv-1" });
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(displayErrorToast).toHaveBeenCalled();
  });

  it("shows an error toast for a non-Axios failure (e.g. a network error)", async () => {
    vi.mocked(
      AgentServerConversationService.readConversationFile,
    ).mockRejectedValue(new Error("Network Error"));

    const { result } = renderHook(() => useReadConversationFile(), {
      wrapper,
    });

    act(() => {
      result.current.mutate({ conversationId: "conv-1" });
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(displayErrorToast).toHaveBeenCalled();
  });
});
