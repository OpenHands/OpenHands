import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import React from "react";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";

const trackDownloadTrajectoryButtonClickedMock = vi.fn();
vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({
    trackDownloadTrajectoryButtonClicked: trackDownloadTrajectoryButtonClickedMock,
  }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/utils/utils", () => ({
  downloadBlob: vi.fn(),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

import { useDownloadConversation } from "#/hooks/use-download-conversation";

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
};

describe("useDownloadConversation - tracking", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(
      AgentServerConversationService,
      "downloadConversation",
    ).mockResolvedValue(new Blob(["data"], { type: "application/zip" }));
  });

  it("calls trackDownloadTrajectoryButtonClicked when download is triggered", async () => {
    const { result } = renderHook(() => useDownloadConversation(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync("conv-123");

    await waitFor(() => {
      expect(trackDownloadTrajectoryButtonClickedMock).toHaveBeenCalledTimes(1);
    });
  });

  it("calls trackDownloadTrajectoryButtonClicked before the API call", async () => {
    const callOrder: string[] = [];
    trackDownloadTrajectoryButtonClickedMock.mockImplementation(() => {
      callOrder.push("tracking");
    });
    vi.spyOn(
      AgentServerConversationService,
      "downloadConversation",
    ).mockImplementation(async () => {
      callOrder.push("api");
      return new Blob();
    });

    const { result } = renderHook(() => useDownloadConversation(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync("conv-123");

    expect(callOrder).toEqual(["tracking", "api"]);
  });

  it("still fires tracking even when the API subsequently rejects", async () => {
    // trackDownloadTrajectoryButtonClicked is called synchronously before
    // the await on downloadConversation, so it always fires regardless of
    // API outcome.
    vi.spyOn(
      AgentServerConversationService,
      "downloadConversation",
    ).mockRejectedValue(new Error("network error"));

    const { result } = renderHook(() => useDownloadConversation(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync("conv-123").catch(() => {});

    expect(trackDownloadTrajectoryButtonClickedMock).toHaveBeenCalledTimes(1);
  });
});
