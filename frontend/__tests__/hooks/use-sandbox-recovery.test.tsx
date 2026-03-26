import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { useSandboxRecovery } from "#/hooks/use-sandbox-recovery";
import { useUnifiedResumeConversationSandbox } from "#/hooks/mutation/use-unified-start-conversation";
import * as customToastHandlers from "#/utils/custom-toast-handlers";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
  initReactI18next: {
    type: "3rdParty",
    init: () => {},
  },
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({
    providers: [{ provider: "github", token: "test-token" }],
  }),
}));

vi.mock("#/utils/custom-toast-handlers");
vi.mock("#/hooks/mutation/use-unified-start-conversation");

describe("useSandboxRecovery", () => {
  let mockMutate: ReturnType<typeof vi.fn>;

  const createWrapper = () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    return ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };

  beforeEach(() => {
    vi.clearAllMocks();

    mockMutate = vi.fn();

    vi.mocked(useUnifiedResumeConversationSandbox).mockReturnValue({
      mutate: mockMutate,
      mutateAsync: vi.fn(),
      isPending: false,
      isSuccess: false,
      isError: false,
      isIdle: true,
      data: undefined,
      error: null,
      reset: vi.fn(),
      status: "idle",
      variables: undefined,
      failureCount: 0,
      failureReason: null,
      submittedAt: 0,
      context: undefined,
    } as unknown as ReturnType<typeof useUnifiedResumeConversationSandbox>);

    // Reset document.visibilityState
    Object.defineProperty(document, "visibilityState", {
      value: "visible",
      writable: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("initial load recovery", () => {
    it("should call resumeSandbox on initial load when conversation is STOPPED", () => {
      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "STOPPED",
          }),
        { wrapper: createWrapper() },
      );

      expect(mockMutate).toHaveBeenCalledTimes(1);
      expect(mockMutate).toHaveBeenCalledWith(
        {
          conversationId: "conv-123",
          providers: [{ provider: "github", token: "test-token" }],
        },
        expect.objectContaining({
          onSuccess: expect.any(Function),
          onError: expect.any(Function),
        }),
      );
    });

    it("should NOT call resumeSandbox on initial load when conversation is RUNNING", () => {
      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "RUNNING",
          }),
        { wrapper: createWrapper() },
      );

      expect(mockMutate).not.toHaveBeenCalled();
    });

    it("should NOT call resumeSandbox when conversationId is undefined", () => {
      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: undefined,
            conversationStatus: "STOPPED",
          }),
        { wrapper: createWrapper() },
      );

      expect(mockMutate).not.toHaveBeenCalled();
    });

    it("should NOT call resumeSandbox when conversationStatus is undefined", () => {
      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: undefined,
          }),
        { wrapper: createWrapper() },
      );

      expect(mockMutate).not.toHaveBeenCalled();
    });

    it("should only call resumeSandbox once per conversation on initial load", () => {
      const { rerender } = renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "STOPPED",
          }),
        { wrapper: createWrapper() },
      );

      expect(mockMutate).toHaveBeenCalledTimes(1);

      // Rerender with same props - should not trigger again
      rerender();

      expect(mockMutate).toHaveBeenCalledTimes(1);
    });

    it("should call resumeSandbox for a new conversation after navigating", async () => {
      const { rerender } = renderHook(
        ({ conversationId }) =>
          useSandboxRecovery({
            conversationId,
            conversationStatus: "STOPPED",
          }),
        {
          wrapper: createWrapper(),
          initialProps: { conversationId: "conv-123" },
        },
      );

      expect(mockMutate).toHaveBeenCalledTimes(1);
      expect(mockMutate).toHaveBeenLastCalledWith(
        expect.objectContaining({ conversationId: "conv-123" }),
        expect.any(Object),
      );

      // Navigate to a different conversation
      rerender({ conversationId: "conv-456" });

      await waitFor(() => {
        expect(mockMutate).toHaveBeenCalledTimes(2);
      });

      expect(mockMutate).toHaveBeenLastCalledWith(
        expect.objectContaining({ conversationId: "conv-456" }),
        expect.any(Object),
      );
    });
  });

  describe("tab focus recovery", () => {
    it("should call resumeSandbox when tab becomes visible and refetch returns STOPPED", async () => {
      // Start with tab hidden
      Object.defineProperty(document, "visibilityState", {
        value: "hidden",
        writable: true,
      });

      const mockRefetch = vi.fn().mockResolvedValue({
        data: { status: "STOPPED" },
      });

      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "RUNNING", // Cached status is RUNNING
            refetchConversation: mockRefetch,
          }),
        { wrapper: createWrapper() },
      );

      // No initial recovery for RUNNING
      expect(mockMutate).not.toHaveBeenCalled();

      // Simulate tab becoming visible
      Object.defineProperty(document, "visibilityState", {
        value: "visible",
        writable: true,
      });

      await act(async () => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      // Refetch should be called to get fresh status
      expect(mockRefetch).toHaveBeenCalledTimes(1);
      // Recovery should trigger because fresh status is STOPPED
      expect(mockMutate).toHaveBeenCalledTimes(1);
    });

    it("should NOT call resumeSandbox when tab becomes visible and refetch returns RUNNING", async () => {
      const mockRefetch = vi.fn().mockResolvedValue({
        data: { status: "RUNNING" },
      });

      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "RUNNING",
            refetchConversation: mockRefetch,
          }),
        { wrapper: createWrapper() },
      );

      // No initial recovery for RUNNING
      expect(mockMutate).not.toHaveBeenCalled();

      // Simulate tab becoming visible
      await act(async () => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      // Refetch was called but status is still RUNNING
      expect(mockRefetch).toHaveBeenCalledTimes(1);
      expect(mockMutate).not.toHaveBeenCalled();
    });

    it("should NOT call resumeSandbox when tab becomes visible but refetchConversation is not provided", async () => {
      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "STOPPED",
            // No refetchConversation provided
          }),
        { wrapper: createWrapper() },
      );

      // Initial load triggers recovery
      expect(mockMutate).toHaveBeenCalledTimes(1);
      mockMutate.mockClear();

      // Simulate tab becoming visible
      await act(async () => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      // No recovery on tab focus without refetchConversation
      expect(mockMutate).not.toHaveBeenCalled();
    });

    it("should NOT call resumeSandbox when tab becomes hidden", async () => {
      const mockRefetch = vi.fn().mockResolvedValue({
        data: { status: "STOPPED" },
      });

      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "STOPPED",
            refetchConversation: mockRefetch,
          }),
        { wrapper: createWrapper() },
      );

      // Initial load triggers recovery
      expect(mockMutate).toHaveBeenCalledTimes(1);
      mockMutate.mockClear();

      // Simulate tab becoming hidden
      Object.defineProperty(document, "visibilityState", {
        value: "hidden",
        writable: true,
      });

      await act(async () => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      // Refetch should NOT be called when tab is hidden
      expect(mockRefetch).not.toHaveBeenCalled();
      expect(mockMutate).not.toHaveBeenCalled();
    });

    it("should clean up visibility event listener on unmount", () => {
      const addEventListenerSpy = vi.spyOn(document, "addEventListener");
      const removeEventListenerSpy = vi.spyOn(document, "removeEventListener");

      const { unmount } = renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "STOPPED",
          }),
        { wrapper: createWrapper() },
      );

      expect(addEventListenerSpy).toHaveBeenCalledWith(
        "visibilitychange",
        expect.any(Function),
      );

      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith(
        "visibilitychange",
        expect.any(Function),
      );
    });

    it("should NOT call resumeSandbox when tab becomes visible while isPending is true", async () => {
      vi.mocked(useUnifiedResumeConversationSandbox).mockReturnValue({
        mutate: mockMutate,
        mutateAsync: vi.fn(),
        isPending: true,
        isSuccess: false,
        isError: false,
        isIdle: false,
        data: undefined,
        error: null,
        reset: vi.fn(),
        status: "pending",
        variables: undefined,
        failureCount: 0,
        failureReason: null,
        submittedAt: 0,
        context: undefined,
      } as unknown as ReturnType<typeof useUnifiedResumeConversationSandbox>);

      const mockRefetch = vi.fn().mockResolvedValue({
        data: { status: "STOPPED" },
      });

      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "RUNNING",
            refetchConversation: mockRefetch,
          }),
        { wrapper: createWrapper() },
      );

      // Simulate tab becoming visible
      await act(async () => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      // Refetch will be called when isPending is true
      expect(mockRefetch).toHaveBeenCalledTimes(1);
      // resumeSandbox should NOT be called
      expect(mockMutate).not.toHaveBeenCalled();
    });

    it("should handle refetch errors gracefully without crashing", async () => {
      const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

      const mockRefetch = vi.fn().mockRejectedValue(new Error("Network error"));

      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "RUNNING",
            refetchConversation: mockRefetch,
          }),
        { wrapper: createWrapper() },
      );

      // Simulate tab becoming visible
      await act(async () => {
        document.dispatchEvent(new Event("visibilitychange"));
      });

      // Refetch was called
      expect(mockRefetch).toHaveBeenCalledTimes(1);
      // Error was logged
      expect(consoleErrorSpy).toHaveBeenCalledWith(
        "Failed to refetch conversation on visibility change:",
        expect.any(Error),
      );
      // No recovery attempt was made (due to error)
      expect(mockMutate).not.toHaveBeenCalled();

      consoleErrorSpy.mockRestore();
    });
  });

  describe("recovery callbacks", () => {
    it("should return isResuming=false when no recovery is in progress", () => {
      const { result } = renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "RUNNING",
          }),
        { wrapper: createWrapper() },
      );

      expect(result.current.isResuming).toBe(false);
    });

    it("should return isResuming=true when mutation is pending", () => {
      vi.mocked(useUnifiedResumeConversationSandbox).mockReturnValue({
        mutate: mockMutate,
        mutateAsync: vi.fn(),
        isPending: true,
        isSuccess: false,
        isError: false,
        isIdle: false,
        data: undefined,
        error: null,
        reset: vi.fn(),
        status: "pending",
        variables: undefined,
        failureCount: 0,
        failureReason: null,
        submittedAt: 0,
        context: undefined,
      } as unknown as ReturnType<typeof useUnifiedResumeConversationSandbox>);

      const { result } = renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "STOPPED",
          }),
        { wrapper: createWrapper() },
      );

      expect(result.current.isResuming).toBe(true);
    });

    it("should call onSuccess callback when recovery succeeds", () => {
      const onSuccess = vi.fn();

      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "STOPPED",
            onSuccess,
          }),
        { wrapper: createWrapper() },
      );

      // Get the onSuccess callback passed to mutate
      const mutateCall = mockMutate.mock.calls[0];
      const options = mutateCall[1];

      // Simulate successful mutation
      act(() => {
        options.onSuccess();
      });

      expect(onSuccess).toHaveBeenCalledTimes(1);
    });

    it("should call onError callback and display toast when recovery fails", () => {
      const onError = vi.fn();
      const testError = new Error("Resume failed");

      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "STOPPED",
            onError,
          }),
        { wrapper: createWrapper() },
      );

      // Get the onError callback passed to mutate
      const mutateCall = mockMutate.mock.calls[0];
      const options = mutateCall[1];

      // Simulate failed mutation
      act(() => {
        options.onError(testError);
      });

      expect(onError).toHaveBeenCalledTimes(1);
      expect(onError).toHaveBeenCalledWith(testError);
      expect(vi.mocked(customToastHandlers.displayErrorToast)).toHaveBeenCalled();
    });

    it("should NOT call resumeSandbox when isPending is true", () => {
      vi.mocked(useUnifiedResumeConversationSandbox).mockReturnValue({
        mutate: mockMutate,
        mutateAsync: vi.fn(),
        isPending: true,
        isSuccess: false,
        isError: false,
        isIdle: false,
        data: undefined,
        error: null,
        reset: vi.fn(),
        status: "pending",
        variables: undefined,
        failureCount: 0,
        failureReason: null,
        submittedAt: 0,
        context: undefined,
      } as unknown as ReturnType<typeof useUnifiedResumeConversationSandbox>);

      renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "STOPPED",
          }),
        { wrapper: createWrapper() },
      );

      // Should not call mutate because isPending is true
      expect(mockMutate).not.toHaveBeenCalled();
    });
  });

  describe("WebSocket disconnect (negative test)", () => {
    it("should NOT have any mechanism to auto-resume on WebSocket disconnect", () => {
      // This test documents the intended behavior: the hook does NOT
      // listen for WebSocket disconnects. Recovery only happens on:
      // 1. Initial page load (STOPPED status)
      // 2. Tab focus (visibilitychange event)
      //
      // There is intentionally NO onDisconnect handler or WebSocket listener.

      const { result } = renderHook(
        () =>
          useSandboxRecovery({
            conversationId: "conv-123",
            conversationStatus: "RUNNING",
          }),
        { wrapper: createWrapper() },
      );

      // The hook should only expose isResuming - no disconnect-related functionality
      expect(result.current).toEqual({
        isResuming: expect.any(Boolean),
      });

      // No calls should have been made for RUNNING status
      expect(mockMutate).not.toHaveBeenCalled();
    });
  });
});
