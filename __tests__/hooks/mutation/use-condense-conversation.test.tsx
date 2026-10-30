/* Regression (OpenHands/OpenHands#18860): a failed condense request used to
 * toast once through the global MutationCache and again through the compact
 * context action's own onError. The hook now opts out of the global toast, so
 * only the caller's toast fires. */
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AxiosError, AxiosHeaders } from "axios";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import { createAgentServerQueryClient } from "#/query-client-config";

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: { condenseConversation: vi.fn() },
  }),
);
const displayErrorToast = vi.fn();
const displaySuccessToast = vi.fn();
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: (...args: unknown[]) => displayErrorToast(...args),
  displaySuccessToast: (...args: unknown[]) => displaySuccessToast(...args),
}));

import { useCondenseConversation } from "#/hooks/mutation/use-condense-conversation";

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

describe("useCondenseConversation", () => {
  beforeEach(() => {
    __resetActiveStoreForTests();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("suppresses the global error toast so the caller's toast is the only one", async () => {
    // A failed Compact context reports a readable reason with a 409; the
    // action's own onError toasts that reason. The global MutationCache must
    // stay silent for this mutation.
    vi.mocked(
      AgentServerConversationService.condenseConversation,
    ).mockRejectedValue(makeAxiosError(409));

    const client = createAgentServerQueryClient();
    const wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(QueryClientProvider, { client }, children);
    const { result } = renderHook(() => useCondenseConversation(), {
      wrapper,
    });

    act(() => {
      result.current.mutate({
        conversationId: "conv-1",
        conversationUrl: null,
        sessionApiKey: null,
      });
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(displayErrorToast).not.toHaveBeenCalled();
    expect(displaySuccessToast).not.toHaveBeenCalled();
  });
});
