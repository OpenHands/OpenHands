import React from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createAgentServerQueryClient } from "#/query-client-config";
import { useSharedConversationEvents } from "#/hooks/query/use-shared-conversation-events";
import * as ToastHandlers from "#/utils/custom-toast-handlers";

const searchSharedEventsMock = vi.hoisted(() => vi.fn());

vi.mock("@openhands/typescript-client/clients", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@openhands/typescript-client/clients")
    >();
  return {
    ...actual,
    SharedClient: vi.fn().mockImplementation(function SharedClient() {
      return { searchSharedEvents: searchSharedEventsMock };
    }),
  };
});

afterEach(() => {
  searchSharedEventsMock.mockReset();
  vi.restoreAllMocks();
});

describe("useSharedConversationEvents", () => {
  it("leaves a missing shared conversation to the page's not-found state, without a toast", async () => {
    const toast = vi
      .spyOn(ToastHandlers, "displayErrorToast")
      .mockImplementation(() => {});
    searchSharedEventsMock.mockRejectedValue(
      new Error('HTTP request failed (404 Not Found): {"detail":"Not Found"}'),
    );
    const client = createAgentServerQueryClient();
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );

    const { result } = renderHook(
      () => useSharedConversationEvents("missing-id"),
      { wrapper },
    );

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(searchSharedEventsMock).toHaveBeenCalledTimes(1);
    expect(toast).not.toHaveBeenCalled();
  });
});
