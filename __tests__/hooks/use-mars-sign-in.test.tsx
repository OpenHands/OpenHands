import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { MarsBridge } from "#/api/mars/mars-tunnel-backend";
import { useMarsSignIn } from "#/hooks/use-mars-sign-in";

function wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

afterEach(() => {
  delete window.marsBridge;
});

describe("useMarsSignIn", () => {
  it("drops a failed sign-out's error once a new sign-in starts", async () => {
    // Arrange
    window.marsBridge = {
      signOut: vi.fn(async () => {
        throw new Error("sign-out failed");
      }),
      savePat: vi.fn(() => new Promise(() => {})),
    } as unknown as MarsBridge;
    const { result } = renderHook(() => useMarsSignIn(), { wrapper });
    act(() => result.current.signOut());
    await waitFor(() => expect(result.current.error).toBe("sign-out failed"));

    // Act
    act(() => result.current.saveToken("dop_v1_token"));

    // Assert
    await waitFor(() => expect(result.current.isSavingToken).toBe(true));
    expect(result.current.error).toBeNull();
  });
});
