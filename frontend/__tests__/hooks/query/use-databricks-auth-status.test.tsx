import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import {
  useDatabricksAuthStatus,
  useDatabricksLogout,
} from "#/hooks/query/use-databricks-auth-status";

const statusMock = vi.fn();
const logoutMock = vi.fn();

vi.mock(
  "#/api/databricks-auth-service/databricks-auth-service.api",
  () => ({
    default: {
      INITIATE_URL: "/auth/databricks/initiate",
      status: (...args: unknown[]) => statusMock(...args),
      logout: (...args: unknown[]) => logoutMock(...args),
    },
  }),
);

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0, staleTime: 0 } },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

describe("useDatabricksAuthStatus", () => {
  beforeEach(() => {
    statusMock.mockReset();
    logoutMock.mockReset();
  });

  it("is disabled by default when enabled=false (no network call)", async () => {
    const wrapper = makeWrapper();
    const { result } = renderHook(
      () => useDatabricksAuthStatus({ enabled: false }),
      { wrapper },
    );

    // No fetch should kick off while disabled.
    expect(statusMock).not.toHaveBeenCalled();
    expect(result.current.data).toBeUndefined();
  });

  it("fetches status when enabled and returns the payload", async () => {
    statusMock.mockResolvedValue({
      configured: true,
      authenticated: true,
      host: "https://adb-1.cloud.databricks.com",
    });

    const wrapper = makeWrapper();
    const { result } = renderHook(() => useDatabricksAuthStatus(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(statusMock).toHaveBeenCalledTimes(1);
    expect(result.current.data).toEqual({
      configured: true,
      authenticated: true,
      host: "https://adb-1.cloud.databricks.com",
    });
  });
});

describe("useDatabricksLogout", () => {
  beforeEach(() => {
    statusMock.mockReset();
    logoutMock.mockReset();
  });

  it("invalidates the status query on success", async () => {
    logoutMock.mockResolvedValue(undefined);
    statusMock.mockResolvedValue({
      configured: true,
      authenticated: true,
      host: "https://adb-1.cloud.databricks.com",
    });

    // Shared client so we can observe invalidation across the two hooks.
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false, gcTime: 0, staleTime: 0 } },
    });
    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );

    // Prime the status cache.
    const { result: statusRes } = renderHook(
      () => useDatabricksAuthStatus(),
      { wrapper },
    );
    await waitFor(() => expect(statusRes.current.isSuccess).toBe(true));
    expect(statusMock).toHaveBeenCalledTimes(1);

    // Fire the logout mutation.
    const { result: logoutRes } = renderHook(() => useDatabricksLogout(), {
      wrapper,
    });
    logoutRes.current.mutate();
    await waitFor(() => expect(logoutMock).toHaveBeenCalledTimes(1));

    // On invalidation the status query should be refetched.
    await waitFor(() => expect(statusMock).toHaveBeenCalledTimes(2));
  });
});
