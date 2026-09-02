import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import { useInstallCanvasExtension } from "#/hooks/mutation/use-manage-canvas-extensions";

const displayErrorToast = vi.fn();
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: (message: string) => displayErrorToast(message),
  displaySuccessToast: vi.fn(),
}));

const createWrapper = () => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(QueryClientProvider, { client }, children);
  };
};

/** The shape the shared TypeScript client throws for a failed request. */
class HttpError extends Error {
  status: number;

  response: unknown;

  constructor(status: number, detail: string) {
    super(
      `HTTP request failed (${status} Bad Request): ${JSON.stringify({ detail })}`,
    );
    this.name = "HttpError";
    this.status = status;
    this.response = { detail };
  }
}

describe("useInstallCanvasExtension", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("surfaces the server's reason instead of the raw HTTP message", async () => {
    const detail =
      "Failed to fetch canvas extension source: Subdirectory 'canvas-puls' not found in local source '/repo'";
    vi.spyOn(CanvasExtensionsService, "install").mockRejectedValue(
      new HttpError(400, detail),
    );

    const { result } = renderHook(() => useInstallCanvasExtension(), {
      wrapper: createWrapper(),
    });
    result.current.mutate({ source: "/repo", repo_path: "canvas-puls" });

    await waitFor(() => expect(displayErrorToast).toHaveBeenCalled());
    expect(displayErrorToast).toHaveBeenCalledWith(detail);
    expect(displayErrorToast).not.toHaveBeenCalledWith(
      expect.stringContaining("HTTP request failed"),
    );
  });
});
