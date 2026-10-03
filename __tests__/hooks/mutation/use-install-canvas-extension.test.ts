import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import {
  useInstallCanvasExtension,
  useRefreshCanvasExtension,
} from "#/hooks/mutation/use-manage-canvas-extensions";
import { CORS_OR_NETWORK_ERROR_MESSAGE } from "#/utils/user-facing-error";

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

  it("keeps the shared disconnect wording when the request never lands", async () => {
    vi.spyOn(CanvasExtensionsService, "install").mockRejectedValue(
      new TypeError("Failed to fetch"),
    );

    const { result } = renderHook(() => useInstallCanvasExtension(), {
      wrapper: createWrapper(),
    });
    result.current.mutate({ source: "/repo" });

    await waitFor(() => expect(displayErrorToast).toHaveBeenCalled());
    expect(displayErrorToast).toHaveBeenCalledWith(
      CORS_OR_NETWORK_ERROR_MESSAGE,
    );
  });
});

describe("useRefreshCanvasExtension", () => {
  const installed = {
    name: "demo",
    version: "0.1.0",
    enabled: true,
    source: "github:example/apps",
    requested_ref: "v1",
    resolved_ref: "abc123",
    repo_path: "demo",
    installed_at: "2026-08-01T00:00:00Z",
    install_path: "/tmp/demo",
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("force-reinstalls from the recorded source coordinates", async () => {
    const install = vi
      .spyOn(CanvasExtensionsService, "install")
      .mockResolvedValue({} as never);

    const { result } = renderHook(() => useRefreshCanvasExtension(), {
      wrapper: createWrapper(),
    });
    result.current.mutate(installed);

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(install).toHaveBeenCalledWith({
      source: "github:example/apps",
      ref: "v1",
      repo_path: "demo",
      force: true,
    });
  });

  it("surfaces the server's validation error", async () => {
    const detail =
      "entrypoint 'dist/missing.js' does not resolve to a file in the extension package";
    vi.spyOn(CanvasExtensionsService, "install").mockRejectedValue(
      new HttpError(422, detail),
    );

    const { result } = renderHook(() => useRefreshCanvasExtension(), {
      wrapper: createWrapper(),
    });
    result.current.mutate(installed);

    await waitFor(() => expect(displayErrorToast).toHaveBeenCalled());
    expect(displayErrorToast).toHaveBeenCalledWith(detail);
  });
});
