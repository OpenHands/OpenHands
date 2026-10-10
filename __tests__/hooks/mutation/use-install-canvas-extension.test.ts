import React from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import CanvasExtensionsService from "#/api/canvas-extensions-service";
import {
  useInstallCanvasExtension,
  useRefreshCanvasExtension,
} from "#/hooks/mutation/use-manage-canvas-extensions";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import { createAgentServerQueryClient } from "#/query-client-config";

const { toastMock } = vi.hoisted(() => ({
  toastMock: Object.assign(vi.fn(), { success: vi.fn() }),
}));
vi.mock("react-hot-toast", () => ({ default: toastMock }));

/** The messages of every error toast shown, in order. */
const errorToastMessages = () =>
  toastMock.mock.calls.map(
    ([content]) =>
      (content as React.ReactElement<{ message: string }>).props.message,
  );

// The app's query client, so its global mutation error toast is in play too.
const createWrapper = () => {
  const client = createAgentServerQueryClient();
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

  it("shows the server's reason once, even when it mentions a failed fetch", async () => {
    const detail =
      "Could not read canvas extension source: Failed to fetch extension from http://127.0.0.1:9/qa-owner/qa-missing";
    vi.spyOn(CanvasExtensionsService, "install").mockRejectedValue(
      new HttpError(400, detail),
    );

    const { result } = renderHook(() => useInstallCanvasExtension(), {
      wrapper: createWrapper(),
    });
    result.current.mutate({ source: "http://127.0.0.1:9/qa-owner/qa-missing" });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(errorToastMessages()).toEqual([detail]);
  });

  it("keeps the shared disconnect wording when the request never lands", async () => {
    vi.spyOn(CanvasExtensionsService, "install").mockRejectedValue(
      new TypeError("Failed to fetch"),
    );

    const { result } = renderHook(() => useInstallCanvasExtension(), {
      wrapper: createWrapper(),
    });
    result.current.mutate({ source: "/repo" });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(errorToastMessages()).toEqual([
      i18n.t(I18nKey.ERROR$CORS_OR_NETWORK),
    ]);
  });

  it("points to the existing card when the app is already installed", async () => {
    vi.spyOn(CanvasExtensionsService, "install").mockRejectedValue(
      new HttpError(
        409,
        "Canvas extension already installed. Use force=true to overwrite.",
      ),
    );

    const { result } = renderHook(() => useInstallCanvasExtension(), {
      wrapper: createWrapper(),
    });
    result.current.mutate({ source: "/repo/demo-page" });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(errorToastMessages()).toEqual([
      I18nKey.SETTINGS$APPS_ALREADY_INSTALLED,
    ]);
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

  it("shows the server's reason when the recorded source is unreachable", async () => {
    const detail =
      "Could not read canvas extension source: Failed to fetch extension from github:example/apps";
    vi.spyOn(CanvasExtensionsService, "install").mockRejectedValue(
      new HttpError(400, detail),
    );

    const { result } = renderHook(() => useRefreshCanvasExtension(), {
      wrapper: createWrapper(),
    });
    result.current.mutate(installed);

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(errorToastMessages()).toEqual([detail]);
  });
});
