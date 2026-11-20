import React from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { AxiosError } from "axios";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useDeleteMcpServer } from "#/hooks/mutation/use-delete-mcp-server";
import { useUpdateMcpServer } from "#/hooks/mutation/use-update-mcp-server";
import { createAgentServerQueryClient } from "#/query-client-config";
import type { MCPServerConfig } from "#/types/mcp-server";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

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

const server: MCPServerConfig = {
  id: "qa_stale",
  type: "stdio",
  name: "qa_stale",
  command: "python3",
};

/** The shape the shared TypeScript client throws for a failed request. */
class HttpError extends Error {
  status: number;

  response: unknown;

  constructor(status: number, detail: string) {
    super(`HTTP request failed (${status}): ${JSON.stringify({ detail })}`);
    this.name = "HttpError";
    this.status = status;
    this.response = { detail };
  }
}

const notFound = () =>
  new HttpError(404, "MCP server 'qa_stale' was not found");

// Every caller of these hooks toasts or shows the failure itself, as below.
const reportError = (error: unknown) =>
  displayErrorToast(retrieveAxiosErrorMessage(error as AxiosError));

describe("MCP server mutations on failure", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    toastMock.mockClear();
  });

  it("shows one error toast when deleting a server fails", async () => {
    vi.spyOn(SettingsService, "deleteMcpServer").mockRejectedValue(notFound());
    const { result } = renderHook(() => useDeleteMcpServer(), {
      wrapper: createWrapper(),
    });

    result.current.mutate(server, { onError: reportError });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(errorToastMessages()).toEqual([
      "MCP server 'qa_stale' was not found",
    ]);
  });

  it("shows one error toast when updating a server fails", async () => {
    vi.spyOn(SettingsService, "patchMcpServer").mockRejectedValue(notFound());
    const { result } = renderHook(() => useUpdateMcpServer(), {
      wrapper: createWrapper(),
    });

    result.current.mutate(
      { serverId: server.id, server: { ...server, enabled: false } },
      { onError: reportError },
    );

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(errorToastMessages()).toEqual([
      'MCP server "qa_stale" no longer exists.',
    ]);
  });
});
