import React from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { AxiosError, AxiosHeaders } from "axios";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useDeleteMcpServer } from "#/hooks/mutation/use-delete-mcp-server";
import { useUpdateMcpServer } from "#/hooks/mutation/use-update-mcp-server";
import { useAddMcpServer } from "#/hooks/mutation/use-add-mcp-server";
import { createAgentServerQueryClient } from "#/query-client-config";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import {
  MCP_RENAME_CREDENTIAL_ERROR,
  REDACTED_MCP_SECRET_VALUE,
} from "#/utils/mcp-config";
import type { MCPServerConfig } from "#/types/mcp-server";

const { toastMock } = vi.hoisted(() => ({
  toastMock: Object.assign(vi.fn(), { success: vi.fn() }),
}));
vi.mock("react-hot-toast", () => ({ default: toastMock }));

const useSettingsMock = vi.fn();
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

const errorToastMessages = () =>
  toastMock.mock.calls.map(
    ([content]) =>
      (content as React.ReactElement<{ message: string }>).props.message,
  );

const createWrapper = () => {
  const client = createAgentServerQueryClient();
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(QueryClientProvider, { client }, children);
  };
};

const createAxios404Error = (message: string) => {
  const headers = new AxiosHeaders({
    "content-type": "application/json",
  });
  return new AxiosError(
    message,
    "ERR_BAD_REQUEST",
    { headers },
    {},
    {
      status: 404,
      statusText: "Not Found",
      data: { error: message },
      headers: {},
      config: { headers },
    },
  );
};

describe("MCP mutation toast deduplication", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    toastMock.mockClear();
    useSettingsMock.mockReturnValue({
      data: {
        mcp_config: {
          qa_stale: {
            transport: "stdio",
            command: "node",
            env: { API_KEY: REDACTED_MCP_SECRET_VALUE },
          },
        },
      },
    });
  });

  it("shows exactly one toast when deleting an MCP server fails", async () => {
    const error = createAxios404Error("MCP server 'qa_stale' was not found");
    vi.spyOn(SettingsService, "deleteMcpServer").mockRejectedValue(error);

    const { result } = renderHook(() => useDeleteMcpServer(), {
      wrapper: createWrapper(),
    });

    const target: MCPServerConfig = {
      id: "qa_stale",
      type: "stdio",
      name: "qa_stale",
      command: "node",
    };

    result.current.mutate(target, {
      onError: (err) => {
        const message = retrieveAxiosErrorMessage(err as AxiosError);
        displayErrorToast(message || "An error occurred");
      },
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(errorToastMessages()).toEqual([
      "MCP server 'qa_stale' was not found",
    ]);
  });

  it("shows exactly one toast when updating an MCP server fails", async () => {
    const error = createAxios404Error("MCP server 'qa_stale' was not found");
    vi.spyOn(SettingsService, "patchMcpServer").mockRejectedValue(error);

    const { result } = renderHook(() => useUpdateMcpServer(), {
      wrapper: createWrapper(),
    });

    const target: MCPServerConfig = {
      id: "qa_stale",
      type: "stdio",
      name: "qa_stale",
      command: "node",
    };

    result.current.mutate(
      { serverId: "qa_stale", server: target },
      {
        onError: (err) => {
          const message = retrieveAxiosErrorMessage(err as AxiosError);
          displayErrorToast(message || "An error occurred");
        },
      },
    );

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(errorToastMessages()).toEqual([
      "MCP server 'qa_stale' was not found",
    ]);
  });

  it("shows exactly one toast when adding an MCP server fails and caller handles error", async () => {
    const error = createAxios404Error("Failed to add MCP server");
    vi.spyOn(SettingsService, "createMcpServer").mockRejectedValue(error);

    const { result } = renderHook(() => useAddMcpServer(), {
      wrapper: createWrapper(),
    });

    const target: MCPServerConfig = {
      id: "qa_new",
      type: "stdio",
      name: "qa_new",
      command: "node",
    };

    result.current.mutate(target, {
      onError: (err) => {
        const message = retrieveAxiosErrorMessage(err as AxiosError);
        displayErrorToast(message || "An error occurred");
      },
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(errorToastMessages()).toEqual(["Failed to add MCP server"]);
  });

  it("does not show a global toast when caller handles error inline without toasting", async () => {
    const error = createAxios404Error("Inline failure");
    vi.spyOn(SettingsService, "createMcpServer").mockRejectedValue(error);

    const { result } = renderHook(() => useAddMcpServer(), {
      wrapper: createWrapper(),
    });

    const target: MCPServerConfig = {
      id: "qa_inline",
      type: "stdio",
      name: "qa_inline",
      command: "node",
    };

    let inlineError = "";
    result.current.mutate(target, {
      onError: (err) => {
        inlineError = retrieveAxiosErrorMessage(err as AxiosError);
      },
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(inlineError).toBe("Inline failure");
    expect(errorToastMessages()).toEqual([]);
  });

  it("shows exactly one toast for client-side MCP_RENAME_CREDENTIAL_ERROR", async () => {
    const { result } = renderHook(() => useUpdateMcpServer(), {
      wrapper: createWrapper(),
    });

    const target: MCPServerConfig = {
      id: "qa_stale",
      type: "stdio",
      name: "new_name",
      command: "node",
    };

    result.current.mutate(
      { serverId: "qa_stale", server: target },
      {
        onError: (err) => {
          if (
            err instanceof Error &&
            err.message === MCP_RENAME_CREDENTIAL_ERROR
          ) {
            displayErrorToast(err.message);
            return;
          }
          const message = retrieveAxiosErrorMessage(err as AxiosError);
          displayErrorToast(message || "An error occurred");
        },
      },
    );

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(errorToastMessages()).toHaveLength(1);
  });
});
