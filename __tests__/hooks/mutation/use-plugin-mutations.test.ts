import React from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { HttpError as ClientHttpError } from "@openhands/typescript-client";
import { beforeEach, describe, expect, it, vi } from "vitest";
import PluginsManagementService from "#/api/plugins-management-service";
import { useInstallPlugin } from "#/hooks/mutation/use-install-plugin";
import { useRefreshPlugin } from "#/hooks/mutation/use-refresh-plugin";
import { useSetPluginEnabled } from "#/hooks/mutation/use-set-plugin-enabled";
import { useUninstallPlugin } from "#/hooks/mutation/use-uninstall-plugin";
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
    super(`HTTP request failed (${status}): ${JSON.stringify({ detail })}`);
    this.name = "HttpError";
    this.status = status;
    this.response = { detail };
  }
}

const SERVER_DETAIL =
  "Failed to fetch plugin source. Check that the source is valid.";

const failingMutations = [
  {
    name: "install",
    method: "installPlugin",
    run: () => {
      const { result } = renderHook(() => useInstallPlugin(), {
        wrapper: createWrapper(),
      });
      result.current.mutate({ source: "qa-not-a-source" });
      return result;
    },
  },
  {
    name: "update",
    method: "refreshPlugin",
    run: () => {
      const { result } = renderHook(() => useRefreshPlugin(), {
        wrapper: createWrapper(),
      });
      result.current.mutate("magic-test");
      return result;
    },
  },
  {
    name: "enable/disable",
    method: "setPluginEnabled",
    run: () => {
      const { result } = renderHook(() => useSetPluginEnabled(), {
        wrapper: createWrapper(),
      });
      result.current.mutate({ name: "magic-test", enabled: false });
      return result;
    },
  },
  {
    name: "uninstall",
    method: "uninstallPlugin",
    run: () => {
      const { result } = renderHook(() => useUninstallPlugin(), {
        wrapper: createWrapper(),
      });
      result.current.mutate("magic-test");
      return result;
    },
  },
] as const;

describe("plugin mutations", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    toastMock.mockClear();
  });

  it.each(failingMutations)(
    "a failed $name shows one toast with the server's reason",
    async ({ method, run }) => {
      vi.spyOn(PluginsManagementService, method).mockRejectedValue(
        new HttpError(400, SERVER_DETAIL),
      );

      const result = run();

      await waitFor(() => expect(result.current.isError).toBe(true));
      expect(errorToastMessages()).toEqual([SERVER_DETAIL]);
    },
  );

  it.each([
    { label: "an empty body", body: "" },
    {
      label: "a plain-text body",
      body: "Bad Gateway: connect ECONNREFUSED 127.0.0.1:18811",
    },
  ])(
    "shows the fallback, not the raw transport text, when the server answers with $label",
    async ({ body }) => {
      vi.spyOn(PluginsManagementService, "installPlugin").mockRejectedValue(
        new ClientHttpError(
          502,
          "Bad Gateway",
          body,
          `HTTP request failed (502 Bad Gateway): ${JSON.stringify(body)}`,
        ),
      );

      const result = failingMutations[0].run();

      await waitFor(() => expect(result.current.isError).toBe(true));
      expect(errorToastMessages()).toEqual([I18nKey.ERROR$GENERIC]);
    },
  );

  it("keeps the shared disconnect wording when the request never lands", async () => {
    vi.spyOn(PluginsManagementService, "installPlugin").mockRejectedValue(
      new TypeError("Failed to fetch"),
    );

    const result = failingMutations[0].run();

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(errorToastMessages()).toEqual([
      i18n.t(I18nKey.ERROR$CORS_OR_NETWORK),
    ]);
  });
});
