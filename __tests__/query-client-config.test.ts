import { AxiosError } from "axios";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createAgentServerQueryClient } from "#/query-client-config";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import * as ToastHandlers from "#/utils/custom-toast-handlers";

afterEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  __resetActiveStoreForTests();
  vi.restoreAllMocks();
});

describe("createAgentServerQueryClient", () => {
  it("does not show a toast when query meta disables toasts", async () => {
    const toastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");
    const client = createAgentServerQueryClient();

    await expect(
      client.fetchQuery({
        queryKey: ["config", "suppressed"],
        queryFn: async () => {
          throw new AxiosError("suppressed query error");
        },
        meta: { disableToast: true },
        retry: false,
      }),
    ).rejects.toThrow("suppressed query error");

    expect(toastSpy).not.toHaveBeenCalled();
  });

  it("shows a toast when query meta does not disable toasts", async () => {
    const toastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");
    const client = createAgentServerQueryClient();

    await expect(
      client.fetchQuery({
        queryKey: ["config", "toast"],
        queryFn: async () => {
          throw new AxiosError("query error with toast");
        },
        retry: false,
      }),
    ).rejects.toThrow("query error with toast");

    expect(toastSpy).toHaveBeenCalledWith("query error with toast");
  });

  it.each([
    { status: 503, cached: true, shouldToast: false },
    { status: 503, cached: false, shouldToast: true },
    { status: 403, cached: true, shouldToast: true },
  ])(
    "handles refresh failure $status with cached=$cached",
    async ({ status, cached, shouldToast }) => {
      const toastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");
      const client = createAgentServerQueryClient();
      const queryKey = ["refresh", status, cached];
      const original = { title: "Existing conversation" };
      if (cached) client.setQueryData(queryKey, original);
      const message = `refresh-${status}-${cached}`;
      await expect(
        client.fetchQuery({
          queryKey,
          queryFn: async () => {
            throw new AxiosError(
              message,
              "ERR_BAD_RESPONSE",
              undefined,
              undefined,
              { status } as never,
            );
          },
          retry: false,
        }),
      ).rejects.toThrow(message);
      expect(toastSpy).toHaveBeenCalledTimes(shouldToast ? 1 : 0);
      expect(client.getQueryData(queryKey)).toEqual(
        cached ? original : undefined,
      );
      if (cached) {
        await client.fetchQuery({
          queryKey,
          queryFn: async () => ({ title: "Recovered" }),
        });
        expect(client.getQueryData(queryKey)).toEqual({ title: "Recovered" });
      }
      client.clear();
    },
  );

  it("does not show raw 401 toasts while the active cloud backend is logged out", async () => {
    const toastSpy = vi.spyOn(ToastHandlers, "displayErrorToast");
    const backend = {
      id: "cloud-expired",
      name: "OpenHands Cloud",
      host: "https://app.all-hands.dev",
      apiKey: "expired-token",
      kind: "cloud",
    };
    window.localStorage.setItem(
      "openhands-backends",
      JSON.stringify([backend]),
    );
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: backend.id, orgId: null }),
    );
    window.sessionStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: backend.id, orgId: null }),
    );
    __resetActiveStoreForTests();
    const client = createAgentServerQueryClient();

    await expect(
      client.fetchQuery({
        queryKey: ["cloud", "logged-out"],
        queryFn: async () => {
          throw new AxiosError(
            "Request failed with status code 401",
            "ERR_BAD_REQUEST",
            undefined,
            undefined,
            { status: 401 } as never,
          );
        },
        retry: false,
      }),
    ).rejects.toThrow("Request failed with status code 401");

    expect(toastSpy).not.toHaveBeenCalled();
  });
});
