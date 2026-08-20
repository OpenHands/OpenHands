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
