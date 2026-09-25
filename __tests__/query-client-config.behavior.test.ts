import { QueryClient } from "@tanstack/react-query";
import { AxiosError } from "axios";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createAgentServerQueryClient } from "#/query-client-config";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import {
  __resetHealthStoreForTests,
  getBackendHealthEntry,
  recordBackendFailure,
} from "#/api/backend-registry/health-store";
import * as ToastHandlers from "#/utils/custom-toast-handlers";

interface ErrorOptions {
  directStatus?: boolean;
  message?: string;
  status?: number;
  url?: string;
}

function createAxiosError({
  directStatus = false,
  message = "Request failed",
  status,
  url,
}: ErrorOptions = {}) {
  const error = new AxiosError(
    message,
    "ERR_BAD_REQUEST",
    url ? ({ url } as never) : undefined,
    undefined,
    !directStatus && status !== undefined ? ({ status } as never) : undefined,
  );
  error.status = directStatus ? status : undefined;
  return error;
}

function createBackend(overrides: Partial<Backend> = {}): Backend {
  return {
    id: "local-backend",
    name: "Local Backend",
    host: "http://localhost:3000",
    apiKey: "test-key",
    kind: "local",
    ...overrides,
  };
}

function activateBackend(backend: Backend) {
  const selection = { backendId: backend.id, orgId: null };
  window.localStorage.setItem("openhands-backends", JSON.stringify([backend]));
  window.localStorage.setItem(
    "openhands-active-backend",
    JSON.stringify(selection),
  );
  window.sessionStorage.setItem(
    "openhands-active-backend",
    JSON.stringify(selection),
  );
  __resetActiveStoreForTests();
}

function executeFailingQuery(
  client: QueryClient,
  error: unknown,
  {
    meta,
    queryKey = ["behavior", "failure"],
  }: {
    meta?: Record<string, unknown>;
    queryKey?: readonly unknown[];
  } = {},
) {
  return client.fetchQuery({
    queryKey,
    queryFn: async () => {
      throw error;
    },
    meta,
    retry: false,
  });
}

function executeFailingMutation(
  client: QueryClient,
  error: unknown,
  meta?: Record<string, unknown>,
) {
  const mutation = client.getMutationCache().build(client, {
    mutationFn: async () => {
      throw error;
    },
    meta,
    retry: false,
  });
  return mutation.execute(undefined);
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  window.localStorage.clear();
  window.sessionStorage.clear();
  delete (window as typeof window & { __OH_QUERY_CLIENT__?: QueryClient })
    .__OH_QUERY_CLIENT__;
  __resetActiveStoreForTests();
  __resetHealthStoreForTests();
});

describe("query client behavior", () => {
  it("records successful queries for string backend identifiers only", async () => {
    const client = createAgentServerQueryClient();
    const malformedBackendId = 42;
    const malformedMeta = { backendId: malformedBackendId } as unknown as {
      backendId: string;
    };
    recordBackendFailure("backend-one", new Error("offline"));
    recordBackendFailure(
      malformedBackendId as unknown as string,
      new Error("offline"),
    );

    await client.fetchQuery({
      queryKey: ["health", "backend-one"],
      queryFn: async () => "healthy",
      meta: { backendId: "backend-one" },
    });
    await client.fetchQuery({
      queryKey: ["health", "backend-two"],
      queryFn: async () => "healthy",
      meta: malformedMeta,
    });
    await client.fetchQuery({
      queryKey: ["health", "unattributed"],
      queryFn: async () => "healthy",
    });

    expect(getBackendHealthEntry("backend-one")).toBeNull();
    expect(
      getBackendHealthEntry(malformedBackendId as unknown as string),
    ).not.toBeNull();
  });

  it.each([
    { queryKey: ["settings"], description: "an unrelated query" },
    { queryKey: ["user", "profile"], description: "another user query" },
    {
      queryKey: ["settings", "authenticated"],
      description: "a non-user query ending in authenticated",
    },
  ])(
    "invalidates authentication after a 401 from $description",
    async ({ queryKey }) => {
      const client = createAgentServerQueryClient();
      const invalidateQueries = vi
        .spyOn(client, "invalidateQueries")
        .mockResolvedValue();
      const error = createAxiosError({ status: 401 });

      await expect(
        executeFailingQuery(client, error, {
          meta: { disableToast: true },
          queryKey,
        }),
      ).rejects.toBe(error);

      expect(invalidateQueries).toHaveBeenCalledWith({
        queryKey: ["user", "authenticated"],
      });
    },
  );

  it("does not recursively invalidate authentication when that query fails", async () => {
    const client = createAgentServerQueryClient();
    const invalidateQueries = vi
      .spyOn(client, "invalidateQueries")
      .mockResolvedValue();
    const error = createAxiosError({ status: 401 });

    await expect(
      executeFailingQuery(client, error, {
        meta: { disableToast: true },
        queryKey: ["user", "authenticated"],
      }),
    ).rejects.toBe(error);

    expect(invalidateQueries).not.toHaveBeenCalled();
  });

  it("recognizes a direct Axios status when a mutation receives a 401", async () => {
    const client = createAgentServerQueryClient();
    const invalidateQueries = vi
      .spyOn(client, "invalidateQueries")
      .mockResolvedValue();
    const error = createAxiosError({ directStatus: true, status: 401 });

    await expect(
      executeFailingMutation(client, error, { disableToast: true }),
    ).rejects.toBe(error);

    expect(invalidateQueries).toHaveBeenCalledWith({
      queryKey: ["user", "authenticated"],
    });
  });

  it("does not invalidate authentication for non-401 failures", async () => {
    const client = createAgentServerQueryClient();
    const invalidateQueries = vi
      .spyOn(client, "invalidateQueries")
      .mockResolvedValue();
    const error = createAxiosError({ status: 500 });

    await expect(
      executeFailingQuery(client, error, { meta: { disableToast: true } }),
    ).rejects.toBe(error);

    expect(invalidateQueries).not.toHaveBeenCalled();
  });

  it("preserves null mutation failures without invalidating authentication", async () => {
    const toast = vi.spyOn(ToastHandlers, "displayErrorToast");
    const client = createAgentServerQueryClient();
    const invalidateQueries = vi
      .spyOn(client, "invalidateQueries")
      .mockResolvedValue();

    await expect(executeFailingMutation(client, null)).rejects.toBeNull();

    expect(invalidateQueries).not.toHaveBeenCalled();
    expect(toast).toHaveBeenCalledWith(expect.any(String));
  });

  it.each([
    {
      directStatus: false,
      url: undefined,
      description: "has no request URL",
    },
    {
      directStatus: false,
      url: "https://cloud.example/api/conversations",
      description: "targets the active cloud host",
    },
    {
      directStatus: true,
      url: "https://cloud.example/api/settings",
      description: "reports 401 directly for the active cloud host",
    },
  ])(
    "suppresses a cloud authentication toast when the request $description",
    async ({ directStatus, url }) => {
      activateBackend(
        createBackend({
          id: "cloud-backend",
          name: "Cloud Backend",
          host: "https://cloud.example///",
          kind: "cloud",
        }),
      );
      const toast = vi.spyOn(ToastHandlers, "displayErrorToast");
      const client = createAgentServerQueryClient();
      const error = createAxiosError({
        directStatus,
        message: "Cloud authentication failed",
        status: 401,
        url,
      });

      await expect(
        executeFailingQuery(client, error, {
          queryKey: ["cloud", url ?? "missing-url"],
        }),
      ).rejects.toBe(error);

      expect(toast).not.toHaveBeenCalled();
    },
  );

  it("shows a non-authentication error from the active cloud host", async () => {
    activateBackend(
      createBackend({
        id: "cloud-backend",
        host: "https://cloud.example",
        kind: "cloud",
      }),
    );
    const toast = vi.spyOn(ToastHandlers, "displayErrorToast");
    const client = createAgentServerQueryClient();
    const error = createAxiosError({
      message: "Cloud service unavailable",
      status: 503,
      url: "https://cloud.example/api/settings",
    });

    await expect(
      executeFailingQuery(client, error, {
        queryKey: ["cloud", "service-unavailable"],
      }),
    ).rejects.toBe(error);

    expect(toast).toHaveBeenCalledWith("Cloud service unavailable");
  });

  it("shows a 401 toast when a cloud request targets another host", async () => {
    activateBackend(
      createBackend({
        id: "cloud-backend",
        host: "https://cloud.example",
        kind: "cloud",
      }),
    );
    const toast = vi.spyOn(ToastHandlers, "displayErrorToast");
    const client = createAgentServerQueryClient();
    const error = createAxiosError({
      message: "Foreign host authentication failed",
      status: 401,
      url: "https://different.example/api",
    });

    await expect(
      executeFailingQuery(client, error, {
        queryKey: ["cloud", "foreign-host"],
      }),
    ).rejects.toBe(error);

    expect(toast).toHaveBeenCalledWith("Foreign host authentication failed");
  });

  it("shows a 401 toast for an active local backend", async () => {
    activateBackend(createBackend());
    const toast = vi.spyOn(ToastHandlers, "displayErrorToast");
    const client = createAgentServerQueryClient();
    const error = createAxiosError({
      message: "Local authentication failed",
      status: 401,
    });

    await expect(
      executeFailingQuery(client, error, {
        queryKey: ["local", "authentication"],
      }),
    ).rejects.toBe(error);

    expect(toast).toHaveBeenCalledWith("Local authentication failed");
  });

  it("deduplicates query toasts until the cooldown expires", async () => {
    vi.useFakeTimers();
    const toast = vi.spyOn(ToastHandlers, "displayErrorToast");
    const client = createAgentServerQueryClient();
    const first = new AxiosError("Repeated query failure");
    const second = new AxiosError("Repeated query failure");

    await expect(
      executeFailingQuery(client, first, {
        queryKey: ["dedupe", "first"],
      }),
    ).rejects.toBe(first);
    await expect(
      executeFailingQuery(client, second, {
        queryKey: ["dedupe", "second"],
      }),
    ).rejects.toBe(second);
    expect(toast).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(3000);
    const afterCooldown = new AxiosError("Repeated query failure");
    await expect(
      executeFailingQuery(client, afterCooldown, {
        queryKey: ["dedupe", "after-cooldown"],
      }),
    ).rejects.toBe(afterCooldown);

    expect(toast).toHaveBeenCalledTimes(2);
  });

  it("uses the translated generic query error when no message is available", async () => {
    vi.useFakeTimers();
    const toast = vi.spyOn(ToastHandlers, "displayErrorToast");
    const client = createAgentServerQueryClient();
    const first = {};
    const duplicate = {};

    await expect(
      executeFailingQuery(client, first, {
        queryKey: ["generic", "first"],
      }),
    ).rejects.toBe(first);
    await expect(
      executeFailingQuery(client, duplicate, {
        queryKey: ["generic", "duplicate"],
      }),
    ).rejects.toBe(duplicate);

    expect(toast).toHaveBeenCalledWith(expect.any(String));
    expect(toast).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(3000);
    const afterCooldown = {};
    await expect(
      executeFailingQuery(client, afterCooldown, {
        queryKey: ["generic", "after-cooldown"],
      }),
    ).rejects.toBe(afterCooldown);

    expect(toast).toHaveBeenCalledTimes(2);
  });

  it("honors mutation toast metadata", async () => {
    const toast = vi.spyOn(ToastHandlers, "displayErrorToast");
    const client = createAgentServerQueryClient();
    const visible = new AxiosError("Visible mutation failure");
    const suppressed = new AxiosError("Suppressed mutation failure");

    await expect(executeFailingMutation(client, visible)).rejects.toBe(visible);
    await expect(
      executeFailingMutation(client, suppressed, { disableToast: true }),
    ).rejects.toBe(suppressed);

    expect(toast).toHaveBeenCalledOnce();
    expect(toast).toHaveBeenCalledWith("Visible mutation failure");
  });

  it("suppresses matching cloud-auth mutation errors", async () => {
    activateBackend(
      createBackend({
        id: "cloud-backend",
        host: "https://cloud.example",
        kind: "cloud",
      }),
    );
    const toast = vi.spyOn(ToastHandlers, "displayErrorToast");
    const client = createAgentServerQueryClient();
    const error = createAxiosError({
      message: "Cloud mutation authentication failed",
      status: 401,
      url: "https://cloud.example/api/settings",
    });

    await expect(executeFailingMutation(client, error)).rejects.toBe(error);

    expect(toast).not.toHaveBeenCalled();
  });

  it("uses the translated generic mutation error when no message is available", async () => {
    const toast = vi.spyOn(ToastHandlers, "displayErrorToast");
    const client = createAgentServerQueryClient();
    const error = {};

    await expect(executeFailingMutation(client, error)).rejects.toBe(error);

    expect(toast).toHaveBeenCalledWith(expect.any(String));
  });
});

describe("query client selection and proxy behavior", () => {
  it("creates one default client and exposes it in development", async () => {
    vi.resetModules();
    const config = await import("#/query-client-config");

    const first = config.getDefaultQueryClient();
    const second = config.getDefaultQueryClient();

    expect(first).toBeInstanceOf(QueryClient);
    expect(second).toBe(first);
    expect(config.getQueryClient()).toBe(first);
    expect(
      (window as typeof window & { __OH_QUERY_CLIENT__?: QueryClient })
        .__OH_QUERY_CLIENT__,
    ).toBe(first);
  });

  it("selects custom clients and forwards proxy reads, calls, and writes", async () => {
    vi.resetModules();
    const config = await import("#/query-client-config");
    const custom = new QueryClient();

    expect(config.setQueryClient(custom)).toBe(custom);
    expect(config.getQueryClient()).toBe(custom);

    config.queryClient.setQueryData(["proxy", "value"], "forwarded");
    expect(custom.getQueryData(["proxy", "value"])).toBe("forwarded");

    const extendedProxy = config.queryClient as QueryClient & {
      marker?: string;
    };
    const extendedClient = custom as QueryClient & { marker?: string };
    extendedProxy.marker = "proxy-write";
    expect(extendedProxy.marker).toBe("proxy-write");
    expect(extendedClient.marker).toBe("proxy-write");

    expect(config.setQueryClient(undefined)).toBe(
      config.getDefaultQueryClient(),
    );
    expect(config.setQueryClient(null)).toBe(config.getDefaultQueryClient());
  });

  it.each([
    { mockApi: "true", expectedExposure: true },
    { mockApi: "false", expectedExposure: false },
  ])(
    "sets window exposure to $expectedExposure outside development when VITE_MOCK_API is $mockApi",
    async ({ expectedExposure, mockApi }) => {
      vi.stubEnv("DEV", false);
      vi.stubEnv("VITE_MOCK_API", mockApi);
      vi.resetModules();
      const config = await import("#/query-client-config");

      const client = config.getDefaultQueryClient();
      const exposed = (
        window as typeof window & { __OH_QUERY_CLIENT__?: QueryClient }
      ).__OH_QUERY_CLIENT__;

      if (expectedExposure) {
        expect(exposed).toBe(client);
      } else {
        expect(exposed).toBeUndefined();
      }
    },
  );
});
