import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AxiosResponse, InternalAxiosRequestConfig } from "axios";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import type { AutomationSpec } from "#/types/automation";
import type { SetupRequestBody } from "#/manifests/types";
import AutomationService from "./automation-service.api";

const PADDED_KEY = "  padded-session-key  ";
const TRIMMED_KEY = "padded-session-key";

// Replace only the network adapter: the real axios instance and the real
// `localAutomationAxios` request interceptor still run, and the adapter
// records the final config each request resolves to. This is what lets the
// suite assert on the headers a real request would carry, rather than on the
// options passed to a mocked axios. It lives in its own file because the
// sibling suite mocks axios module-wide, which would keep the interceptor
// from ever running.
const mocks = vi.hoisted(() => ({
  callCloudProxy: vi.fn(),
  downloadBlob: vi.fn(),
  getTelemetryDistinctId: vi.fn(),
  requests: [] as import("axios").InternalAxiosRequestConfig[],
  failNextPatch: false,
}));

vi.mock("axios", async (importOriginal) => {
  const actual = await importOriginal<typeof import("axios")>();
  actual.default.defaults.adapter = async (
    config: InternalAxiosRequestConfig,
  ): Promise<AxiosResponse> => {
    mocks.requests.push(config);
    if (config.method === "patch" && mocks.failNextPatch) {
      mocks.failNextPatch = false;
      throw new Error("update failed");
    }
    return {
      data:
        config.responseType === "arraybuffer"
          ? new ArrayBuffer(0)
          : {
              id: "created-automation",
              automations: [],
              runs: [],
              sdk_version: "1.36.1",
              tarball_path: "oh-internal://uploads/test.tar",
            },
      status: 200,
      statusText: "OK",
      headers: new actual.default.AxiosHeaders(),
      config,
    };
  };
  return actual;
});

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: mocks.callCloudProxy,
}));

vi.mock("#/services/telemetry", () => ({
  clearPendingLocalTelemetryRevocation: vi.fn(),
  getTelemetryConsent: vi.fn(),
  getTelemetryDistinctId: mocks.getTelemetryDistinctId,
  getTelemetryDistinctIdForConsentSync: vi.fn(),
}));

// `downloadTarball` ends in a DOM download; only that side effect is faked,
// the request it makes still goes through the real pipeline.
vi.mock("#/utils/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/utils/utils")>()),
  downloadBlob: mocks.downloadBlob,
}));

const localBackend: Backend = {
  id: "local-test",
  name: "Local test backend",
  host: "http://localhost:3000",
  apiKey: PADDED_KEY,
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-test",
  name: "Cloud test backend",
  host: "https://app.example.test",
  apiKey: "cloud-api-key",
  kind: "cloud",
};

const spec: AutomationSpec = {
  name: "Imported review",
  prompt: "Review open pull requests.",
  trigger: {
    type: "cron",
    schedule: "0 9 * * *",
    schedule_human: "Daily at 09:00",
  },
  enabled: true,
  repository: "openhands/agent-canvas",
  branch: "main",
  plugins: ["github:openhands/extensions"],
  model: "fast",
  timezone: "America/Los_Angeles",
};

/**
 * One invocation per exported method that can target the local automation
 * service. The enumeration test below fails whenever the class grows a method
 * that is not listed here, so a new method cannot silently skip the header
 * assertions.
 */
const localMethodInvocations: Record<string, () => Promise<unknown>> = {
  syncTelemetryConsent: () => AutomationService.syncTelemetryConsent("granted"),
  getSdkVersion: () => AutomationService.getSdkVersion(),
  listAutomations: () => AutomationService.listAutomations(),
  getAutomations: () => AutomationService.getAutomations(),
  getAutomation: () => AutomationService.getAutomation("automation-1"),
  createAutomation: () => AutomationService.createAutomation(spec),
  updateAutomation: () =>
    AutomationService.updateAutomation("automation-1", { enabled: true }),
  deleteAutomation: () => AutomationService.deleteAutomation("automation-1"),
  dispatchAutomation: () =>
    AutomationService.dispatchAutomation("automation-1"),
  cancelAutomationRun: () => AutomationService.cancelAutomationRun("run-1"),
  listAutomationRuns: () =>
    AutomationService.listAutomationRuns("automation-1"),
  getAutomationRuns: () => AutomationService.getAutomationRuns("automation-1"),
  toggleAutomation: () =>
    AutomationService.toggleAutomation("automation-1", false),
  fetchTarballBytes: () => AutomationService.fetchTarballBytes("automation-1"),
  downloadTarball: () =>
    AutomationService.downloadTarball("automation-1", "bundle"),
  getCapabilities: () => AutomationService.getCapabilities(),
  validateDraft: () => AutomationService.validateDraft({} as SetupRequestBody),
  createAutomationDraft: () =>
    AutomationService.createAutomationDraft({} as SetupRequestBody),
  uploadAutomationTarball: () =>
    AutomationService.uploadAutomationTarball("bundle.tar", new Uint8Array()),
  getGitSyncStatus: () => AutomationService.getGitSyncStatus(),
  updateGitSyncConfig: () =>
    AutomationService.updateGitSyncConfig({ branch: "main" }),
  checkGitSyncConfig: () =>
    AutomationService.checkGitSyncConfig({ branch: "main" }),
  triggerGitSync: () => AutomationService.triggerGitSync(),
  checkHealth: () => AutomationService.checkHealth(),
};

async function invokeLocal(
  name: string,
  invoke: () => Promise<unknown>,
): Promise<InternalAxiosRequestConfig[]> {
  mocks.requests.length = 0;
  try {
    await invoke();
  } catch (error) {
    throw new Error(
      `AutomationService.${name} threw during invocation: ${String(error)}`,
    );
  }
  expect(
    mocks.requests.length,
    `AutomationService.${name} should issue a local request`,
  ).toBeGreaterThan(0);
  return [...mocks.requests];
}

describe("AutomationService local session-key coverage", () => {
  beforeEach(() => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    mocks.requests.length = 0;
    mocks.failNextPatch = false;
    mocks.getTelemetryDistinctId.mockResolvedValue("ph-distinct-id");
    mocks.callCloudProxy.mockResolvedValue({});
  });

  afterEach(() => {
    setActiveSelection(null);
    setRegisteredBackends([]);
    vi.clearAllMocks();
  });

  it("enumerates every exported method so a new method cannot skip the suite", () => {
    const exportedMethods = Object.getOwnPropertyNames(
      AutomationService,
    ).filter(
      (name) =>
        !["length", "name", "prototype"].includes(name) &&
        typeof (AutomationService as unknown as Record<string, unknown>)[
          name
        ] === "function",
    );

    expect(Object.keys(localMethodInvocations).sort()).toEqual(
      exportedMethods.sort(),
    );
  });

  it("sends the trimmed session key on every local request of every method", async () => {
    for (const [name, invoke] of Object.entries(localMethodInvocations)) {
      const requests = await invokeLocal(name, invoke);
      for (const request of requests) {
        expect(
          request.headers.get("X-Session-API-Key"),
          `${name} -> ${String(request.method).toUpperCase()} ${request.url}`,
        ).toBe(TRIMMED_KEY);
      }
    }
  });

  it("omits the header on every local request when the key is blank", async () => {
    setRegisteredBackends([{ ...localBackend, apiKey: "   " }]);

    for (const [name, invoke] of Object.entries(localMethodInvocations)) {
      const requests = await invokeLocal(name, invoke);
      for (const request of requests) {
        expect(
          request.headers.get("X-Session-API-Key"),
          `${name} -> ${String(request.method).toUpperCase()} ${request.url}`,
        ).toBeUndefined();
      }
    }
  });

  it("keeps the pinned import requests on the key, including the cleanup delete", async () => {
    mocks.failNextPatch = true;

    await expect(AutomationService.createAutomation(spec)).rejects.toThrow(
      "update failed",
    );

    expect(mocks.requests.map((request) => request.method)).toEqual([
      "post",
      "patch",
      "delete",
    ]);
    for (const request of mocks.requests) {
      expect(request.baseURL).toBe(localBackend.host);
      expect(request.headers.get("X-Session-API-Key")).toBe(TRIMMED_KEY);
    }
  });

  it("leaves the Cloud paths off the session-key header", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
    mocks.callCloudProxy.mockResolvedValue({ sdk_version: "1.36.2" });

    await AutomationService.getSdkVersion();

    expect(mocks.callCloudProxy).toHaveBeenCalledTimes(1);
    const proxyCall = mocks.callCloudProxy.mock.calls[0][0] as {
      headers: Record<string, string>;
    };
    expect(proxyCall.headers["X-Session-API-Key"]).toBeUndefined();
    expect(mocks.requests).toHaveLength(0);

    await AutomationService.uploadAutomationTarball(
      "bundle.tar",
      new Uint8Array(),
    );

    // The upload posts straight to the cloud host, not through the proxy.
    expect(mocks.callCloudProxy).toHaveBeenCalledTimes(1);
    expect(mocks.requests).toHaveLength(1);
    const [upload] = mocks.requests;
    expect(upload.url).toBe(
      "https://app.example.test/api/automation/v1/uploads?name=bundle.tar",
    );
    expect(upload.headers.get("Authorization")).toBe("Bearer cloud-api-key");
    expect(upload.headers.get("X-Org-Id")).toBe("org-1");
    expect(upload.headers.get("X-Session-API-Key")).toBeUndefined();
  });
});
