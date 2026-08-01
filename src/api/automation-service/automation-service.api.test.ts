import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import type { Automation, AutomationSpec } from "#/types/automation";
import AutomationService from "./automation-service.api";

const {
  localAxios,
  callCloudProxy,
  clearPendingLocalTelemetryRevocation,
  getTelemetryConsent,
  getTelemetryDistinctId,
  getTelemetryDistinctIdForConsentSync,
} = vi.hoisted(() => ({
  localAxios: {
    interceptors: { request: { use: vi.fn() } },
    get: vi.fn(),
    post: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn(),
  },
  callCloudProxy: vi.fn(),
  clearPendingLocalTelemetryRevocation: vi.fn(),
  getTelemetryConsent: vi.fn(),
  getTelemetryDistinctId: vi.fn(),
  getTelemetryDistinctIdForConsentSync: vi.fn(),
}));

vi.mock("axios", () => ({
  default: {
    create: () => localAxios,
    post: vi.fn(),
  },
}));

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy,
}));

vi.mock("#/services/telemetry", () => ({
  clearPendingLocalTelemetryRevocation,
  getTelemetryConsent,
  getTelemetryDistinctId,
  getTelemetryDistinctIdForConsentSync,
}));

const localBackend: Backend = {
  id: "local-test",
  name: "Local test backend",
  host: "http://localhost:3000",
  apiKey: "test-session-key",
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

const createdAutomation: Automation = {
  id: "created-automation",
  name: spec.name,
  prompt: spec.prompt,
  trigger: { type: "cron", schedule: spec.trigger.schedule },
  enabled: true,
  model: spec.model,
  created_at: "2026-07-10T00:00:00Z",
  updated_at: "2026-07-10T00:00:00Z",
};

describe("AutomationService.getSdkVersion", () => {
  beforeEach(() => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    setActiveSelection(null);
    setRegisteredBackends([]);
    vi.clearAllMocks();
  });

  it("fetches the local automation SDK version from the automation sidecar", async () => {
    localAxios.get.mockResolvedValueOnce({ data: { sdk_version: "1.36.1" } });

    await expect(AutomationService.getSdkVersion()).resolves.toBe("1.36.1");

    expect(localAxios.get).toHaveBeenCalledWith("/api/automation/sdk-version", {
      timeout: 5000,
    });
  });

  it("fetches the cloud automation SDK version through the cloud proxy", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
    callCloudProxy.mockResolvedValueOnce({ sdk_version: "1.36.2" });

    await expect(AutomationService.getSdkVersion()).resolves.toBe("1.36.2");

    expect(callCloudProxy).toHaveBeenCalledWith(
      expect.objectContaining({
        backend: cloudBackend,
        method: "GET",
        path: "/api/automation/sdk-version",
        headers: expect.objectContaining({ "X-Org-Id": "org-1" }),
        timeoutSeconds: 5,
      }),
    );
  });

  it("returns null when the SDK version endpoint is unavailable", async () => {
    localAxios.get.mockRejectedValueOnce(new Error("not running"));

    await expect(AutomationService.getSdkVersion()).resolves.toBeNull();
  });
});

describe("AutomationService.syncTelemetryConsent", () => {
  beforeEach(() => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    getTelemetryConsent.mockReturnValue("pending");
    getTelemetryDistinctIdForConsentSync.mockResolvedValue("ph-fe-sync");
    localAxios.post.mockResolvedValue({ data: { consent_granted: true } });
  });

  afterEach(() => {
    setActiveSelection(null);
    setRegisteredBackends([]);
    vi.clearAllMocks();
  });

  it("posts local telemetry consent with the frontend PostHog distinct ID", async () => {
    await AutomationService.syncTelemetryConsent("granted");

    expect(localAxios.post).toHaveBeenCalledWith(
      "/api/automation/v1/telemetry/consent",
      {
        consent_granted: true,
        frontend_distinct_id: "ph-fe-sync",
      },
      { timeout: 5000 },
    );
  });

  it("uses the current telemetry consent when no explicit value is supplied", async () => {
    getTelemetryConsent.mockReturnValue("denied");

    await AutomationService.syncTelemetryConsent();

    expect(localAxios.post).toHaveBeenCalledWith(
      "/api/automation/v1/telemetry/consent",
      {
        consent_granted: false,
        frontend_distinct_id: "ph-fe-sync",
      },
      { timeout: 5000 },
    );
    expect(clearPendingLocalTelemetryRevocation).toHaveBeenCalledWith(
      "ph-fe-sync",
    );
  });

  it("skips cloud backends because cloud consent is handled by auth", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });

    await AutomationService.syncTelemetryConsent("granted");

    expect(localAxios.post).not.toHaveBeenCalled();
  });
});

describe("AutomationService.createAutomation", () => {
  beforeEach(() => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
    localAxios.post.mockResolvedValue({ data: createdAutomation });
    localAxios.patch.mockImplementation(
      async (_path: string, body: Partial<Automation>) => ({
        data: { ...createdAutomation, ...body },
      }),
    );
  });

  afterEach(() => {
    setActiveSelection(null);
    setRegisteredBackends([]);
    vi.clearAllMocks();
  });

  it("creates plugin automations through the preset API and disables them", async () => {
    const created = await AutomationService.createAutomation(spec);

    expect(localAxios.post).toHaveBeenCalledWith(
      "/api/automation/v1/preset/plugin",
      {
        name: spec.name,
        prompt: spec.prompt,
        model: spec.model,
        trigger: {
          type: "event",
          source: "agent-canvas-import",
          on: expect.stringMatching(/^pending\./),
        },
        repos: [
          {
            url: spec.repository,
            ref: spec.branch,
            provider: "github",
          },
        ],
        plugins: [{ source: spec.plugins![0] }],
      },
      {
        baseURL: localBackend.host,
        headers: expect.objectContaining({
          "X-Session-API-Key": localBackend.apiKey,
        }),
      },
    );
    expect(localAxios.patch).toHaveBeenCalledWith(
      "/api/automation/v1/created-automation",
      {
        trigger: {
          type: "cron",
          schedule: spec.trigger.schedule,
          timezone: spec.timezone,
        },
        enabled: false,
      },
      {
        baseURL: localBackend.host,
        headers: expect.objectContaining({
          "X-Session-API-Key": localBackend.apiKey,
        }),
      },
    );
    expect(created.enabled).toBe(false);
  });

  it("uses the prompt preset path when no plugins are configured", async () => {
    await AutomationService.createAutomation({
      ...spec,
      plugins: undefined,
    });

    expect(localAxios.post).toHaveBeenCalledWith(
      "/api/automation/v1/preset/prompt",
      expect.not.objectContaining({ plugins: expect.anything() }),
      expect.any(Object),
    );
  });

  it("applies the imported event trigger while disabling the automation", async () => {
    const eventTrigger = {
      type: "event",
      source: "github",
      on: ["pull_request.opened", "pull_request.synchronize"],
      filter: "repository.full_name == 'openhands/agent-canvas'",
    };

    await AutomationService.createAutomation({
      ...spec,
      trigger: eventTrigger,
    });

    expect(localAxios.patch).toHaveBeenCalledWith(
      "/api/automation/v1/created-automation",
      { trigger: eventTrigger, enabled: false },
      expect.any(Object),
    );
  });

  it("uses the selected cloud backend and organization for both requests", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
    callCloudProxy
      .mockResolvedValueOnce(createdAutomation)
      .mockResolvedValueOnce({ ...createdAutomation, enabled: false });

    const created = await AutomationService.createAutomation(spec);

    expect(callCloudProxy).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        backend: cloudBackend,
        method: "POST",
        path: "/api/automation/v1/preset/plugin",
        body: expect.objectContaining({ name: spec.name }),
        headers: expect.objectContaining({ "X-Org-Id": "org-1" }),
      }),
    );
    expect(callCloudProxy).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        backend: cloudBackend,
        method: "PATCH",
        path: "/api/automation/v1/created-automation",
        body: expect.objectContaining({ enabled: false }),
        headers: expect.objectContaining({ "X-Org-Id": "org-1" }),
      }),
    );
    expect(created.enabled).toBe(false);
  });

  it("removes the inert automation when disabling it fails", async () => {
    const updateError = new Error("update failed");
    localAxios.patch.mockRejectedValueOnce(updateError);

    await expect(AutomationService.createAutomation(spec)).rejects.toBe(
      updateError,
    );

    expect(localAxios.delete).toHaveBeenCalledWith(
      "/api/automation/v1/created-automation",
      {
        baseURL: localBackend.host,
        headers: expect.objectContaining({
          "X-Session-API-Key": localBackend.apiKey,
        }),
      },
    );
  });

  it("includes the timeout in the create request when the spec sets one", async () => {
    await AutomationService.createAutomation({ ...spec, timeout: 1200 });

    expect(localAxios.post).toHaveBeenCalledWith(
      "/api/automation/v1/preset/plugin",
      expect.objectContaining({ timeout: 1200 }),
      expect.any(Object),
    );
  });
});
