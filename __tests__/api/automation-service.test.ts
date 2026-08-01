import { describe, it, expect, vi, beforeEach } from "vitest";
import { AutomationRunStatus } from "#/types/automation";
import type {
  Automation,
  AutomationRun,
  AutomationsResponse,
  AutomationRunsResponse,
} from "#/types/automation";
import type { Backend } from "#/api/backend-registry/types";
import type { InternalAxiosRequestConfig } from "axios";

// Use vi.hoisted to define mocks that will be available during vi.mock hoisting
const {
  mockGet,
  mockPatch,
  mockPost,
  mockDelete,
  mockCallCloudProxy,
  mockGetActive,
  mockGetEffectiveLocal,
  mockGetTelemetryDistinctId,
  capturedInterceptors,
} = vi.hoisted(() => {
  const interceptors: Array<
    (
      config: InternalAxiosRequestConfig,
    ) => InternalAxiosRequestConfig | Promise<InternalAxiosRequestConfig>
  > = [];
  return {
    mockGet: vi.fn(),
    mockPatch: vi.fn(),
    mockPost: vi.fn(),
    mockDelete: vi.fn(),
    mockCallCloudProxy: vi.fn(),
    mockGetActive: vi.fn(),
    mockGetEffectiveLocal: vi.fn(),
    mockGetTelemetryDistinctId: vi.fn(),
    capturedInterceptors: interceptors,
  };
});

vi.mock("axios", () => ({
  default: {
    create: () => ({
      get: mockGet,
      post: mockPost,

      patch: mockPatch,
      delete: mockDelete,
      interceptors: {
        request: {
          use: (
            fn: (
              config: InternalAxiosRequestConfig,
            ) =>
              | InternalAxiosRequestConfig
              | Promise<InternalAxiosRequestConfig>,
          ) => {
            capturedInterceptors.push(fn);
          },
        },
      },
    }),
  },
}));

vi.mock("#/api/cloud/proxy", () => ({
  callCloudProxy: mockCallCloudProxy,
}));

vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: mockGetActive,
  getEffectiveLocalBackend: mockGetEffectiveLocal,
}));

vi.mock("#/services/telemetry", () => ({
  getTelemetryDistinctId: mockGetTelemetryDistinctId,
}));

// Import after mocking
import AutomationService from "#/api/automation-service/automation-service.api";

const localBackend: Backend = {
  id: "local-1",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

const expectedAutomationTelemetryHeaders = {
  "X-OpenHands-Client": "agent_canvas",
  "X-OpenHands-Client-Version": expect.any(String),
  "X-OpenHands-Telemetry-Distinct-Id": "ph-test-distinct-id",
};

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};

/** Build a minimal InternalAxiosRequestConfig for interceptor tests. */
function makeAxiosConfig(
  overrides: Partial<InternalAxiosRequestConfig> = {},
): InternalAxiosRequestConfig {
  const headers = {
    set: vi.fn(),
    get: vi.fn(),
  } as unknown as InternalAxiosRequestConfig["headers"];
  return {
    headers,
    ...overrides,
  } as unknown as InternalAxiosRequestConfig;
}

const mockAutomation: Automation = {
  id: "1",
  name: "Test Automation",
  prompt: "A test automation",
  trigger: { type: "schedule", schedule_human: "Daily at 09:00" },
  enabled: true,
  repository: "acme/repo",
  model: "daily-profile",
  created_at: "2026-01-01T00:00:00Z",
  updated_at: "2026-01-02T00:00:00Z",
};

const mockRun: AutomationRun = {
  id: "run-1",
  status: AutomationRunStatus.PENDING,
  conversation_id: null,
  bash_command_id: null,
  error_detail: null,
  started_at: "2026-01-03T00:00:00Z",
  completed_at: null,
};

describe("AutomationService", () => {
  beforeEach(() => {
    // restoreAllMocks (vs clearAllMocks) re-attaches the original
    // implementations of any class methods spied via vi.spyOn in earlier
    // tests, so the cloud-routing assertions actually exercise the real
    // method bodies instead of stale spies.
    vi.restoreAllMocks();
    mockGet.mockReset();
    mockPatch.mockReset();
    mockPost.mockReset();
    mockDelete.mockReset();
    mockCallCloudProxy.mockReset();
    // Default: active backend is local. Cloud-routing tests override this.
    mockGetActive.mockReset();
    mockGetActive.mockReturnValue({ backend: localBackend, orgId: null });
    mockGetEffectiveLocal.mockReset();
    mockGetEffectiveLocal.mockReturnValue(localBackend);
    mockGetTelemetryDistinctId.mockReset();
    mockGetTelemetryDistinctId.mockResolvedValue("ph-test-distinct-id");
  });

  describe("listAutomations", () => {
    it("fetches paginated automations list with params object", async () => {
      const response: AutomationsResponse = {
        automations: [mockAutomation],
        total: 1,
      };
      mockGet.mockResolvedValue({ data: response });

      const result = await AutomationService.listAutomations({
        limit: 10,
        offset: 5,
      });

      expect(mockGet).toHaveBeenCalledWith("/api/automation/v1", {
        params: { limit: 10, offset: 5 },
      });
      expect(result).toEqual(response);
    });

    it("uses default params when none provided", async () => {
      const response: AutomationsResponse = {
        automations: [],
        total: 0,
      };
      mockGet.mockResolvedValue({ data: response });

      await AutomationService.listAutomations();

      expect(mockGet).toHaveBeenCalledWith("/api/automation/v1", {
        params: { limit: 50, offset: 0 },
      });
    });
  });

  describe("getAutomations", () => {
    it("delegates to listAutomations", async () => {
      const response: AutomationsResponse = {
        automations: [mockAutomation],
        total: 1,
      };
      vi.spyOn(AutomationService, "listAutomations").mockResolvedValue(
        response,
      );

      const result = await AutomationService.getAutomations(10, 5);

      expect(AutomationService.listAutomations).toHaveBeenCalledWith({
        limit: 10,
        offset: 5,
      });
      expect(result).toEqual(response);
    });
  });

  describe("getAutomation", () => {
    it("fetches a single automation by id", async () => {
      mockGet.mockResolvedValue({
        data: mockAutomation,
      });

      const result = await AutomationService.getAutomation("1");

      expect(mockGet).toHaveBeenCalledWith("/api/automation/v1/1");
      expect(result).toEqual(mockAutomation);
    });
  });

  describe("updateAutomation", () => {
    it("patches an automation with the provided body", async () => {
      const updated = { ...mockAutomation, name: "Updated Name" };
      mockPatch.mockResolvedValue({ data: updated });

      const result = await AutomationService.updateAutomation("1", {
        name: "Updated Name",
      });

      expect(mockPatch).toHaveBeenCalledWith("/api/automation/v1/1", {
        name: "Updated Name",
      });
      expect(result).toEqual(updated);
    });

    it("sends model profile updates to the automation API", async () => {
      const updated = { ...mockAutomation, model: "careful-profile" };
      mockPatch.mockResolvedValue({ data: updated });

      const result = await AutomationService.updateAutomation("1", {
        model: "careful-profile",
      });

      expect(mockPatch).toHaveBeenCalledWith("/api/automation/v1/1", {
        model: "careful-profile",
      });
      expect(result).toEqual(updated);
    });
  });

  describe("dispatchAutomation", () => {
    it("posts to the dispatch endpoint", async () => {
      mockPost.mockResolvedValue({ data: mockRun });

      const result = await AutomationService.dispatchAutomation("1");

      expect(mockPost).toHaveBeenCalledWith("/api/automation/v1/1/dispatch");
      expect(result).toEqual(mockRun);
    });
  });

  describe("deleteAutomation", () => {
    it("deletes an automation by id", async () => {
      mockDelete.mockResolvedValue({});

      await AutomationService.deleteAutomation("1");

      expect(mockDelete).toHaveBeenCalledWith("/api/automation/v1/1");
    });
  });

  describe("listAutomationRuns", () => {
    it("fetches runs with params object", async () => {
      const response: AutomationRunsResponse = { runs: [], total: 0 };
      mockGet.mockResolvedValue({ data: response });

      const result = await AutomationService.listAutomationRuns("1", {
        limit: 20,
        offset: 10,
      });

      expect(mockGet).toHaveBeenCalledWith("/api/automation/v1/1/runs", {
        params: { limit: 20, offset: 10 },
      });
      expect(result).toEqual(response);
    });

    it("uses default params when none provided", async () => {
      const response: AutomationRunsResponse = { runs: [], total: 0 };
      mockGet.mockResolvedValue({ data: response });

      await AutomationService.listAutomationRuns("1");

      expect(mockGet).toHaveBeenCalledWith("/api/automation/v1/1/runs", {
        params: { limit: 50, offset: 0 },
      });
    });
  });

  describe("getAutomationRuns", () => {
    it("delegates to listAutomationRuns", async () => {
      const response: AutomationRunsResponse = { runs: [], total: 0 };
      vi.spyOn(AutomationService, "listAutomationRuns").mockResolvedValue(
        response,
      );

      const result = await AutomationService.getAutomationRuns("1", 25, 5);

      expect(AutomationService.listAutomationRuns).toHaveBeenCalledWith("1", {
        limit: 25,
        offset: 5,
      });
      expect(result).toEqual(response);
    });
  });

  describe("toggleAutomation", () => {
    it("delegates to updateAutomation with enabled field", async () => {
      const toggled = { ...mockAutomation, enabled: false };
      vi.spyOn(AutomationService, "updateAutomation").mockResolvedValue(
        toggled,
      );

      const result = await AutomationService.toggleAutomation("1", false);

      expect(AutomationService.updateAutomation).toHaveBeenCalledWith("1", {
        enabled: false,
      });
      expect(result).toEqual(toggled);
    });
  });

  describe("dispatchAutomation", () => {
    it("posts to the dispatch endpoint for local backends", async () => {
      const run = {
        id: "run-1",
        status: "PENDING",
        conversation_id: null,
        bash_command_id: null,
        error_detail: null,
        started_at: "2026-01-01T00:00:00Z",
        completed_at: null,
      };
      mockPost.mockResolvedValue({ data: run });

      const result = await AutomationService.dispatchAutomation("1");

      expect(mockPost).toHaveBeenCalledWith("/api/automation/v1/1/dispatch");
      expect(result).toEqual(run);
    });
  });

  // When the active backend is cloud the local axios instance must be
  // bypassed entirely; calls must route through `callCloudProxy`, which
  // sends them directly to the cloud host from the browser (the automation
  // service grants permissive CORS to API-key requests, automation#185).
  describe("cloud routing", () => {
    beforeEach(() => {
      mockGetActive.mockReturnValue({ backend: cloudBackend, orgId: null });
    });

    it("listAutomations routes to callCloudProxy with pagination in the path", async () => {
      const response: AutomationsResponse = {
        automations: [mockAutomation],
        total: 1,
      };
      mockCallCloudProxy.mockResolvedValue(response);

      const result = await AutomationService.listAutomations({
        limit: 10,
        offset: 5,
      });

      expect(mockCallCloudProxy).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "GET",
        path: "/api/automation/v1?limit=10&offset=5",
        headers: expectedAutomationTelemetryHeaders,
      });
      expect(mockGet).not.toHaveBeenCalled();
      expect(result).toEqual(response);
    });

    it("getAutomation routes to callCloudProxy with the id in the path", async () => {
      mockCallCloudProxy.mockResolvedValue(mockAutomation);

      const result = await AutomationService.getAutomation("abc");

      expect(mockCallCloudProxy).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "GET",
        path: "/api/automation/v1/abc",
        headers: expectedAutomationTelemetryHeaders,
      });
      expect(result).toEqual(mockAutomation);
    });

    it("dispatchAutomation forwards method POST via callCloudProxy", async () => {
      mockCallCloudProxy.mockResolvedValue(mockRun);

      const result = await AutomationService.dispatchAutomation("abc");

      expect(mockCallCloudProxy).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "POST",
        path: "/api/automation/v1/abc/dispatch",
        headers: expectedAutomationTelemetryHeaders,
      });
      expect(mockPost).not.toHaveBeenCalled();
      expect(result).toEqual(mockRun);
    });

    it("updateAutomation forwards method PATCH and body via callCloudProxy", async () => {
      const updated = { ...mockAutomation, enabled: false };
      mockCallCloudProxy.mockResolvedValue(updated);

      const result = await AutomationService.updateAutomation("abc", {
        enabled: false,
      });

      expect(mockCallCloudProxy).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "PATCH",
        path: "/api/automation/v1/abc",
        body: { enabled: false },
        headers: expectedAutomationTelemetryHeaders,
      });
      expect(mockPatch).not.toHaveBeenCalled();
      expect(result).toEqual(updated);
    });

    it("deleteAutomation forwards method DELETE via callCloudProxy", async () => {
      mockCallCloudProxy.mockResolvedValue(undefined);

      await AutomationService.deleteAutomation("abc");

      expect(mockCallCloudProxy).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "DELETE",
        path: "/api/automation/v1/abc",
        headers: expectedAutomationTelemetryHeaders,
      });
      expect(mockDelete).not.toHaveBeenCalled();
    });

    it("dispatchAutomation forwards method POST via callCloudProxy", async () => {
      const run = {
        id: "run-1",
        status: "PENDING",
        conversation_id: null,
        bash_command_id: null,
        error_detail: null,
        started_at: "2026-01-01T00:00:00Z",
        completed_at: null,
      };
      mockCallCloudProxy.mockResolvedValue(run);

      const result = await AutomationService.dispatchAutomation("abc");

      expect(mockCallCloudProxy).toHaveBeenCalledWith({
        backend: cloudBackend,
        method: "POST",
        path: "/api/automation/v1/abc/dispatch",
        headers: expectedAutomationTelemetryHeaders,
      });
      expect(mockPost).not.toHaveBeenCalled();
      expect(result).toEqual(run);
    });

    it("checkHealth calls the cloud host with a fail-fast timeout and returns the upstream status", async () => {
      mockCallCloudProxy.mockResolvedValue({ status: "ok" });

      const result = await AutomationService.checkHealth();

      // Property access (vs whole-object matching) keeps these assertions
      // resilient to future additive CloudProxyRequest fields.
      const call = mockCallCloudProxy.mock.calls[0]![0];
      expect(call.method).toBe("GET");
      expect(call.path).toBe("/api/automation/health");
      expect(call.timeoutSeconds).toBe(5);
      expect(mockGet).not.toHaveBeenCalled();
      expect(result).toEqual({ status: "ok" });
    });

    it("checkHealth resolves to an error status instead of throwing when the cloud call fails", async () => {
      mockCallCloudProxy.mockRejectedValue(new Error("proxy unreachable"));

      const result = await AutomationService.checkHealth();

      expect(result).toEqual({ status: "error" });
    });
  });

  // The interceptor must read the session API key from the active backend
  // registry rather than the build-time VITE_SESSION_API_KEY env var so that
  // the published npm package picks up the runtime-injected key (issue #829).
  describe("localAutomationAxios interceptor", () => {
    it("sets telemetry headers and X-Session-API-Key from the effective local backend apiKey", async () => {
      const interceptor = capturedInterceptors[0];
      expect(interceptor).toBeDefined();

      const backendWithKey: Backend = {
        ...localBackend,
        apiKey: "runtime-injected-key",
      };
      mockGetEffectiveLocal.mockReturnValue(backendWithKey);

      const config = makeAxiosConfig();
      await interceptor(config);

      expect(config.headers.set).toHaveBeenCalledWith(
        "X-OpenHands-Client",
        "agent_canvas",
      );
      expect(config.headers.set).toHaveBeenCalledWith(
        "X-OpenHands-Telemetry-Distinct-Id",
        "ph-test-distinct-id",
      );
      expect(config.headers.set).toHaveBeenCalledWith(
        "X-Session-API-Key",
        "runtime-injected-key",
      );
    });

    it("does not set X-Session-API-Key when backend apiKey is empty", async () => {
      const interceptor = capturedInterceptors[0];
      expect(interceptor).toBeDefined();

      mockGetEffectiveLocal.mockReturnValue({
        ...localBackend,
        apiKey: "",
      });

      const config = makeAxiosConfig();
      await interceptor(config);

      expect(config.headers.set).not.toHaveBeenCalledWith(
        "X-Session-API-Key",
        expect.anything(),
      );
    });

    it("sets baseURL from effective local backend host when not already set", async () => {
      const interceptor = capturedInterceptors[0];
      expect(interceptor).toBeDefined();

      mockGetEffectiveLocal.mockReturnValue({
        ...localBackend,
        host: "http://custom-host:9000",
        apiKey: "key",
      });

      const config = makeAxiosConfig();
      await interceptor(config);

      expect(config.baseURL).toBe("http://custom-host:9000");
    });

    it("does not overwrite an already-set baseURL", async () => {
      const interceptor = capturedInterceptors[0];
      expect(interceptor).toBeDefined();

      mockGetEffectiveLocal.mockReturnValue({
        ...localBackend,
        host: "http://should-not-use:9000",
        apiKey: "key",
      });

      const config = makeAxiosConfig({ baseURL: "http://already-set:8000" });
      await interceptor(config);

      expect(config.baseURL).toBe("http://already-set:8000");
    });
  });
});
