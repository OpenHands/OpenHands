import { describe, it, expect, vi, beforeEach } from "vitest";
import type { InternalAxiosRequestConfig } from "axios";
import { AutomationRunStatus } from "#/types/automation";
import type {
  Automation,
  AutomationRun,
  AutomationSpec,
} from "#/types/automation";
import type { Backend } from "#/api/backend-registry/types";

const {
  mockGet,
  mockPatch,
  mockPost,
  mockPut,
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
    mockPut: vi.fn(),
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
      put: mockPut,
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
  getTelemetryConsent: vi.fn().mockReturnValue("granted"),
  getTelemetryDistinctId: mockGetTelemetryDistinctId,
  getTelemetryDistinctIdForConsentSync: vi
    .fn()
    .mockResolvedValue("test-telemetry-id"),
  clearPendingLocalTelemetryRevocation: vi.fn(),
}));

import AutomationService from "#/api/automation-service/automation-service.api";

const localBackendWithKey: Backend = {
  id: "local-1",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "secret-session-token-123",
  kind: "local",
};

const localBackendWhitespaceKey: Backend = {
  id: "local-2",
  name: "Local Whitespace",
  host: "http://localhost:8000",
  apiKey: "   \t\n  ",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "cloud-token",
  kind: "cloud",
};

const sampleSpec: AutomationSpec = {
  name: "Test Automation",
  prompt: "Test prompt",
  trigger: { type: "schedule", schedule: "0 0 * * *" },
  enabled: true,
};

const mockAutomation: Automation = {
  id: "1",
  name: "Test Automation",
  prompt: "A test automation",
  trigger: { type: "schedule", schedule_human: "Daily at 09:00" },
  enabled: true,
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

function createMockAxiosConfig(
  overrides: Partial<InternalAxiosRequestConfig> = {},
): InternalAxiosRequestConfig {
  const headersMap = new Map<string, string>();
  if (overrides.headers && typeof overrides.headers === "object") {
    Object.entries(overrides.headers).forEach(([k, v]) => {
      if (typeof v === "string") headersMap.set(k.toLowerCase(), v);
    });
  }

  const headers = {
    set: vi.fn((key: string, val: string) => {
      headersMap.set(key.toLowerCase(), val);
    }),
    get: vi.fn((key: string) => headersMap.get(key.toLowerCase())),
    has: vi.fn((key: string) => headersMap.has(key.toLowerCase())),
  } as unknown as InternalAxiosRequestConfig["headers"];

  const { headers: _overrideHeaders, ...restOverrides } = overrides;

  return {
    headers,
    ...restOverrides,
  } as unknown as InternalAxiosRequestConfig;
}

describe("AutomationService - X-Session-API-Key Header Audit (Issue #18076)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetTelemetryDistinctId.mockResolvedValue("test-telemetry-id");
    mockGetActive.mockReturnValue({
      backend: localBackendWithKey,
      selectionSource: "active",
      backendId: "local-1",
    });
    mockGetEffectiveLocal.mockReturnValue(localBackendWithKey);

    mockGet.mockResolvedValue({ data: { items: [], automations: [] } });
    mockPost.mockResolvedValue({ data: { id: "1", status: "ok" } });
    mockPut.mockResolvedValue({ data: { id: "1", status: "ok" } });
    mockPatch.mockResolvedValue({ data: { id: "1", enabled: true } });
    mockDelete.mockResolvedValue({ data: { status: "ok" } });
  });

  describe("1. Real request interceptor audits", () => {
    it("attaches X-Session-API-Key from effective local backend when apiKey is configured", async () => {
      const interceptor = capturedInterceptors[0];
      expect(interceptor).toBeDefined();

      const config = createMockAxiosConfig();
      await interceptor(config);

      expect(config.headers.set).toHaveBeenCalledWith(
        "X-Session-API-Key",
        "secret-session-token-123",
      );
      expect(config.baseURL).toBe("http://localhost:8000");
    });

    it("omits X-Session-API-Key when effective local backend apiKey is empty or whitespace", async () => {
      const interceptor = capturedInterceptors[0];
      expect(interceptor).toBeDefined();

      mockGetEffectiveLocal.mockReturnValue(localBackendWhitespaceKey);

      const config = createMockAxiosConfig();
      await interceptor(config);

      expect(config.headers.set).not.toHaveBeenCalledWith(
        "X-Session-API-Key",
        expect.anything(),
      );
    });

    it("defensively attaches X-Session-API-Key when request carries baseURL but omitted the session key", async () => {
      const interceptor = capturedInterceptors[0];
      expect(interceptor).toBeDefined();

      const config = createMockAxiosConfig({
        baseURL: "http://pinned-host:9000",
      });
      await interceptor(config);

      expect(config.headers.set).toHaveBeenCalledWith(
        "X-Session-API-Key",
        "secret-session-token-123",
      );
      expect(config.baseURL).toBe("http://pinned-host:9000");
    });

    it("preserves an existing X-Session-API-Key when request carries baseURL and explicit header", async () => {
      const interceptor = capturedInterceptors[0];
      expect(interceptor).toBeDefined();

      const config = createMockAxiosConfig({
        baseURL: "http://pinned-host:9000",
        headers: { "X-Session-API-Key": "explicit-pinned-key" } as never,
      });
      await interceptor(config);

      expect(config.headers.set).not.toHaveBeenCalledWith(
        "X-Session-API-Key",
        "secret-session-token-123",
      );
    });
  });

  describe("2. Pinned options path audit (createAutomation)", () => {
    it("passes X-Session-API-Key in headers options for pinned create, patch, and cleanup", async () => {
      mockPost.mockResolvedValue({ data: mockAutomation });
      mockPatch.mockResolvedValue({ data: mockAutomation });

      await AutomationService.createAutomation(sampleSpec);

      expect(mockPost).toHaveBeenCalled();
      const createConfig = mockPost.mock.calls[0][2];
      expect(createConfig?.headers?.["X-Session-API-Key"]).toBe(
        "secret-session-token-123",
      );
      expect(createConfig?.baseURL).toBe("http://localhost:8000");
    });

    it("omits X-Session-API-Key in headers options when pinned backend key is whitespace", async () => {
      mockGetEffectiveLocal.mockReturnValue(localBackendWhitespaceKey);
      mockGetActive.mockReturnValue({
        backend: localBackendWhitespaceKey,
        selectionSource: "active",
        backendId: "local-2",
      });
      mockPost.mockResolvedValue({ data: mockAutomation });

      await AutomationService.createAutomation(sampleSpec);

      expect(mockPost).toHaveBeenCalled();
      const createConfig = mockPost.mock.calls[0][2];
      expect(createConfig?.headers?.["X-Session-API-Key"]).toBeUndefined();
    });
  });

  describe("3. Exported local methods audit", () => {
    it("routes every non-pinned local method through localAutomationAxios", async () => {
      // 1. syncTelemetryConsent
      await AutomationService.syncTelemetryConsent("granted");
      expect(mockPost).toHaveBeenLastCalledWith(
        "/api/automation/v1/telemetry/consent",
        expect.anything(),
        expect.anything(),
      );

      // 2. getSdkVersion
      mockGet.mockResolvedValueOnce({ data: { sdk_version: "1.0.0" } });
      await AutomationService.getSdkVersion();
      expect(mockGet).toHaveBeenLastCalledWith(
        "/api/automation/sdk-version",
        expect.objectContaining({ timeout: 5000 }),
      );

      // 3. listAutomations / getAutomations
      mockGet.mockResolvedValueOnce({
        data: { automations: [mockAutomation] },
      });
      await AutomationService.listAutomations();
      expect(mockGet).toHaveBeenLastCalledWith(
        "/api/automation/v1",
        expect.objectContaining({
          params: expect.objectContaining({ limit: 50, offset: 0 }),
        }),
      );

      // 4. getAutomation
      mockGet.mockResolvedValueOnce({ data: mockAutomation });
      await AutomationService.getAutomation("1");
      expect(mockGet).toHaveBeenLastCalledWith("/api/automation/v1/1");

      // 5. updateAutomation
      mockPatch.mockResolvedValueOnce({ data: mockAutomation });
      await AutomationService.updateAutomation("1", sampleSpec);
      expect(mockPatch).toHaveBeenLastCalledWith(
        "/api/automation/v1/1",
        sampleSpec,
      );

      // 6. deleteAutomation
      mockDelete.mockResolvedValueOnce({ data: { status: "ok" } });
      await AutomationService.deleteAutomation("1");
      expect(mockDelete).toHaveBeenLastCalledWith("/api/automation/v1/1");

      // 7. dispatchAutomation
      mockPost.mockResolvedValueOnce({ data: mockRun });
      await AutomationService.dispatchAutomation("1");
      expect(mockPost).toHaveBeenLastCalledWith(
        "/api/automation/v1/1/dispatch",
      );

      // 8. cancelAutomationRun
      mockPost.mockResolvedValueOnce({ data: mockRun });
      await AutomationService.cancelAutomationRun("run-1");
      expect(mockPost).toHaveBeenLastCalledWith(
        "/api/automation/v1/runs/run-1/cancel",
      );

      // 9. listAutomationRuns / getAutomationRuns
      mockGet.mockResolvedValueOnce({ data: { runs: [mockRun] } });
      await AutomationService.listAutomationRuns("auto-1");
      expect(mockGet).toHaveBeenLastCalledWith(
        "/api/automation/v1/auto-1/runs",
        expect.objectContaining({
          params: expect.objectContaining({ limit: 50, offset: 0 }),
        }),
      );

      // 10. toggleAutomation
      mockPatch.mockResolvedValueOnce({ data: mockAutomation });
      await AutomationService.toggleAutomation("1", false);
      expect(mockPatch).toHaveBeenLastCalledWith("/api/automation/v1/1", {
        enabled: false,
      });

      // 11. fetchTarballBytes
      mockGet.mockResolvedValueOnce({ data: new ArrayBuffer(8) });
      await AutomationService.fetchTarballBytes("1");
      expect(mockGet).toHaveBeenLastCalledWith(
        "/api/automation/v1/1/tarball",
        expect.objectContaining({ responseType: "arraybuffer" }),
      );

      // 12. getCapabilities
      mockGet.mockResolvedValueOnce({
        data: { ready: true, features: [] },
      });
      await AutomationService.getCapabilities();
      expect(mockGet).toHaveBeenLastCalledWith(
        "/api/automation/v1/capabilities",
      );

      // 13. validateDraft
      mockPost.mockResolvedValueOnce({ data: { valid: true } });
      await AutomationService.validateDraft({ prompt: "hi" } as never);
      expect(mockPost).toHaveBeenLastCalledWith(
        "/api/automation/v1/validate",
        expect.anything(),
      );

      // 14. createAutomationDraft
      mockPost.mockResolvedValueOnce({
        data: { id: "draft-1", draft: {} },
      });
      await AutomationService.createAutomationDraft(
        { prompt: "hi" } as never,
        "prompt",
        null,
      );
      expect(mockPost).toHaveBeenLastCalledWith(
        "/api/automation/v1/preset/prompt",
        expect.anything(),
      );

      // 15. createServerDraft
      mockPost.mockResolvedValueOnce({
        data: { id: "draft-2", draft: {} },
      });
      await AutomationService.createServerDraft({
        endpoint: "/v1/preset/prompt",
        draft: {} as never,
      });
      expect(mockPost).toHaveBeenLastCalledWith(
        "/api/automation/v1/drafts",
        expect.anything(),
      );

      // 16. getServerDraft
      mockGet.mockResolvedValueOnce({
        data: { id: "draft-1", draft: {} },
      });
      await AutomationService.getServerDraft("draft-1");
      expect(mockGet).toHaveBeenLastCalledWith(
        "/api/automation/v1/drafts/draft-1",
      );

      // 17. updateServerDraft
      mockPatch.mockResolvedValueOnce({
        data: { id: "draft-1", draft: {} },
      });
      await AutomationService.updateServerDraft("draft-1", {
        draft: {} as never,
      });
      expect(mockPatch).toHaveBeenLastCalledWith(
        "/api/automation/v1/drafts/draft-1",
        expect.anything(),
      );

      // 18. deleteServerDraft
      mockDelete.mockResolvedValueOnce({ data: { status: "ok" } });
      await AutomationService.deleteServerDraft("draft-1");
      expect(mockDelete).toHaveBeenLastCalledWith(
        "/api/automation/v1/drafts/draft-1",
      );

      // 19. listServerDrafts
      mockGet.mockResolvedValueOnce({ data: { drafts: [] } });
      await AutomationService.listServerDrafts();
      expect(mockGet).toHaveBeenLastCalledWith(
        "/api/automation/v1/drafts?limit=50&offset=0",
      );

      // 20. dispatchServerDraft
      mockPost.mockResolvedValueOnce({ data: mockRun });
      await AutomationService.dispatchServerDraft("draft-1");
      expect(mockPost).toHaveBeenLastCalledWith(
        "/api/automation/v1/drafts/draft-1/dispatch",
      );

      // 21. uploadAutomationTarball
      mockPost.mockResolvedValueOnce({
        data: { tarball_path: "tmp/automation.tar.gz" },
      });
      await AutomationService.uploadAutomationTarball(
        "my-tarball",
        new Uint8Array([1, 2, 3]),
      );
      expect(mockPost).toHaveBeenLastCalledWith(
        expect.stringContaining("/api/automation/v1/uploads"),
        expect.anything(),
        expect.anything(),
      );

      // 22. getGitSyncStatus
      mockGet.mockResolvedValueOnce({ data: { configured: false } });
      await AutomationService.getGitSyncStatus();
      expect(mockGet).toHaveBeenLastCalledWith(
        "/api/automation/v1/git-sync/status",
      );

      // 23. updateGitSyncConfig
      mockPut.mockResolvedValueOnce({ data: { configured: true } });
      await AutomationService.updateGitSyncConfig({
        repo_url: "https://github.com/test/repo",
      });
      expect(mockPut).toHaveBeenLastCalledWith(
        "/api/automation/v1/git-sync/config",
        expect.anything(),
      );

      // 24. checkGitSyncConfig
      mockPost.mockResolvedValueOnce({ data: { ok: true } });
      await AutomationService.checkGitSyncConfig({
        repo_url: "https://github.com/test/repo",
      });
      expect(mockPost).toHaveBeenLastCalledWith(
        "/api/automation/v1/git-sync/check",
        expect.anything(),
      );

      // 25. triggerGitSync
      mockPost.mockResolvedValueOnce({ data: { triggered: true } });
      await AutomationService.triggerGitSync();
      expect(mockPost).toHaveBeenLastCalledWith(
        "/api/automation/v1/git-sync/sync",
      );

      // 26. checkHealth
      mockGet.mockResolvedValueOnce({ data: { status: "ok" } });
      await AutomationService.checkHealth();
      expect(mockGet).toHaveBeenLastCalledWith(
        "/api/automation/health",
        expect.objectContaining({ timeout: 5000 }),
      );
    });
  });

  describe("4. Cloud requests audit", () => {
    it("routes through callCloudProxy with no local axios call or session key", async () => {
      mockGetActive.mockReturnValue({
        backend: cloudBackend,
        selectionSource: "active",
        backendId: "cloud-1",
        orgId: "org-123",
      });
      mockCallCloudProxy.mockResolvedValueOnce({ automations: [] });

      await AutomationService.listAutomations();

      expect(mockCallCloudProxy).toHaveBeenCalledWith(
        expect.objectContaining({
          path: expect.stringContaining("/api/automation/v1"),
        }),
      );
      expect(mockGet).not.toHaveBeenCalled();
    });
  });
});
