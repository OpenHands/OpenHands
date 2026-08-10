import axios from "axios";
import {
  clearPendingLocalTelemetryRevocation,
  getTelemetryConsent,
  getTelemetryDistinctId,
  getTelemetryDistinctIdForConsentSync,
  type TelemetryConsent,
} from "#/services/telemetry";
import type {
  Automation,
  AutomationRun,
  AutomationSpec,
  AutomationTrigger,
  AutomationsResponse,
  AutomationRunsResponse,
} from "#/types/automation";
import { AUTOMATION_CREATE_ENDPOINT } from "#/manifests/automation-setup";
import {
  getAutomationEndpoint,
  getAutomationIdEndpoint,
  getImportExportSpec,
} from "#/manifests/automation-interface";
import type {
  DeploymentCapabilities,
  SetupRequestBody,
  ValidateDraftResponse,
} from "#/manifests/types";
import type { Backend, ResolvedActiveBackend } from "../backend-registry/types";
import {
  getActiveBackend,
  getEffectiveLocalBackend,
} from "../backend-registry/active-store";
import { NoBackendAvailableError } from "../agent-server-client-options";
import { callCloudProxy } from "../cloud/proxy";
import {
  AGENT_CANVAS_CLIENT_HEADERS,
  OPENHANDS_TELEMETRY_DISTINCT_ID_HEADER,
} from "../client-source";

const AUTOMATION_BASE_PATH = "/api/automation";

export interface AutomationHealthResponse {
  status: "ok" | "error";
  message?: string;
}

type AutomationSdkVersionResponse =
  | string
  | {
      sdk_version?: unknown;
      version?: unknown;
    };

// Local automation calls go to the automation sidecar that
// `scripts/dev-with-automation.mjs` mounts behind the local agent-server.
// Both backends use the same session API key and the same `X-Session-API-Key`
// header for consistency.
const localAutomationAxios = axios.create();

async function buildAutomationRequestHeaders(
  extraHeaders: Record<string, string> = {},
): Promise<Record<string, string>> {
  const telemetryDistinctId = await getTelemetryDistinctId();
  return {
    ...AGENT_CANVAS_CLIENT_HEADERS,
    ...(telemetryDistinctId
      ? { [OPENHANDS_TELEMETRY_DISTINCT_ID_HEADER]: telemetryDistinctId }
      : {}),
    ...extraHeaders,
  };
}

localAutomationAxios.interceptors.request.use(async (config) => {
  const requestHeaders = await buildAutomationRequestHeaders();
  Object.entries(requestHeaders).forEach(([name, value]) => {
    config.headers.set(name, value);
  });

  // Import uses an explicit baseURL/header pair so its POST, PATCH, and
  // cleanup stay pinned to the backend selected when the mutation started.
  if (config.baseURL) return config;

  // Resolve the local backend on every call so it tracks the
  // currently-active local backend (and any host/key edits made via the
  // manage-backends UI), rather than freezing whatever value the
  // agent-server-config produced at module load time.
  // Using the backend registry (rather than the build-time VITE_SESSION_API_KEY
  // env var) ensures the published npm package picks up the runtime-injected
  // session key that scripts/static-server.mjs seeds into localStorage, fixing
  // the 401 errors reported in issue #829.
  const backend = getEffectiveLocalBackend();
  if (!backend) throw new NoBackendAvailableError();
  // eslint-disable-next-line no-param-reassign
  config.baseURL = backend.host;

  const apiKey = backend.apiKey?.trim();
  if (apiKey) {
    config.headers.set("X-Session-API-Key", apiKey);
  }
  return config;
});

function normalizeAutomationSdkVersion(version: unknown): string | null {
  if (typeof version !== "string") return null;
  const trimmed = version.trim();
  return trimmed || null;
}

function getAutomationSdkVersionFromResponse(
  response: AutomationSdkVersionResponse,
): string | null {
  if (typeof response === "string") {
    return normalizeAutomationSdkVersion(response);
  }
  return (
    normalizeAutomationSdkVersion(response.sdk_version) ??
    normalizeAutomationSdkVersion(response.version)
  );
}

function buildPaginationQuery(limit: number, offset: number): string {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  params.set("offset", String(offset));
  return params.toString();
}

function buildImportedTrigger(spec: AutomationSpec): AutomationTrigger {
  if (spec.trigger.type === "event") {
    return {
      type: "event",
      source: spec.trigger.source,
      on: spec.trigger.on,
      ...(spec.trigger.filter && { filter: spec.trigger.filter }),
    };
  }

  return {
    type: "cron",
    schedule: spec.trigger.schedule,
    timezone: spec.timezone ?? spec.trigger.timezone ?? "UTC",
  };
}

function generatePendingImportEvent(): string {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return `pending.${crypto.randomUUID()}`;
  }
  return `pending.${Date.now()}.${Math.random().toString(36).slice(2, 10)}`;
}

function buildCreateAutomationRequest(spec: AutomationSpec) {
  if (!spec.prompt) {
    throw new Error("An automation prompt is required for import.");
  }

  const { importDefaults } = getImportExportSpec();
  const repos = spec.repository
    ? [
        {
          url: spec.repository,
          ...(spec.branch && { ref: spec.branch }),
          ...(!spec.repository.includes("://") &&
            !spec.repository.startsWith("git@") && {
              provider: importDefaults.repoProvider,
            }),
        },
      ]
    : undefined;

  return {
    path: `${AUTOMATION_BASE_PATH}${getAutomationEndpoint(
      spec.plugins?.length ? "createPlugin" : "createPrompt",
    )}`,
    body: {
      name: spec.name,
      prompt: spec.prompt,
      // Preset creation defaults to enabled. A unique event trigger keeps the
      // new record inert until the real trigger and disabled state are applied
      // together in the follow-up PATCH.
      trigger: {
        type: "event",
        source: importDefaults.placeholderEventSource,
        on: generatePendingImportEvent(),
      },
      ...(spec.model && { model: spec.model }),
      ...(repos && { repos }),
      ...(spec.plugins?.length && {
        plugins: spec.plugins.map((source) => ({ source })),
      }),
      ...(spec.timeout != null && { timeout: spec.timeout }),
    },
  };
}

async function buildPinnedLocalConfig(backend: Backend) {
  const apiKey = backend.apiKey.trim();
  return {
    baseURL: backend.host,
    headers: await buildAutomationRequestHeaders(
      apiKey ? { "X-Session-API-Key": apiKey } : {},
    ),
  };
}

async function buildPinnedCloudHeaders(active: ResolvedActiveBackend) {
  return buildAutomationRequestHeaders(
    active.orgId ? { "X-Org-Id": active.orgId } : {},
  );
}

class AutomationService {
  static async syncTelemetryConsent(
    consent: TelemetryConsent = getTelemetryConsent(),
  ): Promise<void> {
    const active = getActiveBackend();
    if (active.backend.kind !== "local") return;

    const frontendDistinctId = await getTelemetryDistinctIdForConsentSync();
    await localAutomationAxios.post(
      `${AUTOMATION_BASE_PATH}/v1/telemetry/consent`,
      {
        consent_granted: consent === "granted",
        frontend_distinct_id: frontendDistinctId,
      },
      { timeout: 5000 },
    );
    if (consent !== "granted" && frontendDistinctId) {
      clearPendingLocalTelemetryRevocation(frontendDistinctId);
    }
  }

  static async getSdkVersion(): Promise<string | null> {
    const active = getActiveBackend();
    const path = `${AUTOMATION_BASE_PATH}/sdk-version`;

    try {
      let response: AutomationSdkVersionResponse;
      if (active.backend.kind === "cloud") {
        response = await callCloudProxy<AutomationSdkVersionResponse>({
          backend: active.backend,
          method: "GET",
          path,
          headers: await buildPinnedCloudHeaders(active),
          timeoutSeconds: 5,
        });
      } else {
        const { data } =
          await localAutomationAxios.get<AutomationSdkVersionResponse>(path, {
            timeout: 5000,
          });
        response = data;
      }

      return getAutomationSdkVersionFromResponse(response);
    } catch {
      return null;
    }
  }

  static async listAutomations(
    params: { limit?: number; offset?: number } = {},
  ): Promise<AutomationsResponse> {
    const { limit = 50, offset = 0 } = params;
    const active = getActiveBackend().backend;

    if (active.kind === "cloud") {
      return callCloudProxy<AutomationsResponse>({
        backend: active,
        method: "GET",
        path: `${AUTOMATION_BASE_PATH}${getAutomationEndpoint("list")}?${buildPaginationQuery(limit, offset)}`,
        headers: await buildAutomationRequestHeaders(),
      });
    }

    const { data } = await localAutomationAxios.get<AutomationsResponse>(
      `${AUTOMATION_BASE_PATH}${getAutomationEndpoint("list")}`,
      { params: { limit, offset } },
    );
    return data;
  }

  static async getAutomations(
    limit = 50,
    offset = 0,
  ): Promise<AutomationsResponse> {
    return AutomationService.listAutomations({ limit, offset });
  }

  static async getAutomation(id: string): Promise<Automation> {
    const active = getActiveBackend().backend;
    const path = `${AUTOMATION_BASE_PATH}${getAutomationIdEndpoint("detail", id)}`;

    if (active.kind === "cloud") {
      return callCloudProxy<Automation>({
        backend: active,
        method: "GET",
        path,
        headers: await buildAutomationRequestHeaders(),
      });
    }

    const { data } = await localAutomationAxios.get<Automation>(path);
    return data;
  }

  static async createAutomation(spec: AutomationSpec): Promise<Automation> {
    const active = getActiveBackend();
    const { path, body } = buildCreateAutomationRequest(spec);

    let created: Automation;
    if (active.backend.kind === "cloud") {
      created = await callCloudProxy<Automation>({
        backend: active.backend,
        method: "POST",
        path,
        body,
        headers: await buildPinnedCloudHeaders(active),
      });
    } else {
      const { data } = await localAutomationAxios.post<Automation>(
        path,
        body,
        await buildPinnedLocalConfig(active.backend),
      );
      created = data;
    }

    const updatePath = `${AUTOMATION_BASE_PATH}${getAutomationIdEndpoint("detail", created.id)}`;
    const updateBody: Partial<Automation> = {
      trigger: buildImportedTrigger(spec),
      enabled: false,
    };

    try {
      if (active.backend.kind === "cloud") {
        return await callCloudProxy<Automation>({
          backend: active.backend,
          method: "PATCH",
          path: updatePath,
          body: updateBody,
          headers: await buildPinnedCloudHeaders(active),
        });
      }

      const { data } = await localAutomationAxios.patch<Automation>(
        updatePath,
        updateBody,
        await buildPinnedLocalConfig(active.backend),
      );
      return data;
    } catch (updateError) {
      try {
        if (active.backend.kind === "cloud") {
          await callCloudProxy<unknown>({
            backend: active.backend,
            method: "DELETE",
            path: updatePath,
            headers: await buildPinnedCloudHeaders(active),
          });
        } else {
          await localAutomationAxios.delete(
            updatePath,
            await buildPinnedLocalConfig(active.backend),
          );
        }
      } catch (cleanupError) {
        throw new AggregateError(
          [updateError, cleanupError],
          "Failed to disable the imported automation and clean it up.",
        );
      }
      throw updateError;
    }
  }

  static async updateAutomation(
    id: string,
    body: Partial<Automation>,
  ): Promise<Automation> {
    const active = getActiveBackend().backend;
    const path = `${AUTOMATION_BASE_PATH}${getAutomationIdEndpoint("detail", id)}`;

    if (active.kind === "cloud") {
      return callCloudProxy<Automation>({
        backend: active,
        method: "PATCH",
        path,
        body: body as Record<string, unknown>,
        headers: await buildAutomationRequestHeaders(),
      });
    }

    const { data } = await localAutomationAxios.patch<Automation>(path, body);
    return data;
  }

  static async deleteAutomation(id: string): Promise<void> {
    const active = getActiveBackend().backend;
    const path = `${AUTOMATION_BASE_PATH}${getAutomationIdEndpoint("detail", id)}`;

    if (active.kind === "cloud") {
      await callCloudProxy<unknown>({
        backend: active,
        method: "DELETE",
        path,
        headers: await buildAutomationRequestHeaders(),
      });
      return;
    }

    await localAutomationAxios.delete(path);
  }

  static async dispatchAutomation(id: string): Promise<AutomationRun> {
    const active = getActiveBackend().backend;
    const path = `${AUTOMATION_BASE_PATH}${getAutomationIdEndpoint("dispatch", id)}`;

    if (active.kind === "cloud") {
      return callCloudProxy<AutomationRun>({
        backend: active,
        method: "POST",
        path,
        headers: await buildAutomationRequestHeaders(),
      });
    }

    const { data } = await localAutomationAxios.post<AutomationRun>(path);
    return data;
  }

  /**
   * Cancel a pending or running automation run.
   * Backend: POST /api/automation/v1/runs/{run_id}/cancel
   */
  static async cancelAutomationRun(runId: string): Promise<AutomationRun> {
    const active = getActiveBackend().backend;
    const path = `${AUTOMATION_BASE_PATH}/v1/runs/${encodeURIComponent(runId)}/cancel`;

    if (active.kind === "cloud") {
      return callCloudProxy<AutomationRun>({
        backend: active,
        method: "POST",
        path,
        headers: await buildAutomationRequestHeaders(),
      });
    }

    const { data } = await localAutomationAxios.post<AutomationRun>(path);
    return data;
  }

  static async listAutomationRuns(
    id: string,
    params: { limit?: number; offset?: number } = {},
  ): Promise<AutomationRunsResponse> {
    const { limit = 50, offset = 0 } = params;
    const active = getActiveBackend().backend;
    const basePath = `${AUTOMATION_BASE_PATH}${getAutomationIdEndpoint("runs", id)}`;

    if (active.kind === "cloud") {
      return callCloudProxy<AutomationRunsResponse>({
        backend: active,
        method: "GET",
        path: `${basePath}?${buildPaginationQuery(limit, offset)}`,
        headers: await buildAutomationRequestHeaders(),
      });
    }

    const { data } = await localAutomationAxios.get<AutomationRunsResponse>(
      basePath,
      { params: { limit, offset } },
    );
    return data;
  }

  static async getAutomationRuns(
    id: string,
    limit = 50,
    offset = 0,
  ): Promise<AutomationRunsResponse> {
    return AutomationService.listAutomationRuns(id, { limit, offset });
  }

  static async toggleAutomation(
    id: string,
    enabled: boolean,
  ): Promise<Automation> {
    return AutomationService.updateAutomation(id, { enabled });
  }

  static async downloadTarball(id: string, name: string): Promise<void> {
    const active = getActiveBackend().backend;
    const path = `${AUTOMATION_BASE_PATH}${getAutomationIdEndpoint("tarball", id)}`;

    let blob: Blob;
    if (active.kind === "cloud") {
      blob = await callCloudProxy<Blob>({
        backend: active,
        method: "GET",
        path,
        responseType: "blob",
        headers: await buildAutomationRequestHeaders(),
      });
    } else {
      const { data } = await localAutomationAxios.get<Blob>(path, {
        responseType: "blob",
      });
      blob = data;
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.tar`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /**
   * Ask what this deployment supports, before a setup form renders.
   *
   * The setup endpoints answer a contract authored in `OpenHands/extensions`,
   * so their bodies are camelCase where the rest of this service is snake_case.
   */
  static async getCapabilities(): Promise<DeploymentCapabilities> {
    const active = getActiveBackend().backend;
    const path = `${AUTOMATION_BASE_PATH}${getAutomationEndpoint("capabilities")}`;

    if (active.kind === "cloud") {
      return callCloudProxy<DeploymentCapabilities>({
        backend: active,
        method: "GET",
        path,
        headers: await buildAutomationRequestHeaders(),
      });
    }

    const { data } =
      await localAutomationAxios.get<DeploymentCapabilities>(path);
    return data;
  }

  /**
   * Validate a draft without creating it. An invalid draft is a 200 carrying
   * field-addressed errors; only a malformed envelope is a 4xx.
   */
  static async validateDraft(
    body: SetupRequestBody,
  ): Promise<ValidateDraftResponse> {
    const active = getActiveBackend().backend;
    const path = `${AUTOMATION_BASE_PATH}${getAutomationEndpoint("validate")}`;

    if (active.kind === "cloud") {
      return callCloudProxy<ValidateDraftResponse>({
        backend: active,
        method: "POST",
        path,
        body,
        headers: await buildAutomationRequestHeaders(),
      });
    }

    const { data } = await localAutomationAxios.post<ValidateDraftResponse>(
      path,
      body,
    );
    return data;
  }

  /**
   * Create from a draft a setup form derived. Unlike {@link
   * AutomationService.createAutomation}, which exists for import and has to
   * park the record on a placeholder trigger, this sends the finished record in
   * one request.
   */
  static async createAutomationDraft(
    body: SetupRequestBody,
  ): Promise<Record<string, unknown>> {
    const active = getActiveBackend().backend;
    const path = `${AUTOMATION_BASE_PATH}${AUTOMATION_CREATE_ENDPOINT}`;

    if (active.kind === "cloud") {
      return callCloudProxy<Record<string, unknown>>({
        backend: active,
        method: "POST",
        path,
        body,
        headers: await buildAutomationRequestHeaders(),
      });
    }

    const { data } = await localAutomationAxios.post<Record<string, unknown>>(
      path,
      body,
    );
    return data;
  }

  static async checkHealth(): Promise<AutomationHealthResponse> {
    const active = getActiveBackend().backend;
    const path = `${AUTOMATION_BASE_PATH}${getAutomationEndpoint("health")}`;

    try {
      if (active.kind === "cloud") {
        const response = await callCloudProxy<AutomationHealthResponse>({
          backend: active,
          method: "GET",
          path,
          // Fail fast, matching the local branch's 5s timeout below.
          timeoutSeconds: 5,
          headers: await buildAutomationRequestHeaders(),
        });
        return response;
      }

      const { data } = await localAutomationAxios.get<AutomationHealthResponse>(
        path,
        { timeout: 5000 },
      );
      return data;
    } catch {
      return { status: "error" };
    }
  }
}

export default AutomationService;
