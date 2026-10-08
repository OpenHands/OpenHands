import type { CloudRequestOptions } from "@openhands/typescript-client/clients";
import type { Backend } from "../backend-registry/types";
import {
  activeOrgForBackend,
  createCloudClientForRuntime,
  createCloudClient,
} from "./client";
import { recordOrganizationSuspension } from "./organization-suspension-store";

export interface CloudProxyRequest {
  backend: Backend;
  method: CloudRequestOptions["method"];
  path: string;
  body?: unknown;
  headers?: Record<string, string>;
  timeoutSeconds?: number;
  hostOverride?: string;
  authMode?: "bearer" | "session-api-key" | "none";
  sessionApiKey?: string | null;
  responseType?: "blob" | "arrayBuffer";
}

export async function callCloudProxy<TResponse = unknown>(
  req: CloudProxyRequest,
): Promise<TResponse> {
  const client = req.hostOverride
    ? createCloudClientForRuntime(req.backend)
    : createCloudClient(req.backend);
  // The org this request is scoped to, read when the client is built so a
  // later org switch can't be blamed for this response.
  const orgId = activeOrgForBackend(req.backend);

  try {
    return await client.request<TResponse>({
      method: req.method,
      path: req.path,
      body: req.body,
      headers: req.headers,
      timeoutSeconds: req.timeoutSeconds,
      hostOverride: req.hostOverride,
      authMode:
        req.authMode === undefined || req.authMode === "bearer"
          ? "bearer"
          : req.authMode,
      sessionApiKey: req.sessionApiKey,
      responseType: req.responseType,
    });
  } catch (error) {
    recordOrganizationSuspension(req.backend.id, orgId, error);
    throw error;
  }
}
