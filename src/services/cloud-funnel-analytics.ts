import { AGENT_CANVAS_CLIENT_SOURCE } from "#/api/client-source";
import { getBackendTelemetryProperties } from "#/services/telemetry-context";
import { trackEvent } from "#/services/telemetry";

export type CloudConnectionSource =
  | "onboarding"
  | "add_backend_modal"
  | "manage_backends_modal"
  | "cloud_auto_connect";

const CLOUD_CONVERSATION_READY_INSERT_ID_PREFIX = `${AGENT_CANVAS_CLIENT_SOURCE}:cloud_conversation_ready`;

function trackCloudFunnelEvent(
  event: string,
  properties: Record<string, unknown>,
): void {
  void trackEvent(event, properties);
}

function cloudLoginBackendContext() {
  return getBackendTelemetryProperties({
    backendKind: "cloud",
    connectionMethod: "cloud_login",
  });
}

function cloudConversationBackendContext() {
  return getBackendTelemetryProperties({
    backendKind: "cloud",
  });
}

export function trackCloudDeviceAuthorizationStarted(
  _host: string,
  source?: CloudConnectionSource,
  attemptId?: string,
): void {
  trackCloudFunnelEvent("cloud_device_authorization_started", {
    ...cloudLoginBackendContext(),
    attempt_id: attemptId,
    source,
  });
}

export function trackCloudDeviceAuthorizationSucceeded(
  _host: string,
  source?: CloudConnectionSource,
  attemptId?: string,
): void {
  trackCloudFunnelEvent("cloud_device_authorization_succeeded", {
    ...cloudLoginBackendContext(),
    attempt_id: attemptId,
    source,
  });
}

export function trackCloudDeviceAuthorizationFailed({
  source,
  attemptId,
  stage,
  reason,
}: {
  source?: CloudConnectionSource;
  attemptId?: string;
  stage: "authorization_request" | "authorization_poll";
  reason: string;
}): void {
  trackCloudFunnelEvent("cloud_device_authorization_failed", {
    ...cloudLoginBackendContext(),
    attempt_id: attemptId,
    source,
    stage,
    reason,
  });
}

export function trackCloudConversationReady(
  taskId: string,
  conversationId: string,
): void {
  trackCloudFunnelEvent("cloud_conversation_ready", {
    ...cloudConversationBackendContext(),
    $insert_id: `${CLOUD_CONVERSATION_READY_INSERT_ID_PREFIX}:${taskId}`,
    task_id: taskId,
    conversation_id: conversationId,
  });
}
