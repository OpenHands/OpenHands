import React from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCloudCurrentUserId } from "#/hooks/query/use-cloud-current-user-id";
import { useSettings } from "#/hooks/query/use-settings";
import { useTracking } from "#/hooks/use-tracking";
import {
  setTelemetryCloudContext,
  setTelemetryIdentity,
} from "#/services/telemetry";

const CLOUD_COOKIE_BACKEND_ADDED_SESSION_PREFIX =
  "agent_canvas_cloud_cookie_backend_added";
const trackedCloudCookieBackendAddedSessionKeys = new Set<string>();

function getCloudCookieBackendAddedSessionKey(
  backendId: string,
  userId: string,
): string {
  return `${CLOUD_COOKIE_BACKEND_ADDED_SESSION_PREFIX}:${backendId}:${userId}`;
}

function hasTrackedCloudCookieBackendAddedThisSession(
  backendId: string,
  userId: string,
): boolean {
  const key = getCloudCookieBackendAddedSessionKey(backendId, userId);
  if (typeof window === "undefined") return true;

  try {
    return (
      window.sessionStorage.getItem(key) === "true" ||
      trackedCloudCookieBackendAddedSessionKeys.has(key)
    );
  } catch {
    return trackedCloudCookieBackendAddedSessionKeys.has(key);
  }
}

function markCloudCookieBackendAddedTrackedThisSession(
  backendId: string,
  userId: string,
): void {
  const key = getCloudCookieBackendAddedSessionKey(backendId, userId);
  if (typeof window === "undefined") return;

  try {
    window.sessionStorage.setItem(key, "true");
  } catch {
    trackedCloudCookieBackendAddedSessionKeys.add(key);
  }
}

/** Keep Cloud user event context aligned with the active backend. */
export const useTelemetryIdentity = () => {
  const { backend, orgId } = useActiveBackend();
  const { data: settings } = useSettings();
  const { trackBackendAdded } = useTracking();
  const trackBackendAddedRef = React.useRef(trackBackendAdded);
  const userIds = useCloudCurrentUserId();
  const identity = backend.kind === "cloud" ? userIds[backend.id] : undefined;
  const isIdentityLoading = identity?.isLoading ?? true;
  const userId = identity?.userId ?? null;
  const email =
    (settings?.email || settings?.git_user_email)?.trim() || undefined;

  React.useEffect(() => {
    trackBackendAddedRef.current = trackBackendAdded;
  }, [trackBackendAdded]);

  React.useEffect(() => {
    if (backend.kind !== "cloud") {
      setTelemetryCloudContext(null);
      return;
    }

    if (isIdentityLoading) return;

    if (!userId) {
      setTelemetryCloudContext(null);
      void setTelemetryIdentity(null);
      return;
    }

    setTelemetryCloudContext({ userId, email, orgId });
    void setTelemetryIdentity(userId, email ? { email } : {});

    if (
      backend.authMode === "cookie" &&
      email &&
      !hasTrackedCloudCookieBackendAddedThisSession(backend.id, userId)
    ) {
      trackBackendAddedRef.current({
        backendKind: "cloud",
        connectionMethod: "cloud_cookie",
        hasApiKey: false,
        source: "cloud_auto_connect",
      });
      markCloudCookieBackendAddedTrackedThisSession(backend.id, userId);
    }
  }, [
    backend.authMode,
    backend.connectionRevision,
    backend.id,
    backend.kind,
    email,
    isIdentityLoading,
    orgId,
    userId,
  ]);
};
