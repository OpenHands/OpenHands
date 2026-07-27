import React from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCloudCurrentUserId } from "#/hooks/query/use-cloud-current-user-id";
import { useSettings } from "#/hooks/query/use-settings";
import {
  setTelemetryCloudContext,
  setTelemetryIdentity,
} from "#/services/telemetry";

/** Keep Cloud user event context aligned with the active backend. */
export const useTelemetryIdentity = () => {
  const { backend, orgId } = useActiveBackend();
  const { data: settings } = useSettings();
  const userIds = useCloudCurrentUserId();
  const identity = backend.kind === "cloud" ? userIds[backend.id] : undefined;
  const isIdentityLoading = identity?.isLoading ?? true;
  const userId = identity?.userId ?? null;
  const email = settings?.email || settings?.git_user_email || undefined;

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
  }, [
    backend.connectionRevision,
    backend.id,
    backend.kind,
    email,
    isIdentityLoading,
    orgId,
    userId,
  ]);
};
