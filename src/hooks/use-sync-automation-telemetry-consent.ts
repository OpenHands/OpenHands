import React from "react";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  getPendingLocalTelemetryRevocationId,
  getTelemetryConsent,
  subscribeTelemetryConsent,
} from "#/services/telemetry";

export function useSyncAutomationTelemetryConsent() {
  const { backend } = useActiveBackend();
  const consent = React.useSyncExternalStore(
    subscribeTelemetryConsent,
    getTelemetryConsent,
    () => "pending" as const,
  );
  const lastSyncKeyRef = React.useRef<string | null>(null);

  React.useEffect(() => {
    const pendingRevocationId =
      consent === "pending" ? getPendingLocalTelemetryRevocationId() : null;
    if (
      backend.kind !== "local" ||
      (consent === "pending" && !pendingRevocationId)
    ) {
      return;
    }

    const resolvedConsent = consent === "granted" ? "granted" : "denied";
    const pendingActor = consent === "pending" ? pendingRevocationId : "";
    const syncKey = `${backend.id}:${backend.host}:${backend.apiKey ?? ""}:${resolvedConsent}:${pendingActor}`;
    if (lastSyncKeyRef.current === syncKey) return;
    lastSyncKeyRef.current = syncKey;

    void AutomationService.syncTelemetryConsent(resolvedConsent).catch(
      () => {},
    );
  }, [backend.apiKey, backend.host, backend.id, backend.kind, consent]);
}
