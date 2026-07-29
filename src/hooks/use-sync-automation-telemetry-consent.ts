import React from "react";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
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
    if (backend.kind !== "local" || consent === "pending") return;

    const syncKey = `${backend.id}:${backend.host}:${backend.apiKey ?? ""}:${consent}`;
    if (lastSyncKeyRef.current === syncKey) return;
    lastSyncKeyRef.current = syncKey;

    void AutomationService.syncTelemetryConsent(consent).catch(() => {});
  }, [backend.apiKey, backend.host, backend.id, backend.kind, consent]);
}
