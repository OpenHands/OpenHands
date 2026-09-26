import React from "react";
import AutomationService from "#/api/automation-service/automation-service.api";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  getPendingLocalTelemetryRevocationId,
  getTelemetryConsent,
  subscribeTelemetryConsent,
} from "#/services/telemetry";

const INITIAL_RETRY_DELAY_MS = 1000;
const MAX_RETRY_DELAY_MS = 30_000;

interface DesiredConsentSync {
  key: string;
  consent: "granted" | "denied";
}

export function useSyncAutomationTelemetryConsent() {
  const { backend } = useActiveBackend();
  const consent = React.useSyncExternalStore(
    subscribeTelemetryConsent,
    getTelemetryConsent,
    () => "pending" as const,
  );
  const desiredSyncRef = React.useRef<DesiredConsentSync | null>(null);
  const inFlightKeysRef = React.useRef(new Set<string>());
  const lastSuccessfulKeyRef = React.useRef<string | null>(null);
  const retryStateRef = React.useRef<{
    key: string;
    failureCount: number;
  } | null>(null);
  const retryTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );

  React.useEffect(() => {
    const clearRetryTimer = () => {
      if (retryTimerRef.current !== null) {
        clearTimeout(retryTimerRef.current);
        retryTimerRef.current = null;
      }
    };

    const pendingRevocationId =
      consent === "pending" ? getPendingLocalTelemetryRevocationId() : null;
    if (
      backend.kind !== "local" ||
      (consent === "pending" && !pendingRevocationId)
    ) {
      desiredSyncRef.current = null;
      clearRetryTimer();
      retryStateRef.current = null;
      return undefined;
    }

    const resolvedConsent = consent === "granted" ? "granted" : "denied";
    const pendingActor = consent === "pending" ? pendingRevocationId : "";
    const syncKey = `${backend.id}:${backend.connectionRevision ?? 0}:${resolvedConsent}:${pendingActor}`;
    desiredSyncRef.current = { key: syncKey, consent: resolvedConsent };

    if (retryStateRef.current?.key !== syncKey) {
      clearRetryTimer();
      retryStateRef.current = { key: syncKey, failureCount: 0 };
    }

    const startSync = () => {
      const desired = desiredSyncRef.current;
      if (
        desired?.key !== syncKey ||
        lastSuccessfulKeyRef.current === syncKey ||
        inFlightKeysRef.current.has(syncKey)
      ) {
        return;
      }

      inFlightKeysRef.current.add(syncKey);

      void AutomationService.syncTelemetryConsent(desired.consent)
        .then(() => {
          if (desiredSyncRef.current?.key !== syncKey) return;
          lastSuccessfulKeyRef.current = syncKey;
          retryStateRef.current = null;
          clearRetryTimer();
        })
        .catch(() => {
          if (desiredSyncRef.current?.key !== syncKey) return;

          const failureCount =
            retryStateRef.current?.key === syncKey
              ? retryStateRef.current.failureCount + 1
              : 1;
          retryStateRef.current = { key: syncKey, failureCount };
          const retryDelay = Math.min(
            INITIAL_RETRY_DELAY_MS * 2 ** Math.min(failureCount - 1, 10),
            MAX_RETRY_DELAY_MS,
          );
          clearRetryTimer();
          retryTimerRef.current = setTimeout(() => {
            retryTimerRef.current = null;
            startSync();
          }, retryDelay);
        })
        .finally(() => {
          inFlightKeysRef.current.delete(syncKey);
        });
    };

    startSync();

    return () => {
      desiredSyncRef.current = null;
      clearRetryTimer();
    };
  }, [backend.connectionRevision, backend.id, backend.kind, consent]);
}
