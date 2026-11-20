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
  connectionKey: string;
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
  const syncCurrentRef = React.useRef<(() => void) | null>(null);
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
      syncCurrentRef.current = null;
      clearRetryTimer();
      retryStateRef.current = null;
      return undefined;
    }

    const resolvedConsent = consent === "granted" ? "granted" : "denied";
    const pendingActor = consent === "pending" ? pendingRevocationId : "";
    const connectionKey = `${backend.id}:${backend.connectionRevision ?? 0}`;
    const syncKey = `${connectionKey}:${resolvedConsent}:${pendingActor}`;
    desiredSyncRef.current = {
      key: syncKey,
      connectionKey,
      consent: resolvedConsent,
    };

    if (retryStateRef.current?.key !== syncKey) {
      clearRetryTimer();
      retryStateRef.current = { key: syncKey, failureCount: 0 };
    }

    const startSync = () => {
      const desired = desiredSyncRef.current;
      if (
        desired?.key !== syncKey ||
        retryTimerRef.current !== null ||
        lastSuccessfulKeyRef.current === syncKey ||
        inFlightKeysRef.current.has(syncKey)
      ) {
        return;
      }

      inFlightKeysRef.current.add(syncKey);
      // An intervening write makes the prior acknowledgement insufficient if
      // the user returns to that value while this request is still in flight.
      lastSuccessfulKeyRef.current = null;

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
          const latest = desiredSyncRef.current;
          if (
            latest?.connectionKey === connectionKey &&
            latest.key !== syncKey
          ) {
            // The older write may have reached the server after the latest
            // acknowledgement, even if its response failed. Reassert the
            // current consent once it settles, scoped to this connection.
            lastSuccessfulKeyRef.current = null;
          }
          syncCurrentRef.current?.();
        });
    };

    syncCurrentRef.current = startSync;
    startSync();

    return () => {
      desiredSyncRef.current = null;
      syncCurrentRef.current = null;
      clearRetryTimer();
    };
  }, [backend.connectionRevision, backend.id, backend.kind, consent]);
}
