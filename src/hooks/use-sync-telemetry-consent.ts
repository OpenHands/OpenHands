import React from "react";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import {
  clearPendingCloudTelemetryConsent,
  getPendingCloudTelemetryConsent,
  setTelemetryConsent,
  subscribeTelemetryConsent,
} from "#/services/telemetry";
import { useSettings } from "./query/use-settings";

/**
 * Hook to sync PostHog opt-in/out state with the backend setting.
 *
 * Reconciles the browser's capture state with the backend preference.
 *
 * A first-run user can make an explicit choice before Cloud login. That newer
 * browser decision remains pending across local backends, but does not
 * automatically fill unset local backend settings; each local backend can ask
 * for its own persisted preference. The pending decision is written to the
 * Cloud backend after login; until Cloud confirms it, a stale/default backend
 * value must not overwrite it. With no pending browser choice, backend updates
 * remain authoritative so revocation in another tab or via the API is respected.
 *
 * Consent model:
 *   true  → opt in  (user explicitly accepted)
 *   false → opt out (user explicitly denied)
 *   null  → pending (consent not yet collected; PostHog stays opted out until
 *           the user decides)
 */
export const useSyncTelemetryConsent = () => {
  const { backend } = useActiveBackend();
  const { data: settings } = useSettings();
  const { mutate: saveSettings, isPending: isSavingSettings } = useSaveSettings(
    "personal",
    { retry: 2 },
  );
  const pendingBrowserConsent = React.useSyncExternalStore(
    subscribeTelemetryConsent,
    getPendingCloudTelemetryConsent,
    () => null,
  );
  const attemptedSyncRef = React.useRef<string | null>(null);

  React.useEffect(() => {
    if (settings === undefined) return;

    if (pendingBrowserConsent !== null) {
      const backendConsent =
        settings.user_consents_to_analytics === true
          ? "granted"
          : settings.user_consents_to_analytics === false
            ? "denied"
            : null;

      if (backendConsent === pendingBrowserConsent) {
        attemptedSyncRef.current = null;
        if (backend.kind === "cloud") {
          clearPendingCloudTelemetryConsent(pendingBrowserConsent);
        }
        return;
      }

      if (backend.kind !== "cloud") return;

      const syncKey = `${backend.id}:${pendingBrowserConsent}`;
      if (isSavingSettings || attemptedSyncRef.current === syncKey) return;

      attemptedSyncRef.current = syncKey;
      saveSettings({
        user_consents_to_analytics: pendingBrowserConsent === "granted",
      });
      return;
    }

    if (settings.user_consents_to_analytics === null) return;

    void setTelemetryConsent(
      settings.user_consents_to_analytics === true ? "granted" : "denied",
      { syncToCloud: false },
    );
  }, [
    backend.id,
    backend.kind,
    isSavingSettings,
    pendingBrowserConsent,
    saveSettings,
    settings,
  ]);
};
