import React from "react";
import type { BootstrapConfig } from "posthog-js";
import {
  configurePostHogBootstrap,
  configureTelemetry,
  initializePostHogClient,
  type TelemetryConfiguration,
} from "#/services/telemetry";

const POSTHOG_BOOTSTRAP_KEY = "posthog_bootstrap";

function isBootstrapConfig(value: unknown): value is BootstrapConfig {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.distinctID === "string" &&
    typeof candidate.sessionID === "string"
  );
}

function readBootstrapIds(): BootstrapConfig | undefined {
  if (typeof window === "undefined" || typeof sessionStorage === "undefined") {
    return undefined;
  }

  const params = new URLSearchParams(window.location.hash.slice(1));
  const distinctID = params.get("distinct_id");
  const sessionID = params.get("session_id");
  if (distinctID && sessionID) {
    const bootstrap = { distinctID, sessionID };
    try {
      sessionStorage.setItem(POSTHOG_BOOTSTRAP_KEY, JSON.stringify(bootstrap));
    } catch {
      // OAuth continuity is best effort when browser storage is unavailable.
    }
    try {
      window.history.replaceState(
        null,
        "",
        window.location.pathname + window.location.search,
      );
    } catch {
      // Telemetry must never prevent the application from rendering.
    }
    return bootstrap;
  }

  try {
    const stored = sessionStorage.getItem(POSTHOG_BOOTSTRAP_KEY);
    if (!stored) return undefined;

    sessionStorage.removeItem(POSTHOG_BOOTSTRAP_KEY);
    const parsed: unknown = JSON.parse(stored);
    return isBootstrapConfig(parsed) ? parsed : undefined;
  } catch {
    try {
      sessionStorage.removeItem(POSTHOG_BOOTSTRAP_KEY);
    } catch {
      // Ignore unavailable storage.
    }
    return undefined;
  }
}

export function TelemetryProvider({
  children,
  config = {},
}: {
  children: React.ReactNode;
  config?: TelemetryConfiguration;
}) {
  const configuredBootstrap = React.useRef(false);
  const analyticsEnabled = config !== false;
  const apiKey = config === false ? undefined : config.apiKey;
  const apiHost = config === false ? undefined : config.apiHost;
  const uiHost = config === false ? undefined : config.uiHost;

  React.useLayoutEffect(() => {
    configureTelemetry(analyticsEnabled ? { apiKey, apiHost, uiHost } : false);
    if (!configuredBootstrap.current) {
      configurePostHogBootstrap(readBootstrapIds());
      configuredBootstrap.current = true;
    }
  }, [analyticsEnabled, apiHost, apiKey, uiHost]);

  React.useEffect(() => {
    if (analyticsEnabled) {
      void initializePostHogClient().catch(() => {
        // Analytics are optional; the service retries on the next operation.
      });
    }
  }, [analyticsEnabled, apiHost, apiKey, uiHost]);

  return children;
}
