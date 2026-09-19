import React from "react";
import type { BootstrapConfig } from "posthog-js";
import { useTelemetry } from "#/hooks/use-telemetry";
import {
  configurePostHogBootstrap,
  configureTelemetry,
  initializePostHogClient,
  setTelemetryWebsiteAttribution,
  type TelemetryConfiguration,
  type WebsiteHandoffAttribution,
} from "#/services/telemetry";

const POSTHOG_BOOTSTRAP_KEY = "posthog_bootstrap";
const POSTHOG_HANDOFF_PARAM = "oh_ph_handoff";
const consumedHandoffNonces = new Set<string>();

type PostHogHandoff = {
  bootstrap: BootstrapConfig;
  attribution?: WebsiteHandoffAttribution;
};

type StoredHandoff = PostHogHandoff & {
  exp?: number;
  nonce?: string;
};

const ATTRIBUTION_KEYS = [
  "utm_source",
  "utm_medium",
  "utm_campaign",
  "landing_page_category",
  "cta_id",
  "cta_surface",
  "referring_domain_category",
] as const;

function isBootstrapConfig(value: unknown): value is BootstrapConfig {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.distinctID === "string" &&
    typeof candidate.sessionID === "string"
  );
}

function safeSessionStorage(): Storage | null {
  if (typeof window === "undefined") return null;

  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

function safeLocalStorage(): Storage | null {
  if (typeof window === "undefined") return null;

  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function base64UrlDecode(value: string): string {
  const normalized = value.replace(/-/g, "+").replace(/_/g, "/");
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
  const decoded = atob(padded);
  const encoded = Array.from(
    decoded,
    (char) => `%${char.charCodeAt(0).toString(16).padStart(2, "0")}`,
  ).join("");
  return decodeURIComponent(encoded);
}

function sanitizeAttribution(
  value: unknown,
): WebsiteHandoffAttribution | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const source = value as Record<string, unknown>;
  const attribution: WebsiteHandoffAttribution = {};

  for (const key of ATTRIBUTION_KEYS) {
    const candidate = source[key];
    if (typeof candidate === "string" && candidate.trim()) {
      attribution[key] = candidate.trim().slice(0, 80);
    }
  }

  return Object.keys(attribution).length > 0 ? attribution : undefined;
}

function isHandoffNonceConsumed(nonce: string): boolean {
  if (consumedHandoffNonces.has(nonce)) return true;

  try {
    return (
      safeLocalStorage()?.getItem(`${POSTHOG_BOOTSTRAP_KEY}:${nonce}`) ===
      "consumed"
    );
  } catch {
    return false;
  }
}

function markHandoffNonceConsumed(nonce: string): void {
  consumedHandoffNonces.add(nonce);
  try {
    safeLocalStorage()?.setItem(
      `${POSTHOG_BOOTSTRAP_KEY}:${nonce}`,
      "consumed",
    );
  } catch {
    // In-memory replay protection still applies for this page lifetime.
  }
}

function removeHandoffFromUrl(params: URLSearchParams): void {
  params.delete(POSTHOG_HANDOFF_PARAM);
  params.delete("distinct_id");
  params.delete("session_id");
  const nextHash = params.toString();

  try {
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${window.location.search}${nextHash ? `#${nextHash}` : ""}`,
    );
  } catch {
    // Telemetry must never prevent the application from rendering.
  }
}

function parseStructuredHandoff(encoded: string): StoredHandoff | undefined {
  try {
    const parsed: unknown = JSON.parse(base64UrlDecode(encoded));
    if (typeof parsed !== "object" || parsed === null) return undefined;

    const candidate = parsed as Record<string, unknown>;
    if (candidate.v !== 1) return undefined;
    if (typeof candidate.exp !== "number" || candidate.exp < Date.now())
      return undefined;
    if (typeof candidate.nonce !== "string" || !candidate.nonce)
      return undefined;
    if (isHandoffNonceConsumed(candidate.nonce)) return undefined;
    if (typeof candidate.distinct_id !== "string" || !candidate.distinct_id)
      return undefined;
    if (typeof candidate.session_id !== "string" || !candidate.session_id)
      return undefined;

    const handoff = {
      bootstrap: {
        distinctID: candidate.distinct_id.slice(0, 256),
        sessionID: candidate.session_id.slice(0, 256),
      },
      attribution: sanitizeAttribution(candidate.attribution),
      exp: candidate.exp,
      nonce: candidate.nonce,
    };
    markHandoffNonceConsumed(candidate.nonce);
    return handoff;
  } catch {
    return undefined;
  }
}

function readHandoffFromUrl(): PostHogHandoff | null | undefined {
  const params = new URLSearchParams(window.location.hash.slice(1));
  const structured = params.get(POSTHOG_HANDOFF_PARAM);
  const distinctID = params.get("distinct_id");
  const sessionID = params.get("session_id");
  if (!structured && !(distinctID && sessionID)) return undefined;

  const handoff = structured
    ? parseStructuredHandoff(structured)
    : {
        bootstrap: { distinctID: distinctID ?? "", sessionID: sessionID ?? "" },
      };

  if (handoff) {
    try {
      safeSessionStorage()?.setItem(
        POSTHOG_BOOTSTRAP_KEY,
        JSON.stringify(handoff),
      );
    } catch {
      // OAuth continuity is best effort when browser storage is unavailable.
    }
  }

  removeHandoffFromUrl(params);
  return handoff ?? null;
}

function isStoredHandoff(value: unknown): value is StoredHandoff {
  if (isBootstrapConfig(value)) return true;
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Record<string, unknown>;
  return isBootstrapConfig(candidate.bootstrap);
}

function readStoredHandoff(): PostHogHandoff | undefined {
  const storage = safeSessionStorage();
  if (!storage) return undefined;

  try {
    const stored = storage.getItem(POSTHOG_BOOTSTRAP_KEY);
    if (!stored) return undefined;

    storage.removeItem(POSTHOG_BOOTSTRAP_KEY);
    const parsed: unknown = JSON.parse(stored);
    if (!isStoredHandoff(parsed)) return undefined;

    if (isBootstrapConfig(parsed)) return { bootstrap: parsed };
    if (typeof parsed.exp === "number" && parsed.exp < Date.now())
      return undefined;
    return {
      bootstrap: parsed.bootstrap,
      attribution: sanitizeAttribution(parsed.attribution),
    };
  } catch {
    try {
      storage.removeItem(POSTHOG_BOOTSTRAP_KEY);
    } catch {
      // Ignore unavailable storage.
    }
    return undefined;
  }
}

function readPostHogHandoff(): PostHogHandoff | undefined {
  if (typeof window === "undefined") return undefined;
  const urlHandoff = readHandoffFromUrl();
  return urlHandoff === undefined
    ? readStoredHandoff()
    : (urlHandoff ?? undefined);
}

function TelemetryLifecycle() {
  useTelemetry();
  return null;
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
      const handoff = readPostHogHandoff();
      configurePostHogBootstrap(handoff?.bootstrap);
      setTelemetryWebsiteAttribution(handoff?.attribution);
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

  return (
    <>
      {analyticsEnabled ? <TelemetryLifecycle /> : null}
      {children}
    </>
  );
}
