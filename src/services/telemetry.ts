/**
 * Telemetry service for tracking library usage.
 *
 * This module handles anonymous telemetry for the @openhands/agent-canvas package
 * using the PostHog SDK for reliable event delivery with batching, retry logic,
 * and offline support.
 *
 * TRACKING PHILOSOPHY:
 * - Install event (canvas_install): Sent immediately on first use, regardless of consent.
 *   This is anonymous and contains no PII - just basic browser info and a random ID.
 * - Session/custom events: Only sent after user grants consent via the consent modal.
 * - Users can opt out of all future tracking by declining consent.
 *
 * AD BLOCKER BYPASS:
 * By default, telemetry is routed through OpenHands' reverse proxy (z.openhands.dev)
 * to avoid being blocked by ad blockers. Library consumers can override this with:
 * - VITE_POSTHOG_HOST: Custom proxy URL or direct PostHog URL
 * - VITE_POSTHOG_UI_HOST: PostHog UI host (defaults to https://us.posthog.com)
 *
 * IMPORTANT: By default, telemetry is sent to the OpenHands PostHog project.
 * Source builds can override this with VITE_POSTHOG_API_KEY. Precompiled
 * library consumers can pass the same settings to configureTelemetry().
 *
 * Users can disable all telemetry (including install tracking) via:
 * - Setting VITE_DO_NOT_TRACK=1 environment variable
 * - Browser's Do Not Track setting
 */

import type { BootstrapConfig, CaptureResult, PostHog } from "posthog-js";
import { getLockedCloudAuthMode } from "#/api/agent-server-config";
import defaults from "../../config/defaults.json";

import packageJson from "../../package.json";
import {
  AGENT_CANVAS_CLIENT_SOURCE,
  AGENT_CANVAS_CLIENT_VERSION,
} from "#/api/client-source";
import {
  getBackendTelemetryProperties,
  getCloudTelemetryProperties,
  type BackendTelemetryContextInput,
  type CloudTelemetryContextInput,
} from "#/services/telemetry-context";

const TELEMETRY_CONSENT_KEY = "openhands-telemetry-consent";
const TELEMETRY_CONSENT_PENDING_CLOUD_SYNC_KEY =
  "openhands-telemetry-consent-pending-cloud-sync";
const TELEMETRY_CONSENT_PENDING_LOCAL_REVOCATION_KEY =
  "openhands-telemetry-consent-pending-local-revocation";
const TELEMETRY_CONSENT_CHANGE_EVENT = "openhands-telemetry-consent-change";
const TELEMETRY_FIRST_USE_KEY = "openhands-telemetry-first-use";
const TELEMETRY_SESSION_KEY = "openhands-telemetry-session";
const POSTHOG_INSTANCE_NAME = "agent-canvas";
const POSTHOG_PAGEVIEW_CAPTURE_MODE = "history_change";

// Unconfigured source builds use staging. Production release workflows pass
// VITE_POSTHOG_API_KEY explicitly for both the app and library artifacts.
const DEFAULT_POSTHOG_API_KEY: string =
  (import.meta.env.VITE_POSTHOG_API_KEY as string | undefined) ||
  defaults.telemetry.posthogApiKey;

// Default to OpenHands' reverse proxy to bypass ad blockers.
// The proxy at z.openhands.dev routes to PostHog's US region.
// Library consumers can override this with their own proxy or direct PostHog URL.
const DEFAULT_POSTHOG_HOST =
  import.meta.env.VITE_POSTHOG_HOST || "https://z.openhands.dev";

// UI host is needed for PostHog features like toolbar to work correctly
// when using a reverse proxy. Defaults to US region.
const DEFAULT_POSTHOG_UI_HOST =
  import.meta.env.VITE_POSTHOG_UI_HOST || "https://us.posthog.com";

export interface TelemetryConfig {
  /** PostHog project key. Useful for precompiled library consumers. */
  apiKey?: string;
  /** Event ingestion host or reverse proxy. */
  apiHost?: string;
  /** PostHog UI host used by toolbar links and other UI features. */
  uiHost?: string;
}

export type TelemetryConfiguration = TelemetryConfig | false;

export type TelemetryConsent = "granted" | "denied" | "pending";
export type ResolvedTelemetryConsent = Exclude<TelemetryConsent, "pending">;

export interface SetTelemetryConsentOptions {
  /** Do not persist a value mirrored from backend settings back to Cloud. */
  syncToCloud?: boolean;
}

let posthogInstance: PostHog | null = null;
let initializationPromise: Promise<PostHog | null> | null = null;
let pendingBootstrap: BootstrapConfig | undefined;
let telemetryConfig: TelemetryConfig = {};
let telemetryDisabled = false;

interface TelemetryIdentity {
  distinctId: string;
  properties: Record<string, unknown>;
}

// undefined means that Cloud identity has not resolved yet; null means that it
// resolved without a user. This distinction prevents startup from resetting a
// persisted identity while the current account is still loading.
let desiredTelemetryIdentity: TelemetryIdentity | null | undefined;
let desiredIdentityRevision = 0;
let appliedIdentityRevision = -1;

const CANVAS_EVENT_PROPERTIES = Object.freeze({
  client_source: AGENT_CANVAS_CLIENT_SOURCE,
  client_version: AGENT_CANVAS_CLIENT_VERSION,
  package_name: packageJson.name,
  package_version: packageJson.version,
});

let telemetryBackendContext = getBackendTelemetryProperties({});
let telemetryCloudContext = getCloudTelemetryProperties();

export function setTelemetryBackendContext(
  context: BackendTelemetryContextInput,
): void {
  telemetryBackendContext = getBackendTelemetryProperties(context);
}

export function setTelemetryCloudContext(
  context: CloudTelemetryContextInput | null,
): void {
  telemetryCloudContext = getCloudTelemetryProperties(context);
}

function addCanvasEventProperties(
  event: CaptureResult | null,
): CaptureResult | null {
  if (!event) return null;

  return {
    ...event,
    properties: {
      ...telemetryBackendContext,
      ...telemetryCloudContext,
      ...event.properties,
      ...CANVAS_EVENT_PROPERTIES,
    },
  };
}

function restorePostHogConsent(posthog: PostHog): void {
  if (telemetryDisabled || getTelemetryConsent() !== "granted") {
    posthog.opt_out_capturing();
  } else {
    posthog.opt_in_capturing();
  }
}

function resetPostHogIdentity(posthog: PostHog, resetDeviceId = false): void {
  posthog.reset(resetDeviceId);
  appliedIdentityRevision = -1;
  // PostHog reset clears its own consent persistence, so immediately restore
  // the canonical Canvas decision kept in localStorage.
  restorePostHogConsent(posthog);
}

function applyDesiredTelemetryIdentity(posthog: PostHog): void {
  if (desiredTelemetryIdentity === undefined || !isTelemetryEnabled()) return;

  const desiredId = desiredTelemetryIdentity?.distinctId;
  const currentId = posthog.get_property("$user_id");
  if (currentId != null && currentId !== desiredId) {
    resetPostHogIdentity(posthog);
  }

  if (desiredTelemetryIdentity === null) {
    appliedIdentityRevision = desiredIdentityRevision;
    return;
  }

  if (
    posthog.get_property("$user_id") !== desiredTelemetryIdentity.distinctId ||
    appliedIdentityRevision !== desiredIdentityRevision
  ) {
    posthog.identify(
      desiredTelemetryIdentity.distinctId,
      desiredTelemetryIdentity.properties,
    );
    appliedIdentityRevision = desiredIdentityRevision;
  }
}

function propertiesEqual(
  left: Record<string, unknown>,
  right: Record<string, unknown>,
): boolean {
  const keys = Object.keys(left);
  return (
    keys.length === Object.keys(right).length &&
    keys.every((key) => left[key] === right[key])
  );
}

/**
 * Configure the single Canvas telemetry client before its first use.
 * Passing false disables telemetry and install tracking for embedded hosts.
 */
export function configureTelemetry(config: TelemetryConfiguration): void {
  if (config === false) {
    if (telemetryDisabled) return;
    telemetryDisabled = true;
    posthogInstance?.opt_out_capturing();
    notifyTelemetryConsentListeners();
    return;
  }

  const wasDisabled = telemetryDisabled;
  telemetryDisabled = false;

  if (!posthogInstance && !initializationPromise) {
    const definedConfig = Object.fromEntries(
      Object.entries(config).filter(([, value]) => value !== undefined),
    ) as TelemetryConfig;
    telemetryConfig = { ...telemetryConfig, ...definedConfig };
  }

  if (wasDisabled) {
    if (posthogInstance) {
      restorePostHogConsent(posthogInstance);
      applyDesiredTelemetryIdentity(posthogInstance);
    }
    notifyTelemetryConsentListeners();
  }
}

function getResolvedTelemetryConfig(): Required<TelemetryConfig> | null {
  if (telemetryDisabled) return null;

  return {
    apiKey: telemetryConfig.apiKey || DEFAULT_POSTHOG_API_KEY,
    apiHost: telemetryConfig.apiHost || DEFAULT_POSTHOG_HOST,
    uiHost: telemetryConfig.uiHost || DEFAULT_POSTHOG_UI_HOST,
  };
}

/**
 * Check if we're in a browser environment
 */
function isBrowser(): boolean {
  return typeof window !== "undefined" && typeof localStorage !== "undefined";
}

/**
 * Lazily load PostHog to avoid SSR/Node.js issues.
 * PostHog is a browser-only library, so we dynamically import it only when needed.
 */
async function getPostHog(): Promise<PostHog | null> {
  if (!isBrowser()) {
    return null;
  }

  if (posthogInstance) {
    return posthogInstance;
  }

  try {
    const { default: posthog } = await import("posthog-js");
    return posthog;
  } catch {
    // Failed to load PostHog - telemetry will be disabled
    return null;
  }
}

/**
 * Check if telemetry is disabled via environment variable or browser setting.
 * Works in both Node.js and browser (Vite) environments.
 */
function isDoNotTrackEnabled(): boolean {
  if (telemetryDisabled) {
    return true;
  }

  // Check Vite environment variable (browser)
  if (
    typeof import.meta !== "undefined" &&
    import.meta.env?.VITE_DO_NOT_TRACK === "1"
  ) {
    return true;
  }

  // Check Node.js environment variable (SSR/testing)
  if (typeof process !== "undefined" && process.env?.DO_NOT_TRACK === "1") {
    return true;
  }

  // Check browser's navigator.doNotTrack standard
  if (
    typeof navigator !== "undefined" &&
    (navigator.doNotTrack === "1" ||
      // @ts-expect-error - Some browsers use window.doNotTrack
      (typeof window !== "undefined" && window.doNotTrack === "1"))
  ) {
    return true;
  }

  return false;
}

/**
 * Initialize PostHog SDK.
 *
 * @param enableCapturing - If true, enable capturing immediately (for install tracking).
 *                          If false, start with capturing disabled (for consent-gated tracking).
 */
export function configurePostHogBootstrap(
  bootstrap: BootstrapConfig | undefined,
): void {
  if (!posthogInstance) {
    pendingBootstrap = bootstrap;
  }
}

export async function initializePostHogClient(
  enableCapturing = false,
): Promise<PostHog | null> {
  if (posthogInstance) {
    return posthogInstance;
  }

  if (initializationPromise) {
    return initializationPromise;
  }

  initializationPromise = (async () => {
    const config = getResolvedTelemetryConfig();
    if (!config) {
      return null;
    }

    const posthog = await getPostHog();
    if (!posthog) {
      return null;
    }

    // A named instance isolates Canvas configuration, consent, identity, and
    // persistence from a host application's default PostHog singleton.
    const initializedPostHog = posthog.init(
      config.apiKey,
      {
        api_host: config.apiHost,
        ui_host: config.uiHost,
        opt_out_capturing_by_default: !enableCapturing,
        persistence: "localStorage",
        persistence_name: POSTHOG_INSTANCE_NAME,
        consent_persistence_name: `${POSTHOG_INSTANCE_NAME}-consent`,
        person_profiles: "always",
        capture_pageview: POSTHOG_PAGEVIEW_CAPTURE_MODE,
        autocapture: true,
        disable_session_recording: true,
        bootstrap: pendingBootstrap,
        before_send: addCanvasEventProperties,
      },
      POSTHOG_INSTANCE_NAME,
    );
    if (!initializedPostHog) return null;

    posthogInstance = initializedPostHog;
    pendingBootstrap = undefined;

    if (telemetryDisabled) {
      posthogInstance.opt_out_capturing();
    } else if (getTelemetryConsent() === "granted") {
      posthogInstance.opt_in_capturing();
      applyDesiredTelemetryIdentity(posthogInstance);
    } else if (!enableCapturing) {
      posthogInstance.opt_out_capturing();
    }

    return posthogInstance;
  })();

  try {
    return await initializationPromise;
  } finally {
    if (!posthogInstance) {
      initializationPromise = null;
    }
  }
}

/**
 * Get user's telemetry consent preference
 */
export function getTelemetryConsent(): TelemetryConsent {
  if (!isBrowser()) {
    return "pending";
  }

  // Check environment variable for opt-out
  if (isDoNotTrackEnabled()) {
    return "denied";
  }

  if (getLockedCloudAuthMode() === "cookie") {
    return "granted";
  }

  try {
    const consent = localStorage.getItem(TELEMETRY_CONSENT_KEY);
    if (consent === "granted" || consent === "denied") {
      return consent;
    }
  } catch {
    // Ignore storage errors
  }

  return "pending";
}

/**
 * Return an explicit browser choice that still needs to survive a Cloud login.
 * It remains pending across local backends so their settings cannot consume a
 * decision that must still be applied after the user connects to Cloud.
 */
export function getPendingCloudTelemetryConsent(): ResolvedTelemetryConsent | null {
  if (!isBrowser()) return null;

  try {
    const consent = localStorage.getItem(
      TELEMETRY_CONSENT_PENDING_CLOUD_SYNC_KEY,
    );
    return consent === "granted" || consent === "denied" ? consent : null;
  } catch {
    return null;
  }
}

/** Return the pre-reset actor whose local Automation consent must be revoked. */
export function getPendingLocalTelemetryRevocationId(): string | null {
  if (!isBrowser()) return null;

  try {
    return localStorage.getItem(TELEMETRY_CONSENT_PENDING_LOCAL_REVOCATION_KEY);
  } catch {
    return null;
  }
}

export function subscribeTelemetryConsent(listener: () => void): () => void {
  if (!isBrowser()) return () => {};
  const handleStorage = (event: StorageEvent) => {
    if (
      event.key === TELEMETRY_CONSENT_KEY ||
      event.key === TELEMETRY_CONSENT_PENDING_CLOUD_SYNC_KEY
    ) {
      listener();
    }
  };
  window.addEventListener(TELEMETRY_CONSENT_CHANGE_EVENT, listener);
  window.addEventListener("storage", handleStorage);

  return () => {
    window.removeEventListener(TELEMETRY_CONSENT_CHANGE_EVENT, listener);
    window.removeEventListener("storage", handleStorage);
  };
}

function notifyTelemetryConsentListeners(): void {
  if (isBrowser())
    window.dispatchEvent(new Event(TELEMETRY_CONSENT_CHANGE_EVENT));
}

function markTelemetryConsentForCloudSync(
  consent: ResolvedTelemetryConsent,
): void {
  if (!isBrowser()) return;

  try {
    localStorage.setItem(TELEMETRY_CONSENT_PENDING_CLOUD_SYNC_KEY, consent);
  } catch {
    // Ignore storage errors; the in-browser consent decision still applies.
  }
}

function markPendingLocalTelemetryRevocation(distinctId: string | null): void {
  if (!isBrowser() || !distinctId) return;

  try {
    localStorage.setItem(
      TELEMETRY_CONSENT_PENDING_LOCAL_REVOCATION_KEY,
      distinctId,
    );
  } catch {
    // Ignore storage errors; browser capture is still disabled below.
  }
}

export function clearPendingLocalTelemetryRevocation(
  expectedDistinctId: string,
): void {
  if (!isBrowser()) return;

  try {
    if (
      localStorage.getItem(TELEMETRY_CONSENT_PENDING_LOCAL_REVOCATION_KEY) !==
      expectedDistinctId
    ) {
      return;
    }
    localStorage.removeItem(TELEMETRY_CONSENT_PENDING_LOCAL_REVOCATION_KEY);
  } catch {
    // Ignore storage errors.
  }
}

export function clearPendingCloudTelemetryConsent(
  expected?: ResolvedTelemetryConsent,
): void {
  if (!isBrowser()) return;

  try {
    if (
      expected !== undefined &&
      localStorage.getItem(TELEMETRY_CONSENT_PENDING_CLOUD_SYNC_KEY) !==
        expected
    ) {
      return;
    }
    localStorage.removeItem(TELEMETRY_CONSENT_PENDING_CLOUD_SYNC_KEY);
    notifyTelemetryConsentListeners();
  } catch {
    // Ignore storage errors.
  }
}

/**
 * Set user's telemetry consent preference
 */
export async function setTelemetryConsent(
  consent: ResolvedTelemetryConsent,
  { syncToCloud = true }: SetTelemetryConsentOptions = {},
): Promise<void> {
  if (!isBrowser()) {
    return;
  }

  try {
    const previousConsent = getTelemetryConsent();
    const unchanged = previousConsent === consent;
    localStorage.setItem(TELEMETRY_CONSENT_KEY, consent);
    if (telemetryDisabled) return;
    if (unchanged) return;

    // Reuse an initialized client synchronously so a same-flush identify()
    // cannot run before consent is applied. Only the cold path awaits import.
    const posthog = posthogInstance ?? (await initializePostHogClient());
    if (!posthog) {
      return;
    }

    if (consent === "granted") {
      posthog.opt_in_capturing();
      applyDesiredTelemetryIdentity(posthog);
    } else {
      markPendingLocalTelemetryRevocation(posthog.get_distinct_id?.() ?? null);
      if (posthog.get_property?.("$user_id") != null) {
        resetPostHogIdentity(posthog);
      } else {
        posthog.opt_out_capturing();
      }
    }
  } catch {
    // Ignore storage errors
  } finally {
    // Notify UI/backend/identity reconcilers only after the browser capture state
    // reflects this decision. Otherwise a pre-login grant can trigger an
    // identify while PostHog is still opted out and lose funnel continuity.
    if (syncToCloud) {
      markTelemetryConsentForCloudSync(consent);
    }
    notifyTelemetryConsentListeners();
  }
}

/**
 * Declare the current Cloud identity. The telemetry service applies it only
 * after consent and owns all reset/account-switch semantics.
 */
export async function setTelemetryIdentity(
  distinctId: string | null,
  properties: Record<string, unknown> = {},
): Promise<void> {
  const nextIdentity = distinctId === null ? null : { distinctId, properties };
  const unchanged =
    desiredTelemetryIdentity === nextIdentity ||
    (desiredTelemetryIdentity !== undefined &&
      desiredTelemetryIdentity !== null &&
      nextIdentity !== null &&
      desiredTelemetryIdentity.distinctId === nextIdentity.distinctId &&
      propertiesEqual(desiredTelemetryIdentity.properties, properties));
  if (unchanged) return;

  desiredTelemetryIdentity = nextIdentity;
  desiredIdentityRevision += 1;
  appliedIdentityRevision = -1;

  if (!isTelemetryEnabled()) return;
  const posthog = posthogInstance ?? (await initializePostHogClient());
  if (posthog && isTelemetryEnabled()) {
    // Read the desired identity after the await so a newer account always wins
    // if identity changes while the SDK is loading.
    applyDesiredTelemetryIdentity(posthog);
  }
}

/**
 * Check if telemetry is enabled (user has granted consent)
 */
export function isTelemetryEnabled(): boolean {
  return getTelemetryConsent() === "granted";
}

/**
 * Check if first use event has already been sent
 */
function hasFirstUseSent(): boolean {
  if (!isBrowser()) {
    return false;
  }

  try {
    return localStorage.getItem(TELEMETRY_FIRST_USE_KEY) === "true";
  } catch {
    return false;
  }
}

/**
 * Mark first use event as sent
 */
function markFirstUseSent(): void {
  if (!isBrowser()) {
    return;
  }

  try {
    localStorage.setItem(TELEMETRY_FIRST_USE_KEY, "true");
  } catch {
    // Ignore storage errors
  }
}

/**
 * Track the initial install of the library.
 *
 * IMPORTANT: This is sent immediately on first use, regardless of consent status.
 * This allows us to track library adoption even if users haven't made a consent choice yet.
 *
 * The event is:
 * - Completely anonymous (no PII, just a random PostHog distinct_id)
 * - Sent only once per installation (tracked via localStorage, persists across sessions)
 * - Still respects DO_NOT_TRACK environment variable and browser setting
 *
 * Users who want complete privacy can:
 * - Set VITE_DO_NOT_TRACK=1 or browser's Do Not Track
 * - Later deny consent to prevent all future tracking
 */
export async function trackInstall(): Promise<void> {
  // Respect hard opt-out via environment variable or browser setting
  if (isDoNotTrackEnabled()) {
    return;
  }

  // Already sent install event (persisted in localStorage - survives app relaunches)
  if (hasFirstUseSent()) {
    return;
  }

  // Initialize PostHog with capturing enabled (for this one event)
  const posthog = await initializePostHogClient(true);
  if (!posthog || isDoNotTrackEnabled()) {
    return;
  }

  // Temporarily enable capturing if it was disabled
  const wasOptedOut = posthog.has_opted_out_capturing?.() ?? false;
  if (wasOptedOut) {
    posthog.opt_in_capturing();
  }

  // Capture the install event
  posthog.capture("canvas_install", {
    platform: typeof navigator !== "undefined" ? navigator.platform : "unknown",
    user_agent:
      typeof navigator !== "undefined" ? navigator.userAgent : "unknown",
    referrer: typeof document !== "undefined" ? document.referrer : "",
    url_origin: typeof window !== "undefined" ? window.location.origin : "",
    embedded: typeof window !== "undefined" && window.self !== window.top,
  });

  // Mark as sent (stored in localStorage - persists across browser sessions)
  markFirstUseSent();

  // Restore opt-out state if user hasn't granted consent yet
  // This ensures we only send the install event, not subsequent events
  const currentConsent = getTelemetryConsent();
  if (currentConsent !== "granted") {
    posthog.opt_out_capturing();
  }
}

/**
 * Check if session start event has already been sent (this browser session)
 */
function hasSessionSent(): boolean {
  if (!isBrowser()) {
    return false;
  }

  try {
    return sessionStorage.getItem(TELEMETRY_SESSION_KEY) === "true";
  } catch {
    return false;
  }
}

/**
 * Mark session start event as sent (uses sessionStorage so it resets on new tabs/sessions)
 */
function markSessionSent(): void {
  if (!isBrowser()) {
    return;
  }

  try {
    sessionStorage.setItem(TELEMETRY_SESSION_KEY, "true");
  } catch {
    // Ignore storage errors
  }
}

/** Return the shared client only when a consented capture is safe to emit. */
async function getPostHogForConsentedCapture(): Promise<PostHog | null> {
  if (!isTelemetryEnabled() || getTelemetryConsent() !== "granted") return null;

  const posthog = await initializePostHogClient();
  if (!posthog || !isTelemetryEnabled()) return null;

  // The browser preference is the canonical capture decision. PostHog may
  // still carry an older opt-out marker while backend consent is loading or
  // after a previous backend temporarily reported a stale value. Heal that
  // drift at the event boundary so capture() cannot silently discard an event
  // that the user has explicitly allowed.
  if (posthog.has_opted_out_capturing?.()) {
    posthog.opt_in_capturing();
  }

  applyDesiredTelemetryIdentity(posthog);

  return posthog;
}

/**
 * Track a session start event.
 * Called each time a new browser session starts (respects consent).
 * Uses sessionStorage for deduplication - only sends once per browser session.
 */
export async function trackSessionStart(): Promise<void> {
  // Already sent session event this browser session
  if (hasSessionSent()) {
    return;
  }

  const posthog = await getPostHogForConsentedCapture();
  if (!posthog) return;

  posthog.capture("canvas_new_session", {
    is_first_use: !hasFirstUseSent(),
  });

  // Mark as sent for this session
  markSessionSent();
}

/** Return the active PostHog distinct ID when consent allows capture. */
export async function getTelemetryDistinctId(): Promise<string | null> {
  const posthog = await getPostHogForConsentedCapture();
  return posthog?.get_distinct_id?.() ?? null;
}

/** Return the PostHog distinct ID for local consent sync without emitting capture. */
export async function getTelemetryDistinctIdForConsentSync(): Promise<
  string | null
> {
  if (!isBrowser()) return null;

  if (getTelemetryConsent() !== "granted") {
    const pendingRevocationId = getPendingLocalTelemetryRevocationId();
    if (pendingRevocationId) return pendingRevocationId;
  }

  if (telemetryDisabled || isDoNotTrackEnabled()) return null;

  const posthog = await initializePostHogClient();
  return posthog?.get_distinct_id?.() ?? null;
}

/**
 * Track a custom event (respects consent).
 */
export async function trackEvent(
  eventName: string,
  properties: Record<string, unknown> = {},
): Promise<void> {
  const posthog = await getPostHogForConsentedCapture();
  if (!posthog) return;

  posthog.capture(eventName, properties);
}

/** Track an exception through the same consent-aware client as custom events. */
export async function trackException(
  error: unknown,
  properties: Record<string, unknown> = {},
): Promise<void> {
  const posthog = await getPostHogForConsentedCapture();
  if (!posthog) return;

  posthog.captureException(error, properties);
}

/**
 * Clear all telemetry data (for privacy/GDPR requests)
 */
export async function clearTelemetryData(): Promise<void> {
  if (!isBrowser()) {
    return;
  }

  try {
    markPendingLocalTelemetryRevocation(
      posthogInstance?.get_distinct_id?.() ?? null,
    );
    localStorage.removeItem(TELEMETRY_CONSENT_KEY);
    localStorage.removeItem(TELEMETRY_FIRST_USE_KEY);
  } catch {
    // Continue clearing the in-memory and SDK identity if storage is blocked.
  }
  clearPendingCloudTelemetryConsent();
  try {
    sessionStorage.removeItem(TELEMETRY_SESSION_KEY);
  } catch {
    // Continue clearing the in-memory and SDK identity if storage is blocked.
  }

  telemetryBackendContext = getBackendTelemetryProperties({});
  telemetryCloudContext = getCloudTelemetryProperties();
  desiredTelemetryIdentity = null;
  desiredIdentityRevision += 1;
  appliedIdentityRevision = -1;

  try {
    if (posthogInstance) {
      resetPostHogIdentity(posthogInstance, true);
    }
  } catch {
    // A reset failure must not leave capture enabled after a privacy clear.
    try {
      posthogInstance?.opt_out_capturing();
    } catch {
      // Telemetry failures must not break the application.
    }
  } finally {
    notifyTelemetryConsentListeners();
  }
}
