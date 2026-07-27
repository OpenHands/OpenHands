import React from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ServerClient } from "@openhands/typescript-client/clients";
import OpenHandsLogoWhite from "#/assets/branding/openhands-logo-white.svg?react";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { SegmentedToggle } from "#/components/features/files-tab/segmented-toggle";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { useNavigation } from "#/context/navigation-context";
import { useBackendsHealth } from "#/hooks/query/use-backends-health";
import { useTracking } from "#/hooks/use-tracking";
import type { CloudConnectionSource } from "#/services/cloud-funnel-analytics";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getLockedCloudHost } from "#/api/agent-server-config";
import { isOpenHandsCloudHost } from "#/api/device-flow-client";
import {
  assertAgentServerVersionIsSupported,
  getDisplayAgentServerVersion,
} from "#/api/agent-server-compatibility";
import ChevronDownSmallIcon from "#/icons/chevron-down-small.svg?react";
import { I18nKey } from "#/i18n/declaration";
import type { Backend, BackendKind } from "#/api/backend-registry/types";
import { getUserFacingConnectionErrorMessage } from "#/utils/user-facing-error";
import { cn } from "#/utils/utils";
import {
  modalTitleLgClassName,
  modalTitleLgMediumClassName,
} from "#/utils/modal-classes";
import { getBackendStatusLabel } from "./backend-status-label";
import { BackendStatusDot } from "./backend-status-dot";
import { DeviceFlowAuth } from "./device-flow-auth";

export type BackendFormMode = "add" | "edit";

interface BackendConnectionTestMetadata {
  agentServerVersion: string | null;
}

interface BackendFormModalProps {
  mode: BackendFormMode;
  /** Required when `mode === "edit"`. */
  backend?: Backend;
  onClose: () => void;
  /** Analytics surface for the `backend_added` event (add mode only). */
  source?: BackendAddedSource;
  /** Hide the close button and disable backdrop/escape dismissal. Used for locked Cloud first-run. */
  hideCloseButton?: boolean;
}

/**
 * Seed the default backend kind from the host. Uses proper hostname-suffix
 * matching (via {@link isOpenHandsCloudHost}) rather than a substring test, so
 * a look-alike host such as `all-hands-testing.dev` isn't misread as cloud.
 *
 * This is only a *default*: a self-hosted OpenHands Cloud/Enterprise instance
 * on a truly custom domain is indistinguishable from a local agent-server by
 * host alone, so the manual add form lets the user override the kind
 * explicitly (see the Type selector in ManualConnectionColumn).
 */
function inferKindFromHost(host: string): BackendKind {
  return isOpenHandsCloudHost(host) ? "cloud" : "local";
}

/**
 * Returns true for hostnames that represent a local / private-network address.
 * Used by normalizeHost to choose http:// instead of https://.
 */
function isLocalAddress(hostname: string): boolean {
  // Strip IPv6 bracket notation: [::1] → ::1
  const h = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  // IPv6 loopback, any-address, and named loopback
  if (h === "localhost" || h === "::1" || h === "::" || h === "0.0.0.0")
    return true;
  // 127.x.x.x loopback range + IPv4-mapped loopback (::ffff:127.x.x.x)
  if (/^127\./.test(h) || /^::ffff:127\./i.test(h)) return true;
  // RFC 1918 private ranges
  if (/^10\./.test(h)) return true;
  if (/^192\.168\./.test(h)) return true;
  if (/^172\.(1[6-9]|2\d|3[01])\./.test(h)) return true;
  // IPv6 link-local (fe80::/10) and unique local (fc00::/7)
  if (/^fe[89ab][0-9a-f]:/i.test(h)) return true;
  if (/^f[cd][0-9a-f]{2}:/i.test(h)) return true;
  // mDNS / Bonjour (.local)
  if (h.endsWith(".local")) return true;
  // Single-label hostnames (no dots, no colons) are local network names.
  // Colons are excluded so bare IPv6 addresses don't accidentally match.
  if (!h.includes(".") && !h.includes(":")) return true;
  return false;
}

function normalizeHost(host: string): string {
  const trimmed = host.trim().replace(/\/+$/, "");
  if (!trimmed) return "";
  // Already has an explicit scheme — respect it.
  if (/^https?:\/\//i.test(trimmed)) return trimmed;
  // Extract the pure hostname for scheme selection, handling three cases:
  //   [::1]:8080  → bracket IPv6 notation → extract ::1
  //   ::1         → bare IPv6 (multiple colons, no bracket) → whole string
  //   host:port   → regular host:port → part before the colon
  const bracketMatch = trimmed.match(/^\[([^\]]+)\]/);
  const hostname = bracketMatch
    ? bracketMatch[1]
    : (trimmed.match(/:/g) ?? []).length > 1
      ? trimmed
      : trimmed.split(":")[0];
  const scheme = isLocalAddress(hostname) ? "http" : "https";
  return `${scheme}://${trimmed}`;
}

/**
 * Returns true when `host` represents a reachable backend URL.
 *
 * Rules (applied in order):
 *   1. Must be non-empty after trimming.
 *   2. Must contain no whitespace — spaces can never appear in a host/port.
 *   3. After normalisation (bare hosts get `https://` prepended), must parse
 *      as a valid http or https URL with a non-empty hostname.
 */
function isValidHostUrl(host: string): boolean {
  const trimmed = host.trim();
  if (!trimmed) return false;
  // Spaces anywhere in the input are an immediate rejection.
  if (/\s/.test(trimmed)) return false;
  const normalized = normalizeHost(trimmed);
  if (!normalized) return false;
  try {
    const url = new URL(normalized);
    return (
      (url.protocol === "http:" || url.protocol === "https:") &&
      url.hostname.length > 0
    );
  } catch {
    return false;
  }
}

const DEFAULT_OPENHANDS_CLOUD_HOST = "https://app.all-hands.dev";

export type BackendConnectionMethod = "manual" | "cloud_login";

export type BackendAddedSource = CloudConnectionSource;

function getConnectionTestFailedTitle(
  t: ReturnType<typeof useTranslation>["t"],
  host: string,
): string {
  return t(I18nKey.BACKEND$CONNECTION_TEST_FAILED, {
    host,
    interpolation: { escapeValue: false },
  });
}

function getConnectionErrorDetail(error: unknown): string | null {
  return getUserFacingConnectionErrorMessage(error);
}

function getConnectionTestFailedMessage(title: string, error: unknown): string {
  const detail = getConnectionErrorDetail(error);
  return detail ? `${title}\n${detail}` : title;
}

async function testBackendConnection(
  backend: Pick<Backend, "host" | "apiKey" | "kind">,
): Promise<BackendConnectionTestMetadata> {
  // Cloud backends authenticate via OAuth; preflight GET is not applicable.
  if (backend.kind !== "local") return { agentServerVersion: null };

  const serverInfo = await new ServerClient(
    getAgentServerClientOptions({
      host: backend.host,
      sessionApiKey: backend.apiKey || null,
      timeout: 5000,
    }),
  ).getServerInfo();
  assertAgentServerVersionIsSupported(serverInfo);
  return { agentServerVersion: getDisplayAgentServerVersion(serverInfo) };
}

/**
 * Live status row for the edit form: shows a connection dot, a
 * "Local"/"Cloud" label, and the agent server's reported version when
 * available. Replaces the legacy local/cloud radio fieldset (kind is
 * now inferred from the host).
 */
function BackendStatusBadge({
  backend,
  testIdRoot,
}: {
  backend: Backend;
  testIdRoot: string;
}) {
  const { t } = useTranslation("openhands");
  const healthByBackendId = useBackendsHealth([backend]);
  const health = healthByBackendId[backend.id];
  const isConnected = health?.isConnected ?? null;
  const disabled = health?.disabled === true;
  const consecutiveFailures = health?.consecutiveFailures ?? 0;
  const lastError = health?.lastError ?? null;

  const { data: version } = useQuery({
    queryKey: ["backend-version", backend.host, backend.apiKey],
    queryFn: async () => {
      const info = await new ServerClient(
        getAgentServerClientOptions({
          host: backend.host,
          sessionApiKey: backend.apiKey || null,
          timeout: 5000,
        }),
      ).getServerInfo();
      return getDisplayAgentServerVersion(info);
    },
    retry: false,
    staleTime: 60_000,
    enabled: backend.kind === "local" && !disabled,
  });

  const statusLabel = getBackendStatusLabel(t, backend, health);

  const kindLabel =
    backend.kind === "cloud"
      ? t(I18nKey.BACKEND$KIND_CLOUD)
      : t(I18nKey.BACKEND$KIND_LOCAL);

  return (
    <div className="flex flex-col gap-2">
      <div
        data-testid={`${testIdRoot}-status`}
        className="flex items-center gap-3 text-sm"
      >
        <BackendStatusDot isConnected={isConnected} />
        <span className="text-white" data-testid={`${testIdRoot}-status-label`}>
          {statusLabel}
        </span>
        <span className="text-tertiary-alt">·</span>
        <span className="text-[var(--oh-text-tertiary)]">{kindLabel}</span>
        {version ? (
          <span
            className="text-xs text-[var(--oh-muted)]"
            data-testid={`${testIdRoot}-version`}
          >
            {t(I18nKey.BACKEND$VERSION_LABEL, { version })}
          </span>
        ) : null}
      </div>

      {disabled ? (
        <div
          data-testid={`${testIdRoot}-status-error`}
          className="flex flex-col gap-1 rounded-md border border-red-500/40 bg-red-500/10 p-3 text-sm"
        >
          <span className="font-semibold text-red-300">
            {t(I18nKey.BACKEND$HEALTH_FAILED_TITLE)}
          </span>
          <span className="text-xs text-[var(--oh-text-tertiary)]">
            {t(I18nKey.BACKEND$HEALTH_FAILED_DETAIL, {
              count: consecutiveFailures,
            })}
          </span>
          {lastError ? (
            <span
              data-testid={`${testIdRoot}-status-error-message`}
              className="text-xs text-red-300 whitespace-pre-wrap break-words"
            >
              {lastError}
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export interface BackendFormSubmitPayload {
  name: string;
  host: string;
  apiKey: string;
  kind: BackendKind;
}

interface UseBackendFormOptions {
  initialName?: string;
  initialHost?: string;
  initialApiKey?: string;
  /**
   * Called to test the connection. The hook does NOT call
   * `testBackendConnection` directly so callers can inject a
   * wrapped version (e.g. with extra logging or different timeout).
   * Should throw on failure.
   */
  onTestConnection: (
    payload: BackendFormSubmitPayload,
  ) => Promise<BackendConnectionTestMetadata>;
  /** Called after a successful connection test and persistence. */
  onSuccess: (metadata: BackendConnectionTestMetadata) => void;
  /** Require a non-empty API key even when the host looks local. */
  requireApiKey?: boolean;
  /**
   * When provided, completely replaces the default submit flow
   * (onTestConnection + onSuccess). The hook still manages form state
   * and canSubmit validation, but the caller owns error handling and
   * success side effects. Should throw on failure.
   */
  onSubmitOverride?: (payload: BackendFormSubmitPayload) => Promise<void>;
}

/**
 * Shared hook for the backend-form state used by both `BackendForm`
 * (edit/add mode) and `ManualConnectionColumn` (add-mode-only column).
 * Encapsulates name / host / apiKey fields, `connectionError`,
 * `isSubmitting`, and the shared `handleSubmit` flow.
 */
function useBackendForm({
  initialName = "",
  initialHost = "",
  initialApiKey = "",
  onTestConnection,
  onSuccess,
  requireApiKey = false,
  onSubmitOverride,
}: UseBackendFormOptions) {
  const { t } = useTranslation("openhands");

  const [name, setName] = React.useState(initialName);
  const [host, setHost] = React.useState(initialHost);
  const [apiKey, setApiKey] = React.useState(initialApiKey);
  const [connectionError, setConnectionError] = React.useState<string | null>(
    null,
  );
  const [isSubmitting, setIsSubmitting] = React.useState(false);
  const [kindOverride, setKindOverride] = React.useState<BackendKind | null>(
    null,
  );

  // Kind follows host inference until the user explicitly picks a type, then
  // respects that choice. A custom-domain OHE can't be distinguished from a
  // custom-domain local agent-server by host alone, so ManualConnectionColumn
  // exposes `setKind` (a Type selector) to let the user declare it.
  const kind = kindOverride ?? inferKindFromHost(host);
  const needsApiKey = requireApiKey || kind !== "local";
  const canSubmit =
    name.trim().length > 0 &&
    isValidHostUrl(host) &&
    (!needsApiKey || apiKey.trim().length > 0);

  const handleSubmit = React.useCallback(
    async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();
      if (!canSubmit || isSubmitting) return;

      const payload: BackendFormSubmitPayload = {
        name: name.trim(),
        host: normalizeHost(host),
        apiKey: apiKey.trim(),
        kind,
      };

      setConnectionError(null);
      setIsSubmitting(true);

      try {
        // When onSubmitOverride is provided, it completely replaces the
        // default flow (onTestConnection + onSuccess).
        if (onSubmitOverride) {
          await onSubmitOverride(payload);
        } else {
          const metadata = await onTestConnection(payload);
          onSuccess(metadata);
        }
      } catch (error) {
        setConnectionError(
          getConnectionTestFailedMessage(
            getConnectionTestFailedTitle(t, payload.host),
            error,
          ),
        );
      } finally {
        setIsSubmitting(false);
      }
    },
    [
      canSubmit,
      isSubmitting,
      name,
      host,
      apiKey,
      kind,
      onTestConnection,
      onSuccess,
      requireApiKey,
      onSubmitOverride,
      t,
    ],
  );

  return {
    name,
    setName,
    host,
    setHost,
    apiKey,
    setApiKey,
    connectionError,
    setConnectionError,
    isSubmitting,
    kind,
    setKind: setKindOverride,
    canSubmit,
    handleSubmit,
  };
}

export interface BackendFormProps {
  mode: BackendFormMode;
  /** Required when `mode === "edit"`. */
  backend?: Backend;
  /**
   * Called after the form is submitted and the backend has been
   * persisted. Use this to dismiss a containing modal, advance an
   * onboarding step, etc.
   */
  onSubmitted: () => void;
  /**
   * Optional render slot rendered in place of the default
   * Save / Cancel button row, so callers (e.g. the onboarding flow)
   * can re-skin the action area while still owning submission via the
   * standard `<form onSubmit>` flow. Receives the form's submit-ready
   * state.
   */
  renderActions?: (state: {
    canSubmit: boolean;
    isSubmitting: boolean;
    testIdRoot: string;
  }) => React.ReactNode;
  /** Used to disambiguate test ids across the same screen. */
  testIdRoot?: string;
  /** When true, the host field is pre-filled and disabled. */
  hostReadOnly?: boolean;
  /**
   * When true, a non-empty API key is required for submission regardless
   * of the inferred backend kind.  The standard add form allows empty
   * keys for local backends; the auth-gate screen needs to enforce one.
   */
  requireApiKey?: boolean;
  /**
   * When true, hides the name/host/API-key inputs (and related inline
   * errors) while keeping the action row visible — used by onboarding
   * after a successful connection probe.
   */
  hideConfigurationFields?: boolean;
  /**
   * Replace the default synchronous add/update-and-close submit with a
   * custom async handler.  The form builds the payload, validates
   * client-side, then hands it to this callback. If the callback throws,
   * the form remains open so the caller can surface errors.
   */
  onSubmitOverride?: (payload: BackendFormSubmitPayload) => Promise<void>;
}

/**
 * Reusable form body for adding / editing a backend. Renders the
 * common name / host / API-key inputs plus the kind selector
 * (radio buttons in `add` mode, status badge in `edit` mode).
 *
 * Rendered as a `<form>`, so consumers should put any extra controls
 * either inside `renderActions` or as siblings inside a wrapping
 * element — but submission flows through the standard form submit so
 * Enter-to-submit still works.
 */
export function BackendForm({
  mode,
  backend,
  onSubmitted,
  renderActions,
  testIdRoot: explicitTestIdRoot,
  hostReadOnly,
  requireApiKey,
  hideConfigurationFields = false,
  onSubmitOverride,
}: BackendFormProps) {
  const { t } = useTranslation("openhands");
  const { addBackend, updateBackend } = useActiveBackendContext();

  // In edit mode preserve the existing backend's kind so that renaming or
  // rotating the API key on a cloud backend (e.g. an OHE/enterprise instance
  // on a custom domain) does not silently downgrade it to "local" and switch
  // the auth header from `Authorization: Bearer` to `X-Session-API-Key`.
  // Only infer from the host when adding a new backend.
  const fixedKind: BackendKind | null =
    mode === "edit" && backend ? backend.kind : null;

  const {
    name,
    setName,
    host,
    setHost,
    apiKey,
    setApiKey,
    connectionError,
    setConnectionError,
    isSubmitting,
    kind: inferredKind,
    handleSubmit: runSubmit,
  } = useBackendForm({
    initialName: backend?.name ?? "",
    initialHost: backend?.host ?? "",
    initialApiKey: backend?.apiKey ?? "",
    onTestConnection: testBackendConnection,
    onSuccess: async () => {
      const payload: BackendFormSubmitPayload = {
        name: name.trim(),
        host: normalizeHost(host),
        apiKey: apiKey.trim(),
        kind: fixedKind ?? inferredKind,
      };
      if (mode === "edit" && backend) {
        updateBackend(backend.id, payload);
      } else {
        addBackend(payload);
      }
      onSubmitted();
    },
    requireApiKey,
    onSubmitOverride,
  });

  // Inline validation: only show errors after the user has left a field.
  const [nameTouched, setNameTouched] = React.useState(false);
  const [hostTouched, setHostTouched] = React.useState(false);

  const kind = fixedKind ?? inferredKind;
  const testIdRoot =
    explicitTestIdRoot ?? (mode === "edit" ? "edit-backend" : "add-backend");

  const needsApiKey = requireApiKey || kind !== "local";
  const canSubmit =
    name.trim().length > 0 &&
    isValidHostUrl(host) &&
    (!needsApiKey || apiKey.trim().length > 0);

  // Error messages — only surfaced after the user has blurred the field.
  const nameError =
    nameTouched && !name.trim() ? t(I18nKey.BACKEND$NAME_REQUIRED) : undefined;
  const hostError = hostTouched
    ? !host.trim()
      ? t(I18nKey.BACKEND$HOST_REQUIRED)
      : !isValidHostUrl(host)
        ? t(I18nKey.BACKEND$HOST_INVALID)
        : undefined
    : undefined;

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    if (!canSubmit) {
      // Mark all validated fields as touched so inline errors become visible
      // (e.g. user pressed Enter before filling required fields).
      setNameTouched(true);
      setHostTouched(true);
      return;
    }
    await runSubmit(event);
  };

  return (
    <form
      data-testid={`${testIdRoot}-form`}
      onSubmit={handleSubmit}
      className="flex flex-col gap-4"
    >
      <div
        data-testid={`${testIdRoot}-configuration-fields`}
        className={cn(
          "flex flex-col gap-4",
          hideConfigurationFields && "hidden",
        )}
      >
        <SettingsInput
          testId={`${testIdRoot}-name`}
          name={`${testIdRoot}-name`}
          type="text"
          label={t(I18nKey.BACKEND$NAME_LABEL)}
          value={name}
          onChange={(value) => {
            setName(value);
            setConnectionError(null);
          }}
          onBlur={() => setNameTouched(true)}
          // eslint-disable-next-line i18next/no-literal-string -- example placeholder, not user-facing copy
          placeholder="Production"
          className="w-full"
          showRequiredTag
          error={nameError}
        />

        <SettingsInput
          testId={`${testIdRoot}-host`}
          name={`${testIdRoot}-host`}
          type="text"
          label={t(I18nKey.BACKEND$HOST_LABEL)}
          value={host}
          onChange={
            hostReadOnly
              ? undefined
              : (value) => {
                  setHost(value);
                  setConnectionError(null);
                }
          }
          onBlur={() => setHostTouched(true)}
          placeholder={DEFAULT_OPENHANDS_CLOUD_HOST}
          className="w-full"
          showRequiredTag
          error={hostError}
          isDisabled={hostReadOnly}
        />

        <SettingsInput
          testId={`${testIdRoot}-api-key`}
          name={`${testIdRoot}-api-key`}
          type="password"
          label={t(I18nKey.BACKEND$KEY_LABEL)}
          value={apiKey}
          onChange={(value) => {
            setApiKey(value);
            setConnectionError(null);
          }}
          placeholder=""
          className="w-full"
        />

        {connectionError ? (
          <div
            role="alert"
            data-testid={`${testIdRoot}-error`}
            className="rounded-md border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300 whitespace-pre-wrap break-words"
          >
            {connectionError}
          </div>
        ) : null}

        {mode === "edit" && backend && (
          <BackendStatusBadge backend={backend} testIdRoot={testIdRoot} />
        )}
      </div>

      {renderActions ? (
        renderActions({
          canSubmit: canSubmit && !isSubmitting,
          isSubmitting,
          testIdRoot,
        })
      ) : (
        <div className="flex justify-end gap-2 mt-2 w-full">
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onSubmitted}
            testId={`${testIdRoot}-cancel`}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            type="submit"
            variant="primary"
            isDisabled={!canSubmit || isSubmitting}
            testId={`${testIdRoot}-submit`}
          >
            {t(I18nKey.BACKEND$SAVE)}
          </BrandButton>
        </div>
      )}
    </form>
  );
}

// ── Add-mode two-column layout ──────────────────────────────────────

/**
 * @spec BM-002 — Adding a backend auto-switches the active selection to it
 * (BM-001), so a backend-scoped detail page the user is viewing now belongs
 * to the previous backend. Redirect to that section's list so they never see
 * stale data, mirroring the switch-backend redirect in BackendSelector.
 */
function useRedirectAfterAddBackend() {
  const { currentPath, navigate } = useNavigation();
  return React.useCallback(() => {
    if (/^\/automations\/[^/]+/.test(currentPath)) navigate("/automations");
    else if (/^\/conversations\/[^/]+/.test(currentPath))
      navigate("/conversations");
  }, [currentPath, navigate]);
}

interface BackendConnectionOptionsProps {
  onConnected: (
    payload: BackendFormSubmitPayload,
    connectionMethod: BackendConnectionMethod,
    metadata?: BackendConnectionTestMetadata,
  ) => void;
  testIdRoot?: string;
  initialManualBackend?: Partial<
    Pick<BackendFormSubmitPayload, "name" | "host" | "apiKey">
  >;
  requireManualApiKey?: boolean;
  manualSubmitLabel?: React.ReactNode;
  manualSubmittingLabel?: React.ReactNode;
  manualSubmitTestId?: string;
  analyticsSource?: CloudConnectionSource;
}

/**
 * Manual agent-server connection plus OpenHands Cloud OAuth login.
 * Used by both the Add Backend modal and the onboarding backend step so
 * supported backend choices stay consistent across first-run and settings UI.
 */
export function BackendConnectionOptions({
  onConnected,
  testIdRoot = "add-backend",
  initialManualBackend,
  requireManualApiKey = false,
  manualSubmitLabel,
  manualSubmittingLabel,
  manualSubmitTestId,
  analyticsSource,
}: BackendConnectionOptionsProps) {
  const { t } = useTranslation("openhands");
  const lockedCloudHost = getLockedCloudHost();

  if (lockedCloudHost) {
    return (
      <div
        data-testid={`${testIdRoot}-connection-options`}
        className="flex justify-center"
      >
        <div className="w-full">
          <CloudLoginColumn
            onConnected={onConnected}
            testIdRoot={testIdRoot}
            lockedHost={lockedCloudHost}
            analyticsSource={analyticsSource}
          />
        </div>
      </div>
    );
  }

  return (
    <div
      data-testid={`${testIdRoot}-connection-options`}
      className="flex flex-col gap-6 md:flex-row"
    >
      <div className="flex-1 min-w-0">
        <ManualConnectionColumn
          onConnected={onConnected}
          testIdRoot={testIdRoot}
          initialBackend={initialManualBackend}
          requireApiKey={requireManualApiKey}
          submitLabel={manualSubmitLabel ?? t(I18nKey.BACKEND$CONNECT)}
          submittingLabel={
            manualSubmittingLabel ??
            t(I18nKey.ONBOARDING$BACKEND_STATUS_CHECKING)
          }
          submitTestId={manualSubmitTestId}
        />
      </div>

      <div className="flex-1 min-w-0">
        <CloudLoginColumn
          onConnected={onConnected}
          testIdRoot={testIdRoot}
          analyticsSource={analyticsSource}
        />
      </div>
    </div>
  );
}

interface ManualConnectionColumnProps {
  onConnected: (
    payload: BackendFormSubmitPayload,
    connectionMethod: BackendConnectionMethod,
    metadata?: BackendConnectionTestMetadata,
  ) => void;
  testIdRoot: string;
  initialBackend?: Partial<
    Pick<BackendFormSubmitPayload, "name" | "host" | "apiKey">
  >;
  requireApiKey: boolean;
  submitLabel: React.ReactNode;
  submittingLabel: React.ReactNode;
  submitTestId?: string;
}

/**
 * Manual connection via Host + API Key. Designed for self-hosted agent servers
 * and self-hosted OpenHands Cloud with API key auth.
 */
function ManualConnectionColumn({
  onConnected,
  testIdRoot,
  initialBackend,
  requireApiKey,
  submitLabel,
  submittingLabel,
  submitTestId,
}: ManualConnectionColumnProps) {
  const { t } = useTranslation("openhands");

  const {
    name,
    setName,
    host,
    setHost,
    apiKey,
    setApiKey,
    connectionError,
    setConnectionError,
    isSubmitting,
    kind,
    setKind,
    canSubmit,
    handleSubmit,
  } = useBackendForm({
    initialName: initialBackend?.name ?? "",
    initialHost: initialBackend?.host ?? "",
    initialApiKey: initialBackend?.apiKey ?? "",
    onTestConnection: testBackendConnection,
    onSuccess: (metadata) => {
      onConnected(
        {
          name: name.trim(),
          host: normalizeHost(host),
          apiKey: apiKey.trim(),
          kind,
        },
        "manual",
        metadata,
      );
    },
    requireApiKey,
  });

  return (
    <form
      data-testid={`${testIdRoot}-form`}
      onSubmit={handleSubmit}
      className="flex flex-col gap-4 flex-1 min-w-0"
    >
      <div className="flex flex-col gap-1">
        <SettingsInput
          testId={`${testIdRoot}-name`}
          name={`${testIdRoot}-name`}
          type="text"
          label={t(I18nKey.BACKEND$NAME_LABEL)}
          value={name}
          onChange={(value) => {
            setName(value);
            setConnectionError(null);
          }}
          // eslint-disable-next-line i18next/no-literal-string -- example placeholder, not user-facing copy
          placeholder="e.g. My Server"
          className="w-full"
        />
        <p className="text-xs text-[var(--oh-muted)]">
          {t(I18nKey.BACKEND$NAME_HELPER)}
        </p>
      </div>

      <div className="flex flex-col gap-1">
        <SettingsInput
          testId={`${testIdRoot}-host`}
          name={`${testIdRoot}-host`}
          type="text"
          label={t(I18nKey.BACKEND$HOST_LABEL)}
          value={host}
          onChange={(value) => {
            setHost(value);
            setConnectionError(null);
          }}
          // eslint-disable-next-line i18next/no-literal-string -- example value, not translatable
          placeholder="http://localhost:8000"
          className="w-full"
        />
        <p
          className="text-xs text-[var(--oh-muted)]"
          data-testid={`${testIdRoot}-host-helper`}
        >
          {t(I18nKey.BACKEND$HOST_HELPER)}
        </p>
      </div>

      <div className="flex flex-col items-start gap-2.5">
        <span className="text-sm">{t(I18nKey.BACKEND$KIND_LABEL)}</span>
        <SegmentedToggle<BackendKind>
          value={kind}
          options={[
            { value: "local", label: t(I18nKey.BACKEND$KIND_LOCAL) },
            { value: "cloud", label: t(I18nKey.BACKEND$KIND_CLOUD) },
          ]}
          onChange={(value) => setKind(value)}
          ariaLabel={t(I18nKey.BACKEND$KIND_LABEL)}
          testId={`${testIdRoot}-kind`}
        />
      </div>

      <SettingsInput
        testId={`${testIdRoot}-api-key`}
        name={`${testIdRoot}-api-key`}
        type="password"
        label={t(I18nKey.BACKEND$KEY_LABEL)}
        value={apiKey}
        onChange={(value) => {
          setApiKey(value);
          setConnectionError(null);
        }}
        // eslint-disable-next-line i18next/no-literal-string -- example value, not translatable
        placeholder="sk-••••••••••"
        className="w-full"
      />

      {connectionError ? (
        <div
          role="alert"
          data-testid={`${testIdRoot}-error`}
          className="rounded-md border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-300 whitespace-pre-wrap break-words"
        >
          {connectionError}
        </div>
      ) : null}

      <BrandButton
        type="submit"
        variant="secondary"
        isDisabled={!canSubmit || isSubmitting}
        testId={submitTestId ?? `${testIdRoot}-submit`}
        className="w-full text-center"
      >
        {isSubmitting ? submittingLabel : submitLabel}
      </BrandButton>
    </form>
  );
}

interface CloudLoginColumnProps {
  onConnected: (
    payload: BackendFormSubmitPayload,
    connectionMethod: BackendConnectionMethod,
    metadata?: BackendConnectionTestMetadata,
  ) => void;
  testIdRoot: string;
  lockedHost?: string;
  analyticsSource?: CloudConnectionSource;
}

/**
 * One-click OAuth login with OpenHands Cloud. Includes an "Advanced"
 * disclosure for users who self-host OpenHands Cloud and need to override the
 * host.
 */
function CloudLoginColumn({
  onConnected,
  testIdRoot,
  lockedHost,
  analyticsSource,
}: CloudLoginColumnProps) {
  const { t } = useTranslation("openhands");

  const [advancedOpen, setAdvancedOpen] = React.useState(false);
  const [customHost, setCustomHost] = React.useState("");

  const effectiveHost =
    lockedHost ?? (customHost.trim() || DEFAULT_OPENHANDS_CLOUD_HOST);

  const handleLoginSuccess = (apiKey: string) => {
    onConnected(
      {
        name: "OpenHands Cloud",
        host: normalizeHost(effectiveHost),
        apiKey,
        kind: "cloud",
      },
      "cloud_login",
    );
  };

  return (
    <div className="flex flex-1 min-w-0 flex-col items-center gap-3 pb-8">
      <div className="flex flex-col items-center gap-1">
        <OpenHandsLogoWhite width={56} height={56} aria-hidden />

        <h4
          className={modalTitleLgMediumClassName}
          data-testid={`${testIdRoot}-cloud-title`}
        >
          {t(I18nKey.BACKEND$CLOUD_TITLE)}
        </h4>
      </div>

      <p className="text-center text-sm leading-relaxed text-[var(--oh-muted)]">
        {t(I18nKey.BACKEND$CLOUD_DESCRIPTION)}
      </p>

      <DeviceFlowAuth
        host={effectiveHost}
        onSuccess={handleLoginSuccess}
        testIdRoot={testIdRoot}
        analyticsSource={analyticsSource}
      />

      {lockedHost ? null : (
        <div className="w-full">
          <button
            type="button"
            onClick={() => setAdvancedOpen((open) => !open)}
            aria-expanded={advancedOpen}
            data-testid={`${testIdRoot}-advanced-toggle`}
            className="flex w-full cursor-pointer items-center justify-center gap-1 text-center text-xs text-[var(--oh-muted)] transition-colors hover:text-content-2"
          >
            <span>{t(I18nKey.BACKEND$ADVANCED)}</span>
            <ChevronDownSmallIcon
              className={cn(
                "h-4 w-4 shrink-0 text-muted transition-transform",
                advancedOpen && "rotate-180",
              )}
              aria-hidden
            />
          </button>
          <div
            className={cn(
              "pt-2",
              !advancedOpen && "pointer-events-none invisible",
            )}
            aria-hidden={!advancedOpen}
          >
            <SettingsInput
              testId={`${testIdRoot}-cloud-host`}
              name={`${testIdRoot}-cloud-host`}
              type="text"
              label={t(I18nKey.BACKEND$HOST_LABEL)}
              value={customHost}
              onChange={setCustomHost}
              placeholder={DEFAULT_OPENHANDS_CLOUD_HOST}
              className="w-full"
            />
            <p className="mt-1 text-xs text-[var(--oh-muted)]">
              {t(I18nKey.BACKEND$LOGIN_CLOUD_HINT)}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function AddBackendConnectionOptions({
  onClose,
  source,
}: {
  onClose: () => void;
  source: BackendAddedSource;
}) {
  const { addBackend } = useActiveBackendContext();
  const redirectAfterAdd = useRedirectAfterAddBackend();
  const { trackBackendAdded } = useTracking();

  const handleConnected = React.useCallback(
    (
      payload: BackendFormSubmitPayload,
      connectionMethod: BackendConnectionMethod,
      metadata?: BackendConnectionTestMetadata,
    ) => {
      addBackend(payload);
      trackBackendAdded({
        backendKind: payload.kind,
        connectionMethod,
        hasApiKey: Boolean(payload.apiKey),
        source,
        agentServerVersion: metadata?.agentServerVersion,
      });
      redirectAfterAdd();
      onClose();
    },
    [addBackend, redirectAfterAdd, onClose, trackBackendAdded, source],
  );

  return (
    <BackendConnectionOptions
      onConnected={handleConnected}
      analyticsSource={source}
    />
  );
}

// ── Modal wrappers ──────────────────────────────────────────────────

/**
 * Modal wrapper. In **add** mode it renders a two-column layout
 * (manual connection | OR | Cloud login). In **edit** mode it wraps
 * the standard `BackendForm`.
 */
export function BackendFormModal({
  mode,
  backend,
  onClose,
  source = "add_backend_modal",
  hideCloseButton = false,
}: BackendFormModalProps) {
  const { t } = useTranslation("openhands");

  if (mode === "add") {
    return (
      <ModalBackdrop
        onClose={hideCloseButton ? undefined : onClose}
        closeOnEscape={false}
        closeOnBackdropClick={!hideCloseButton}
        aria-label={t(I18nKey.BACKEND$ADD_TITLE)}
      >
        <div
          data-testid={
            hideCloseButton ? "onboarding-modal" : "add-backend-modal"
          }
          className={cn(
            "relative rounded-xl border border-[var(--oh-border)] bg-base-secondary",
            modalWidthClassName("xl"),
            MODAL_MAX_WIDTH_VIEWPORT,
          )}
        >
          {hideCloseButton ? null : (
            <ModalCloseButton onClose={onClose} testId="add-backend-close" />
          )}
          {/* Header - hide in locked Cloud first-run mode for cleaner UX */}
          {hideCloseButton ? null : (
            <div className="px-6 pt-6 pb-2 pr-12">
              <h2 className={modalTitleLgClassName}>
                {t(I18nKey.BACKEND$ADD_TITLE)}
              </h2>
            </div>
          )}

          <div className={cn("px-6 pb-6", hideCloseButton ? "pt-6" : "pt-2")}>
            <AddBackendConnectionOptions onClose={onClose} source={source} />
          </div>
        </div>
      </ModalBackdrop>
    );
  }

  // Edit mode — single-column form (unchanged)
  const testIdRoot = "edit-backend";
  return (
    <ModalBackdrop
      onClose={onClose}
      closeOnEscape={false}
      aria-label={t(I18nKey.BACKEND$EDIT_TITLE)}
    >
      <div
        data-testid={`${testIdRoot}-modal`}
        className={cn(
          "relative bg-base-secondary p-6 rounded-xl flex flex-col gap-4 border border-[var(--oh-border)]",
          modalWidthClassName("md"),
        )}
      >
        <ModalCloseButton onClose={onClose} testId={`${testIdRoot}-close`} />
        <h2 className={cn("pr-6", modalTitleLgClassName)}>
          {t(I18nKey.BACKEND$EDIT_TITLE)}
        </h2>
        <BackendForm
          mode="edit"
          backend={backend}
          onSubmitted={onClose}
          testIdRoot={testIdRoot}
        />
      </div>
    </ModalBackdrop>
  );
}
