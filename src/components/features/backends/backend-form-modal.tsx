import React from "react";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { ChevronDown, Globe, Info, Monitor } from "lucide-react";
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
import ExternalLinkIcon from "#/icons/external-link.svg?react";
import ServerIcon from "#/icons/server.svg?react";
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
const LOCAL_BACKEND_COMMAND = "agent-canvas --backend-only --port 8001";
const LOCAL_AGENT_SERVER_DOCS_URL =
  "https://github.com/OpenHands/OpenHands/blob/main/docs/DEVELOPMENT.md#alternative-development-workflows";
const REMOTE_AGENT_SERVER_DOCS_URL =
  "https://github.com/OpenHands/OpenHands/blob/main/docs/SELF_HOSTING.md";
const DEPLOYMENT_OPTIONS_URL =
  "https://docs.openhands.dev/overview/introduction";
export type BackendConnectionMethod = "manual" | "cloud_login";

export type BackendAddedSource = CloudConnectionSource;
type AddBackendOption = "cloud" | "agent-server";
type AgentServerLocation = "local" | "remote";

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
  /** Fix the persisted backend kind instead of inferring it from the host. */
  fixedKind?: BackendKind;
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
  fixedKind,
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
  const kind = fixedKind ?? kindOverride ?? inferKindFromHost(host);
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
      fixedKind,
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
  fixedKind?: BackendKind;
  showKindSelector?: boolean;
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
  fixedKind,
  showKindSelector = true,
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
    fixedKind,
  });

  return (
    <form
      data-testid={`${testIdRoot}-form`}
      onSubmit={handleSubmit}
      className="flex flex-col gap-4 flex-1 min-w-0"
    >
      <SettingsInput
        testId={`${testIdRoot}-name`}
        name={`${testIdRoot}-name`}
        type="text"
        label={t(I18nKey.BACKEND$NAME_LABEL)}
        hint={t(I18nKey.BACKEND$NAME_HELPER)}
        value={name}
        onChange={(value) => {
          setName(value);
          setConnectionError(null);
        }}
        // eslint-disable-next-line i18next/no-literal-string -- example placeholder, not user-facing copy
        placeholder="e.g. My Server"
        className="w-full"
      />

      <SettingsInput
        testId={`${testIdRoot}-host`}
        name={`${testIdRoot}-host`}
        type="text"
        label={t(I18nKey.BACKEND$HOST_LABEL)}
        hint={t(I18nKey.BACKEND$HOST_HELPER)}
        value={host}
        onChange={(value) => {
          setHost(value);
          setConnectionError(null);
        }}
        // eslint-disable-next-line i18next/no-literal-string -- example value, not translatable
        placeholder="http://localhost:8000"
        className="w-full"
      />

      {showKindSelector ? (
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
      ) : null}

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
  /** Omit repeated branding when a surrounding chooser already names Cloud. */
  showBranding?: boolean;
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
  showBranding = true,
}: CloudLoginColumnProps) {
  const { t } = useTranslation("openhands");

  const [advancedOpen, setAdvancedOpen] = React.useState(false);
  const [customHost, setCustomHost] = React.useState("");
  const advancedPanelId = `${testIdRoot}-advanced-panel`;

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
    <div className="flex w-full min-w-0 flex-col items-center gap-3">
      {showBranding ? (
        <div className="flex flex-col items-center gap-1">
          <OpenHandsLogoWhite width={56} height={56} aria-hidden />

          <h4
            className={modalTitleLgMediumClassName}
            data-testid={`${testIdRoot}-cloud-title`}
          >
            {t(I18nKey.BACKEND$CLOUD_TITLE)}
          </h4>
        </div>
      ) : null}

      <DeviceFlowAuth
        host={effectiveHost}
        onSuccess={handleLoginSuccess}
        testIdRoot={testIdRoot}
        analyticsSource={analyticsSource}
        className="w-full items-center"
        idleDescription={
          <p
            className="text-center text-sm leading-relaxed text-[var(--oh-muted)]"
            data-testid={`${testIdRoot}-cloud-description`}
          >
            {t(I18nKey.BACKEND$CLOUD_DESCRIPTION)}
          </p>
        }
        // Host overriding only matters before the flow starts, so it rides the
        // same idle-only slot as the description instead of sitting under the
        // authorization status.
        idleFooter={
          lockedHost ? null : (
            // Keep the Advanced disclosure at the original compact width even
            // though the surrounding cloud panel spans the full chooser.
            <div className="mx-auto w-full max-w-md">
              <button
                type="button"
                onClick={() => setAdvancedOpen((open) => !open)}
                aria-expanded={advancedOpen}
                aria-controls={advancedPanelId}
                data-testid={`${testIdRoot}-advanced-toggle`}
                className="flex w-full cursor-pointer items-center justify-center gap-1 text-center text-xs text-[var(--oh-muted)] transition-colors hover:text-content-2"
              >
                <span>{t(I18nKey.BACKEND$ADVANCED)}</span>
                <ChevronDownSmallIcon
                  className={cn(
                    "h-4 w-4 shrink-0 text-muted transition-transform duration-200 ease-out",
                    advancedOpen && "rotate-180",
                  )}
                  aria-hidden
                />
              </button>
              {/* Height animates through `grid-template-rows` (0fr ↔ 1fr) so no
                  max-height guess is needed; the inner clip is what makes `0fr`
                  collapse. Content stays mounted so a typed host survives, and
                  `inert` keeps the collapsed field out of the tab order. */}
              <div
                className={cn(
                  "grid transition-[grid-template-rows] duration-200 ease-out motion-reduce:transition-none",
                  advancedOpen ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
                )}
              >
                <div className="overflow-hidden">
                  <div
                    id={advancedPanelId}
                    data-testid={`${testIdRoot}-advanced-panel`}
                    aria-hidden={!advancedOpen}
                    inert={!advancedOpen ? true : undefined}
                    className={cn(
                      "pt-3 transition-opacity duration-200 ease-out motion-reduce:transition-none",
                      advancedOpen ? "opacity-100" : "opacity-0",
                    )}
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
              </div>
            </div>
          )
        }
      />
    </div>
  );
}

interface BackendOptionTabProps {
  value: AddBackendOption;
  selectedValue: AddBackendOption;
  title: string;
  description: string;
  icon: React.ReactNode;
  onSelect: (value: AddBackendOption) => void;
  panelId: string;
  testId: string;
}

/**
 * Presents a connection-method tab: icon, title, and a one-line subtitle.
 *
 * The shared tablist owns the outer border, so these buttons meet cleanly at
 * the center. Selection uses a bottom bar instead of recoloring the full
 * outline, preserving the group as one visual control.
 */
function BackendOptionTab({
  value,
  selectedValue,
  title,
  description,
  icon,
  onSelect,
  panelId,
  testId,
}: BackendOptionTabProps) {
  const isSelected = value === selectedValue;
  const tabId = `${testId}-tab`;

  return (
    <button
      id={tabId}
      type="button"
      role="tab"
      aria-selected={isSelected}
      aria-controls={panelId}
      data-testid={testId}
      onClick={() => onSelect(value)}
      className={cn(
        "relative flex min-h-16 w-full cursor-pointer items-center gap-3 px-3 py-3 text-left transition-colors",
        "first:border-r first:border-r-[var(--oh-border)]",
        "focus-visible:z-10 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-blue-300",
        isSelected
          ? "bg-[var(--oh-surface-raised)] text-white after:absolute after:inset-x-0 after:bottom-0 after:h-0.5 after:bg-primary"
          : "text-[var(--oh-muted)] hover:bg-[var(--oh-surface-raised)] hover:text-white",
      )}
    >
      <span
        className="flex size-8 shrink-0 items-center justify-center"
        aria-hidden
      >
        {icon}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm font-medium">{title}</span>
        <span className="mt-0.5 block text-xs leading-tight text-[var(--oh-muted)]">
          {description}
        </span>
      </span>
    </button>
  );
}

interface AnimatedPanelHeightProps {
  children: React.ReactNode;
}

/**
 * Animates a panel between content-driven heights.
 *
 * A ResizeObserver keeps the wrapper synchronized as tabs or nested
 * disclosures change. The transition class remains mounted before the
 * observer publishes a new height; adding it in the same render as the height
 * would give the browser no previous painted value to interpolate from.
 */
function AnimatedPanelHeight({ children }: AnimatedPanelHeightProps) {
  const contentRef = React.useRef<HTMLDivElement>(null);
  const [height, setHeight] = React.useState<number>();

  React.useLayoutEffect(() => {
    const content = contentRef.current;
    if (!content) return undefined;

    const measure = () => {
      const nextHeight = content.getBoundingClientRect().height;
      // jsdom reports zero-sized layout boxes; leaving height automatic there
      // keeps component tests representative without changing browser behavior.
      setHeight(nextHeight > 0 ? nextHeight : undefined);
    };

    measure();
    if (typeof ResizeObserver === "undefined") return undefined;

    const observer = new ResizeObserver(measure);
    observer.observe(content);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      data-testid="add-backend-panel-height"
      style={height === undefined ? undefined : { height }}
      className="overflow-hidden transition-[height] duration-300 ease-in-out motion-reduce:transition-none"
    >
      <div ref={contentRef}>{children}</div>
    </div>
  );
}

/**
 * Collects every setup instruction for the selected location into one
 * collapsible note.
 *
 * All of the guidance lives here — the intro, the example command (local) or
 * the host formats to use with a tunnel (remote), and the docs link — so users
 * never have to leave the modal to find out what belongs in the host field.
 * It starts collapsed so the connection form remains compact; users can
 * reveal the complete instructions without leaving the modal.
 *
 * Open/close uses a `grid-template-rows` transition (`0fr` ↔ `1fr`) so the
 * height animates without a fixed `max-height` guess. Content stays mounted
 * so the toggle does not remount the docs link on every expand.
 */
function AgentServerGuidance({ location }: { location: AgentServerLocation }) {
  const { t } = useTranslation("openhands");
  const [isExpanded, setIsExpanded] = React.useState(false);
  const isRemote = location === "remote";
  const title = isRemote
    ? t(I18nKey.BACKEND$REMOTE_SETUP_TITLE)
    : t(I18nKey.BACKEND$BEFORE_CONNECT_TITLE);
  const description = isRemote
    ? t(I18nKey.BACKEND$REMOTE_SETUP_DESCRIPTION)
    : t(I18nKey.BACKEND$LOCAL_SETUP_DESCRIPTION);
  const docsHref = isRemote
    ? REMOTE_AGENT_SERVER_DOCS_URL
    : LOCAL_AGENT_SERVER_DOCS_URL;
  const docsLabel = isRemote
    ? t(I18nKey.BACKEND$REMOTE_SETUP_DOCS)
    : t(I18nKey.BACKEND$LOCAL_SETUP_DOCS);
  const testIdRoot = isRemote ? "add-backend-remote" : "add-backend-local";
  const toggleId = `${testIdRoot}-guidance-toggle`;
  const bodyId = `${testIdRoot}-guidance-body`;

  return (
    <aside
      data-testid={`${testIdRoot}-guidance`}
      className="rounded-lg bg-[var(--oh-surface-raised)] text-sm text-[var(--oh-muted)]"
    >
      {/* Heading wraps the button so the accordion keeps a real heading in the
          document outline while the whole row stays clickable. */}
      <h4 className="text-white">
        <button
          id={toggleId}
          type="button"
          onClick={() => setIsExpanded((expanded) => !expanded)}
          aria-expanded={isExpanded}
          aria-controls={bodyId}
          data-testid={toggleId}
          className={cn(
            // Padding lives on the button rather than the card so the hover
            // fill spans the full row.
            "flex w-full cursor-pointer items-center justify-between gap-2 rounded-lg px-3 py-2 text-left font-medium",
            "transition-colors hover:bg-[var(--oh-interactive-hover)]",
            "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-300",
          )}
        >
          <span className="flex min-w-0 items-center gap-2">
            <Info
              className="size-4 shrink-0 text-[var(--oh-muted)]"
              aria-hidden
            />
            <span className="truncate">{title}</span>
          </span>
          <ChevronDown
            className={cn(
              "size-5 shrink-0 text-[var(--oh-muted)] transition-transform duration-200 ease-out",
              isExpanded && "rotate-180",
            )}
            aria-hidden
          />
        </button>
      </h4>

      {/* Outer grid owns the height animation; the inner overflow clip is
          required so `1fr` measures the content while `0fr` fully collapses. */}
      <div
        className={cn(
          "grid transition-[grid-template-rows] duration-200 ease-out motion-reduce:transition-none",
          isExpanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
        )}
      >
        <div className="overflow-hidden">
          <div
            id={bodyId}
            data-testid={bodyId}
            role="region"
            aria-labelledby={toggleId}
            aria-hidden={!isExpanded}
            inert={!isExpanded ? true : undefined}
            className={cn(
              "flex flex-col gap-2 px-3 pb-3 transition-opacity duration-200 ease-out motion-reduce:transition-none",
              isExpanded ? "opacity-100" : "opacity-0",
            )}
          >
            <p className="leading-5">{description}</p>

            {isRemote ? (
              <div>
                <h5 className="font-medium text-white">
                  {t(I18nKey.BACKEND$REMOTE_CONNECTION_TITLE)}
                </h5>
                <p className="mt-1 leading-5">
                  {t(I18nKey.BACKEND$REMOTE_CONNECTION_DESCRIPTION)}
                </p>
              </div>
            ) : (
              <code className="block break-words font-mono text-xs text-white">
                {LOCAL_BACKEND_COMMAND}
              </code>
            )}

            <a
              href={docsHref}
              target="_blank"
              rel="noopener noreferrer"
              data-testid={`${testIdRoot}-docs-link`}
              tabIndex={isExpanded ? undefined : -1}
              className="inline-flex w-fit items-center gap-1.5 text-primary hover:underline"
            >
              <span>{docsLabel}</span>
              <ExternalLinkIcon className="size-4 shrink-0" aria-hidden />
            </a>
          </div>
        </div>
      </div>
    </aside>
  );
}

function AddBackendChooser({
  onConnected,
  source,
}: {
  onConnected: (
    payload: BackendFormSubmitPayload,
    connectionMethod: BackendConnectionMethod,
    metadata?: BackendConnectionTestMetadata,
  ) => void;
  source: BackendAddedSource;
}) {
  const { t } = useTranslation("openhands");
  const [selectedOption, setSelectedOption] =
    React.useState<AddBackendOption>("cloud");
  const [agentServerLocation, setAgentServerLocation] =
    React.useState<AgentServerLocation>("local");
  const panelId = "add-backend-selected-panel";
  const selectedTabId = `add-backend-option-${selectedOption}-tab`;
  const isCloudSelected = selectedOption === "cloud";

  return (
    <div data-testid="add-backend-chooser" className="flex flex-col">
      <div
        role="tablist"
        aria-label={t(I18nKey.BACKEND$CHOOSER_TITLE)}
        className="grid grid-cols-2 overflow-hidden rounded-lg border border-[var(--oh-border)]"
      >
        <BackendOptionTab
          value="cloud"
          selectedValue={selectedOption}
          title={t(I18nKey.BACKEND$CLOUD_TITLE)}
          description={t(I18nKey.BACKEND$CLOUD_OPTION_DESCRIPTION)}
          icon={
            <OpenHandsLogoWhite
              width={32}
              height={32}
              data-testid="add-backend-option-cloud-logo"
            />
          }
          onSelect={setSelectedOption}
          panelId={panelId}
          testId="add-backend-option-cloud"
        />
        <BackendOptionTab
          value="agent-server"
          selectedValue={selectedOption}
          title={t(I18nKey.BACKEND$AGENT_SERVER_TITLE)}
          description={t(I18nKey.BACKEND$AGENT_SERVER_OPTION_DESCRIPTION)}
          icon={<ServerIcon className="size-6" />}
          onSelect={setSelectedOption}
          panelId={panelId}
          testId="add-backend-option-agent-server"
        />
      </div>

      <div className="mt-6">
        <AnimatedPanelHeight>
          <section
            id={panelId}
            role="tabpanel"
            aria-labelledby={selectedTabId}
            className="min-w-0"
          >
            {isCloudSelected ? (
              /* Padding keeps the CTA and Advanced host field from hugging the
                 border; when Advanced expands the panel grows with the content
                 instead of squeezing it into the reserved min-height. */
              <div
                data-testid="add-backend-cloud-panel"
                className={cn(
                  "relative isolate flex min-h-[13.5rem] w-full items-center justify-center rounded-xl border border-[var(--oh-border)] px-5 py-6",
                  // A soft greyscale halo sits behind the CTA; -z-10 keeps it under the
                  // content rather than washing over it.
                  "before:pointer-events-none before:absolute before:inset-0 before:-z-10 before:rounded-xl",
                  "before:bg-[radial-gradient(75%_75%_at_50%_50%,rgba(255,255,255,0.1)_0%,rgba(255,255,255,0)_70%)]",
                )}
              >
                <CloudLoginColumn
                  onConnected={onConnected}
                  testIdRoot="add-backend"
                  analyticsSource={source}
                  showBranding={false}
                />
              </div>
            ) : (
              <div
                data-testid="add-backend-agent-server-panel"
                className="mx-auto w-full max-w-xl"
              >
                {/* Rules on either side center the toggle and read as a
                    divider between the chooser and the connection form. */}
                <div className="flex items-center gap-3">
                  <span
                    className="h-px flex-1 bg-[var(--oh-border)]"
                    aria-hidden
                  />
                  <SegmentedToggle<AgentServerLocation>
                    value={agentServerLocation}
                    options={[
                      {
                        value: "local",
                        label: t(I18nKey.BACKEND$KIND_LOCAL),
                        icon: <Monitor aria-hidden />,
                      },
                      {
                        value: "remote",
                        label: t(I18nKey.BACKEND$KIND_REMOTE),
                        icon: <Globe aria-hidden />,
                      },
                    ]}
                    onChange={setAgentServerLocation}
                    ariaLabel={t(I18nKey.BACKEND$AGENT_SERVER_LOCATION)}
                    testId="add-backend-location"
                  />
                  <span
                    className="h-px flex-1 bg-[var(--oh-border)]"
                    aria-hidden
                  />
                </div>

                <div className="mt-4 flex flex-col gap-4">
                  <AgentServerGuidance
                    key={agentServerLocation}
                    location={agentServerLocation}
                  />
                  <ManualConnectionColumn
                    onConnected={onConnected}
                    testIdRoot="add-backend"
                    requireApiKey={agentServerLocation === "remote"}
                    submitLabel={t(I18nKey.BACKEND$CONNECT)}
                    submittingLabel={t(
                      I18nKey.ONBOARDING$BACKEND_STATUS_CHECKING,
                    )}
                    fixedKind="local"
                    showKindSelector={false}
                  />
                </div>
              </div>
            )}
          </section>
        </AnimatedPanelHeight>
      </div>
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
  const lockedCloudHost = getLockedCloudHost();

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

  if (lockedCloudHost) {
    return (
      <CloudLoginColumn
        onConnected={handleConnected}
        testIdRoot="add-backend"
        lockedHost={lockedCloudHost}
        analyticsSource={source}
      />
    );
  }

  return <AddBackendChooser onConnected={handleConnected} source={source} />;
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
            "relative max-h-[92vh] w-[720px] overflow-y-auto rounded-xl border border-[var(--oh-border)] bg-base-secondary p-6",
            MODAL_MAX_WIDTH_VIEWPORT,
          )}
        >
          {hideCloseButton ? null : (
            <ModalCloseButton onClose={onClose} testId="add-backend-close" />
          )}
          {hideCloseButton ? null : (
            <div className="pr-8">
              <h2 className={modalTitleLgClassName}>
                {t(I18nKey.BACKEND$CHOOSER_TITLE)}
              </h2>
              <p
                className="mt-2 text-sm leading-6 text-[var(--oh-muted)]"
                data-testid="add-backend-description"
              >
                {t(I18nKey.BACKEND$CHOOSER_DESCRIPTION)}{" "}
                <a
                  href={DEPLOYMENT_OPTIONS_URL}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={t(I18nKey.BACKEND$DEPLOYMENT_OPTIONS)}
                  data-testid="add-backend-deployment-options-link"
                  className="text-primary hover:underline"
                >
                  {t(I18nKey.CTA$LEARN_MORE)}
                  <ExternalLinkIcon
                    className="ml-1 inline size-3.5 align-[-0.125em]"
                    aria-hidden
                  />
                </a>
              </p>
            </div>
          )}

          <div className={hideCloseButton ? undefined : "mt-5"}>
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
