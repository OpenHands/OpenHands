import React from "react";
import { useTranslation } from "react-i18next";
import { CheckCircle2, Loader2, AlertCircle } from "lucide-react";
import { isNoBackend } from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import {
  getAgentServerFormDefaults,
  getLockedCloudHost,
  isAuthRequired,
  isSameCloudHost,
} from "#/api/agent-server-config";
import { DEFAULT_LOCAL_BACKEND_NAME } from "#/api/backend-registry/default-backend";
import {
  BackendConnectionOptions,
  type BackendConnectionMethod,
  type BackendFormSubmitPayload,
} from "#/components/features/backends/backend-form-modal";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { useBackendsHealth } from "#/hooks/query/use-backends-health";
import { useTracking } from "#/hooks/use-tracking";
import { I18nKey } from "#/i18n/declaration";
import ChevronDownSmallIcon from "#/icons/chevron-down-small.svg?react";
import { cn } from "#/utils/utils";
import { getBackendStatusLabel } from "#/components/features/backends/backend-status-label";

interface CheckBackendStepProps {
  onBack?: () => void;
  onNext: () => void;
  /**
   * Dismisses the entire onboarding modal. Called when Cloud login succeeds
   * in locked-to-Cloud mode: there the Cloud login IS the onboarding
   * completion, so the modal must disappear immediately rather than
   * advancing to the next slide (which previously flickered — the next
   * slide flashed before the root gate tore the modal down). Standard mode
   * still walks the user through agent/LLM setup via `onNext`.
   */
  onClose?: () => void;
}

function ConnectionBanner({
  backend,
  isConnected,
  lastError,
}: {
  backend: Backend;
  isConnected: boolean | null;
  lastError: string | null;
}) {
  const { t } = useTranslation("openhands");

  if (isConnected === true) {
    return (
      <div
        role="status"
        data-testid="onboarding-backend-connected"
        className={cn(
          "flex items-start gap-3 rounded-xl border border-green-500/40 bg-green-500/10 px-4 py-3",
        )}
      >
        <CheckCircle2 className="mt-0.5 size-5 shrink-0 text-green-400" />
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium text-green-200">
            {t(I18nKey.ONBOARDING$BACKEND_CONNECTED_TITLE)}
          </span>
          <span className="text-xs text-green-200/80">
            {t(I18nKey.ONBOARDING$BACKEND_CONNECTED_BODY)}
          </span>
        </div>
      </div>
    );
  }

  if (isConnected === false) {
    const statusLabel = getBackendStatusLabel(t, backend, {
      isConnected,
      lastError,
    });
    return (
      <div
        role="alert"
        data-testid="onboarding-backend-disconnected"
        className="flex items-start gap-3 rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3"
      >
        <AlertCircle className="mt-0.5 size-5 shrink-0 text-red-400" />
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium text-red-200">
            {statusLabel}
          </span>
          <span className="text-xs text-red-200/80">
            {t(I18nKey.ONBOARDING$BACKEND_DISCONNECTED_BODY)}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      role="status"
      data-testid="onboarding-backend-checking"
      className="flex items-start gap-3 rounded-xl border border-white/10 bg-white/5 px-4 py-3"
    >
      <Loader2 className="mt-0.5 size-5 shrink-0 animate-spin text-[var(--oh-text-tertiary)]" />
      <span className="text-sm text-[var(--oh-text-tertiary)]">
        {t(I18nKey.ONBOARDING$BACKEND_CHECKING)}
      </span>
    </div>
  );
}

/**
 * First onboarding step: add the initial backend when none is selected,
 * or edit/check the active backend with a contextual health banner.
 */
export function CheckBackendStep({
  onBack,
  onNext,
  onClose,
}: CheckBackendStepProps) {
  const { t } = useTranslation("openhands");
  const { active, addBackend, setActive, updateBackend } =
    useActiveBackendContext();
  const { trackBackendAdded } = useTracking();
  const { backend } = active;
  const noBackendSelected = isNoBackend(backend);
  const lockedCloudHost = getLockedCloudHost();

  // In locked-Cloud mode, a reachable backend that is NOT the locked
  // Cloud host (e.g. a stale Local backend from localStorage) must be
  // forced through Cloud login replacement. Treat it as if no backend
  // were selected for render purposes so the Cloud login UI shows and
  // the connected-backend "Next" shortcut is suppressed. The real
  // `noBackendSelected` still controls whether handleConnected calls
  // `addBackend` or `updateBackend`, so the stale backend gets replaced.
  const lockedCloudHostMismatch =
    lockedCloudHost !== null &&
    !noBackendSelected &&
    !(
      backend.kind === "cloud" && isSameCloudHost(backend.host, lockedCloudHost)
    );
  const treatAsNoBackend = noBackendSelected || lockedCloudHostMismatch;

  const defaults = React.useMemo(() => getAgentServerFormDefaults(), []);
  const backendForForm = treatAsNoBackend
    ? {
        id: "onboarding-local-backend-draft",
        name: DEFAULT_LOCAL_BACKEND_NAME,
        host: defaults.baseUrl,
        apiKey: defaults.sessionApiKey,
        kind: "local" as const,
      }
    : backend;
  const healthByBackendId = useBackendsHealth(
    noBackendSelected ? [] : [backend],
  );
  const isConnected = treatAsNoBackend
    ? null
    : (healthByBackendId[backend.id]?.isConnected ?? null);
  const lastError = treatAsNoBackend
    ? null
    : (healthByBackendId[backend.id]?.lastError ?? null);
  const [configurationOpen, setConfigurationOpen] = React.useState(false);

  React.useEffect(() => {
    if (isConnected === true) {
      setConfigurationOpen(false);
    }
  }, [isConnected]);

  const hideConfigurationFields = isConnected === true && !configurationOpen;

  const handleConnected = React.useCallback(
    (
      payload: BackendFormSubmitPayload,
      connectionMethod: BackendConnectionMethod,
    ) => {
      if (noBackendSelected) {
        addBackend(payload);
      } else {
        // When the host changes (e.g. replacing a stale Cloud backend
        // with the locked Cloud host), the persisted active org_id is
        // keyed to the OLD host's org list and would be sent as an
        // invalid X-Org-Id to the new host. Clear it so the new
        // backend starts org-less; the user picks an org on the new
        // host. (Local-only edits are no-ops here since active.orgId
        // is already null for Local backends.)
        const hostChanged = payload.host !== backend.host;
        updateBackend(backend.id, payload);
        if (hostChanged && active.orgId !== null) {
          setActive(backend.id, null);
        }
      }
      trackBackendAdded({
        backendKind: payload.kind,
        connectionMethod,
        hasApiKey: Boolean(payload.apiKey),
        source: "onboarding",
      });
      // In locked-to-Cloud mode, Cloud login IS the onboarding
      // completion: dismiss the modal immediately so the user never
      // sees the next slide (Choose Agent) flash before the root gate
      // tears the modal down — the flicker reported on PR #1389.
      // Standard mode still walks the user through agent/LLM setup.
      if (lockedCloudHost !== null && payload.kind === "cloud" && onClose) {
        onClose();
      } else {
        onNext();
      }
    },
    [
      active.orgId,
      addBackend,
      backend.host,
      backend.id,
      lockedCloudHost,
      noBackendSelected,
      onClose,
      onNext,
      setActive,
      trackBackendAdded,
      updateBackend,
    ],
  );

  const actionRowClassName = cn(
    "sticky bottom-0 mt-2 flex items-center gap-2 bg-base-secondary pt-4 pb-7",
    onBack ? "justify-between" : "justify-end",
  );
  const titleKey = treatAsNoBackend
    ? lockedCloudHost
      ? I18nKey.ONBOARDING$LOGIN_TO_CLOUD_TITLE
      : I18nKey.BACKEND$ADD_TITLE
    : I18nKey.ONBOARDING$BACKEND_TITLE;

  return (
    <div
      data-testid="onboarding-step-check-backend"
      className="flex flex-col gap-6"
    >
      <header className="flex flex-col gap-2">
        <h2 className="text-2xl font-medium text-white">{t(titleKey)}</h2>
        {treatAsNoBackend ? null : (
          <p
            data-testid="onboarding-backend-subtitle"
            className="text-sm text-[var(--oh-muted)]"
          >
            {t(I18nKey.ONBOARDING$BACKEND_SUBTITLE)}
          </p>
        )}
      </header>

      {treatAsNoBackend ? null : (
        <ConnectionBanner
          backend={backendForForm}
          isConnected={isConnected}
          lastError={lastError}
        />
      )}

      {isConnected === true ? (
        <button
          type="button"
          onClick={() => setConfigurationOpen((open) => !open)}
          aria-expanded={configurationOpen}
          data-testid="onboarding-backend-show-configuration"
          className="flex w-full cursor-pointer items-center justify-center gap-1 text-center text-xs text-[var(--oh-muted)] transition-colors hover:text-content-2"
        >
          <span>
            {configurationOpen
              ? t(I18nKey.ONBOARDING$BACKEND_HIDE_CONFIGURATION)
              : t(I18nKey.ONBOARDING$BACKEND_SHOW_CONFIGURATION)}
          </span>
          <ChevronDownSmallIcon
            className={cn(
              "h-4 w-4 shrink-0 text-muted transition-transform",
              configurationOpen && "rotate-180",
            )}
            aria-hidden
          />
        </button>
      ) : null}

      <div data-testid="onboarding-backend-configuration-fields">
        {!hideConfigurationFields ? (
          <BackendConnectionOptions
            onConnected={handleConnected}
            testIdRoot="onboarding-backend"
            initialManualBackend={backendForForm}
            requireManualApiKey={isAuthRequired()}
            manualSubmitLabel={t(I18nKey.ONBOARDING$NEXT)}
            manualSubmittingLabel={t(I18nKey.SETTINGS$SAVING)}
            manualSubmitTestId="onboarding-backend-next"
            analyticsSource="onboarding"
          />
        ) : null}
      </div>

      {hideConfigurationFields ? (
        <div className={actionRowClassName}>
          {onBack ? (
            <BrandButton
              testId="onboarding-backend-back"
              type="button"
              variant="secondary"
              onClick={onBack}
            >
              {t(I18nKey.ONBOARDING$BACK)}
            </BrandButton>
          ) : null}
          <BrandButton
            testId="onboarding-backend-next"
            type="button"
            variant="primary"
            onClick={onNext}
          >
            {t(I18nKey.ONBOARDING$NEXT)}
          </BrandButton>
        </div>
      ) : null}
    </div>
  );
}
