import React from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { AcpConflictWarnings } from "#/components/features/settings/acp-conflict-warnings";
import { AcpAuthStatusBanner } from "#/components/features/settings/acp-auth-status-banner";
import { AcpSecretField } from "#/components/features/settings/acp-secret-field";
import { I18nKey } from "#/i18n/declaration";
import { useAcpAuthStatus } from "#/hooks/query/use-acp-auth-status";
import { useAcpCredentialForm } from "#/hooks/use-acp-credential-form";
import { useActiveBackend } from "#/contexts/active-backend-context";
import {
  getAcpProviderDisplayName,
  getAcpPreferredDefaultModel,
} from "#/constants/acp-providers";
import { useApplyOnboardingAgentProfile } from "#/hooks/mutation/use-apply-onboarding-agent-profile";
import { type OnboardingAgentId } from "./choose-agent-step";

interface SetupAcpSecretsStepProps {
  /** ACP provider whose credentials we're collecting (e.g. ``"claude-code"``).
   * Typed as {@link OnboardingAgentId} — the same type the onboarding modal
   * tracks — so a mistyped key is a compile error rather than a silently empty
   * form. Providers without a credentials entry (``"openhands"``) simply yield
   * no fields. */
  providerKey: OnboardingAgentId;
  /**
   * Whether this is the currently visible onboarding slide. The modal mounts
   * every slide at once, so we only run the (subprocess-spinning) login probe
   * once the user has actually reached this step — by which point the backend
   * is confirmed connected.
   */
  isActive: boolean;
  onBack: () => void;
  onNext: () => void;
}

/**
 * Onboarding credentials step for ACP providers (Claude Code, Codex, Gemini
 * CLI). The fields are derived from {@link getAcpProviderSecrets}: the API key
 * + optional base URL (from the SDK registry) plus the per-provider
 * credentials a *containerized* agent-server needs (Codex ``auth.json``, the
 * Claude OAuth token, the Gemini Vertex service-account JSON + project/location).
 * Each field maps 1:1 to a **global secret** whose name equals the env var the
 * agent-server exports into the provider subprocess, so saving here is the same
 * as adding the secret under Settings → Secrets.
 *
 * The step is **optional on a backend that can fall back to a host login** (a
 * native agent-server where the user has already run ``claude``/``codex``/
 * ``gcloud`` login) and **required otherwise** — a fresh Docker container or a
 * cloud backend has no host login, so the agent can't authenticate without
 * credentials. Required-ness is capability-driven (see {@link backendRequiresAcpCredentials}):
 * we never block "Next" when the login probe detects an existing session, and
 * we never block a native dev whose host login we just can't classify.
 *
 * Empty fields are never written (a deliberate skip), and a field whose secret
 * already exists shows an "already saved" placeholder and is left untouched
 * unless the user types a replacement.
 */
export function SetupAcpSecretsStep({
  providerKey,
  isActive,
  onBack,
  onNext,
}: SetupAcpSecretsStepProps) {
  const { t } = useTranslation("openhands");
  const activeBackend = useActiveBackend();
  // Login detection via AcpService (provider status commands run through the
  // agent-server bash endpoint) — see issue #964.
  const { status: authStatus, isChecking: isCheckingAuth } = useAcpAuthStatus(
    providerKey,
    { enabled: isActive },
  );

  const {
    fields,
    values,
    setValue,
    secretExists,
    hasValueFor,
    conflicts,
    consumesFileCredentials,
    save,
    isSaving,
  } = useAcpCredentialForm(providerKey);

  const applyAgentProfile = useApplyOnboardingAgentProfile();
  const providerName = getAcpProviderDisplayName(providerKey) ?? providerKey;

  const isAuthenticated = authStatus === "authenticated";
  // Required when the backend can't fall back to a host login (see component
  // docstring). Cloud never has one; a local backend that probes as logged-out
  // is a fresh container — require credentials there, but stay permissive when
  // the probe resolves "unknown" so a native dev is never blocked.
  const required = backendRequiresAcpCredentials(
    activeBackend.backend.kind,
    authStatus,
  );
  // Satisfied once the user has an actual credential for the provider — a
  // masked ``secret`` field (blob, OAuth token, or API key), typed now or
  // previously saved. A base URL or GCP project/location alone can't
  // authenticate anything, so it doesn't count — and neither does a file blob
  // on a backend that can't materialise it (cloud, agent-canvas#1016): the
  // save flow warns it's orphaned, so it can't be what satisfies the gate.
  // An existing login also satisfies it.
  const satisfied =
    isAuthenticated ||
    fields.some(
      (field) =>
        field.secret &&
        (!field.multiline || consumesFileCredentials) &&
        hasValueFor(field.name),
    );
  const blockNext = required && !satisfied;
  // While the probe is still classifying a local backend, hold Next: the gate
  // may be about to come up "unauthenticated", and the checking banner is
  // already showing. A probe that *completes* as "unknown" stays permissive.
  const nextDisabled = isSaving || blockNext || (isCheckingAuth && !satisfied);

  const handleNext = async () => {
    if (await save()) {
      // Land the user's ACP choice on the active AGENT profile so the next
      // conversation launches that provider (no LLM profile/key needed — the
      // subprocess owns its LLM). Best-effort; never blocks advancing. This
      // step never renders for "openhands" (the modal shows SetupLlmStep
      // there), so the guard just narrows the type for `acp_server`.
      if (providerKey !== "openhands") {
        await applyAgentProfile({
          agent_kind: "acp",
          acp_server: providerKey,
          acp_model: getAcpPreferredDefaultModel(providerKey) ?? undefined,
        });
      }
      onNext();
    }
  };

  return (
    <div
      data-testid="onboarding-step-setup-acp-secrets"
      className="flex flex-col gap-6"
    >
      <header className="flex flex-col gap-2">
        <h2 className="text-2xl font-medium text-white">
          {t(I18nKey.ONBOARDING$ACP_SECRETS_TITLE)}
        </h2>
        <p className="text-sm text-[var(--oh-muted)]">
          {t(I18nKey.ONBOARDING$ACP_SECRETS_SUBTITLE, {
            provider: providerName,
          })}
        </p>
        {required && !isAuthenticated ? (
          <p
            data-testid="onboarding-acp-secrets-required-note"
            className="text-sm text-[var(--oh-muted)]"
          >
            {t(I18nKey.ONBOARDING$ACP_SECRETS_REQUIRED_NOTE, {
              provider: providerName,
            })}
          </p>
        ) : (
          authStatus !== "authenticated" && (
            // When already signed in, the success banner below already says to
            // leave the fields blank, so this general reminder would be redundant.
            <p className="text-sm text-[var(--oh-muted)]">
              {t(I18nKey.ONBOARDING$ACP_SECRETS_SUBSCRIPTION_NOTE)}
            </p>
          )
        )}
      </header>

      <AcpAuthStatusBanner
        status={authStatus}
        isChecking={isCheckingAuth}
        providerName={providerName}
        testIdPrefix="onboarding-acp-auth"
      />

      <div className="flex flex-col gap-5">
        {fields.map((field) => (
          <AcpSecretField
            key={field.name}
            field={field}
            value={values[field.name] ?? ""}
            onChange={(value) => setValue(field.name, value)}
            alreadySet={secretExists(field.name)}
            testId={`onboarding-acp-secret-${field.name}`}
            showOptionalTag
          />
        ))}
      </div>

      <AcpConflictWarnings conflicts={conflicts} />

      {blockNext ? (
        <p
          data-testid="onboarding-acp-secrets-blocked"
          className="text-sm text-amber-300"
        >
          {t(I18nKey.ONBOARDING$ACP_SECRETS_REQUIRED_BLOCKED)}
        </p>
      ) : null}

      <div className="sticky bottom-0 flex items-center justify-between gap-2 bg-base-secondary pt-4 pb-7">
        <BrandButton
          testId="onboarding-acp-secrets-back"
          type="button"
          variant="secondary"
          onClick={onBack}
          isDisabled={isSaving}
        >
          {t(I18nKey.ONBOARDING$BACK)}
        </BrandButton>
        <BrandButton
          testId="onboarding-acp-secrets-next"
          type="button"
          variant="primary"
          isDisabled={nextDisabled}
          onClick={handleNext}
        >
          {isSaving ? t(I18nKey.SETTINGS$SAVING) : t(I18nKey.ONBOARDING$NEXT)}
        </BrandButton>
      </div>
    </div>
  );
}

/**
 * Whether the credential step must be satisfied before advancing, given the
 * active backend kind and the ACP login-probe result.
 *
 * - **cloud** → always required: a remote backend has no host CLI login to fall
 *   back on.
 * - **local + ``"unauthenticated"``** → required: the probe ran and found no
 *   login, i.e. a fresh containerized agent-server.
 * - **local + ``"authenticated"`` / ``"unknown"``** → optional: either a login
 *   exists, or the probe couldn't classify it (CLI missing, odd output) — in
 *   which case we stay permissive rather than block a working native dev.
 *
 * Exported for unit testing the matrix without rendering the modal.
 */
export function backendRequiresAcpCredentials(
  backendKind: "local" | "cloud",
  authStatus: "authenticated" | "unauthenticated" | "unknown",
): boolean {
  if (authStatus === "authenticated") return false;
  if (backendKind === "cloud") return true;
  return authStatus === "unauthenticated";
}
