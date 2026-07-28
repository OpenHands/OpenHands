import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { NavigationLink } from "#/components/shared/navigation-link";
import type { ManifestPrerequisitesResult } from "#/hooks/query/use-manifest-prerequisites";
import type { ManifestPrerequisites } from "#/manifests/types";

/** Where the user goes to satisfy each kind of prerequisite. */
const INTEGRATIONS_PATH = "/mcp";
const SECRETS_PATH = "/settings/secrets";

export interface ManifestPrerequisitesStepProps {
  requires: ManifestPrerequisites;
  prerequisites: ManifestPrerequisitesResult;
}

/**
 * Stages 3 and 4 — what has to be in place before the form is worth filling in.
 *
 * Credentials are listed by the name and help text the manifest declares. The
 * host reports only whether each one is present and sends the user to the
 * secrets screen to supply it, so no credential value passes through here.
 */
export function ManifestPrerequisitesStep({
  requires,
  prerequisites,
}: ManifestPrerequisitesStepProps) {
  const { t } = useTranslation("openhands");
  const {
    blockingIntegrations,
    warningIntegrations,
    missingSecrets,
    isBlocked,
  } = prerequisites;

  const message = isBlocked
    ? requires.onUnmet.message
    : requires.onWarn?.message;
  const showIntegrationsLink =
    blockingIntegrations.length > 0 || warningIntegrations.length > 0;

  return (
    <div className="flex flex-col gap-4" data-testid="manifest-prerequisites">
      {message && <p className="text-sm text-[var(--oh-muted)]">{message}</p>}

      {[...blockingIntegrations, ...warningIntegrations].map(
        ({ requirement, entry }) => (
          <div key={requirement.id} className="flex flex-col gap-1">
            <span className="text-sm">{entry?.name ?? requirement.id}</span>
            <span className="text-xs text-[var(--oh-muted)]">
              {requirement.reason}
            </span>
          </div>
        ),
      )}

      {missingSecrets.map((secret) => (
        <div key={secret.key} className="flex flex-col gap-1">
          <span className="text-sm">{secret.label}</span>
          <span className="text-xs text-[var(--oh-muted)]">{secret.help}</span>
        </div>
      ))}

      <div className="flex flex-wrap gap-4 text-sm">
        {showIntegrationsLink && (
          <NavigationLink to={INTEGRATIONS_PATH} className="underline">
            {t(I18nKey.SETUP$MANAGE_INTEGRATIONS)}
          </NavigationLink>
        )}
        {missingSecrets.length > 0 && (
          <NavigationLink to={SECRETS_PATH} className="underline">
            {t(I18nKey.SETUP$MANAGE_SECRETS)}
          </NavigationLink>
        )}
      </div>
    </div>
  );
}
