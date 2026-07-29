import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { NavigationLink } from "#/components/shared/navigation-link";
import type { SetupPrerequisitesResult } from "#/hooks/query/use-manifest-prerequisites";

/** Where the user goes to connect an integration. */
const INTEGRATIONS_PATH = "/mcp";

export interface SetupPrerequisitesStepProps {
  prerequisites: SetupPrerequisitesResult;
}

/**
 * Stage 3 — what has to be connected before the form is worth filling in.
 *
 * Each integration is listed with the reason the manifest gives for needing it,
 * so the user is told why rather than just what.
 */
export function SetupPrerequisitesStep({
  prerequisites,
}: SetupPrerequisitesStepProps) {
  const { t } = useTranslation("openhands");
  const { blockingIntegrations, warningIntegrations } = prerequisites;

  return (
    <div className="flex flex-col gap-4" data-testid="setup-prerequisites">
      {[...blockingIntegrations, ...warningIntegrations].map(
        ({ id, requirement, entry }) => (
          <div key={id} className="flex flex-col gap-1">
            <span className="text-sm">{entry?.name ?? id}</span>
            <span className="text-xs text-[var(--oh-muted)]">
              {requirement.message}
            </span>
          </div>
        ),
      )}

      <NavigationLink to={INTEGRATIONS_PATH} className="text-sm underline">
        {t(I18nKey.SETUP$MANAGE_INTEGRATIONS)}
      </NavigationLink>
    </div>
  );
}
