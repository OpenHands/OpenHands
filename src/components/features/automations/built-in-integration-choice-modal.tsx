import { useTranslation } from "react-i18next";
import {
  ArrowRight,
  CircleCheck,
  Clock,
  Link2,
  Settings,
  SlidersHorizontal,
} from "lucide-react";
import type { IntegrationCatalogEntry } from "@openhands/extensions/integrations";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import { BrandBadge } from "#/components/shared/badge";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";

interface BuiltInIntegrationChoiceModalProps {
  /** The built-in integration that already covers the selected automation. */
  integration: IntegrationCatalogEntry;
  onClose: () => void;
  /** Fired for the "Use the integration" action. */
  onUseBuiltIn: () => void;
  /** Fired for the "Continue with polling setup" action. */
  onContinuePolling: () => void;
}

/**
 * Shown before an 'Issue to PR' automation is set up on a cloud instance whose
 * built-in integration (Settings > Integrations) provides the same workflow:
 * the user picks between that integration and the polling automation.
 */
export function BuiltInIntegrationChoiceModal({
  integration,
  onClose,
  onUseBuiltIn,
  onContinuePolling,
}: BuiltInIntegrationChoiceModalProps) {
  const { t } = useTranslation("openhands");
  const { name } = integration;

  const builtInFeatures = [
    t(I18nKey.AUTOMATION_SETUP_CHOICE$BUILT_IN_FEATURE_READY),
    t(I18nKey.AUTOMATION_SETUP_CHOICE$BUILT_IN_FEATURE_EVENT_DRIVEN),
    t(I18nKey.AUTOMATION_SETUP_CHOICE$BUILT_IN_FEATURE_WORKFLOW),
  ];
  const pollingFeatures = [
    {
      Icon: Link2,
      label: t(I18nKey.AUTOMATION_SETUP_CHOICE$POLLING_FEATURE_MCP, { name }),
    },
    {
      Icon: Clock,
      label: t(I18nKey.AUTOMATION_SETUP_CHOICE$POLLING_FEATURE_SCHEDULE),
    },
    {
      Icon: SlidersHorizontal,
      label: t(I18nKey.AUTOMATION_SETUP_CHOICE$POLLING_FEATURE_CUSTOMIZE),
    },
  ];

  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.AUTOMATION_SETUP_CHOICE$TITLE)}
    >
      <div
        data-testid="built-in-integration-choice-modal"
        className="relative flex w-full max-w-3xl flex-col rounded-xl border border-border bg-base-secondary"
      >
        <ModalCloseButton
          onClose={onClose}
          testId="built-in-integration-choice-modal-close"
        />
        <header className="flex-shrink-0 px-6 pb-4 pt-6">
          <h2 className={cn("pr-6", modalTitleLgClassName)}>
            {t(I18nKey.AUTOMATION_SETUP_CHOICE$TITLE)}
          </h2>
          <p className="mt-2 text-sm text-muted">
            {t(I18nKey.AUTOMATION_SETUP_CHOICE$DESCRIPTION, { name })}
          </p>
        </header>
        <div className="flex flex-col gap-3 px-6 pb-6 sm:flex-row">
          <div
            data-testid="built-in-integration-choice-option-built-in"
            className="flex flex-1 flex-col gap-3 rounded-xl border border-primary bg-surface-raised p-4"
          >
            <div className="flex items-start justify-between gap-2">
              <McpLogoBadge entry={integration} />
              <BrandBadge className="shrink-0 whitespace-nowrap px-2.5 py-1 text-xs">
                {t(I18nKey.SETTINGS$SKILLS_RECOMMENDED)}
              </BrandBadge>
            </div>
            <div className="flex flex-col gap-1">
              <h3 className="text-sm font-semibold text-contrast">
                {t(I18nKey.AUTOMATION_SETUP_CHOICE$BUILT_IN_TITLE, { name })}
              </h3>
              <p className="text-xs leading-relaxed text-tertiary-light">
                {t(I18nKey.AUTOMATION_SETUP_CHOICE$BUILT_IN_DESCRIPTION, {
                  name,
                })}
              </p>
            </div>
            <ul className="flex flex-col gap-2">
              {builtInFeatures.map((feature) => (
                <li
                  key={feature}
                  className="flex items-center gap-2 text-sm text-contrast"
                >
                  <CircleCheck
                    className="size-4 shrink-0 text-primary"
                    aria-hidden
                  />
                  {feature}
                </li>
              ))}
            </ul>
            <BrandButton
              type="button"
              variant="primary"
              className="mt-auto flex items-center justify-center gap-2"
              testId="built-in-integration-choice-use-built-in"
              onClick={onUseBuiltIn}
            >
              {t(I18nKey.AUTOMATION_SETUP_CHOICE$BUILT_IN_ACTION, { name })}
              <ArrowRight className="size-4 shrink-0" aria-hidden />
            </BrandButton>
          </div>

          <div
            data-testid="built-in-integration-choice-option-polling"
            className="flex flex-1 flex-col gap-3 rounded-xl border border-border bg-surface-raised p-4"
          >
            <McpLogoBadge
              entry={null}
              fallback={<Settings className="h-5 w-5" strokeWidth={2.25} />}
            />
            <div className="flex flex-col gap-1">
              <h3 className="text-sm font-semibold text-contrast">
                {t(I18nKey.AUTOMATION_SETUP_CHOICE$POLLING_TITLE)}
              </h3>
              <p className="text-xs leading-relaxed text-tertiary-light">
                {t(I18nKey.AUTOMATION_SETUP_CHOICE$POLLING_DESCRIPTION, {
                  name,
                })}
              </p>
            </div>
            <ul className="flex flex-col gap-2">
              {pollingFeatures.map(({ Icon, label }) => (
                <li
                  key={label}
                  className="flex items-center gap-2 text-sm text-contrast"
                >
                  <Icon
                    className="size-4 shrink-0 text-tertiary-light"
                    aria-hidden
                  />
                  {label}
                </li>
              ))}
            </ul>
            <BrandButton
              type="button"
              variant="tertiary"
              className="mt-auto flex items-center justify-center gap-2"
              testId="built-in-integration-choice-continue-polling"
              onClick={onContinuePolling}
            >
              {t(I18nKey.AUTOMATION_SETUP_CHOICE$POLLING_ACTION)}
              <ArrowRight className="size-4 shrink-0" aria-hidden />
            </BrandButton>
          </div>
        </div>
      </div>
    </ModalBackdrop>
  );
}
