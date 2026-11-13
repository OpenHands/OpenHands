import React from "react";
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
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import { pillToggleClassName } from "#/utils/pill-toggle-classes";

/**
 * Whether the current user has already connected the built-in integration
 * (`connected`) or still has to (`connect`).
 */
export type BuiltInIntegrationState = "connected" | "connect";

/**
 * How a connected user asks OpenHands for work from inside the integration.
 * An instruction to the agent rather than UI copy, so it is intentionally in
 * English and not localized.
 */
const USAGE_EXAMPLE = "@openhands Please complete this task!";

interface BuiltInIntegrationChoiceModalProps {
  /** The built-in integration that already covers the selected automation. */
  integration: IntegrationCatalogEntry;
  state: BuiltInIntegrationState;
  onClose: () => void;
  /** Fired for the "Connect to the integration" action. */
  onUseBuiltIn: () => void;
  /** Fired for the "Continue with polling setup" action. */
  onContinuePolling: () => void;
}

/**
 * Shown before an automation is set up on a cloud instance whose built-in
 * integration (Settings > Integrations) provides the same workflow. A user
 * who has connected that integration is shown how to use it; one who has not
 * is offered to connect it. The polling automation is the alternative either
 * way.
 */
export function BuiltInIntegrationChoiceModal({
  integration,
  state,
  onClose,
  onUseBuiltIn,
  onContinuePolling,
}: BuiltInIntegrationChoiceModalProps) {
  const { t } = useTranslation("openhands");
  const { name } = integration;
  const isConnected = state === "connected";
  const [selectedTab, setSelectedTab] = React.useState<"built-in" | "polling">(
    "built-in",
  );

  const tabs = [
    {
      id: "built-in" as const,
      label: t(I18nKey.AUTOMATION_SETUP_CHOICE$BUILT_IN_TAB),
      recommended: true,
    },
    {
      id: "polling" as const,
      label: t(I18nKey.AUTOMATION_SETUP_CHOICE$MCP_TAB),
      recommended: false,
    },
  ];
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
        className="relative flex setup-dialog-width max-w-xl flex-col rounded-xl border border-border bg-base-secondary"
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
            {t(
              isConnected
                ? I18nKey.AUTOMATION_SETUP_CHOICE$CONNECTED_DESCRIPTION
                : I18nKey.AUTOMATION_SETUP_CHOICE$DESCRIPTION,
              { name },
            )}
          </p>
        </header>
        <div
          role="tablist"
          aria-label={t(I18nKey.AUTOMATION_SETUP_CHOICE$TITLE)}
          className="flex w-full flex-wrap items-center gap-2 px-6"
        >
          {tabs.map(({ id, label, recommended }) => {
            const isActive = selectedTab === id;
            return (
              <button
                key={id}
                type="button"
                role="tab"
                id={`built-in-integration-choice-tab-${id}`}
                aria-selected={isActive}
                aria-controls={`built-in-integration-choice-panel-${id}`}
                data-testid={`built-in-integration-choice-tab-${id}`}
                onClick={() => setSelectedTab(id)}
                className={cn(
                  pillToggleClassName(isActive),
                  "flex-1 justify-center text-center",
                )}
              >
                {label}
                {recommended && (
                  <span
                    className={cn(
                      "shrink-0 rounded-full px-2 py-0.5 text-xs font-medium leading-none",
                      isActive
                        ? "bg-contrast-foreground text-contrast"
                        : "bg-contrast text-contrast-foreground",
                    )}
                  >
                    {t(I18nKey.SETTINGS$SKILLS_RECOMMENDED)}
                  </span>
                )}
              </button>
            );
          })}
        </div>
        <div
          role="tabpanel"
          id={`built-in-integration-choice-panel-${selectedTab}`}
          aria-labelledby={`built-in-integration-choice-tab-${selectedTab}`}
          className="px-6 pb-6 pt-4"
        >
          {selectedTab === "built-in" ? (
            <div
              data-testid="built-in-integration-choice-option-built-in"
              className="flex flex-col gap-4 rounded-xl border border-border bg-surface-raised p-4"
            >
              <div className="flex items-start gap-3">
                <McpLogoBadge entry={integration} />
                <div className="flex flex-col gap-1">
                  <h3 className="text-sm font-semibold text-contrast">
                    {name}
                  </h3>
                  <p className="text-xs leading-relaxed text-tertiary-light">
                    {t(I18nKey.AUTOMATION_SETUP_CHOICE$BUILT_IN_DESCRIPTION, {
                      name,
                    })}
                  </p>
                </div>
              </div>
              {isConnected ? (
                <code
                  data-testid="built-in-integration-choice-example"
                  className="block rounded-md border border-border-subtle bg-base-secondary px-3 py-2 font-mono text-xs text-contrast"
                >
                  {USAGE_EXAMPLE}
                </code>
              ) : (
                <ul className="flex flex-col gap-2 py-3">
                  {builtInFeatures.map((feature) => (
                    <li
                      key={feature}
                      className="flex items-center gap-2 text-sm text-contrast"
                    >
                      <CircleCheck
                        className="size-4 shrink-0 text-success"
                        aria-hidden
                      />
                      {feature}
                    </li>
                  ))}
                </ul>
              )}
              <div className="flex justify-end border-t border-border pt-4">
                <BrandButton
                  type="button"
                  variant="primary"
                  className="flex items-center justify-center gap-2 bg-contrast text-contrast-foreground"
                  testId="built-in-integration-choice-use-built-in"
                  onClick={isConnected ? onClose : onUseBuiltIn}
                >
                  {isConnected
                    ? t(I18nKey.BUTTON$CLOSE)
                    : t(I18nKey.AUTOMATION_SETUP_CHOICE$CONNECT_ACTION, {
                        name,
                      })}
                  {!isConnected && (
                    <ArrowRight className="size-4 shrink-0" aria-hidden />
                  )}
                </BrandButton>
              </div>
            </div>
          ) : (
            <div
              data-testid="built-in-integration-choice-option-polling"
              className="flex flex-col gap-4 rounded-xl border border-border bg-surface-raised p-4"
            >
              <div className="flex items-start gap-3">
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
              </div>
              <ul className="flex flex-col gap-2 py-3">
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
              <div className="flex justify-end border-t border-border pt-4">
                <BrandButton
                  type="button"
                  variant="tertiary"
                  className="flex items-center justify-center gap-2"
                  testId="built-in-integration-choice-continue-polling"
                  onClick={onContinuePolling}
                >
                  {t(I18nKey.AUTOMATION_SETUP_CHOICE$POLLING_ACTION)}
                  <ArrowRight className="size-4 shrink-0" aria-hidden />
                </BrandButton>
              </div>
            </div>
          )}
        </div>
      </div>
    </ModalBackdrop>
  );
}
