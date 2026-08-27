import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import type { IntegrationCatalogEntry as MarketplaceEntry } from "@openhands/extensions/integrations";
import { RecommendedAutomationsLauncher } from "#/components/features/automations/recommended-automations-launcher";
import { BrandButton } from "#/components/features/settings/brand-button";
import { McpLogoBadge } from "#/components/features/mcp-logo-badge";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { I18nKey } from "#/i18n/declaration";
import { modalTitleLgClassName } from "#/utils/modal-classes";

interface IntegrationAutomationRecommendationsModalProps {
  entry: MarketplaceEntry;
  onClose: () => void;
  returnFocusTo?: HTMLElement | null;
}

/**
 * Optional next steps shown after a marketplace integration is installed.
 * Recommendation membership comes from the automation catalog's integration
 * requirements; this component never owns provider-specific automation ids.
 */
export function IntegrationAutomationRecommendationsModal({
  entry,
  onClose,
  returnFocusTo,
}: IntegrationAutomationRecommendationsModalProps) {
  const { t } = useTranslation("openhands");
  const [isChildOverlayOpen, setIsChildOverlayOpen] = useState(false);
  const skipButtonRef = useRef<HTMLButtonElement>(null);
  const title = t(I18nKey.MCP$AUTOMATION_RECOMMENDATIONS_TITLE, {
    name: entry.name,
  });

  useEffect(() => {
    const previouslyFocused =
      returnFocusTo ??
      (document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null);
    skipButtonRef.current?.focus();

    return () => {
      if (previouslyFocused?.isConnected) previouslyFocused.focus();
    };
  }, [returnFocusTo]);

  return (
    <ModalBackdrop
      onClose={onClose}
      closeOnEscape={!isChildOverlayOpen}
      closeOnBackdropClick={!isChildOverlayOpen}
      aria-label={title}
    >
      <div
        data-testid="integration-automation-recommendations-modal"
        data-integration-id={entry.id}
        className="relative flex max-h-[85vh] w-[min(960px,calc(100vw-2rem))] flex-col rounded-xl border border-[var(--oh-border)] bg-base-secondary"
      >
        <ModalCloseButton
          onClose={onClose}
          testId="integration-automation-recommendations-close"
        />

        <header className="flex flex-shrink-0 items-start gap-3 px-6 pb-3 pt-6">
          <McpLogoBadge entry={entry} size="md" />
          <div className="min-w-0 pr-8">
            <h2 className={modalTitleLgClassName}>{title}</h2>
            <p className="mt-2 text-sm text-muted">
              {t(I18nKey.MCP$AUTOMATION_RECOMMENDATIONS_DESCRIPTION)}
            </p>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-5 custom-scrollbar-always">
          <RecommendedAutomationsLauncher
            integrationId={entry.id}
            onLaunched={onClose}
            onChildOverlayChange={setIsChildOverlayOpen}
          />
        </div>

        <footer className="flex flex-shrink-0 justify-end border-t border-[var(--oh-border)] px-6 py-4">
          <BrandButton
            ref={skipButtonRef}
            type="button"
            variant="secondary"
            testId="integration-automation-recommendations-skip"
            onClick={onClose}
          >
            {t(I18nKey.MCP$AUTOMATION_RECOMMENDATIONS_SKIP)}
          </BrandButton>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
