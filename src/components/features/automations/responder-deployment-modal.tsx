import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import {
  VISIBLE_RESPONDER_DEPLOYMENT_TARGETS,
  resolveResponderDeploymentOption,
} from "#/utils/responder-deployment";

interface ResponderDeploymentModalProps {
  isOpen: boolean;
  onClose: () => void;
  /** Fired for the "Continue with local setup" action. */
  onContinueLocal: () => void;
  /** Fired for an external-URL action (e.g. OpenHands Cloud integrations). */
  onOpenUrl: (url: string) => void;
}

export function ResponderDeploymentModal({
  isOpen,
  onClose,
  onContinueLocal,
  onOpenUrl,
}: ResponderDeploymentModalProps) {
  const { t } = useTranslation("openhands");

  if (!isOpen) return null;

  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.RESPONDER_DEPLOYMENT$TITLE)}
    >
      <div
        data-testid="responder-deployment-modal"
        className="relative flex w-full max-w-3xl flex-col rounded-xl border border-[var(--oh-border)] bg-base-secondary"
      >
        <ModalCloseButton
          onClose={onClose}
          testId="responder-deployment-modal-close"
        />
        <header className="flex-shrink-0 px-6 pb-4 pt-6">
          <h2 className={cn("pr-6", modalTitleLgClassName)}>
            {t(I18nKey.RESPONDER_DEPLOYMENT$TITLE)}
          </h2>
          <p className="mt-2 text-sm text-muted">
            {t(I18nKey.RESPONDER_DEPLOYMENT$DESCRIPTION)}
          </p>
        </header>
        <div className="flex flex-col gap-3 px-6 pb-6 sm:flex-row">
          {VISIBLE_RESPONDER_DEPLOYMENT_TARGETS.map((target) => {
            const option = resolveResponderDeploymentOption(target);
            const { action } = option;
            return (
              <div
                key={option.target}
                data-testid={option.testId}
                className="flex flex-1 flex-col gap-3 rounded-xl border border-[var(--oh-border)] bg-surface-raised p-4"
              >
                <div className="flex flex-col gap-1">
                  <h3 className="text-sm font-semibold text-white">
                    {t(option.titleKey)}
                  </h3>
                  <p className="text-xs leading-relaxed text-tertiary-light">
                    {t(option.descriptionKey)}
                  </p>
                </div>
                <BrandButton
                  type="button"
                  variant="primary"
                  className="mt-auto"
                  testId={option.primaryActionTestId}
                  onClick={() => {
                    if (action.kind === "launch-local") {
                      onContinueLocal();
                    } else {
                      onOpenUrl(action.url);
                    }
                  }}
                >
                  {t(option.primaryActionKey)}
                </BrandButton>
              </div>
            );
          })}
        </div>
      </div>
    </ModalBackdrop>
  );
}
