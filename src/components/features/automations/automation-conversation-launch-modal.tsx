import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import {
  type AutomationConversationLaunchRequest,
  useLaunchAutomationConversation,
} from "./use-launch-automation-conversation";

interface AutomationConversationLaunchModalProps {
  request: AutomationConversationLaunchRequest | null;
  onClose: () => void;
}

export function AutomationConversationLaunchModal({
  request,
  onClose,
}: AutomationConversationLaunchModalProps) {
  const { t } = useTranslation("openhands");
  const { launchAutomationConversation, isLaunching } =
    useLaunchAutomationConversation();
  const launchedRequestRef = useRef<AutomationConversationLaunchRequest | null>(
    null,
  );

  useEffect(() => {
    if (!request || launchedRequestRef.current === request || isLaunching) {
      return;
    }

    launchedRequestRef.current = request;
    launchAutomationConversation(request, {
      onSuccess: onClose,
      onError: () => {
        displayErrorToast(t(I18nKey.ERROR$GENERIC));
        onClose();
      },
    });
  }, [isLaunching, launchAutomationConversation, onClose, request, t]);

  if (!request) return null;

  return (
    <ModalBackdrop
      onClose={onClose}
      closeOnEscape={!isLaunching}
      closeOnBackdropClick={!isLaunching}
      aria-label={t(I18nKey.HOME$CREATING_CONVERSATION)}
    >
      <div
        data-testid="automation-conversation-launch-modal"
        className="relative flex w-full max-w-md flex-col gap-4 rounded-xl border border-[var(--oh-border)] bg-base-secondary p-6 text-center"
      >
        <ModalCloseButton
          onClose={onClose}
          disabled={isLaunching}
          testId="automation-conversation-launch-close"
        />
        <div className="flex justify-center pt-2">
          <LoadingSpinner size="large" />
        </div>
        <h2 className="pr-6 text-lg font-semibold text-content">
          {t(I18nKey.HOME$CREATING_CONVERSATION)}
        </h2>
      </div>
    </ModalBackdrop>
  );
}
