import React, { useEffect } from "react";
import { createPortal } from "react-dom";
import { useTranslation } from "react-i18next";
import { TFunction } from "i18next";
import {
  useModalStore,
  ModalConfigMap,
  ModalInstance,
} from "#/stores/modal-store";
import { ConfirmDeleteModal } from "#/components/features/conversation-panel/confirm-delete-modal";
import { ConfirmStopModal } from "#/components/features/conversation-panel/confirm-stop-modal";
import { ExitConversationModal } from "#/components/features/conversation-panel/exit-conversation-modal";
import { SettingsModal } from "#/components/shared/modals/settings/settings-modal";
import { FeedbackModal } from "#/components/features/feedback/feedback-modal";
import { MetricsModal } from "#/components/features/conversation/metrics-modal/metrics-modal";
import { SystemMessageModal } from "#/components/features/conversation-panel/system-message-modal";
import { SkillsModal } from "#/components/features/conversation-panel/skills-modal";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { CreateApiKeyModal } from "#/components/features/settings/create-api-key-modal";
import { DeleteApiKeyModal } from "#/components/features/settings/delete-api-key-modal";
import { NewApiKeyModal } from "#/components/features/settings/new-api-key-modal";
import { LaunchMicroagentModal } from "#/components/features/chat/microagent/launch-microagent-modal";
import { MicroagentManagementLearnThisRepoModal } from "#/components/features/microagent-management/microagent-management-learn-this-repo-modal";
import { MicroagentManagementUpsertMicroagentModal } from "#/components/features/microagent-management/microagent-management-upsert-microagent-modal";
import { ConfigureModal } from "#/components/features/settings/project-management/configure-modal";
import { AuthModal } from "#/components/features/waitlist/auth-modal";
import { ReauthModal } from "#/components/features/waitlist/reauth-modal";
import { EmailVerificationModal } from "#/components/features/waitlist/email-verification-modal";
import { AnalyticsConsentFormModal } from "#/components/features/analytics/analytics-consent-form-modal";
import { SetupPaymentModal } from "#/components/features/payment/setup-payment-modal";
import { CancelSubscriptionModal } from "#/components/features/payment/cancel-subscription-modal";
import { DangerModal } from "#/components/shared/modals/confirmation-modals/danger-modal";
import { I18nKey } from "#/i18n/declaration";

function renderModal(modal: ModalInstance, onClose: () => void, t: TFunction) {
  switch (modal.type) {
    case "confirm-delete": {
      const props = modal.props as ModalConfigMap["confirm-delete"];
      return (
        <ConfirmDeleteModal
          conversationTitle={props.conversationTitle}
          onConfirm={props.onConfirm}
          onClose={onClose}
        />
      );
    }
    case "confirm-stop": {
      const props = modal.props as ModalConfigMap["confirm-stop"];
      return <ConfirmStopModal onConfirm={props.onConfirm} onClose={onClose} />;
    }
    case "exit-conversation": {
      const props = modal.props as ModalConfigMap["exit-conversation"];
      return (
        <ExitConversationModal onConfirm={props.onConfirm} onClose={onClose} />
      );
    }
    case "settings": {
      const props = modal.props as ModalConfigMap["settings"];
      return <SettingsModal settings={props.settings} onClose={onClose} />;
    }
    case "feedback": {
      const props = modal.props as ModalConfigMap["feedback"];
      return <FeedbackModal polarity={props.polarity} onClose={onClose} />;
    }
    case "metrics": {
      return <MetricsModal />;
    }
    case "system-message": {
      const props = modal.props as ModalConfigMap["system-message"];
      return <SystemMessageModal systemMessage={props.systemMessage} />;
    }
    case "skills": {
      return <SkillsModal />;
    }
    case "confirmation": {
      const props = modal.props as ModalConfigMap["confirmation"];
      return (
        <ConfirmationModal
          text={props.text}
          onConfirm={props.onConfirm}
          onClose={onClose}
        />
      );
    }
    case "create-api-key": {
      const props = modal.props as ModalConfigMap["create-api-key"];
      return (
        <CreateApiKeyModal
          onKeyCreated={props.onKeyCreated}
          onClose={onClose}
        />
      );
    }
    case "delete-api-key": {
      const props = modal.props as ModalConfigMap["delete-api-key"];
      return (
        <DeleteApiKeyModal
          keyToDelete={props.keyToDelete}
          onDeleted={props.onDeleted}
          onClose={onClose}
        />
      );
    }
    case "new-api-key": {
      const props = modal.props as ModalConfigMap["new-api-key"];
      return (
        <NewApiKeyModal
          newlyCreatedKey={props.newlyCreatedKey}
          onClose={onClose}
        />
      );
    }
    case "launch-microagent": {
      const props = modal.props as ModalConfigMap["launch-microagent"];
      return (
        <LaunchMicroagentModal
          eventId={props.eventId}
          selectedRepo={props.selectedRepo}
          onLaunch={props.onLaunch}
          isLoading={props.isLoading}
          onClose={onClose}
        />
      );
    }
    case "learn-this-repo": {
      const props = modal.props as ModalConfigMap["learn-this-repo"];
      return (
        <MicroagentManagementLearnThisRepoModal
          onConfirm={props.onConfirm}
          onCancel={onClose}
          isLoading={props.isLoading}
        />
      );
    }
    case "upsert-microagent": {
      const props = modal.props as ModalConfigMap["upsert-microagent"];
      return (
        <MicroagentManagementUpsertMicroagentModal
          onConfirm={props.onConfirm}
          onCancel={onClose}
          isLoading={props.isLoading}
          isUpdate={props.isUpdate}
        />
      );
    }
    case "end-session": {
      const props = modal.props as ModalConfigMap["end-session"];
      return (
        <DangerModal
          title={t(I18nKey.MODAL$END_SESSION_TITLE)}
          description={t(I18nKey.MODAL$END_SESSION_MESSAGE)}
          buttons={{
            danger: {
              text: t(I18nKey.BUTTON$END_SESSION),
              onClick: () => {
                props.onConfirm();
                onClose();
              },
            },
            cancel: {
              text: t(I18nKey.BUTTON$CANCEL),
              onClick: onClose,
            },
          }}
        />
      );
    }
    case "configure-integration": {
      const props = modal.props as ModalConfigMap["configure-integration"];
      return (
        <ConfigureModal
          platform={props.platform}
          platformName={props.platformName}
          integrationData={props.integrationData}
          onConfirm={props.onConfirm}
          onLink={props.onLink}
          onUnlink={props.onUnlink}
          onClose={onClose}
        />
      );
    }
    case "auth": {
      const props = modal.props as ModalConfigMap["auth"];
      return (
        <AuthModal
          githubAuthUrl={props.githubAuthUrl}
          appMode={props.appMode}
          authUrl={props.authUrl}
          providersConfigured={props.providersConfigured}
          emailVerified={props.emailVerified}
          hasDuplicatedEmail={props.hasDuplicatedEmail}
        />
      );
    }
    case "reauth": {
      return <ReauthModal />;
    }
    case "email-verification": {
      const props = modal.props as ModalConfigMap["email-verification"];
      return <EmailVerificationModal userId={props.userId} />;
    }
    case "analytics-consent": {
      return <AnalyticsConsentFormModal onClose={onClose} />;
    }
    case "setup-payment": {
      return <SetupPaymentModal />;
    }
    case "cancel-subscription": {
      const props = modal.props as ModalConfigMap["cancel-subscription"];
      return (
        <CancelSubscriptionModal endDate={props.endDate} onClose={onClose} />
      );
    }
    default:
      return null;
  }
}

export function ModalRoot() {
  const { t } = useTranslation();
  const { modalStack, closeModal, topModal } = useModalStore();

  // Handle ESC key - close topmost modal (if allowed)
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      const top = topModal();
      if (e.key === "Escape" && top && top.props.closeOnEscape !== false) {
        closeModal();
      }
    };

    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [topModal, closeModal]);

  if (modalStack.length === 0) return null;

  const portalRoot = document.getElementById("modal-portal-exit");
  if (!portalRoot) return null;

  return createPortal(
    <>
      {modalStack.map((modal, index) => {
        const isTopModal = index === modalStack.length - 1;
        // Higher base z-index (1000) to stay above toasts, command palette, etc.
        const zIndex = 1000 + index * 10;
        const allowBackdropClose = modal.props.closeOnBackdrop !== false;

        const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
          if (
            e.target === e.currentTarget &&
            isTopModal &&
            allowBackdropClose
          ) {
            closeModal();
          }
        };

        return (
          <div
            key={modal.id}
            className="fixed inset-0 flex items-center justify-center"
            style={{ zIndex }}
          >
            <div
              className="fixed inset-0 bg-black opacity-60"
              onClick={handleBackdropClick}
            />
            <div className="relative">{renderModal(modal, closeModal, t)}</div>
          </div>
        );
      })}
    </>,
    portalRoot,
  );
}
