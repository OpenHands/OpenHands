import { useTranslation } from "react-i18next";
import { SecretForm } from "#/components/features/settings/secrets-settings/secret-form";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  ModalBody,
} from "#/components/shared/modals/modal-body";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import { I18nKey } from "#/i18n/declaration";
import { completeSecretRequest } from "#/services/secret-request";
import { useSecretRequestStore } from "#/stores/secret-request-store";

/**
 * Collects a secret the agent asked for with `request_secret`, over the
 * current conversation. The value goes straight to the secrets store; the
 * agent only learns the outcome.
 */
export function SecretRequestModal() {
  const { t } = useTranslation("openhands");
  const { conversationId } = useOptionalConversationId();
  const request = useSecretRequestStore((s) =>
    s.requests.find((r) => r.conversationId === conversationId),
  );

  if (!request) return null;

  const title = t(I18nKey.SECRETS$REQUEST_TITLE);

  return (
    <ModalBackdrop
      closeOnEscape={false}
      closeOnBackdropClick={false}
      aria-label={title}
    >
      <ModalBody
        testID="secret-request-modal"
        width="md"
        className={MODAL_MAX_WIDTH_VIEWPORT}
      >
        <div className="flex flex-col gap-2 self-start">
          <BaseModalTitle title={title} />
          {request.reason ? (
            <BaseModalDescription description={request.reason} />
          ) : null}
        </div>
        <div className="w-full">
          {/* SecretForm also calls onCancel after a successful save; the
              request is already complete by then, so that call is a no-op. */}
          <SecretForm
            key={request.toolCallId}
            mode="add"
            selectedSecret={null}
            defaultName={request.name}
            keepOpenOnError
            onSaved={(secret) => void completeSecretRequest(request, secret)}
            onCancel={() => void completeSecretRequest(request, null)}
          />
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
