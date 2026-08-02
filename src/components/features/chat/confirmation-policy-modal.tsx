import { useEffect, useState } from "react";
import type { ConfirmationPolicyBase } from "@openhands/typescript-client";
import { useTranslation } from "react-i18next";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";
import {
  getConfirmationPolicySessionScope,
  setSessionConfirmationPolicy,
} from "#/services/confirmation-policy-session";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { cn } from "#/utils/utils";

export type ConfirmationPolicyMode =
  | "always-approve"
  | "always-confirm"
  | "confirm-risky";

const POLICY_BY_MODE: Record<ConfirmationPolicyMode, ConfirmationPolicyBase> = {
  "always-approve": { kind: "NeverConfirm" },
  "always-confirm": { kind: "AlwaysConfirm" },
  "confirm-risky": {
    kind: "ConfirmRisky",
    threshold: "HIGH",
    confirm_unknown: true,
  },
};

export function getConfirmationPolicyMode(
  policy: ConfirmationPolicyBase,
): ConfirmationPolicyMode | null {
  const discriminator = String(policy.kind ?? policy.type ?? "").toLowerCase();
  if (discriminator.includes("never") || discriminator === "never") {
    return "always-approve";
  }
  if (discriminator.includes("risk")) return "confirm-risky";
  if (discriminator.includes("always") || discriminator === "always") {
    return "always-confirm";
  }
  return null;
}

interface ConfirmationPolicyModalProps {
  conversationId: string;
  conversationUrl?: string | null;
  sessionApiKey?: string | null;
  onClose: () => void;
}

export function ConfirmationPolicyModal({
  conversationId,
  conversationUrl,
  sessionApiKey,
  onClose,
}: ConfirmationPolicyModalProps) {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const [currentMode, setCurrentMode] = useState<ConfirmationPolicyMode | null>(
    null,
  );
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    let active = true;
    setIsLoading(true);
    AgentServerConversationService.getConfirmationPolicy(
      conversationId,
      conversationUrl,
      sessionApiKey,
    ).then(
      (policy) => {
        if (!active) return;
        setCurrentMode(getConfirmationPolicyMode(policy));
        setIsLoading(false);
      },
      (error: unknown) => {
        if (!active) return;
        displayErrorToast(
          error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
        );
        onClose();
      },
    );
    return () => {
      active = false;
    };
  }, [conversationId, conversationUrl, onClose, sessionApiKey, t]);

  const selectPolicy = (mode: ConfirmationPolicyMode) => {
    // Capture the invoking connection before the async request begins. A
    // backend switch while the request is in flight must not redirect the
    // successful preference to the newly active backend.
    const sessionScope = getConfirmationPolicySessionScope(backend);
    setIsSaving(true);
    AgentServerConversationService.setConfirmationPolicy(
      conversationId,
      POLICY_BY_MODE[mode],
      conversationUrl,
      sessionApiKey,
    ).then(
      () => {
        setCurrentMode(mode);
        setSessionConfirmationPolicy(sessionScope, POLICY_BY_MODE[mode]);
        displaySuccessToast(
          t(I18nKey.SLASH_COMMAND$CONFIRM_UPDATED, {
            mode: t(
              mode === "always-approve"
                ? I18nKey.SLASH_COMMAND$CONFIRM_ALWAYS_APPROVE
                : mode === "always-confirm"
                  ? I18nKey.SLASH_COMMAND$CONFIRM_EVERY_ACTION
                  : I18nKey.SLASH_COMMAND$CONFIRM_RISKY_ONLY,
            ),
          }),
        );
        onClose();
      },
      (error: unknown) => {
        displayErrorToast(
          error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
        );
        setIsSaving(false);
      },
    );
  };

  const options: Array<{
    mode: ConfirmationPolicyMode;
    label: string;
  }> = [
    {
      mode: "always-approve",
      label: t(I18nKey.SLASH_COMMAND$CONFIRM_ALWAYS_APPROVE),
    },
    {
      mode: "always-confirm",
      label: t(I18nKey.SLASH_COMMAND$CONFIRM_EVERY_ACTION),
    },
    {
      mode: "confirm-risky",
      label: t(I18nKey.SLASH_COMMAND$CONFIRM_RISKY_ONLY),
    },
  ];

  return (
    <ModalBackdrop
      aria-label={t(I18nKey.SCHEMA$CONFIRMATION_MODE$LABEL)}
      onClose={isSaving ? undefined : onClose}
      closeOnEscape={!isSaving}
      closeOnBackdropClick={!isSaving}
    >
      <ModalBody
        testID="confirmation-policy-modal"
        width="sm"
        className="relative items-start border border-[var(--oh-border)] !gap-4"
      >
        <ModalCloseButton
          onClose={onClose}
          disabled={isSaving}
          testId="close-confirmation-policy-modal"
        />
        <div className="flex flex-col gap-2 pr-6">
          <BaseModalTitle title={t(I18nKey.SCHEMA$CONFIRMATION_MODE$LABEL)} />
          <BaseModalDescription>
            {t(I18nKey.SLASH_COMMAND$CONFIRM_MODAL_DESCRIPTION)}
          </BaseModalDescription>
        </div>

        {isLoading ? (
          <div className="flex min-h-28 w-full items-center justify-center">
            <LoadingSpinner size="small" />
          </div>
        ) : (
          <div className="flex w-full flex-col gap-2">
            {options.map(({ mode, label }) => (
              <BrandButton
                key={mode}
                type="button"
                variant="secondary"
                isDisabled={isSaving}
                aria-busy={isSaving}
                aria-pressed={currentMode === mode}
                ariaLabel={label}
                onClick={() => selectPolicy(mode)}
                startContent={
                  <span aria-hidden className="w-4">
                    {currentMode === mode ? "✓" : ""}
                  </span>
                }
                className={cn(
                  "w-full justify-start text-left",
                  currentMode === mode &&
                    "border-primary bg-[var(--oh-interactive-hover)]",
                )}
              >
                {label}
              </BrandButton>
            ))}
          </div>
        )}
      </ModalBody>
    </ModalBackdrop>
  );
}
