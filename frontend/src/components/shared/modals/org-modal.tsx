import React from "react";
import { ModalBackdrop } from "./modal-backdrop";
import { ModalBody } from "./modal-body";
import {
  BaseModalTitle,
  BaseModalDescription,
} from "./confirmation-modals/base-modal";
import { ModalButtonGroup } from "./modal-button-group";

interface OrgModalProps {
  testId?: string;
  title: string;
  description?: React.ReactNode;
  children?: React.ReactNode;
  primaryButtonText: string;
  secondaryButtonText?: string;
  onPrimaryClick?: () => void;
  onClose: () => void;
  isLoading?: boolean;
  primaryButtonType?: "button" | "submit";
  primaryButtonTestId?: string;
  secondaryButtonTestId?: string;
  ariaLabel?: string;
  asForm?: boolean;
  formAction?: (formData: FormData) => void;
  fullWidthButtons?: boolean;
}

export function OrgModal({
  testId,
  title,
  description,
  children,
  primaryButtonText,
  secondaryButtonText,
  onPrimaryClick,
  onClose,
  isLoading = false,
  primaryButtonType = "button",
  primaryButtonTestId,
  secondaryButtonTestId,
  ariaLabel,
  asForm = false,
  formAction,
  fullWidthButtons = false,
}: OrgModalProps) {
  const content = (
    <>
      <div className="flex flex-col gap-2 w-full">
        <BaseModalTitle title={title} />
        {description && (
          <BaseModalDescription>{description}</BaseModalDescription>
        )}
        {children}
      </div>
      <ModalButtonGroup
        primaryText={primaryButtonText}
        secondaryText={secondaryButtonText}
        onPrimaryClick={onPrimaryClick}
        onSecondaryClick={onClose}
        isLoading={isLoading}
        primaryType={primaryButtonType}
        primaryTestId={primaryButtonTestId}
        secondaryTestId={secondaryButtonTestId}
        fullWidth={fullWidthButtons}
      />
    </>
  );

  const modalBodyClassName =
    "items-start rounded-xl p-6 w-sm flex flex-col gap-6 bg-modal-background modal-box-shadow";

  return (
    <ModalBackdrop
      onClose={isLoading ? undefined : onClose}
      aria-label={ariaLabel}
    >
      {asForm ? (
        <form
          data-testid={testId}
          action={formAction}
          noValidate
          className={modalBodyClassName}
        >
          {content}
        </form>
      ) : (
        <ModalBody testID={testId} className={modalBodyClassName}>
          {content}
        </ModalBody>
      )}
    </ModalBackdrop>
  );
}
