/**
 * False-positive reason modal.
 * @spec PROJETOSIN-188 — finding-fp-modal
 */

import React from "react";
import {
  Modal,
  ModalBody,
  ModalContent,
  ModalFooter,
  ModalHeader,
} from "@heroui/react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { modalTitleLgMediumClassName } from "#/utils/modal-classes";

interface FindingFpModalProps {
  isOpen: boolean;
  isPending: boolean;
  errorMessage?: string | null;
  onCancel: () => void;
  onSubmit: (reason: string) => void;
}

export function FindingFpModal({
  isOpen,
  isPending,
  errorMessage,
  onCancel,
  onSubmit,
}: FindingFpModalProps) {
  const { t } = useTranslation("openhands");
  const [reason, setReason] = React.useState("");
  const [touched, setTouched] = React.useState(false);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const restoreFocusRef = React.useRef<HTMLElement | null>(null);
  const wasOpenRef = React.useRef(false);
  const trimmed = reason.trim();
  const invalid = touched && trimmed.length === 0;
  const disableAnimation = import.meta.env.MODE === "test";

  // Capture opener during the open transition, before HeroUI Modal moves focus.
  if (isOpen && !wasOpenRef.current) {
    const active = document.activeElement;
    if (active instanceof HTMLElement) {
      restoreFocusRef.current = active;
    }
  }

  React.useLayoutEffect(() => {
    if (!isOpen && wasOpenRef.current) {
      const toRestore = restoreFocusRef.current;
      restoreFocusRef.current = null;
      if (toRestore && document.contains(toRestore)) {
        toRestore.focus();
      }
    }
    wasOpenRef.current = isOpen;
  }, [isOpen]);

  React.useEffect(() => {
    if (!isOpen) {
      setReason("");
      setTouched(false);
      return undefined;
    }
    const frame = requestAnimationFrame(() => {
      textareaRef.current?.focus();
    });
    return () => cancelAnimationFrame(frame);
  }, [isOpen]);

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    setTouched(true);
    if (!trimmed) return;
    onSubmit(trimmed);
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onCancel}
      isDismissable={!isPending}
      isKeyboardDismissDisabled={isPending}
      hideCloseButton
      disableAnimation={disableAnimation}
      size="md"
      placement="center"
      classNames={{
        backdrop: "bg-black/60",
        base: "border border-[var(--oh-border)] bg-[var(--oh-surface)] text-white",
        header: "border-b-0 pb-0",
        body: "py-4",
        footer: "border-t-0 pt-0",
      }}
    >
      <ModalContent>
        {() => (
          <form
            data-testid="finding-fp-modal"
            aria-labelledby="finding-fp-modal-title"
            onSubmit={handleSubmit}
          >
            <ModalHeader>
              <h2
                id="finding-fp-modal-title"
                className={modalTitleLgMediumClassName}
              >
                {t(I18nKey.FINDINGS$FP_MODAL_TITLE)}
              </h2>
            </ModalHeader>
            <ModalBody>
              <label
                htmlFor="finding-fp-reason"
                className="block text-sm text-[var(--oh-text-secondary)]"
              >
                {t(I18nKey.FINDINGS$FP_REASON_LABEL)}
              </label>
              <textarea
                ref={textareaRef}
                id="finding-fp-reason"
                data-testid="finding-fp-reason"
                rows={4}
                value={reason}
                aria-invalid={invalid || undefined}
                aria-describedby={
                  invalid || errorMessage
                    ? "finding-fp-reason-error"
                    : undefined
                }
                placeholder={t(I18nKey.FINDINGS$FP_REASON_PLACEHOLDER)}
                className="mt-1 w-full rounded-md border border-[var(--oh-border)] bg-base-secondary px-3 py-2 text-sm text-white"
                onChange={(event) => setReason(event.target.value)}
                onBlur={() => setTouched(true)}
              />
              {(invalid || errorMessage) && (
                <p
                  id="finding-fp-reason-error"
                  className="mt-1 text-xs text-[var(--oh-color-danger)]"
                  role="alert"
                >
                  {errorMessage ?? t(I18nKey.FINDINGS$FP_REASON_REQUIRED)}
                </p>
              )}
            </ModalBody>
            <ModalFooter className="gap-3">
              <BrandButton
                type="button"
                variant="secondary"
                testId="finding-fp-cancel"
                onClick={onCancel}
                isDisabled={isPending}
              >
                {t(I18nKey.FINDINGS$FP_CANCEL)}
              </BrandButton>
              <BrandButton
                type="submit"
                variant="danger"
                testId="finding-fp-submit"
                isDisabled={isPending || trimmed.length === 0}
                aria-busy={isPending}
              >
                {t(I18nKey.FINDINGS$FP_SUBMIT)}
              </BrandButton>
            </ModalFooter>
          </form>
        )}
      </ModalContent>
    </Modal>
  );
}
