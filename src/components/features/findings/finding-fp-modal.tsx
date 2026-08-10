/**
 * False-positive reason modal.
 * @spec PROJETOSIN-188 — finding-fp-modal
 */

import React from "react";
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
  const trimmed = reason.trim();
  const invalid = touched && trimmed.length === 0;

  React.useEffect(() => {
    if (!isOpen) {
      setReason("");
      setTouched(false);
      return;
    }
    const frame = requestAnimationFrame(() => {
      textareaRef.current?.focus();
    });
    return () => cancelAnimationFrame(frame);
  }, [isOpen]);

  React.useEffect(() => {
    if (!isOpen) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCancel();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isOpen, onCancel]);

  if (!isOpen) return null;

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    setTouched(true);
    if (!trimmed) return;
    onSubmit(trimmed);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div
        className="absolute inset-0 bg-black/60"
        onClick={onCancel}
        role="presentation"
      />
      <form
        data-testid="finding-fp-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="finding-fp-modal-title"
        className="relative z-10 w-full max-w-md rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface)] p-6"
        onSubmit={handleSubmit}
      >
        <h2 id="finding-fp-modal-title" className={modalTitleLgMediumClassName}>
          {t(I18nKey.FINDINGS$FP_MODAL_TITLE)}
        </h2>

        <label
          htmlFor="finding-fp-reason"
          className="mt-4 block text-sm text-[var(--oh-text-secondary)]"
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
            invalid || errorMessage ? "finding-fp-reason-error" : undefined
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

        <div className="mt-6 flex justify-end gap-3">
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
        </div>
      </form>
    </div>
  );
}
