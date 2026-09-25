import React, { useState } from "react";
import { FileJson, X } from "lucide-react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { parseRcaContext, type RcaContext } from "#/utils/rca-context";
import { Typography } from "#/ui/typography";
import { cn } from "#/utils/utils";
import {
  formControlBorderClassName,
  formControlSurfaceClassName,
  formControlTransitionClassName,
} from "#/utils/form-control-classes";

interface RcaImportControlProps {
  /** Currently-attached RCA context, owned by the conversation store. */
  value: RcaContext | null;
  onChange: (rcaContext: RcaContext | null) => void;
  disabled?: boolean;
}

interface RcaImportModalProps {
  value: RcaContext | null;
  onChange: (rcaContext: RcaContext | null) => void;
  onClose: () => void;
}

function RcaImportModal({ value, onChange, onClose }: RcaImportModalProps) {
  const { t } = useTranslation("openhands");
  const [draft, setDraft] = useState(() =>
    value ? JSON.stringify(value, null, 2) : "",
  );
  const [isInvalid, setIsInvalid] = useState(false);

  const handleApply = () => {
    let parsed: unknown;
    try {
      parsed = JSON.parse(draft);
    } catch {
      setIsInvalid(true);
      return;
    }
    const rcaContext = parseRcaContext(parsed);
    if (!rcaContext) {
      setIsInvalid(true);
      return;
    }
    onChange(rcaContext);
    onClose();
  };

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        data-testid="rca-import-modal"
        className={cn(
          "relative bg-base-secondary p-6 rounded-xl flex flex-col gap-4 border border-border max-h-[80vh]",
          modalWidthClassName("md"),
          MODAL_MAX_WIDTH_VIEWPORT,
        )}
      >
        <ModalCloseButton onClose={onClose} testId="rca-import-close" />
        <Typography.H2 className="pr-6">
          {t(I18nKey.RCA$IMPORT_TITLE)}
        </Typography.H2>
        <p className="text-sm text-contrast">
          {t(I18nKey.RCA$IMPORT_DESCRIPTION)}
        </p>
        <textarea
          data-testid="rca-json-input"
          value={draft}
          onChange={(event) => {
            setDraft(event.target.value);
            setIsInvalid(false);
          }}
          placeholder={t(I18nKey.RCA$IMPORT_PLACEHOLDER)}
          rows={10}
          className="w-full resize-y rounded-md border border-border bg-surface p-3 font-mono text-sm text-foreground"
        />
        {isInvalid && (
          <p data-testid="rca-import-error" className="text-sm text-red-400">
            {t(I18nKey.RCA$IMPORT_INVALID)}
          </p>
        )}
        <div className="flex justify-end gap-3">
          <BrandButton
            testId="rca-import-cancel"
            type="button"
            variant="secondary"
            onClick={onClose}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            testId="rca-import-apply"
            type="button"
            variant="primary"
            onClick={handleApply}
            isDisabled={!draft.trim()}
          >
            {t(I18nKey.RCA$IMPORT_APPLY)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}

/**
 * Pill trigger next to the plugin picker that opens the RCA paste modal, plus
 * a removable chip once a context is attached. Generic by design: the JSON
 * schema is tool-agnostic, so HolmesGPT or any other RCA producer can supply
 * it.
 */
export function RcaImportControl({
  value,
  onChange,
  disabled = false,
}: RcaImportControlProps) {
  const { t } = useTranslation("openhands");
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        data-testid="open-rca-import"
        onClick={() => setIsOpen(true)}
        disabled={disabled}
        className={cn(
          "flex flex-row items-center gap-2 rounded-full px-2.5 py-1 text-contrast",
          formControlBorderClassName,
          formControlSurfaceClassName,
          formControlTransitionClassName,
          disabled
            ? "cursor-not-allowed opacity-50"
            : "cursor-pointer hover:bg-surface-raised",
        )}
      >
        <FileJson size={16} className="shrink-0" aria-hidden />
        <span className="text-sm font-normal leading-5">
          {t(I18nKey.RCA$ADD_CONTEXT)}
        </span>
      </button>

      {value && (
        <span
          data-testid="rca-attached-chip"
          className={cn(
            "flex flex-row items-center gap-1.5 rounded-full px-2.5 py-1 text-xs text-contrast",
            formControlBorderClassName,
            formControlSurfaceClassName,
          )}
        >
          {t(I18nKey.RCA$CONTEXT_ATTACHED)}
          <button
            type="button"
            data-testid="remove-rca-context"
            aria-label={t(I18nKey.COMMON$REMOVE)}
            onClick={() => onChange(null)}
            disabled={disabled}
            className="flex h-3.5 w-3.5 items-center justify-center rounded-full hover:bg-interactive-hover disabled:opacity-50"
          >
            <X size={12} aria-hidden />
          </button>
        </span>
      )}

      {isOpen && (
        <RcaImportModal
          value={value}
          onChange={onChange}
          onClose={() => setIsOpen(false)}
        />
      )}
    </>
  );
}
