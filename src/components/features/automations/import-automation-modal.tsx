import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { AutomationSpec } from "#/types/automation";
import { formatEventOn } from "#/utils/automation-schedule";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";

interface ImportAutomationModalProps {
  isOpen: boolean;
  spec: AutomationSpec | null;
  isImporting: boolean;
  onClose: () => void;
  onImport: () => void;
}

function PreviewField({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="text-xs font-medium uppercase tracking-wide text-muted">
        {label}
      </dt>
      <dd className="whitespace-pre-wrap break-words text-sm text-content">
        {value}
      </dd>
    </div>
  );
}

export function ImportAutomationModal({
  isOpen,
  spec,
  isImporting,
  onClose,
  onImport,
}: ImportAutomationModalProps) {
  const { t } = useTranslation("openhands");

  if (!isOpen || !spec) return null;

  const trigger =
    spec.trigger.type === "event"
      ? [spec.trigger.source, formatEventOn(spec.trigger.on)]
          .filter(Boolean)
          .join(": ")
      : [
          spec.trigger.schedule_human ?? spec.trigger.schedule,
          spec.timezone ?? spec.trigger.timezone,
        ]
          .filter(Boolean)
          .join(" · ");

  return (
    <ModalBackdrop
      onClose={isImporting ? undefined : onClose}
      closeOnEscape={!isImporting}
      closeOnBackdropClick={!isImporting}
      aria-label={t(I18nKey.AUTOMATIONS$IMPORT)}
    >
      <div
        data-testid="import-automation-modal"
        className="relative flex max-h-[85vh] w-[min(36rem,calc(100vw-2rem))] flex-col rounded-xl border border-[var(--oh-border)] bg-base-secondary"
      >
        <ModalCloseButton
          onClose={onClose}
          testId="import-automation-modal-close"
          disabled={isImporting}
        />
        <header className="px-6 pb-4 pt-6">
          <h2 className={cn("pr-6", modalTitleLgClassName)}>
            {t(I18nKey.AUTOMATIONS$IMPORT)}
          </h2>
          <p className="mt-2 text-sm text-tertiary-light">
            {t(I18nKey.AUTOMATIONS$IMPORT_PREVIEW_DESCRIPTION)}
          </p>
        </header>

        <dl className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto border-y border-[var(--oh-border)] px-6 py-5">
          <PreviewField label={t(I18nKey.AUTOMATIONS$NAME)} value={spec.name} />
          <PreviewField
            label={t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER)}
            value={trigger}
          />
          <PreviewField
            label={t(I18nKey.AUTOMATIONS$PROMPT)}
            value={spec.prompt ?? ""}
          />
          {spec.plugins && spec.plugins.length > 0 ? (
            <PreviewField
              label={t(I18nKey.AUTOMATIONS$DETAIL$PLUGINS)}
              value={spec.plugins.join(", ")}
            />
          ) : null}
        </dl>

        <footer className="px-6 py-5">
          <p className="mb-4 text-sm text-tertiary-light">
            {t(I18nKey.AUTOMATIONS$IMPORT_DISABLED_NOTICE)}
          </p>
          <div className="flex justify-end gap-3">
            <BrandButton
              testId="import-automation-cancel"
              type="button"
              variant="secondary"
              onClick={onClose}
              isDisabled={isImporting}
            >
              {t(I18nKey.AUTOMATIONS$CANCEL)}
            </BrandButton>
            <BrandButton
              testId="import-automation-confirm"
              type="button"
              variant="primary"
              onClick={onImport}
              isDisabled={isImporting}
              aria-busy={isImporting}
            >
              {isImporting
                ? t(I18nKey.AUTOMATIONS$IMPORTING)
                : t(I18nKey.AUTOMATIONS$IMPORT)}
            </BrandButton>
          </div>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
