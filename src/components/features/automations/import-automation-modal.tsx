import { useRef, useState, type DragEvent } from "react";
import { FileUp } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { AUTOMATION_FILE_FORMAT_DOCS_URL } from "#/manifests/automation-interface";
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
  onFile: (file: File) => void;
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

function takeDroppedFile(event: DragEvent<HTMLElement>): File | null {
  const files = event.dataTransfer?.files;
  return files && files.length > 0 ? files[0]! : null;
}

function ImportAutomationPicker({ onFile }: { onFile: (file: File) => void }) {
  const { t } = useTranslation("openhands");
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const openFinder = () => inputRef.current?.click();

  return (
    <div className="flex flex-col gap-5 px-6 pb-6">
      <p className="text-sm leading-relaxed text-tertiary-light">
        {t(I18nKey.AUTOMATIONS$IMPORT_EXPLAIN)}{" "}
        <a
          href={AUTOMATION_FILE_FORMAT_DOCS_URL}
          target="_blank"
          rel="noopener noreferrer"
          data-testid="import-automation-format-docs"
          className="text-white no-underline transition-colors hover:text-white"
        >
          {t(I18nKey.AUTOMATIONS$IMPORT_FORMAT_DOCS)}
        </a>
      </p>
      <p className="text-sm text-tertiary-light">
        {t(I18nKey.AUTOMATIONS$IMPORT_DISABLED_NOTICE)}
      </p>

      <input
        ref={inputRef}
        type="file"
        accept="application/json,.json"
        className="hidden"
        data-testid="automations-import-file"
        onChange={() => {
          const input = inputRef.current;
          const file = input?.files?.[0];
          if (input) {
            input.value = "";
          }
          if (file) onFile(file);
        }}
      />

      <div
        data-testid="import-automation-dropzone"
        data-active={isDragging ? "true" : "false"}
        onClick={openFinder}
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={(event) => {
          if (event.currentTarget.contains(event.relatedTarget as Node)) {
            return;
          }
          setIsDragging(false);
        }}
        onDrop={(event) => {
          event.preventDefault();
          setIsDragging(false);
          const file = takeDroppedFile(event);
          if (file) onFile(file);
        }}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border border-dashed px-6 py-10 text-center",
          isDragging
            ? "border-[var(--oh-focus)] bg-[var(--oh-interactive-hover)]"
            : "border-[var(--oh-border)] bg-[var(--oh-surface)]",
        )}
      >
        <FileUp className="size-8 text-[var(--oh-muted)]" aria-hidden />
        <p className="text-sm font-medium text-content">
          {t(I18nKey.AUTOMATIONS$IMPORT_DROPZONE)}
        </p>
        <p className="text-xs text-tertiary-light">
          {t(I18nKey.AUTOMATIONS$IMPORT_OR)}
        </p>
        <BrandButton
          testId="import-automation-choose-file"
          type="button"
          variant="secondary"
          onClick={(event) => {
            event?.stopPropagation();
            openFinder();
          }}
        >
          {t(I18nKey.AUTOMATIONS$IMPORT_CHOOSE_FILE)}
        </BrandButton>
      </div>
    </div>
  );
}

export function ImportAutomationModal({
  isOpen,
  spec,
  isImporting,
  onClose,
  onImport,
  onFile,
}: ImportAutomationModalProps) {
  const { t } = useTranslation("openhands");

  if (!isOpen) return null;

  const trigger = spec
    ? spec.trigger.type === "event"
      ? [spec.trigger.source, formatEventOn(spec.trigger.on)]
          .filter(Boolean)
          .join(": ")
      : [
          spec.trigger.schedule_human ?? spec.trigger.schedule,
          spec.timezone ?? spec.trigger.timezone,
        ]
          .filter(Boolean)
          .join(" · ")
    : "";

  return (
    <ModalBackdrop
      onClose={isImporting ? undefined : onClose}
      closeOnEscape={!isImporting}
      closeOnBackdropClick={!isImporting}
      aria-label={t(I18nKey.AUTOMATIONS$IMPORT)}
    >
      <div
        data-testid="import-automation-modal"
        data-view={spec ? "preview" : "picker"}
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
          {spec ? (
            <p className="mt-2 text-sm text-tertiary-light">
              {t(I18nKey.AUTOMATIONS$IMPORT_PREVIEW_DESCRIPTION)}
            </p>
          ) : null}
        </header>

        {spec ? (
          <>
            <dl className="mx-6 flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface)] px-4 py-4">
              <PreviewField
                label={t(I18nKey.AUTOMATIONS$NAME)}
                value={spec.name}
              />
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
          </>
        ) : (
          <ImportAutomationPicker onFile={onFile} />
        )}
      </div>
    </ModalBackdrop>
  );
}
