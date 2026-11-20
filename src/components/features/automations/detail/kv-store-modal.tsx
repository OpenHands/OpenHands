import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import XMarkIcon from "#/icons/x-mark.svg?react";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { useAutomationKv } from "#/hooks/query/use-automation-kv";
import type { AutomationKvEntry } from "#/types/automation";
import { formatDate } from "#/utils/format-relative-time";
import { modalTitleLgMediumClassName } from "#/utils/modal-classes";

interface KvStoreModalProps {
  automationId: string;
  onClose: () => void;
}

function KvEntryRow({ entry }: { entry: AutomationKvEntry }) {
  const { t, i18n } = useTranslation("openhands");
  return (
    <li
      data-testid="automation-kv-entry"
      className="flex flex-col gap-2 rounded-lg border border-border p-3"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="break-all font-mono text-sm font-medium text-contrast">
          {entry.key}
        </span>
        <span className="text-xs text-muted">
          {t(I18nKey.AUTOMATIONS$DETAIL$KV_UPDATED, {
            date: formatDate(entry.updated_at, i18n.language),
          })}
        </span>
      </div>
      <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md bg-surface-deep p-2 font-mono text-xs text-content dark:bg-black/30">
        {JSON.stringify(entry.value, null, 2)}
      </pre>
    </li>
  );
}

function KvStoreContent({ automationId }: { automationId: string }) {
  const { t } = useTranslation("openhands");
  const { data, isError } = useAutomationKv(automationId, true);

  if (isError) {
    return (
      <p role="alert" className="text-sm text-danger">
        {t(I18nKey.AUTOMATIONS$DETAIL$KV_ERROR)}
      </p>
    );
  }
  if (!data) {
    return (
      <p className="text-sm italic text-muted">
        {t(I18nKey.AUTOMATIONS$DETAIL$KV_LOADING)}
      </p>
    );
  }
  if (data.entries.length === 0) {
    return (
      <p className="text-sm italic text-muted">
        {t(I18nKey.AUTOMATIONS$DETAIL$KV_EMPTY)}
      </p>
    );
  }
  return (
    <>
      {data.total > data.entries.length ? (
        <p className="mb-3 text-xs text-muted">
          {t(I18nKey.AUTOMATIONS$DETAIL$KV_TRUNCATED, {
            shown: data.entries.length,
            total: data.total,
          })}
        </p>
      ) : null}
      <ul className="flex flex-col gap-3">
        {data.entries.map((entry) => (
          <KvEntryRow key={entry.key} entry={entry} />
        ))}
      </ul>
    </>
  );
}

/** Read-only view of an automation's key-value store, loaded when it opens. */
export function KvStoreModal({ automationId, onClose }: KvStoreModalProps) {
  const { t } = useTranslation("openhands");
  const title = t(I18nKey.AUTOMATIONS$DETAIL$KV_STORE);

  return (
    <ModalBackdrop onClose={onClose} aria-label={title}>
      <div
        data-testid="automation-kv-modal"
        className="flex dialog-content-height w-full max-w-3xl flex-col overflow-y-auto overscroll-contain rounded-xl border border-border bg-surface"
      >
        <div className="sticky top-0 z-10 flex shrink-0 items-center justify-between gap-4 bg-surface px-6 pb-4 pt-6">
          <h2 className={modalTitleLgMediumClassName}>{title}</h2>
          <button
            type="button"
            onClick={onClose}
            className="text-muted hover:text-foreground"
            aria-label={t(I18nKey.BUTTON$CLOSE)}
          >
            <XMarkIcon className="size-5" />
          </button>
        </div>
        <div className="px-6 pb-6">
          <KvStoreContent automationId={automationId} />
        </div>
      </div>
    </ModalBackdrop>
  );
}
